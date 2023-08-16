"""Meters."""

from typing import List
from collections import deque

import torch
import numpy as np
from yacs.config import CfgNode
from torch.utils.tensorboard import SummaryWriter

from ltc.utils.metrics import calculate_metrics
import ltc.utils.misc as misc
import ltc.utils.plot_utils as plot_utils

import ltc.utils.logging as logging
logger = logging.get_logger(__name__)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size: int):
        """

        :param window_size:
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value: float):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self,
                 epoch_iters: int,
                 cfg: CfgNode,
                 global_step: int,
                 writer: SummaryWriter = None):
        """

        :param epoch_iters: the overall number of iterations of one epoch
        :param cfg:
        :param global_step:
        :param writer:
        """
        self._cfg = cfg
        self.writer = writer
        self.epoch_iters = epoch_iters

        self.loss_dict = {
            "loss": ScalarMeter(cfg.LOG_PERIOD),
            "loss_ce": ScalarMeter(cfg.LOG_PERIOD),
            "loss_mse": ScalarMeter(cfg.LOG_PERIOD)
        }

        self.loss_sums = {
            "loss": 0.0,
            "loss_ce": 0.0,
            "loss_mse": 0.0
        }

        self._iter_log_window = cfg.LOG_PERIOD
        self._ignore_idx = cfg.MODEL.PAD_IGNORE_IDX
        if isinstance(cfg.MODEL.PAD_IGNORE_IDX, int):
            self._ignore_idx = [cfg.MODEL.PAD_IGNORE_IDX]
        self._ignore_idx = self._ignore_idx + cfg.TRAIN.EVAL_IGNORE_LABELS
        logger.info(f"Ignored label idxs: {self._ignore_idx}")

        self.metrics_dict = {
            "MoF": [],
            "Edit": [],
            "F1@10": [],
            "F1@25": [],
            "F1@50": [],
        }
        self.num_samples = 0
        self.lr = None
        self.global_iter = global_step

    def reset(self):
        """
        Reset the Meter.
        """
        for loss in self.loss_dict.values():
            loss.reset()
        for name in self.loss_sums.keys():
            self.loss_sums[name] = 0.0
        self.metrics_dict = {name: [] for name in self.metrics_dict.keys()}
        self.lr = None
        self.num_samples = 0

    def update_stats(self,
                     target: torch.Tensor,
                     prediction: torch.Tensor,
                     loss_dict: dict,
                     lr: float,
                     ):
        """

        :param target:
        :param prediction:
        :param loss_dict:
        :param lr:
        :return:
        """
        mb_size = target.shape[0]
        target = target.detach().cpu()
        prediction = prediction.detach().cpu()
        video_metrics = calculate_metrics(target, prediction, self._ignore_idx)
        for name, score in video_metrics.items():
            self.metrics_dict[name].append(score)

        if self.writer:
            for name, metric_val in video_metrics.items():
                self.writer.add_scalar(f"4-Train_Metric/{name}", metric_val, global_step=self.global_iter)

        for name, loss in loss_dict.items():
            loss = loss.item()
            if name not in self.loss_dict:
                self.loss_dict[name] = ScalarMeter(self._cfg.LOG_PERIOD)
                self.loss_sums[name] = 0.0

            self.loss_dict[name].add_value(loss)
            self.loss_sums[name] += loss * mb_size
            if self.writer:
                self.writer.add_scalar(f"2-Loss/train/{name}",
                                       loss,
                                       global_step=self.global_iter)
        self.lr = lr
        self.num_samples += mb_size
        self.global_iter += 1

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        Log the stats of the current iteration.

        :param cur_epoch: the number of current epoch.
        :param cur_iter: the number of current iteration.
        :return:
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "lr": f"{self.lr:.6f}",
            "mem": int(np.ceil(mem_usage)),
        }

        for name, metric_list in self.metrics_dict.items():
            metric_val = np.median(metric_list[-self._iter_log_window:])
            stats[name] = f"{metric_val:.4f}"
        for name, loss in self.loss_dict.items():
            stats[name] = f"{loss.get_win_median():.5f}"

        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the end of epochs stats.

        :param cur_epoch:
        :return:
        """
        mem_usage = misc.gpu_mem_usage()
        avg_loss = {}

        for name, loss in self.loss_sums.items():
            avg_loss[name] = np.round(loss / self.num_samples, decimals=4)

        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "loss": avg_loss,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        stats.update(avg_loss)
        for name, metric_list in self.metrics_dict.items():
            metric_mean = np.mean(metric_list)
            stats[name] = metric_mean
            if self.writer:
                self.writer.add_scalar(f"3-Avg_Train_Metric/{name}",
                                       metric_mean,
                                       global_step=cur_epoch)
                
        if self.writer:
            for name, loss_val in avg_loss.items():
                self.writer.add_scalar(f"2-Loss/train/avg_{name}", loss_val, global_step=cur_epoch)

        logging.log_json_stats(stats)
        if self.writer:
            self.writer.add_scalar("5-LR/train_epoch/lr", self.lr, cur_epoch)
        return avg_loss['loss']


class ValMeter(object):
    """
    Measures validation stats.
    """
    def __init__(self, max_iter: int, cfg: CfgNode, writer: SummaryWriter = None):
        """

        :param max_iter:  the max number of iteration of the current epoch.
        :param cfg:
        :param writer:
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.colors = plot_utils.generate_distinct_colors(n=cfg.MODEL.NUM_CLASSES,
                                                          random_seed=115)
        self._iter_log_window = cfg.LOG_PERIOD
        self.loss_sums = {
            "loss": 0.0,
            "loss_ce": 0.0,
            "loss_mse": 0.0
        }

        self._ignore_idx = cfg.MODEL.PAD_IGNORE_IDX
        if isinstance(cfg.MODEL.PAD_IGNORE_IDX, int):
            self._ignore_idx = [cfg.MODEL.PAD_IGNORE_IDX]
        self._ignore_idx = self._ignore_idx + cfg.TRAIN.EVAL_IGNORE_LABELS
        logger.info(f"Ignored label idxs: {self._ignore_idx}")
        self.writer = writer
        self.metrics_dict = {
            "MoF": [],
            "Edit": [],
            "F1@10": [],
            "F1@25": [],
            "F1@50": [],
        }
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.metrics_dict = {name: [] for name in self.metrics_dict.keys()}
        for name in self.loss_sums.keys():
            self.loss_sums[name] = 0.0
        self.num_samples = 0

    def update_stats(self,
                     target: torch.Tensor,
                     prediction: torch.Tensor,
                     loss_dict: dict
                     ):
        video_metrics = calculate_metrics(target,
                                          prediction,
                                          self._ignore_idx)
        mb_size = target.shape[0]
        for name, score in video_metrics.items():
            self.metrics_dict[name].append(score)

        for name, loss in loss_dict.items():
            loss = loss.item()
            self.loss_sums[name] += loss * mb_size

        self.num_samples += mb_size
        return video_metrics

    def log_iter_stats(self, cur_epoch: int, cur_iter: int):
        """

        :param cur_epoch:
        :param cur_iter:
        :return:
        """

        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "mem": int(np.ceil(mem_usage)),
        }
        for name, metric_list in self.metrics_dict.items():
            metric_val = np.median(metric_list[-self._iter_log_window:])
            stats[name] = metric_val
        logging.log_json_stats(stats)

    def visualize_prediction_result(self, vis_data: List, cur_epoch: int):
        """
        Generate and add visualization of predictions to tensorboard.
        :param vis_data: a list of dictionaries. Each dictionary contains the target
                         and prediction array (numpy) with video name
        :param cur_epoch:
        :return:
        """
        res_list = []
        if len(vis_data) % 2 != 0:
            vis_data = vis_data[:-1]
        for data in vis_data:
            gt_lbl, gt_lens = plot_utils.summarize_list(data['target'].tolist())
            pred_lbl, pred_lens = plot_utils.summarize_list(data['pred'].tolist())

            gt_res = plot_utils.generate_image_for_segmentation(gt_lbl, gt_lens,
                                                                colors=self.colors,
                                                                height=10,
                                                                white_label=self._cfg.DATA.BACKGROUND_INDICES)
            pred_res = plot_utils.generate_image_for_segmentation(pred_lbl, pred_lens,
                                                                  colors=self.colors,
                                                                  height=10,
                                                                  white_label=self._cfg.DATA.BACKGROUND_INDICES)
            res_list.append({"gt": gt_res,
                             "pred": pred_res,
                             "video_name": data['video_name'],
                             "len": sum(gt_lens)})

        fig = plot_utils.create_result_fig(res_list)

        if self.writer:
            self.writer.add_figure("Val_Pred", fig, global_step=cur_epoch)

    def log_epoch_stats(self, cur_epoch: int):
        """
         Log the stats of the current epoch.
        :param cur_epoch: the number of current epoch.
        :return:
        """
        mem_usage = misc.gpu_mem_usage()
        avg_loss = {}
        for name, loss in self.loss_sums.items():
            avg_loss[name] = np.round(loss / self.num_samples, decimals=4)
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "avg_loss": avg_loss,
            "mem": int(np.ceil(mem_usage)),
        }

        if self.writer:
            for name, loss_val in avg_loss.items():
                self.writer.add_scalar(f"2-Val_Loss/avg_{name}", loss_val, global_step=cur_epoch)

        # will be used to select best model
        metrics_sum = 0
        for name, metric_list in self.metrics_dict.items():
            metric_mean = np.mean(metric_list)
            stats[name] = metric_mean
            if self.writer:
                self.writer.add_scalar(f"1-Val_Metric/{name}", metric_mean, global_step=cur_epoch)
            if name == 'Edit':
                metric_mean = metric_mean / 100.0
            metrics_sum += metric_mean

        out_metrics = {'metrics_sum': metrics_sum}
        logging.log_json_stats(stats)
        return out_metrics
