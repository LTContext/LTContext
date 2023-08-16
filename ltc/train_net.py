from typing import Dict
import os
import pprint
import json

from tqdm import tqdm
from yacs.config import CfgNode

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from ltc.model.loss import get_loss_func
import ltc.model.optimizer as optim
import ltc.utils.checkpoint as cu
import ltc.utils.misc as misc
from ltc.dataset import loader
from ltc.model import model_builder
from ltc.utils.meters import TrainMeter, ValMeter

import ltc.utils.logging as logging

logger = logging.get_logger(__name__)


def train_epoch(
        train_loader: DataLoader,
        model: nn.Module,
        optimizer: Optimizer,
        train_meter: TrainMeter,
        cur_epoch: int,
        cfg: CfgNode,
        device: torch.device.type
):
    """
    Perform the training for one epoch.
    :param train_loader: video features training loader.
    :param model: the model to train
    :param optimizer: the optimizer to perform optimization on the model's parameters.
    :param train_meter: training meters to log the training performance.
    :param cur_epoch: current epoch of training.
    :param cfg: configs. Details can be found in ltc/config/defaults.py
    :param device: device to use (cuda or cpu)
    :return:

    """
    model.train()
    loss_func = get_loss_func(cfg)
    lr = optim.get_epoch_lr(optimizer)
    train_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for cur_iter, (batch_dict) in train_bar:
        misc.move_to_device(batch_dict, device)
        logits = model(batch_dict['features'], batch_dict['masks'])

        loss_dict = loss_func(logits, batch_dict['targets'])
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        optimizer.step()

        status_text = f"Train [Epoch {cur_epoch}/{cfg.SOLVER.MAX_EPOCH}]" \
                      f" [{cur_iter}/{len(train_loader)}] [Loss: {loss_dict['loss'].item():.4f} ]"
        train_bar.set_description(status_text)

        prediction = misc.prepare_prediction(logits)

        # Update and log stats.
        train_meter.update_stats(target=batch_dict['targets'],
                                 prediction=prediction,
                                 loss_dict=loss_dict,
                                 lr=lr)

        train_meter.log_iter_stats(cur_epoch, cur_iter)

    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(
        val_loader: DataLoader,
        model: nn.Module,
        val_meter: ValMeter,
        cur_epoch: int,
        device: torch.device.type,
        cfg: CfgNode) -> Dict:
    """
    Evaluate the model on the val set.
    :param val_loader: data loader to provide validation data.
    :param model: model to evaluate the performance.
    :param val_meter: meter instance to record and calculate the metrics.
    :param cur_epoch: number of the current epoch of training.
    :param device: device to use (cuda or cpu)
    :param cfg:
    :return:
         a dictionary of validation metrics
    """
    logger.info(f"Validation started")
    model.eval()
    loss_func = get_loss_func(cfg)
    visualization_samples = []
    vis_period = round(len(val_loader) / 50)
    vis_period = 1 if vis_period == 0 else vis_period

    for cur_iter, (batch_dict) in tqdm(enumerate(val_loader), total=len(val_loader)):
        misc.move_to_device(batch_dict, device)
        logits = model(batch_dict['features'], batch_dict['masks'])
        loss_dict = loss_func(logits, batch_dict['targets'])

        prediction = misc.prepare_prediction(logits)

        targets = batch_dict['targets'].cpu()
        prediction = prediction.cpu()

        val_meter.update_stats(targets, prediction, loss_dict)

        if visualization_samples == 0 or cur_iter % vis_period == 0:
            visualization_samples.append({"target": targets.squeeze(0).numpy(),
                                          "pred": prediction.squeeze(0).numpy(),
                                          "video_name": f"{batch_dict['video_name'][0][0]}"})

        val_meter.log_iter_stats(cur_epoch, cur_iter)

    # Log epoch stats.
    val_meter.visualize_prediction_result(vis_data=visualization_samples, cur_epoch=cur_epoch)
    val_metrics = val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()
    return val_metrics


def train(cfg: CfgNode):
    """
    Train an action segmentation model for many epochs on train set and validate it on validation set
    :param cfg: config file. Details can be found in ltc/config/defaults.py
    :return:
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    logging.setup_logging(cfg.OUTPUT_DIR, cfg.EXPR_NUM)

    # Print config.
    logger.info("Training with config:")
    logger.info(pprint.pformat(cfg))

    model = model_builder.build_model(cfg)
    logger.info(f"Number of params: {misc.params_to_string(misc.params_count(model))}")

    if cfg.NUM_GPUS > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transfer the model to device(s)
    model = model.to(device)

    optimizer = optim.construct_optimizer(model, cfg)
    scheduler = optim.construct_lr_scheduler(cfg, optimizer)

    # Load a checkpoint to resume training if applicable.
    checkpoint_path = cu.get_path_to_checkpoint(cfg)
    if os.path.isfile(checkpoint_path):
        checkpoint_epoch = cu.load_checkpoint(
            checkpoint_path,
            model=model,
            num_gpus=cfg.NUM_GPUS,
            optimizer=optimizer,
            lr_scheduler=scheduler
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    # Create tensorboard summary writer
    writer = SummaryWriter(cfg.SUMMARY_PATH)
    writer.add_text(f"Config_EXPR_NUM={cfg.EXPR_NUM}", pprint.pformat(cfg).replace("\n", "\n\n"))

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    test_loader = loader.construct_loader(cfg, cfg.TRAIN.EVAL_SPLIT)

    # Create meters.
    train_meter = TrainMeter(
        len(train_loader), cfg, start_epoch * (len(train_loader)), writer
    )
    val_meter = ValMeter(len(test_loader), cfg, writer)

    # Print summary path.
    logger.info("Summary path {}".format(cfg.SUMMARY_PATH))

    with open(os.path.join(cfg.CONFIG_LOG_PATH, "config.yaml"), "w") as json_file:
        json.dump(cfg, json_file, indent=2)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    best_metrics_sum = -1
    eval_metrics = {}

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, device
        )

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            logger.info("Saving checkpoint")
            best_metrics_sum = cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                cfg.EXPR_NUM,
                model,
                optimizer,
                scheduler,
                eval_metrics,
                best_metrics_sum,
                cur_epoch,
                cfg
            )

        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            eval_metrics = eval_epoch(test_loader, model, val_meter, cur_epoch, device, cfg)

        # Update the learning rate.
        scheduler.step()

    writer.flush()
