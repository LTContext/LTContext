
import os
import numpy as np

import torch
from yacs.config import CfgNode

import ltc.utils.logging as logging


logger = logging.get_logger(__name__)


def move_to_device(batch_dict, device, keys=('features',
                                             'targets',
                                             'masks')):
    for key in keys:
        if key in batch_dict:
            batch_dict[key] = batch_dict[key].to(device)
    return batch_dict


def load_cfg_from_path(cfg_path):
    from yacs.config import CfgNode
    with open(cfg_path, "r") as f:
        cfg = CfgNode.load_cfg(f)
    return cfg


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


def is_eval_epoch(cfg: CfgNode, cur_epoch: int):
    """
     Determine if the model should be evaluated at the current epoch.
    :param cfg:
    :param cur_epoch:
    :return:
    """
    return (
        cur_epoch + 1
    ) % cfg.TRAIN.EVAL_PERIOD == 0 or cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH


def find_latest_experiment(path):
    if not os.path.exists(path):
        return 0

    list_of_experiments = os.listdir(path)
    list_of_int_experiments = []
    for exp in list_of_experiments:
        try:
            int_exp = int(exp)
        except ValueError:
            continue
        list_of_int_experiments.append(int_exp)

    if len(list_of_int_experiments) == 0:
        return 0

    return max(list_of_int_experiments)


def find_latest_log(path):
    if not os.path.exists(path):
        return 0
    list_of_logs = sorted(os.listdir(path))
    # expr_00X.log
    expr_num = list_of_logs[-1].split("_")[-1]
    expr_num = int(expr_num.split(".")[0])
    return expr_num


def check_path(path):
    os.makedirs(path, exist_ok=True)
    return path


def prepare_logits(logits, cfg):
    if cfg.MODEL.LOSS_FUNC == 'ce_mse' and logits.ndim == 3:
        return logits.unsqueeze(0)
    else:
        return logits


def prepare_prediction(outputs, argmax_dim=1):
    if outputs.ndim == 4:
        out_logits = outputs[-1]
    else:
        out_logits = outputs

    pred = torch.argmax(out_logits.data, dim=argmax_dim)
    return pred


def exists(val):
    return val is not None


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, precision)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, precision)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)
