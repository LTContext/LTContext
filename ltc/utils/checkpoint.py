"""Functions that handle saving and loading of checkpoints."""

from typing import Dict
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from yacs.config import CfgNode

import ltc.utils.logging as logging

logger = logging.get_logger(__name__)


def make_checkpoint_dir(path_to_expr: str, expr_num: int):
    """
    Creates the checkpoint directory (if not present already).
    :param path_to_expr:
    :param expr_num:
    :return:
    """
    checkpoint_dir = os.path.join(path_to_expr, "checkpoints", str(expr_num))
    # Create the checkpoint dir from the master process
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except Exception:
            pass
    return checkpoint_dir


def get_checkpoint_dir(path_to_expr: str, resume_expr_num: int):
    """

    :param path_to_expr:
    :param resume_expr_num:
    :return:
    """
    return os.path.join(path_to_expr, "checkpoints", str(resume_expr_num))


def get_path_to_checkpoint(cfg: CfgNode):
    """
    Get the full path to a checkpoint file to load.
    :param cfg: the path to the folder of the current experiment.

    :return:
    """
    checkpoint_path = ""
    if cfg.TRAIN.AUTO_RESUME and has_checkpoint(
        cfg.OUTPUT_DIR, cfg.TRAIN.RESUME_EXPR_NUM
    ):
        logger.info("Load from last checkpoint.")
        checkpoint_path = get_last_checkpoint(
            cfg.OUTPUT_DIR, cfg.TRAIN.RESUME_EXPR_NUM
        )
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_path = cfg.TRAIN.CHECKPOINT_FILE_PATH
    return checkpoint_path


def get_last_checkpoint(path_to_expr: str, resume_expr_num: int):
    """

    :param path_to_expr:
    :param resume_expr_num:
    :return:
    """
    d = get_checkpoint_dir(path_to_expr, resume_expr_num)
    names = os.listdir(d) if os.path.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    assert len(names), "No checkpoints found in '{}'.".format(d)
    # Sort the checkpoints by epoch.
    name = sorted(names)[-1]
    return os.path.join(d, name)


def has_checkpoint(path_to_expr: str, expr_num: int):
    """

    :param path_to_expr:
    :param expr_num:
    :return:
    """
    d = get_checkpoint_dir(path_to_expr, expr_num)
    files = os.listdir(d) if os.path.exists(d) else []
    return any("checkpoint" in f for f in files)


def is_checkpoint_epoch(cur_epoch: int, checkpoint_period: int):
    """
    Determine if a checkpoint should be saved on current epoch.
    :param cur_epoch: current number of epoch of the model.
    :param checkpoint_period: the frequency of checkpointing.
    :return:
    """
    if checkpoint_period == -1:
        return False
    return (cur_epoch + 1) % checkpoint_period == 0


def save_checkpoint(path_to_expr: str,
                    expr_num: int,
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    lr_scheduler: LRScheduler,
                    eval_metrics: Dict,
                    best_metrics_sum: float,
                    epoch: int,
                    cfg: CfgNode):
    """

    :param path_to_expr:
    :param expr_num:
    :param model:
    :param optimizer:
    :param lr_scheduler:
    :param eval_metrics:
    :param best_metrics_sum:
    :param epoch:
    :param cfg:
    :return:
    """
    # Ensure that the checkpoint dir exists.
    os.makedirs(get_checkpoint_dir(path_to_expr, expr_num), exist_ok=True)
    saving_model = model.module if cfg.NUM_GPUS > 1 else model
    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": saving_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint.
    name = "checkpoint_epoch_{:05d}.pyth".format(epoch + 1)
    save_path = os.path.join(get_checkpoint_dir(path_to_expr, expr_num), name)
    torch.save(checkpoint, save_path)

    if 'metrics_sum' in eval_metrics:
        if best_metrics_sum < eval_metrics['metrics_sum']:
            best_metrics_sum = eval_metrics['metrics_sum']
            save_path = os.path.join(get_checkpoint_dir(path_to_expr, expr_num),
                                     "best_checkpoint.pyth")
            torch.save(checkpoint, save_path)

    return best_metrics_sum


def load_checkpoint(
        path_to_checkpoint: str,
        model: nn.Module,
        num_gpus: int,
        optimizer: optim.Optimizer,
        lr_scheduler: LRScheduler,

):
    """
    Load the checkpoint from the given file.
    :param path_to_checkpoint: path to the checkpoint to load.
    :param model: model to load the weights from the checkpoint.
    :param num_gpus: number of gpus
    :param optimizer: optimizer to load the historical state.
    :param lr_scheduler: lr_scheduler to load state
    :return:
    """
    assert os.path.exists(path_to_checkpoint), "Checkpoint '{}' not found".format(
        path_to_checkpoint
    )
    logger.info(f"Loading from {path_to_checkpoint}")
    model_to_load = model.module if num_gpus > 1 else model

    # Load the checkpoint on CPU to avoid GPU mem spike.
    checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
    model_to_load.load_state_dict(checkpoint["model_state"])

    optimizer.load_state_dict(checkpoint["optimizer_state"])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])

    if "epoch" in checkpoint.keys():
        epoch = checkpoint["epoch"]
    else:
        epoch = -1
    return epoch


def load_model(
        path_to_checkpoint: str,
        model: nn.Module,
        num_gpus: int,
):
    """
    Load the trained model from the given file.
    :param path_to_checkpoint: path to the checkpoint to load.
    :param model: model to load the weights from the checkpoint.
    :param num_gpus: number of gpus
    :return:
    """
    assert os.path.exists(path_to_checkpoint), "Checkpoint '{}' not found".format(
        path_to_checkpoint
    )
    logger.info(f"Loading from {path_to_checkpoint}")
    model_to_load = model.module if num_gpus > 1 else model

    # Load the checkpoint on CPU to avoid GPU mem spike.
    checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
    model_to_load.load_state_dict(checkpoint["model_state"])

