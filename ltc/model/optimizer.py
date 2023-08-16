"""Optimizer."""

from yacs.config import CfgNode
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from ltc.utils.lr_policy import IdentityPolicy


def construct_optimizer(model: torch.nn.Module, cfg: CfgNode) -> Optimizer:
    """
     Construct a stochastic gradient descent or ADAM optimizer.
     Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    :param model: model to perform stochastic gradient descent
    :param cfg: configs of hyper-parameters of SGD or ADAM,
    includes base learning rate,  weight_decay, and etc.
    :return:
    """
    optim_params = filter(lambda p: p.requires_grad, model.parameters())

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "nadam":
        return torch.optim.NAdam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def construct_lr_scheduler(cfg: CfgNode, optimizer: Optimizer) -> LRScheduler:
    """
     Construct a learning scheduler.
    :param cfg: configs of hyper-parameters
    :param optimizer:
    :return:
    """
    if cfg.SOLVER.LR_POLICY == 'identity':
        return IdentityPolicy(optimizer)
    elif cfg.SOLVER.LR_POLICY == "constant_cosine_decay":
        identity = LinearLR(optimizer, start_factor=1.0)
        main_lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.SOLVER.T_MAX - cfg.SOLVER.WARMUP_EPOCHS,
            eta_min=cfg.SOLVER.ETA_MIN,
        )
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[identity, main_lr_scheduler],
            milestones=[cfg.SOLVER.WARMUP_EPOCHS]
        )
        return lr_scheduler
    elif cfg.SOLVER.LR_POLICY == "cosine_linear_warmup":
        main_lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.SOLVER.T_MAX - cfg.SOLVER.WARMUP_EPOCHS,
            eta_min=cfg.SOLVER.ETA_MIN,
        )
        warmup_lr_scheduler = LinearLR(
            optimizer, start_factor=cfg.SOLVER.WARMUP_FACTOR, total_iters=cfg.SOLVER.WARMUP_EPOCHS
        )
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[cfg.SOLVER.WARMUP_EPOCHS]
                )
        return lr_scheduler
    else:
        raise NotImplementedError(f"LR scheduler {cfg.SOLVER.LR_POLICY} is not found.")


def get_epoch_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
