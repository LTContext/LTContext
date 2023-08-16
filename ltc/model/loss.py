from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from yacs.config import CfgNode
import ltc.utils.logging as logging

logger = logging.get_logger(__name__)


class CEplusMSE(nn.Module):
    """
    Loss from MS-TCN paper. CrossEntropy + MSE
    https://arxiv.org/abs/1903.01945
    """
    def __init__(self, cfg: CfgNode):
        super(CEplusMSE, self).__init__()
        ignore_idx = cfg.MODEL.PAD_IGNORE_IDX
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_idx)
        self.mse = nn.MSELoss(reduction='none')
        self.mse_fraction = cfg.MODEL.MSE_LOSS_FRACTION
        self.mse_clip_val = cfg.MODEL.MSE_LOSS_CLIP_VAL
        self.num_classes = cfg.MODEL.NUM_CLASSES

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict:
        """
        :param logits: [n_stages, batch_size, n_classes, seq_len]
        :param targets: [batch_size, seq_len]
        :return:
        """
        loss_dict = {"loss": 0.0, "loss_ce": 0.0, "loss_mse": 0.0}
        for p in logits:
            loss_dict['loss_ce'] += self.ce(rearrange(p, "b n_classes seq_len -> (b seq_len) n_classes"),
                                            rearrange(targets, "b seq_len -> (b seq_len)"))

            loss_dict['loss_mse'] += torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                                                                     F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                                                            min=0,
                                                            max=self.mse_clip_val))

        loss_dict['loss'] = loss_dict['loss_ce'] + self.mse_fraction * loss_dict['loss_mse']

        return loss_dict


def get_loss_func(cfg: CfgNode):
    """
     Retrieve the loss given the loss name.
    :param cfg:
    :return:
    """
    if cfg.MODEL.LOSS_FUNC == 'ce_mse':
        return CEplusMSE(cfg)
    else:
        raise NotImplementedError("Loss {} is not supported".format(cfg.LOSS.TYPE))

