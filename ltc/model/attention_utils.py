# Some part of this code is inspired from an early version of xformers lib https://github.com/facebookresearch/xformers
import math
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


import ltc.utils.logging as logging
from ltc.utils.misc import exists

logger = logging.get_logger(__name__)


def _matmul_with_mask(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if mask is None:
        return a @ b
    att = a @ b
    if mask.dtype == torch.bool:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(att.shape[0], -1, -1)
        # mask is presumed false == ignore
        att[~mask] = float("-inf")
    else:
        raise ValueError("Attention Mask need to be boolean with False to ignore")

    return att


def scaled_query_key_mult(
    q: torch.Tensor,
    k: torch.Tensor,
    att_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    # TODO assume we have (N, S, hs) instead of (B, nh, S, hs), with N = B x nh
    # this is needed due to limitations in sparse_bmm for now

    # Self-attend: (N, S, hs) x (N, hs, S) -> (N, S, S)
    q = q / math.sqrt(k.size(-1))

    att = _matmul_with_mask(q, k.transpose(-2, -1), att_mask)

    return att


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    att_mask: Optional[torch.Tensor],
    position_bias: Optional[torch.Tensor] = None,
    dropout: Optional[torch.nn.Module] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor]]:

    att = scaled_query_key_mult(q, k, att_mask=att_mask)

    if position_bias is not None:
        att += position_bias

    out_mask = torch.ones_like(q)
    # to avoid blocks that are all zero (i.e. softmax over rows with all zeros)
    if (att_mask.sum((-2, -1)) == 0).any():
        att[(att_mask.sum((-2, -1)) == 0)] = 0.00001
        out_mask[(att_mask.sum((-2, -1)) == 0), :, :] = 0

    att = torch.softmax(att, dim=att.ndim - 1)
    #  Optional dropout, could be part of the masking in the future
    att = dropout(att)
    # Get to the predicted values, for all heads
    # y = att @ v  # (N, S, S) x (N, S, hs) -> (N, S, hs)
    y = att @ v
    y = y * out_mask
    return y, att


class ScaledDotProduct(nn.Module):
    """
    This code is inspired from xformers lib
    """
    def __init__(
            self,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.attn_drop = nn.Dropout(dropout, inplace=False)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                att_mask: Optional[torch.Tensor] = None,
                position_bias: Optional[torch.Tensor] = None,
                return_attn_matrix: bool = False,
                ) -> Union[Tuple[torch.Tensor, torch.Tensor]]:
        """

        :param q:
        :param k:
        :param v:
        :param att_mask:
        :param position_bias:
        :param return_attn_matrix:
        :return:
        """
        # Convenience, create an attention mask if a tensor was passed
        # Attend: (B x nh, S, hs) x (B x nh, hs, S) -> (B x nh, S, S)
        y, att_matrix = scaled_dot_product_attention(
            q=q, k=k, v=v,
            att_mask=att_mask,
            dropout=self.attn_drop,
            position_bias=position_bias,
        )
        if return_attn_matrix:
            return y, att_matrix
        else:
            return y


def patchify(ts: Tensor, patch_size: int, stride: int):
    """
    Convert a tensor into patches (windows) of size 'window_size' with overlap of 'stride'

    :param ts: a Tensor of shape [batch_size, hidden_dim, seq_len]
    :param patch_size: an integer of patch size
    :param stride: an integer of overlap between windows
    :return:
        Tensor of shape
        [batch_size, hidden_dim, num_windows, window_size]
    """
    patches = ts.unfold(2, patch_size, stride)  # [bs, d, nw, patch_size]
    return patches


def convert_to_patches(x: Tensor,
                       patch_size: int,
                       masks: Tensor = None,
                       overlapping_patches: bool = False):
    """
    Reshape a tensor into 1d windows ( 1d patches) of 'window_size' size. It pads the tensor if it is needed.

    :param x: Tensor of shape [batch_size, hidden_dim, seq_len]
    :param patch_size: the size of each patch
    :param masks: Tensor of shape [batch_size, 1, seq_len]
    :param overlapping_patches: whether the patches are overlapped or not
    :return:
         patches: a Tensor of shape [batch_size, hidden_dim, num_patches, patch_size]
         padding_masks: a Tensor of shape [batch_size, 1, num_patches, patch_size]
    """
    bs, dim, seq_len = x.shape
    if seq_len % patch_size != 0:
        pad_size = patch_size - (seq_len % patch_size)
    else:
        pad_size = 0
    x, padded_masks = pad_sequence(x, pad_size, masks)

    if overlapping_patches:
        half_pad = (patch_size//2, patch_size//2)
        padded_x, padding_mask = pad_sequence(x, pad_size=half_pad, masks=padded_masks)
        patches = patchify(padded_x, 2*patch_size, stride=patch_size)
        padding_mask = patchify(padding_mask, 2*patch_size, stride=patch_size)
    else:
        patches = patchify(x, patch_size, stride=patch_size)
        padding_mask = patchify(padded_masks, patch_size, stride=patch_size)
    return patches, padding_mask


def pad_sequence(x: Tensor, pad_size: Union[int, Tuple[int, int]], masks: Tensor = None):
    """
     Pad the sequence to have a length of as the given size
    :param x: a torch.Tensor of shape [batch_size, feat_dim, seq_len]
    :param pad_size: int or tuple of ints the amount that the sequence need to be padded
    :param masks: batch padding masks which corresponds
        to the padding done during batchification of sequences
    :return:
        padded_x: a torch.Tensor of shape [batch_size, feat_dim, size]
        padding_mask: a torch.Tensor of shape [batch_size, feat_dim, size]
    """
    bs, dim, seq_len = x.shape

    if isinstance(pad_size, int):
        pad_size = (0, pad_size)
    if not exists(masks):
        masks = torch.ones(bs, 1, seq_len).bool()
        masks = masks.to(x.device)
    if pad_size[-1] <= 0:
        return x, masks
    padded_x = F.pad(x, pad=pad_size, value=0.0)
    padding_mask = F.pad(masks, pad=pad_size, value=False)
    return padded_x, padding_mask
