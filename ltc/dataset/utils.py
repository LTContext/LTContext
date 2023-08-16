from typing import List, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange


import logging
logger = logging.getLogger(__name__)


def sequence_collate(batch: List[Dict], pad_ignore_idx: int = -100):
    """

    :param batch: list of batches to collate
    :param pad_ignore_idx: the integer value to use for padding targets
    :return:
        padded inputs
        'features': torch.Tensor [batch_size, input_dim_size, sequence_length]
        'targets': torch.Tensor [batch_size, sequence_length]
        'masks': torch.Tensor [batch_size, 1, sequence_length]
        'video_name': list of video names
    """
    batch_features = [rearrange(item['features'], "dim seq_len -> seq_len dim") for item in batch]
    batch_targets = [item['targets'] for item in batch]
    batch_vid_names = [[item['video_name'] for item in batch]]

    # for pad_sequence the sequence length should be in the first dimension
    batch_feat_tensor = pad_sequence(batch_features, batch_first=True)  # [B, T, D]
    batch_target_tensor = pad_sequence(batch_targets, batch_first=True, padding_value=pad_ignore_idx)  # [B, T]

    masks = torch.where(batch_target_tensor == pad_ignore_idx, 0, 1)  # [batch_size, max_seq_len]
    # assert torch.equal(length_of_sequences,
    #                    masks.sum(-1)), "Masks doesn't match the un-padded sequence sizes"
    masks = masks[:, None, :].bool()
    batch_dict = {"features": rearrange(batch_feat_tensor, "b seq_len dim -> b dim seq_len"),
                  "targets": batch_target_tensor,
                  "video_name": batch_vid_names,
                  "masks": masks}

    return batch_dict


def load_segmentations(segm_path, action_to_idx):
    """

    :param segm_path: path to segmentation ground truth
    :param action_to_idx: dictionary to map action label to action id
    :return:
        numpy array of action ids
    """
    with open(segm_path, 'r') as f:
        actions = map(lambda l: l.strip(), f.readlines())
    segmentation = list(map(lambda ac: action_to_idx[ac], actions))
    return np.array(segmentation, dtype=np.int16)


def conform_temporal_sizes(tensor, seq_len):
    """

    :param tensor: torch.Tensor of shape [T] or [B, T]
    :param seq_len:
    :return:
    """
    if tensor.shape[0] == seq_len:
        return tensor
    # logger.info(f"Resizing label's size from {tensor.shape[0]} to {seq_len}")
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if len(tensor.shape) == 1:
        tensor = tensor[None, None, :]
    elif len(tensor.shape) == 2:
        tensor = tensor[None, :]
    resized_tensor = F.interpolate(tensor.float(), size=seq_len, mode='nearest')
    return resized_tensor.squeeze().long()
