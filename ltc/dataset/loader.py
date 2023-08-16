"""Data loader."""

from functools import partial
import torch

from torch.utils.data.sampler import RandomSampler
from .breakfast import Breakfast
from .assembly101 import Assembly101
from .utils import sequence_collate


_DATASETS = {
    "breakfast": Breakfast,
    "assembly101": Assembly101,
}


def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    :param cfg:
    :param split:  the split of the data loader. 'train', 'test', 'eval'
    :return:
    """
    assert split in ["train", "test", "val"]
    dataset_name = cfg.TRAIN.DATASET

    if split in ["train"]:
        shuffle = True
        batch_size = cfg.TRAIN.BATCH_SIZE

    elif split in ["test", "val"]:
        shuffle = False
        batch_size = cfg.TRAIN.EVAL_BATCH_SIZE

    # Construct the dataset
    dataset = _DATASETS[dataset_name](cfg, split)
    custom_collate_fn = partial(sequence_collate, pad_ignore_idx=cfg.MODEL.PAD_IGNORE_IDX)
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return loader
