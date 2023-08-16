import os
from os.path import join

from yacs.config import CfgNode

import numpy as np
import torch
from torch.utils.data import Dataset

from ltc.dataset.utils import load_segmentations
from ltc.dataset.utils import conform_temporal_sizes
import ltc.utils.logging as logging

logger = logging.get_logger(__name__)


class VideoDataset(Dataset):
    def __init__(self, cfg: CfgNode, mode: str):
        assert mode in [
            "train",
            "test",
            "val",
        ], "Split '{}' not supported".format(mode)

        self._mode = mode
        self._cfg = cfg

        self._video_meta = {}
        self._path_to_data = cfg.DATA.PATH_TO_DATA_DIR
        self._video_sampling_rate = cfg.DATA.FRAME_SAMPLING_RATE
        self._construct_loader()
        self._dataset_size = len(self._path_to_features)

    def _construct_loader(self):
        """
        Construct the list of features and segmentations.
        """
        video_list_file = join(self._path_to_data,
                               "splits",
                               f"{self._mode}.split{self._cfg.DATA.CV_SPLIT_NUM}.bundle")
        assert os.path.isfile(video_list_file), f"Video list file {video_list_file} not found."

        with open(video_list_file, 'r') as f:
            list_of_videos = list(map(lambda l: l.strip(), f.readlines()))

        with open(join(self._path_to_data, "mapping.txt"), 'r') as f:
            lines = map(lambda l: l.strip().split(), f.readlines())
            action_to_idx = {action: int(str_idx) for str_idx, action in lines}

        num_videos = int(len(list_of_videos) * self._cfg.DATA.DATA_FRACTION)
        logger.info(f"Using {self._cfg.DATA.DATA_FRACTION*100}% of {self._mode} data.")

        self._path_to_features = []
        self._segmentations = []
        self._video_names = []
        for gt_filename in list_of_videos[:num_videos]:
            video_id = os.path.splitext(gt_filename)[0]
            feat_filename = video_id + ".npy"
            feat_path = join(self._path_to_data, "features", feat_filename)
            assert os.path.isfile(feat_path), f"Feature {feat_path} not found."
            self._path_to_features.append(feat_path)
            gt_path = join(self._path_to_data, "groundTruth", gt_filename)
            self._segmentations.append(load_segmentations(gt_path, action_to_idx))
            self._video_names.append(video_id)

    def _load_features(self, feature_path: str):
        features = np.load(feature_path)
        features = features.astype(np.float32)
        features = features[:, ::self._video_sampling_rate]  # [D, T]
        return features

    def __getitem__(self, index: int):
        """

        :param index:
        :return: sample dict containing:
         'features': torch.Tensor [batch_size, input_dim_size, sequence_length]
         'targets': torch.Tensor [batch_size, sequence_length]
        """
        sample = {}
        targets = self._segmentations[index]
        sample['targets'] = torch.tensor(targets).long()[::self._video_sampling_rate]  # [T]

        feature_path = self._path_to_features[index]
        sample['features'] = torch.tensor(self._load_features(feature_path))  # [D, T]
        seq_length = sample['features'].shape[-1]
        sample['targets'] = conform_temporal_sizes(sample['targets'], seq_length)

        sample['video_name'] = self._video_names[index]

        return sample

    def __len__(self):
        """
        :return: the number of data point (videos) in the dataset.
        """
        return self._dataset_size


