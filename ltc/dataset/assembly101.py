from typing import Dict
import os
from os.path import join, isfile

import numpy as np
import lmdb
import pandas as pd
import torch
from torch.utils.data import Dataset
from numpy.lib.format import open_memmap

from ltc.dataset.utils import conform_temporal_sizes
import ltc.utils.logging as logging
from tqdm import tqdm

logger = logging.get_logger(__name__)

VIEWS = ['C10095_rgb', 'C10115_rgb', 'C10118_rgb', 'C10119_rgb', 'C10379_rgb', 'C10390_rgb', 'C10395_rgb', 'C10404_rgb',
         'HMC_21176875_mono10bit', 'HMC_84346135_mono10bit', 'HMC_21176623_mono10bit', 'HMC_84347414_mono10bit',
         'HMC_21110305_mono10bit', 'HMC_84355350_mono10bit', 'HMC_21179183_mono10bit', 'HMC_84358933_mono10bit']


class Assembly101(Dataset):
    def __init__(self, cfg, mode):
        """

        """
        assert mode in [
            "train",
            "val",
        ], "Split '{}' not supported".format(mode)

        self._mode = mode
        self._cfg = cfg
        self._load_type = cfg.DATA.LOAD_TYPE
        assert self._load_type in ['lmdb', 'numpy'], \
            "Available loading types are: ['lmdb', 'numpy']"

        self._path_to_data = self._cfg.DATA.PATH_TO_DATA_DIR
        self._video_sampling_rate = cfg.DATA.FRAME_SAMPLING_RATE

        if self._load_type == 'lmdb':
            self.env = None
            self.frames_format = "{}/{}_{:010d}.jpg"

        self._data = self._construct_loader()
        self._dataset_size = len(self._data)
        logger.info(f"Number {mode} input sequences is: {self._dataset_size}")

    def _construct_loader(self):
        """
        Construct the list of features and segmentations.
        """
        annotation_path = join(os.getcwd(), "data/assembly101", "coarse-annotations")
        path_to_csvfile = join(os.getcwd(), "data/assembly101", f"{self._mode}.csv")
        assert os.path.exists(path_to_csvfile), f"Split csv file not found at {path_to_csvfile}"

        data_df = pd.read_csv(path_to_csvfile)
        actions_df = pd.read_csv(join(annotation_path, "actions.csv"))

        num_videos = int(self._cfg.DATA.DATA_FRACTION * len(data_df))
        data_df = data_df[:num_videos]
        video_data = []
        for idx, entry in tqdm(data_df.iterrows(), total=num_videos):
            sample = entry.to_dict()
            if self._load_type == 'numpy':
                feature_path = join(self._path_to_data,
                                    "TSM_features",
                                    entry['video_id'],
                                    entry['view'],
                                    "features.npy")
                if not isfile(feature_path):
                    logger.warn(f"Numpy feature map "
                                f"{sample['action_type']}_{sample['video_id']}_{sample['view']} not found.")
                    continue
                sample['feat_path'] = feature_path

            segm_filename = f"{sample['action_type']}_{sample['video_id']}.txt"
            segm_path = join(annotation_path, "coarse_labels", segm_filename)
            segm, start_frame, end_frame = self._load_segmentations(segm_path, actions_df)
            if len(segm) == 0:
                continue
            if end_frame - start_frame < 1:
                continue

            sample['segm'] = segm
            sample['start_frame'] = start_frame
            sample['end_frame'] = min(end_frame, sample['video_end_frame'])
            video_data.append(sample)

        return video_data

    def _init_db(self):
        db_feat_path = join(self._path_to_data, "db_TSM_features")
        self.env = {view: lmdb.open(join(db_feat_path, view),
                                    readonly=True,
                                    readahead=False,
                                    meminit=False,
                                    lock=False) for view in VIEWS}

    def _load_features(self, data_dict: Dict):
        """

        :param data_dict:
        :return:
            the feature map with shape of [D, T] where T is the temporal dimension
        """
        if self._load_type == 'numpy':
            features = open_memmap(data_dict['feat_path'], mode='r')  # [D, T]
            features = features[:, data_dict['start_frame']:data_dict['end_frame']:self._video_sampling_rate]  # [D, T]
        else:
            elements = []
            view = data_dict['view']
            with self.env[view].begin(write=False) as e:
                for i in range(data_dict['start_frame'], data_dict['end_frame']):
                    key = join(data_dict['video_id'], self.frames_format.format(view, view, i))
                    frame_data = e.get(key.strip().encode('utf-8'))
                    if frame_data is None:
                        logger.info(f"No data found for key={key}.")
                        exit(2)
                    frame_data = np.frombuffer(frame_data, 'float32')
                    elements.append(frame_data)

            features = np.array(elements).T  # [D, T]

        return features

    def _load_segmentations(self, segm_path, actions_df):
        segment_labels = []
        start_indices = []
        end_indices = []
        with open(segm_path, 'r') as f:
            lines = list(map(lambda s: s.split("\n"), f.readlines()))
            for line in lines:
                start, end, lbl = line[0].split("\t")[:-1]
                start_indices.append(int(start))
                end_indices.append(int(end))
                action_id = actions_df.loc[actions_df['action_cls'] == lbl, 'action_id']
                segm_len = int(end) - int(start)
                segment_labels.append(np.full(segm_len, fill_value=action_id.item()))

        segmentation = np.concatenate(segment_labels)
        num_frames = segmentation.shape[0]
        # start and end frame idx @30fps
        start_frame = min(start_indices)
        end_frame = max(end_indices)
        assert num_frames == (end_frame-start_frame), \
            "Length of Segmentation doesn't match with clip length."

        return segmentation, start_frame, end_frame

    def __getitem__(self, index: int):
        """

        :param index:
        :return: sample dict containing:
         'features': torch.Tensor [D, T]
         'targets': torch.Tensor [T]
        """
        if self._load_type == 'lmdb' and self.env is None:
            self._init_db()

        sample = {}
        data_dict = self._data[index]
        targets = data_dict['segm']
        sample['targets'] = torch.tensor(targets).long()[::self._video_sampling_rate]  # [T]
        sample['features'] = torch.tensor(self._load_features(data_dict))  # [D, T]

        seq_length = sample['features'].shape[-1]
        sample['video_name'] = join(data_dict['action_type'], data_dict['video_id'], data_dict['view'])

        if np.abs(sample['targets'].shape[0] - seq_length) > 200:
            logger.warn(f"Down/Up-sampling targets from {sample['targets'].shape[0]} to {seq_length}")
        sample['targets'] = conform_temporal_sizes(sample['targets'], seq_length)

        return sample

    def __len__(self):
        """
        Returns:
            (int): the number of videos or features in the dataset.
        """
        return self._dataset_size


