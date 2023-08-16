from typing import List, Dict
from torch import Tensor

from .base import Metric
from .segmentation import MoFAccuracyMetric
from .fully_supervised import F1Score, Edit


def calculate_metrics(target: Tensor, pred: Tensor, ignored_class_ids: List[int]) -> Dict:
    """
    Calculates the action segmentation metrics (MoF, Edit, F1@{10, 25, 50}) for a video
    :param target: a Tensor of shape [batch_size, sequence_len]
    :param pred: a Tensor of shape [batch_size, sequence_len]
    :param ignored_class_ids: a list of class ids to ignore during calculation
    :return:
      a dict of metrics values
    """
    assert target.shape[0] == 1, "Batch size should be one for validation due to limitation of metric functions."

    result_dict = {}
    mof_func = MoFAccuracyMetric(ignore_ids=ignored_class_ids)
    edit_func = Edit(ignore_ids=ignored_class_ids)
    f1_func = F1Score(ignore_ids=ignored_class_ids)

    mof_func.add(target, pred)
    edit_func.add(target, pred)
    f1_func.add(target, pred)

    result_dict['MoF'] = mof_func.summary()
    result_dict['Edit'] = edit_func.summary()
    f1_dict = f1_func.summary()
    result_dict.update(f1_dict)
    return result_dict


def get_metric(name):
    return {
        "MoF": MoFAccuracyMetric,
        "F1": F1Score,
        "Edit": Edit
    }[name]
