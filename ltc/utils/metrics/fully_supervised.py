# this code is a modified version of https://github.com/yassersouri/MuCon/tree/main/src/core/metrics
from collections import defaultdict
from typing import List, Iterable

from torch import Tensor
import numpy as np

from .base import Metric
from ltc.utils.metrics.external.mstcn_code import edit_score, f_score


class Edit(Metric):
    def __init__(self, ignore_ids: Iterable[int] = (), window_size: int = 1):
        super(Edit, self).__init__(window_size=window_size)
        self.ignore_ids = ignore_ids
        self.reset()

    def reset(self):
        self.values = []
        self.deque.clear()

    def get_deque_median(self):
        return np.median(self.deque)

    def add(
        self, targets: Tensor, predictions: Tensor
    ) -> float:
        """

        :param targets: torch tensor with shape [batch_size, seq_len]
        :param predictions: torch tensor with shape [batch_size, seq_len]
        :return:
        """
        targets, predictions = np.array(targets), np.array(predictions)
        for target, pred in zip(targets, predictions):
            mask = np.logical_not(np.isin(target, self.ignore_ids))
            target = target[mask]
            pred = pred[mask]

            current_score = edit_score(
                recognized=pred.tolist(),
                ground_truth=target.tolist(),
                ignored_classes=self.ignore_ids,
            )

            self.values.append(current_score)
            self.deque.append(current_score)

        return current_score

    def summary(self) -> float:
        if len(self.values) > 0:
            return np.array(self.values).mean()
        else:
            return 0.0


class F1Score(Metric):
    def __init__(
        self,
        overlaps: List[float] = (0.1, 0.25, 0.5),
        ignore_ids: List[int] = (),
        window_size: int = 1,
        num_classes: int = None,
    ):
        super(F1Score, self).__init__(window_size=window_size)
        self.overlaps = overlaps
        self.ignore_ids = ignore_ids
        self.num_classes = num_classes
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        if self.num_classes is None:
            shape = (len(self.overlaps), 1)
        else:
            shape = (len(self.overlaps), self.num_classes)
        self.tp = np.zeros(shape)
        self.fp = np.zeros(shape)
        self.fn = np.zeros(shape)
        self.deque.clear()

    def get_deque_median(self):
        medians = {}
        aggregate_scores = defaultdict(list)
        for score_dict in self.deque:
            for n, v in score_dict.items():
                aggregate_scores[n].append(v)
        for name, scores in aggregate_scores.items():
            medians[name] = np.median(scores)
        return medians

    def add(
            self,
            targets: Tensor,
            predictions: Tensor
    ) -> dict:
        """

        :param targets: tensor of shape [batch_size, seq_len]
        :param predictions: tensor of shape [batch_size, seq_len]
        :return:
        """

        targets = np.array(targets)
        predictions = np.array(predictions)
        for target, pred in zip(targets, predictions):
            current_result = {}
            mask = np.logical_not(np.isin(target, self.ignore_ids))
            target = target[mask]
            pred = pred[mask]

            for s in range(len(self.overlaps)):
                tp1, fp1, fn1 = f_score(
                    pred.tolist(),
                    target.tolist(),
                    self.overlaps[s],
                    ignored_classes=self.ignore_ids,
                )
                self.tp[s] += tp1
                self.fp[s] += fp1
                self.fn[s] += fn1

                current_f1 = self.get_f1_score(tp1, fp1, fn1)
                current_result[f"F1@{int(self.overlaps[s]*100)}"] = current_f1
                self.deque.append(current_result)

        return current_result

    def summary(self) -> dict:
        result = {}
        for s in range(len(self.overlaps)):
            f1_per_class = self.get_f1_score(tp=self.tp[s], fp=self.fp[s], fn=self.fn[s])
            result[f"F1@{int(self.overlaps[s]*100)}"] = np.mean(f1_per_class)

        return result

    @staticmethod
    def get_vectorized_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
        """
        Args:
            tp: [num_classes]
            fp: [num_classes]
            fn: [num_classes]
        Returns:
            [num_classes]
        """
        return 2 * tp / (2 * tp + fp + fn + 0.00001)

    @staticmethod
    def get_f1_score(tp: float, fp: float, fn: float) -> float:
        if tp + fp != 0.0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        else:
            precision = 0.0
            recall = 0.0

        if precision + recall != 0.0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return f1
