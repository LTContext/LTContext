from collections import deque


class Metric:
    def __init__(self, window_size):
        # keep the last 'window_size' number of results
        self.deque = deque(maxlen=window_size)

    def add(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        result = self.add(*args, **kwargs)

        return result

    def get_deque_median(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__
