from torch.optim.lr_scheduler import _LRScheduler


class IdentityPolicy(_LRScheduler):
    def step(self):
        pass

