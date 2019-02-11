from bisect import bisect_right
import math

import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """multi-step learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be main.tex list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """cosine annealing scheduler with warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer,
        T_max,
        eta_min,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )

        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return self.get_lr_warmup()
        else:
            return self.get_lr_cos_annealing()

    def get_lr_warmup(self):
        if self.warmup_method == "constant":
            warmup_factor = self.warmup_factor
        elif self.warmup_method == "linear":
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr * warmup_factor
            for base_lr in self.base_lrs
        ]

    def get_lr_cos_annealing(self):
        last_epoch = self.last_epoch - self.warmup_iters
        T_max = self.T_max - self.warmup_iters
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * last_epoch / T_max)) / 2
                for base_lr in self.base_lrs]


class PiecewiseCyclicalLinearLR(torch.optim.lr_scheduler._LRScheduler):
    """Set the learning rate of each parameter group using piecewise
    cyclical linear schedule.

    When last_epoch=-1, sets initial lr as lr.
    
    Args:    
        c: cycle length
        alpha1: lr upper bound of cycle
        alpha2: lr lower bound of cycle

    _Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs
    https://arxiv.org/pdf/1802.10026
    _Exploring loss function topology with cyclical learning rates
    https://arxiv.org/abs/1702.04283
    """

    def __init__(self, optimizer, c, alpha1=1e-2, alpha2=5e-4, last_epoch=-1):

        self.c = c
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        super(PiecewiseCyclicalLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        lrs = []
        for _ in range(len(self.base_lrs)):
            ti = ((self.last_epoch - 1) % self.c + 1) / self.c
            if 0 <= ti <= 0.5:
                lr = (1 - 2 * ti) * self.alpha1 + 2 * ti * self.alpha2
            elif 0.5 < ti <= 1.0:
                lr = (2 - 2 * ti) * self.alpha2 + (2 * ti - 1) * self.alpha1
            else:
                raise ValueError('t(i) is out of range [0,1].')
            lrs.append(lr)

        return lrs


class PolyLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, power=0.9, max_epoch=4e4, last_epoch=-1):
        self.power = power
        self.max_epoch = max_epoch
        self.last_epoch = last_epoch
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            lr = base_lr * (1.0 - (self.last_epoch / self.max_epoch)) ** self.power
            lrs.append(lr)

        return lrs
