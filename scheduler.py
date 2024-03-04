import torch
import numpy as np

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, num_epochs: int):
        self.warmup = warmup_epochs
        self.max_num_iters = num_epochs
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch <= self.warmup:
            return (epoch + 1) * 1.0 / (self.warmup + 1)
        return  0.5 * (1 + np.cos(np.pi * (epoch - self.warmup) / (self.max_num_iters - self.warmup)))
