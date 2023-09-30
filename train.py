from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .magic import reprint

__all__ = ["loopa", "ACC_METRIC", "EarlyStopper"]

Metrics = dict[
    str, 
    tuple[
        Callable[[torch.Tensor, torch.Tensor], Any],
        Callable[[list[Any]], float],
    ] | Callable[[torch.Tensor, torch.Tensor], float]]

@reprint
def _loopa(*, model: nn.Module, dataloader: DataLoader, device: str,
           loss_fn, optim, metrics: Metrics,
           is_train: bool = True, accum_grad: int = 1):
    
    metric_lists = defaultdict(list)
    for metric, val in metrics.items():
        try:
            iter(val)
        except TypeError:
            metrics[metric] = (val, np.mean)
    if is_train:
        optim.zero_grad()
    for i, (X, y) in enumerate(tqdm(dataloader)):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y) / accum_grad
        
        if is_train:
            loss.backward()
            if not (i + 1) % accum_grad:
                optim.step()
                optim.zero_grad()
            
        with torch.no_grad():
            for metric, (fn, _) in metrics.items():
                if metric == "loss":
                    metric_lists["loss"].append(loss.item() * accum_grad)
                else:
                    metric_lists[metric].append(fn(y_pred, y))
    
    if is_train and (i + 1) % accum_grad:
        # optim.step()
        optim.zero_grad()
    
    metric_results = {}
    for key, (_, agg) in metrics.items():
        metric_results[key] = agg(metric_lists[key])
    
    return metric_results

@reprint
def loopa(model: nn.Module, dataloader: DataLoader, *, device: str,
           loss_fn=None, optim=None, metrics: Metrics,
           is_train: bool = True, accum_grad: int = 1):
    if is_train:
        model.train()
        return _loopa(model=model, dataloader=dataloader, device=device,
                      loss_fn=loss_fn, optim=optim, metrics=metrics, accum_grad=accum_grad,
                      is_train=is_train)
    
    with torch.no_grad():
        model.eval()
        ret = _loopa(model=model, dataloader=dataloader, device=device,
                     loss_fn=loss_fn, optim=optim, metrics=metrics, accum_grad=accum_grad,
                     is_train=is_train)
        model.train()
        return ret

##########################################################################

try:
    from plotter import History
except ImportError:
    pass

@dataclass
class EarlyStopper:
    model: nn.Module
    save_path: str
    bound_history: "History"
    loss: str = "loss",
    patience: int = 3
    min_delta: float = 0
    
    def __post_init__(self):
        self.best_epoch: int = -1 if not len(self.bound_history) else self.get_losses().argmin() + 1
        self.best_loss = np.inf if not len(self.bound_history) else self.get_losses().min()
        
    def __str__(self):
        if self.loss.startswith("-"):
            metric = self.loss[1:]
            sign = -1
        else:
            metric = self.loss
            sign = 1
        return f"based on metric: {metric}, best epoch: {self.best_epoch}, best value: {sign * self.best_loss}"
    
    def get_losses(self):
        if self.loss.startswith("-"):
            metric = self.loss[1:]
            sign = -1
        else:
            metric = self.loss
            sign = 1
        return np.array([sign * res[metric]
                         for res in self.bound_history.val])
    
    def __call__(self):
        """
        saves model on improvement
        returns True if training should stop else False
        """ 
        losses = self.get_losses()
        if losses[-1] <= self.best_loss:
            self.best_loss = losses[-1]
            self.best_epoch = len(self.bound_history)
            torch.save(self.model.state_dict(), self.save_path)
            return False
        return len(losses) >= self.patience + 1 and \
                np.all(losses[-self.patience:] >= self.best_loss + self.min_delta)

def mean_metric(sum_of_metrics_func: Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]):
    # ! if return is torch.Tensor, it has to be a scalar
    collector = lambda y_pred, y_true: (sum_of_metrics_func, len(y_true))
    def aggregator(results):
        correct, total = map(sum, zip(*results))
        ret = correct / total
        return ret.item() if isinstance(ret, torch.Tensor) else ret
    return collector, aggregator
