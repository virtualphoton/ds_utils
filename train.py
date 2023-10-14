import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Any
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from .magic import reprint
except ImportError:
    warn("Couldn't load magic!")
    reprint = lambda t: t

__all__ = ["loopa", "ACC_METRIC", "EarlyStopper"]

ListOfMetrics = list[
    str |
    tuple[
        str,
        Callable[[torch.Tensor, torch.Tensor], Any], # metric calculation
        Callable[[list[Any]], float],                # metric aggregation
    ]
]

def to(X, device):
    if isinstance(X, dict):
        for key, val in X.items():
            X[key] = val.to(device)
        return X
    if isinstance(X, torch.Tensor):
        return X.to(device)
    raise RuntimeError(f"Incorrect type {type(X)}")

@reprint
def _loopa(*, model: nn.Module, dataloader: DataLoader, device: str,
           loss_fn, optim, metrics: ListOfMetrics,
           is_train: bool = True, accum_grad: int = 1,
           scheduler: torch.optim.lr_scheduler.LRScheduler | None = None):
    
    metric_lists = defaultdict(list)
    _metrics = []
    for metric in metrics:
        if isinstance(metric, str):
            if metric not in METRICS:
                raise RuntimeError(f"couldn't find metric: {metric}")
            _metrics.append((metric, *METRICS[metric]))
        else:
            _metrics.append(metric)
    metrics = _metrics
    do_loss = is_train or "loss" in next(zip(*metrics))
    
    optim.zero_grad() if is_train else None
        
    for i, (X, y) in enumerate(tqdm(dataloader, desc="train phase" if is_train else "val phase")):
        X, y = to(X, device), to(y, device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y) / accum_grad if do_loss else None
        
        if is_train:
            loss.backward()
            if not (i + 1) % accum_grad:
                optim.step()
                optim.zero_grad()
            
        with torch.no_grad():
            for metric, fn, _ in metrics:
                if metric == "loss":
                    metric_lists["loss"].append(loss.item() * accum_grad)
                else:
                    metric_lists[metric].append(fn(y_pred, y))
    if scheduler is not None and is_train:
        scheduler.step()
    if is_train and (i + 1) % accum_grad:
        # optim.step()
        optim.zero_grad()
    
    metric_results = {}
    for key, _, agg in metrics:
        metric_results[key] = agg(metric_lists[key])
    
    return metric_results

@reprint
def loopa(model: nn.Module, dataloader: DataLoader, *, device: str,
           loss_fn=None, optim=None, metrics: ListOfMetrics,
           is_train: bool = True, accum_grad: int = 1,
           scheduler: torch.optim.lr_scheduler.LRScheduler | None = None):
    if is_train:
        model.train()
        return _loopa(model=model, dataloader=dataloader, device=device,
                      loss_fn=loss_fn, optim=optim, metrics=metrics, accum_grad=accum_grad,
                      is_train=is_train, scheduler=scheduler)
    
    with torch.no_grad():
        model.eval()
        ret = _loopa(model=model, dataloader=dataloader, device=device,
                     loss_fn=loss_fn, optim=optim, metrics=metrics, accum_grad=accum_grad,
                     is_train=is_train, scheduler=scheduler)
        model.train()
        return ret

##########################################################################

try:
    from plotter import History
except ImportError:
    pass

@dataclass
class State:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    history: "History"
    path: str
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    
    def __post_init__(self):
        if os.path.exists(self.path):
            warn(f"Saver: {self.path} already exists!")
    
    def save(self):
        state_dict = dict(model=self.model.state_dict(),
                          optimizer=self.optimizer.state_dict(),
                          history=self.history)
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler
        torch.save(state_dict, self.path)
        self.save_history()
    
    def load_inplace(self):
        chkp = torch.load(self.path)
        self.model.load_state_dict(chkp["model"])
        self.optimizer.load_state_dict(chkp["model"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(chkp["scheduler"])
        
        self.history.train = chkp["history"].train
        self.history.val = chkp["history"].val
        self.history.drop_query = chkp["history"].drop_query
        
    def save_history(self):
        # for learning curves, need full history not just until the best
        torch.save(dict(history=self.history),
                   f"{self.path}.history")
        
    def load_history(self):
        return torch.load(f"{self.path}.history")["history"]
    
    def as_tuple(self):
        return self.model, self.optimizer, self.history, self.scheduler
    
@dataclass
class EarlyStopper:
    state: State
    loss: str = "loss"
    patience: int | None = 3
    min_delta: float = 0
    
    def __post_init__(self):
        self.best_epoch: int = -1 if not len(self.state.history) else self.get_losses().argmin() + 1
        self.best_loss = np.inf if not len(self.state.history) else self.get_losses().min()
        
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
                         for res in self.state.history.val
                         if metric in res])
    
    def __call__(self):
        """
        saves model on improvement
        returns True if training should stop else False
        """ 
        losses = self.get_losses()
        if not len(losses):
            return False
        
        if losses[-1] <= self.best_loss:
            self.best_loss = losses[-1]
            self.best_epoch = len(self.state.history)
            self.state.save()
            return False
        return len(losses) > self.patience and np.all(losses[-self.patience:] > self.best_loss + self.min_delta)

def mean_metric(sum_of_metrics_func: Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]):
    # ! if return is torch.Tensor, it has to be a scalar
    collector = lambda y_pred, y_true: (sum_of_metrics_func(y_pred, y_true), len(y_true))
    def aggregator(results):
        correct, total = map(sum, zip(*results))
        ret = correct / total
        return ret.item() if isinstance(ret, torch.Tensor) else ret
    return collector, aggregator

METRICS = {
    "loss": [None, np.mean],
    "acc": mean_metric(lambda y_pred, y_true: (y_pred.argmax(-1) == y_true).sum()),
}
