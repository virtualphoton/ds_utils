import os
from dataclasses import dataclass
from warnings import warn

import numpy as np
import torch
import torch.nn as nn

try:
    from plotter import History
except ImportError:
    try:
        from .plotter import History
    except ImportError:
        raise ImportError("download `plotter.py`")

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
                          history=self.history.state_dict())
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        torch.save(state_dict, self.path)
        self.save_history()
    
    def load_inplace(self):
        chkp = torch.load(self.path, "cpu")
        self.model.load_state_dict(chkp["model"])
        self.optimizer.load_state_dict(chkp["optimizer"])
        self.history.load_state_dict(chkp["history"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(chkp["scheduler"])
        
    def save_history(self):
        # for learning curves, need full history not just until the best
        torch.save(self.history.state_dict(),
                   f"{self.path}.history")
        
    def load_history(self):
        his = History()
        his.load_state_dict(torch.load(f"{self.path}.history"))
        return his
    
    def as_tuple(self):
        return self.model, self.optimizer, self.history, self.scheduler
    
@dataclass
class EarlyStopper:
    state: State
    loss: str = "loss"
    patience: int | None = 3
    min_delta: float = 0
    
    _last_history_len: int = None
    
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
    
    def hist_changed(self):
        return len(self.state.history) != self._last_history_len
    
    def __call__(self):
        """
        saves model on improvement
        returns True if training should stop else False
        """
        if not self.hist_changed(): return False
        
        self._last_history_len = len(self.state.history)
        losses = self.get_losses()
        
        if not len(losses):
            return False
        
        self._last_history_len = len(losses)
        
        if losses[-1] <= self.best_loss:
            self.best_loss = losses[-1]
            self.best_epoch = len(self.state.history)
            self.state.save()
            return False
        return len(losses) > self.patience and np.all(losses[-self.patience:] > self.best_loss + self.min_delta)
