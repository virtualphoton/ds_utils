import dataclasses
from pathlib import Path
from dataclasses import dataclass
from warnings import warn
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    print("couldn't load scheduler")

from .plotter import History

@dataclass
class State:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    _: dataclasses.KW_ONLY
    history: History = None
    scheduler: Optional["LRScheduler"] = None
    path: Path = None
    
    def __post_init__(self):
        assert self.path is not None
        self.path = Path(self.path)
        
        if self.history is None:
            self.history = History()
        
        if self.path.exists():
            warn(f"Saver: {self.path} already exists!")
        else:
            self.path.mkdir()
        self.load_history = self.__load_history
    
    def save(self):
        state_dict = dict(model=self.model.state_dict(),
                          optimizer=self.optimizer.state_dict(),
                          history=self.history.state_dict())
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        torch.save(state_dict, self.path / "model.chkp")
        self.save_history()
    
    def load_inplace(self):
        chkp = torch.load(self.path / "model.chkp", "cpu")
        self.model.load_state_dict(chkp["model"])
        self.optimizer.load_state_dict(chkp["optimizer"])
        self.history.load_state_dict(chkp["history"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(chkp["scheduler"])
        
    def save_history(self):
        # for learning curves, need full history not just until the best
        torch.save(self.history.state_dict(),
                   self.path / "history.chkp")
    
    def as_tuple(self) -> tuple[nn.Module, torch.optim.Optimizer, History, Optional["LRScheduler"]]:
        return self.model, self.optimizer, self.history, self.scheduler
    
    @staticmethod
    def load_history(path):
        his = History()
        his.load_state_dict(torch.load(path / "history.chkp"))
        return his
    
    def __load_history(self):
        return State.load_history(self.path)
    
    @staticmethod
    def load_model_into(path, model):
        chkp = torch.load(path / "model.chkp", "cpu")
        model.load_state_dict(chkp["model"])
        return model
    
@dataclass
class EarlyStopper:
    state: State
    loss: str = "loss"
    patience: int | None = 3
    min_epochs: int = 1
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
        
        if len(losses) < self.min_epochs:
            if len(losses):
                self.state.save()
            return False
        
        self._last_history_len = len(losses)
        
        if losses[-1] <= self.best_loss:
            self.best_loss = losses[-1]
            self.best_epoch = len(self.state.history)
            self.state.save()
            return False
        return len(losses) > self.patience and np.all(losses[-self.patience:] > self.best_loss + self.min_delta)
