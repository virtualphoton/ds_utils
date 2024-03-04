import dataclasses
import inspect
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Any, Optional
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import Config
from .magic import reprint

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    print("couldn't load scheduler")

ListOfMetrics = list[
    str |
    tuple[
        str,
        Callable[[torch.Tensor, torch.Tensor], Any], # metric calculation
        Callable[[list[Any]], float],                # metric aggregation
    ]
]

def fetch_device(device, delta):
    # +1 from caller; +1 because this method is called
    # delta - number of intermediate step between global and caller of this method
    if device is None:
        device = inspect.stack()[2 + delta][0].f_globals.get("device", None)
    return device

@dataclass
class TrainConfig(Config):
    _: dataclasses.KW_ONLY
    model: nn.Module = None
    dataloader: DataLoader = None
    device: str = None
    loss_fn: Any = None
    optim: Any = None
    metrics: ListOfMetrics = None
    is_train: bool = None
    accum_grad: int = None
    scheduler: Optional["LRScheduler"] = None
    mixed_precision: bool = None
    epoch: int | None = None
    
    def with_state(self, state):
        self.model = state.model
        self.optim = state.optimizer
        self.scheduler = state.scheduler
        return self
    
    def __post_init__(self):
        self.device = fetch_device(self.device, delta=1)
        if self.device is None:
            raise RuntimeError("device must be either passed as kward, or be defined as global")


def get_train_val_loaders(train_set, val_set, *, batch_size, num_workers=0,
                          collate_fn=None, pin_memory=False) -> tuple[DataLoader, DataLoader]:
    params = dict(batch_size=batch_size, num_workers=num_workers,
                  collate_fn=collate_fn, pin_memory=pin_memory)
    train_loader = DataLoader(train_set, shuffle=True, **params)
    val_loader = DataLoader(val_set, shuffle=False, **params)
    return train_loader, val_loader

@contextmanager
def dummy_ctx(*args, **kwargs):
    yield


__all__ = ["loopa", "ACC_METRIC", "EarlyStopper"]

def to(X, device):
    if isinstance(X, list | tuple):
        return type(X)([to(val, device) for val in X])
    if isinstance(X, dict):
        for key, val in X.items():
            X[key] = to(val, device)
        return X
    if isinstance(X, torch.Tensor):
        return X.to(device)
    raise RuntimeError(f"Incorrect type {type(X)}")

@reprint
def _loopa(*, model: nn.Module, dataloader: DataLoader, device: str,
           loss_fn, optim, metrics: ListOfMetrics,
           is_train: bool, accum_grad: int,
           scheduler: "LRScheduler",
           mixed_precision: bool, epoch: int | None):
    
    metric_lists = defaultdict(list)
    do_loss = is_train or "loss" in next(zip(*metrics))
    
    optim.zero_grad()
    scaler = torch.cuda.amp.GradScaler()
        
    for i, batch in enumerate(tqdm(dataloader, desc="train phase" if is_train else "val phase")):
        batch = to(batch, device)
        if isinstance(batch, dict):
            X, y = batch.pop("X"), batch.pop("y")
            kwargs = batch
        elif isinstance(batch, (tuple, list)):
            X, y = batch
            kwargs = {}
        else:
            X = y = batch
            kwargs = {}
            
        if epoch is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scheduler.step(epoch + i / len(dataloader))
        
        with (torch.autocast(device_type='cuda', dtype=torch.float16) if mixed_precision else dummy_ctx()):
            y_pred = model(X, **kwargs)
            loss = loss_fn(y_pred, y, **kwargs) / accum_grad if do_loss else None
        
        if is_train:
            (scaler.scale(loss) if mixed_precision else loss).backward()
            if not (i + 1) % accum_grad:
                if mixed_precision:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad()
            
        with torch.no_grad():
            for metric, fn, _ in metrics:
                if metric == "loss":
                    metric_lists["loss"].append(loss.detach().cpu() * accum_grad)
                else:
                    metric_lists[metric].append(fn(y_pred, y, **kwargs))
    if is_train and (i + 1) % accum_grad:
        optim.step()
        optim.zero_grad()
    
    if epoch is None: scheduler.step()
    
    metric_results = {}
    for key, _, agg in metrics:
        metric_results[key] = agg(metric_lists[key])
    
    return metric_results

class DummyOptim:
    def zero_grad(self, *args, **kwargs): pass
    def step(self, *args, **kwargs): pass

@reprint
def loopa(model: nn.Module, dataloader: DataLoader, device: str, *,
           loss_fn=None, optim=None, metrics: ListOfMetrics,
           is_train: bool = True, accum_grad: int = 1,
           scheduler: Optional["LRScheduler"] = None,
           mixed_precision: bool = False, epoch: int | None = None):
    # preparation function for _loopa
    metrics_unified = []
    
    for metric in metrics:
        match metric:
            case [name, collector]:
                metrics_unified.append((name, collector, lambda _: None))
            case [_, _, _]:
                metrics_unified.append(metric)
            case str():
                if metric not in METRICS:
                    raise RuntimeError(f"couldn't find metric: {metric}")
                metrics_unified.append((metric, *METRICS[metric]))
            case _:
                raise RuntimeError("incorrect metric definition: {metric}")
    
    optim = optim if is_train else DummyOptim()
    scheduler = scheduler if scheduler is not None and is_train else DummyOptim()
    with (torch.no_grad() if not is_train else dummy_ctx()):
        model.train(is_train)
        ret = _loopa(model=model, dataloader=dataloader, device=device,
                     loss_fn=loss_fn, optim=optim, metrics=metrics_unified, accum_grad=accum_grad,
                     is_train=is_train, scheduler=scheduler,
                     mixed_precision=mixed_precision, epoch=epoch)
        model.train(not is_train)
    return ret

@reprint
def one_epoch(params, train_loader, val_loader, history, plotter):
    tr = loopa(**params, dataloader=train_loader, is_train=True)
    val = loopa(**params, dataloader=val_loader, is_train=False)
    history.push_epoch(tr, val)
    plotter.plot()
    
def mean_metric(sum_of_metrics_func: Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]):
    # ! if return is torch.Tensor, it has to be a scalar
    collector = lambda y_pred, y_true: (sum_of_metrics_func(y_pred, y_true), len(y_true))
    def aggregator(results):
        correct, total = map(sum, zip(*results))
        ret = correct / total
        return ret.item() if isinstance(ret, torch.Tensor) else ret
    return collector, aggregator

def save_into(container: list):
    def inner(y_pred, _):
        container.append(y_pred.detach().cpu())
    return inner

METRICS = {
    "loss": [None, np.mean],
    "acc": mean_metric(lambda y_pred, y_true: (y_pred.argmax(-1) == y_true).sum()),
}
