import warnings
from collections import defaultdict
from typing import Callable, Any, Optional
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
from contextlib import contextmanager
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    print("couldn't load scheduler")

from .magic import reprint

def get_train_val(train_set, val_set, batch_size, num_workers=0, collate_fn=None):
    params = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    train_loader = DataLoader(train_set, shuffle=True, **params)
    val_loader = DataLoader(val_set, shuffle=False, **params)
    return train_loader, val_loader

@contextmanager
def dummy_ctx(*args, **kwargs):
    yield


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
    if isinstance(X, list | tuple):
        return [to(val, device) for val in X]
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
           do_scale: bool, epoch: int | None):
    
    metric_lists = defaultdict(list)
    do_loss = is_train or "loss" in next(zip(*metrics))
    
    optim.zero_grad()
    scaler = torch.cuda.amp.GradScaler()
        
    for i, (X, y) in enumerate(tqdm(dataloader, desc="train phase" if is_train else "val phase")):
        if epoch is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scheduler.step(epoch + i / len(dataloader))
            
        
        X, y = to(X, device), to(y, device)
        
        with (torch.autocast(device_type='cuda', dtype=torch.float16) if do_scale else dummy_ctx()):
            y_pred = model(X)
            loss = loss_fn(y_pred, y) / accum_grad if do_loss else None
        
        if is_train:
            (scaler.scale(loss) if do_scale else loss).backward()
            if not (i + 1) % accum_grad:
                if do_scale:
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
                    metric_lists[metric].append(fn(y_pred, y))    
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
           do_scale: bool = False, epoch: int | None = None):
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
                     do_scale=do_scale, epoch=epoch)
        model.train(not is_train)
    return ret
    
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
