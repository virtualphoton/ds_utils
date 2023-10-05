from functools import wraps
from typing import TypeVar, Iterable, Hashable

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

class Apply(nn.Module):
    def __init__(self, func, *func_args, **func_kwargs):
        super().__init__()
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs
    
    def forward(self, X):
        return self.func(X, *self.func_args, **self.func_kwargs)
    
Arr = TypeVar("Arr", torch.Tensor, np.array)

def normalize(x: Arr, dim: int = -1, ord: int | float | str= 2, eps: float = 1e-6) -> Arr:
    if isinstance(x, np.ndarray):
        norm = np.linalg.norm(x, ord=ord, axis=dim, keepdims=True)
    else:
        norm = x.norm(dim=dim, p=ord, keepdim=True)
    return x / (norm + eps)

def train_val_test_split(*arrays, val_size: float, test_size: float = 0, **kwargs):
    rets_train_val = train_test_split(*arrays, test_size=test_size + val_size, **kwargs)
    
    val_size_cond = val_size / (test_size + val_size)
    rets_val_test = train_test_split(*rets_train_val[1::2], train_size=val_size_cond, **kwargs)
    
    train, val, test = rets_train_val[::2], rets_val_test[::2], rets_val_test[1::2]
    
    return sum(zip(train, val, test), start=())

def device_default(f):
    @wraps(f)
    def inner(*args, device=None, **kwargs):
        if device is None:
            device = f.__globals__["device"]
        return f(*args, **kwargs, device=device)
    return inner

def map_idx(values: Iterable[Hashable]) -> dict[Hashable, int]:
    return {value : idx for idx, value in enumerate(values)}

