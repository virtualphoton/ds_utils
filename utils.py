import builtins
import dataclasses

from toolz import compose
from collections.abc import Mapping
from functools import wraps

import numpy as np

def map(funcs, *args):
    try:
        iter(funcs)
        funcs = compose(*funcs)
    except TypeError:
        pass
    return list(builtins.map(funcs, *args))

def filter(*args):
    return list(builtins.filter(*args))

class Config(Mapping):
    # works only for dataclass instances
    # ! for variable to be defined as field, it must have annotation
    # all defined as None will be skipped, for they are treated as default
    def __len__(self):
        return len(dataclasses.fields(type(self)))
    
    def __iter__(self):
        return iter([i.name
                     for i in dataclasses.fields(type(self))
                     if self[i.name] is not None])
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, val):
        setattr(self, key, val)
        
    def __ior__(self, dic: Mapping):
        assert isinstance(dic, Mapping)
        for key, val in dic.items():
            self[key] = val
        return self
    
    def __or__(self, dic: Mapping):
        assert isinstance(dic, Mapping)
        for key, val in dic.items():
            self[key] = val
        return dict(self) | dic

def cast_all_lists_to_np(f):
    @wraps(f)
    def inner(*args, **kwargs):
        args = [np.array(arg) if isinstance(arg, list) else arg
                for arg in args]
        kwargs = {key : np.array(val) if isinstance(val, list) else val
                  for key, val in kwargs.items()}
        return f(*args, **kwargs)
    return inner
