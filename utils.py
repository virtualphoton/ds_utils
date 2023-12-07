import builtins

from toolz import compose
from collections.abc import Mapping
import dataclasses

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
