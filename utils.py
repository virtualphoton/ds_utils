import builtins

import torch.nn as nn
from toolz import compose

def map(funcs, *args):
    try:
        iter(funcs)
        funcs = compose(*funcs)
    except TypeError:
        pass
    return list(builtins.map(funcs, *args))

def filter(*args):
    return list(builtins.filter(*args))
