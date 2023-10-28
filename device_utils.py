import os
from functools import wraps, partial

import torch


def get_computing_device(dev = 0):
    if torch.cuda.is_available():
        if os.environ.get("IMLADRIS", "False").lower() == "true":
            # 1, 2 - A4000
            # 0, 3-6 - quadro
            physical_to_torch = {1:0, 2:1, 0:2, 3:3, 4:4, 5:5, 6:6}
            device = torch.device(f'cuda:{physical_to_torch[dev]}')
        else:
            device = torch.device("cuda")
        
    else:
        device = torch.device('cpu')
    return device

device = get_computing_device(6)
print(device)


def device_default(f, globs=None):
    if isinstance(f, dict):
        return partial(device_default, globs=f)
    @wraps(f)
    def inner(*args, device=None, **kwargs):
        if device is None:
            device = (globs or f.__globals__)["device"]
        return f(*args, **kwargs, device=device)
    return inner