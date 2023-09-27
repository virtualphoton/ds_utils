import torch
import torch.nn as nn

class Apply(nn.Module):
    def __init__(self, func, *func_args, **func_kwargs):
        super().__init__()
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs
    
    def forward(self, X):
        return self.func(X, *self.func_args, **self.func_kwargs)

def normalize(x: torch.Tensor, dim=-1, norm=2):
    return x / x.norm(dim=dim, p=norm, keepdim=True)
