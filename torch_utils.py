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
    
def normalize(x: torch.Tensor, dim: int = -1, norm: int | float | str= 2,
              eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, p=norm, keepdim=True) + eps)
