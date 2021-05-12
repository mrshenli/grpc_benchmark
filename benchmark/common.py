import torch
from torch import Tensor
from typing import Tuple

def identity(x: Tensor) -> Tensor:
    return x


@torch.jit.script
def identity_script(x: Tensor) -> Tensor:
    return x

# grpc + cuda requries sync=True
def heavy(x: Tensor, sync: bool = False) -> Tensor:
    for _ in range(100):
        x *= 2.0
        x /= 2.0
    if sync:
        torch.cuda.current_stream(0).synchronize()
    return x


@torch.jit.script
def heavy_script(x: Tensor, sync: bool = False) -> Tensor:
    for _ in range(100):
        x *= 2.0
        x /= 2.0
    if sync:
        torch.cuda.current_stream(0).synchronize()
    return x
