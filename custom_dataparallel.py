from torch.nn.parallel._functions import Scatter, Gather
from typing import Any

import torch
import torch.nn as nn

def _is_namedtuple(obj: Any) -> bool:
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )

def scatter_kwargs_custom(
    inputs,
    kwargs,
    target_gpus,
    chunk_size,
    dim: int = 0,
):
    r"""Scatter with support for kwargs dictionary"""
    scattered_inputs = scatter_custom(inputs, target_gpus, dim, chunk_size=chunk_size) if inputs else []
    scattered_kwargs = scatter_custom(kwargs, target_gpus, dim, chunk_size=chunk_size) if kwargs else []
    if len(scattered_inputs) < len(scattered_kwargs):
        scattered_inputs.extend(() for _ in range(len(scattered_kwargs) - len(scattered_inputs)))
    elif len(scattered_kwargs) < len(inputs):
        scattered_kwargs.extend({} for _ in range(len(scattered_inputs) - len(scattered_kwargs)))
    return tuple(scattered_inputs), tuple(scattered_kwargs)

def scatter_custom(inputs, target_gpus, dim=0, chunk_size=None):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, chunk_size, dim, obj)
        if _is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for targets in target_gpus]
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None  # type: ignore[assignment]
    return res
    

class CustomDataParallel(nn.DataParallel):
    def __init__(self, model, chunk_size=None):
        super(CustomDataParallel, self).__init__(model)
        self.chunk_size = chunk_size
        
    def scatter(
        self,
        inputs,
        kwargs,
        device_ids):
        return scatter_kwargs_custom(inputs, kwargs, device_ids, chunk_size=self.chunk_size, dim=self.dim)

