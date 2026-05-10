from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch

_CACHE_TYPES = ["static", "dynamic", "dynamic_subset", "subset"]


def _gather(x, t) -> torch.Tensor:
    """
    Args:
        x: [bsz, seqlen, ...]
        t: [bsz] or [bsz, num_steps]
    Returns:
        x_t: [bsz, 1 / num_steps, ...]
    """
    # expand shape
    _shape = x.shape
    # _t = t.long()[(slice(None),) + (None,) * (len(_shape) - 1)].repeat(1, 1, *(_shape[2:]))
    t = t.long()
    if t.ndim == 1:
        t = t[:, None]
    _t = t[(slice(None), slice(None)) + (None,) * (len(_shape) - 2)].repeat(1, 1, *(_shape[2:]))
    return torch.gather(x, 1, _t)


def _scatter(x, t, y) -> None:
    """
    Args:
        x: [bsz, seqlen, ...]
        t: [bsz]
        y: [bsz, klen, ...]
    """
    # expand shape
    _shape = x.shape
    # _t = t.long()[(slice(None),) + (None,) * (len(_shape) - 1)].repeat(1, 1, *(_shape[2:]))
    t = t.long()
    if t.ndim == 1:
        t = t[:, None]
    _t = t[(slice(None), slice(None)) + (None,) * (len(_shape) - 2)].repeat(1, 1, *(_shape[2:]))
    x.scatter_(1, _t, y)


@dataclass(kw_only=True)
class CachedState:
    value: Optional[torch.Tensor] = None
    cache_type: str = "dynamic"  # dynamic update of dense attention, used for default autoregressive generation

    def __post_init__(self):
        assert self.cache_type in _CACHE_TYPES, f"cache_type must be one of {_CACHE_TYPES}"

    def update(self, value: torch.Tensor | None, seq_dim=None, timestep=None, cache_type=None) -> torch.Tensor:
        if value is None:
            return
        if cache_type is not None:
            assert self.cache_type == cache_type, f"cache_type must be {self.cache_type}, but got {cache_type}"

        if self.cache_type == "static":
            self.value = value

        elif self.cache_type == "dynamic":
            assert seq_dim is not None, "seq_dim must be provided for dynamic cache"
            # key_states: (bsz, n_heads, cur_length, head_dim)
            if self.value is not None:
                value = torch.cat([self.value, value], dim=seq_dim)
            self.value = value

        elif self.cache_type == "subset":
            assert self.value is not None, "key_states must be allocated for subset cache"
            assert timestep is not None, "timestep must be provided for subset cache"
            # states: (..., seq_dim, ...)
            self.value = self.value.to(value)
            # timestep = layout_utils._expand_index(self.value, timestep.long())
            # self.value.scatter_(seq_dim, timestep, value)
            self.value[:, timestep : timestep + 1] = value

        elif self.cache_type == "dynamic_subset":
            assert self.value is not None, "key_states must be allocated for subset cache"
            assert timestep is not None, "timestep must be provided for subset cache"
            # states: (..., seq_dim, ...)
            self.value = self.value.to(value)
            # self.value.scatter_(seq_dim, timestep[:, None, None].repeat(1, 1, value.shape[-1]), value)
            assert seq_dim is None or seq_dim == 1
            _scatter(self.value, timestep, value)
        else:
            raise ValueError(f"Invalid cache type: {self.cache_type}")
        return self.value

    def alloc(self, shape=None, fn=None, device=None, dtype=torch.float32):
        if self.cache_type in {"subset", "dynamic_subset"}:
            if not fn:
                self.value = torch.zeros(shape, dtype=dtype, device=device)
            else:
                self.value = fn(shape, device=device)
