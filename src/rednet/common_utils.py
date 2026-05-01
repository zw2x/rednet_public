from typing import Union, Callable
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys
import tree
import rich
import random
import inspect
import numpy as np
import pandas as pd
import scipy.spatial

import os
import dotenv

dotenv.load_dotenv()
import logging


import torch
import torch.nn.functional as F

__all__ = [
    "check_nan",
    "sample_multinomial",
    "mask_mean",
    "move_to_cuda",
    "move_to_device",
    "apply_tree",
    "collate",
    "resolve_config",
    "is_immutable",
    "get_func_kwargs",
    "print_batch",
    "random_int",
    "to_one_hot",
    "masked_fill",
    "collect_params",
    "to_pairwise_mask",
    "exists",
    "default",
    "add_default",
    "unique_ids",
    "to_tensor",
    "crop_by_mask",
    "_select_polymer_entities",
    "_make_atomsite",
    "query_points",
]


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def check_nan(inputs: dict[str, torch.Tensor], *, prefix=""):
    # check nan
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any():
                msg = f"NaN detected in {k}"
                if prefix:
                    msg = f"{prefix}: {msg}"
                raise ValueError(msg)


def sample_multinomial(probs: torch.Tensor, num_samples: int = 1, *, replacement: bool = True) -> torch.Tensor:
    # probs (..., n_classes)
    _shape = probs.shape[:-1]
    probs = probs.view(-1, probs.shape[-1])
    _samples = torch.multinomial(probs, num_samples=num_samples, replacement=replacement)
    return _samples.view(*_shape, num_samples)


def mask_mean(value: torch.Tensor, mask: torch.BoolTensor, dim=None, keepdim=False, eps=1e-6) -> torch.Tensor:
    assert value.shape == mask.shape, (value.shape, mask.shape)
    _sum = torch.sum(value * mask, dim=dim, keepdim=keepdim)
    _mean = _sum / (torch.sum(mask, dim=dim, keepdim=keepdim) + eps)
    return _mean


def move_to_cuda(batch: dict[str, torch.Tensor]):
    _out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            _out[k] = v.cuda()
        # elif isinstance(v, rigid_utils.Rigid):
        #     _out[k] = v.to(device="cuda")
        else:
            _out[k] = v
    return _out


def move_to_device(batch: dict[str, torch.Tensor], device: str | None = None):
    _out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            _out[k] = v.to(device=device)
        else:
            _out[k] = v
    return _out


def apply_tree(fn: Callable, inputs):
    return tree.map_structure(fn, inputs)


def _pad_multidim_tensors(tensors: list[torch.Tensor], pad_value=0, pad_dims=(0,)):
    maxlens = [max([t.shape[i] for t in tensors]) for i in pad_dims]
    padded = []
    for v in tensors:
        _shape = [[0, 0]] * v.ndim
        for _maxlen, _pad_dim in zip(maxlens, pad_dims):
            _shape[_pad_dim] = [0, _maxlen - v.shape[_pad_dim]]
        pad_shape = []
        for _sh in reversed(_shape):
            pad_shape += _sh
        v = F.pad(v, tuple(pad_shape), value=pad_value)
        padded.append(v)
    return padded


def collate(
    samples: list[dict[str, torch.Tensor]], ignored_keys=None, keys=None, pad_values=None, dtypes=None, pad_dims=None
):
    batch = {}
    pad_values = pad_values or {}
    pad_dims = pad_dims or {}
    dtypes = dtypes or {}

    # def _pad_tensor(k, v: torch.Tensor, maxlen):
    #     v = F.pad(v, (0, 0) * (v.ndim - 1) + (0, maxlen - v.shape[0]), value=pad_values.get(k, 0))
    #     if k in dtypes:
    #         v = v.to(dtype=dtypes[k])
    #     return v

    _sample = samples[0]
    keys = keys or sorted(_sample.keys())
    for key in keys:
        if key not in _sample:
            continue
        if ignored_keys and key in ignored_keys:
            continue
        _value = _sample[key]
        if isinstance(_value, torch.Tensor):
            if samples[0][key].ndim == 0:
                # scalar
                batch[key] = torch.tensor([sample[key] for sample in samples])
            # else:
            #     maxlen = max(sample[key].shape[0] for sample in samples)
            #     batch[key] = torch.stack([_pad_tensor(key, sample[key], maxlen) for sample in samples], dim=0)
            else:
                try:
                    batch[key] = torch.stack(
                        _pad_multidim_tensors(
                            [sample[key] for sample in samples],
                            pad_value=pad_values.get(key, 0),
                            pad_dims=pad_dims.get(key, (0,)),
                        ),
                        dim=0,
                    )
                except KeyError as e:
                    for sample in samples:
                        print_batch(sample)
                    raise e
                if dtypes.get(key):
                    batch[key] = batch[key].to(dtype=dtypes[key])
        else:
            batch[key] = [sample[key] for sample in samples]
        # elif is_immutable(_value):
        #     batch[key] = [sample[key] for sample in samples]
        # else:
        #     raise ValueError(f"Unsupported type: {type(_value)}")

    return batch


def resolve_config(config: Union[dict, DictConfig, Path], to_container=False) -> Union[dict, DictConfig]:
    if config is None:
        return
    if isinstance(config, Path):
        config = OmegaConf.load(config)
    if to_container and isinstance(config, DictConfig):
        config = OmegaConf.to_container(config)
    assert isinstance(config, (dict, DictConfig)), type(config)
    return config


def is_immutable(obj):
    return isinstance(obj, (str, int, float, bool, tuple, type(None), frozenset, bytes))


def get_func_kwargs(func, input_kwargs, ignore_varkw=False):
    spec = inspect.getfullargspec(func)
    if spec.varkw and not ignore_varkw:
        # pass everything to func
        return input_kwargs
    pass_to_func_keys = []
    if spec.kwonlyargs:
        pass_to_func_keys.extend(spec.kwonlyargs)
    if spec.defaults:
        with_default_args = spec.args[-len(spec.defaults) :]
        pass_to_func_keys.extend(with_default_args)
    pass_to_func_kwargs = {k: v for k, v in input_kwargs.items() if k in pass_to_func_keys}
    return pass_to_func_kwargs


def print_batch(batch: dict[str, torch.Tensor], desc=None):
    _out = ({k: (x.shape, x.dtype) if isinstance(x, (torch.Tensor, np.ndarray)) else x for k, x in batch.items()},)
    if desc:
        _out = (desc,) + _out
    rich.print(*_out)


def random_int():
    return random.randrange(0, sys.maxsize)


def to_one_hot(x: torch.Tensor, num_cls, dtype=torch.float32, device=None):
    device = device or x.device
    return F.one_hot(x.long(), num_classes=num_cls).to(dtype=dtype, device=device)


def masked_fill(
    x: torch.Tensor,
    fill_mask: torch.BoolTensor,
    fill_value: float = 0.0,
    num_lead_dims: int = 0,
    num_trail_dims: int = 0,
    normalize: bool = False,
) -> torch.Tensor:
    assert x.shape[num_lead_dims : x.ndim - num_trail_dims] == fill_mask.shape, (x.shape, fill_mask.shape)
    _shape = [1] * num_lead_dims + list(fill_mask.shape) + [1] * num_trail_dims
    fill_mask = fill_mask.bool().reshape(*_shape)
    if normalize:
        fill_value = 1.0 / x.shape[-1]
    x = x.masked_fill(fill_mask, fill_value)
    return x


# collect haiku params from model_params. this is used to get params from alphafold2
def collect_params(model_params, prefix: str):
    params = {}
    for k, v in model_params.items():
        if k.startswith(prefix):
            stem = k[len(prefix.strip("/")) :]
            if stem:
                params[stem.strip("/")] = v
            else:
                params.update(v)
    return params


def to_pairwise_mask(x, ops="and"):
    if ops == "and":
        return x.unsqueeze(-2) * x.unsqueeze(-1)  # (..., N, 1) * (..., 1, N) -> (..., N, N)
    elif ops == "eq":
        return x.unsqueeze(-2) == x.unsqueeze(-1)  # (..., N, 1) == (..., 1, N) -> (..., N, N)
    elif ops == "diff":
        return x.unsqueeze(-2) - x.unsqueeze(-1)  # (..., N, 1) - (..., 1, N) -> (..., N, N)
    else:
        raise ValueError(f"Unsupported operation: {ops}. Supported operations are 'and' and 'eq'.")


def add_default(x, out):
    if exists(out) and exists(x):
        out = out + x
    elif exists(x):
        out = x
    return out


def unique_ids(index, return_values=True, return_index=False):
    # order preserving
    mask = np.concatenate(([True], index[1:] != index[:-1]))
    if return_values:
        values = index[mask]
    if return_index:
        indices = np.where(mask)[0]
    if return_index and return_values:
        return values, indices
    elif return_values:
        return values
    elif return_index:
        return indices
    else:
        raise ValueError("At least one of return_values or return_index must be True.")


def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return x


def to_tensor(x):
    return tree.map_structure(_to_tensor, x)


def crop_by_mask(sample: dict[str, torch.Tensor], crop_mask: torch.BoolTensor) -> dict[str, torch.Tensor]:
    crop_mask = crop_mask.bool()

    def _crop(k, v):
        if isinstance(v, torch.Tensor):
            return v[crop_mask]
        else:
            return v

    _sample = {k: _crop(k, v) for k, v in sample.items()}

    return _sample


_LOG_LEVEL = getattr(logging, (os.environ["LOG_LEVEL"]).upper())


def _setup_root_logger(log_level=_LOG_LEVEL) -> None:
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


_setup_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    log = logging.getLogger(name)
    # print(f"Logger name: {name} {_LOG_LEVEL}")
    log.setLevel(_LOG_LEVEL)
    return log


def _select_polymer_entities(metadata, selected_types=["polypeptide(L)"]):
    _types = set(selected_types)
    ent_ids = [e for e, t in zip(metadata["_entity_poly.entity_id"], metadata["_entity_poly.type"]) if t in _types]
    return ent_ids


def _make_atomsite(atomsite: pd.DataFrame, entity_ids, only_ca=True):
    xyz = atomsite[["_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z"]].to_numpy()
    label_atom_id = atomsite["_atom_site.label_atom_id"].to_numpy()
    label_asym_id = atomsite["_atom_site.label_asym_id"].to_numpy()
    label_entity_id = atomsite["_atom_site.label_entity_id"].to_numpy()
    label_seq_id = atomsite["_atom_site.label_seq_id"].to_numpy()
    mask = np.isin(label_entity_id, entity_ids)
    if only_ca:
        mask = mask & (label_atom_id == "CA")
    out = {
        "xyz": xyz[mask],
        "label_atom_id": label_atom_id[mask],
        "label_asym_id": label_asym_id[mask],
        "label_seq_id": label_seq_id[mask],
    }
    return out


def query_points(xyz):
    tree = scipy.spatial.KDTree(xyz)
    index = tree.query_ball_point(xyz, r=10.0)
    pairs = np.array([[i, j] for i, js in enumerate(index) for j in js if i < j])
    return pairs
