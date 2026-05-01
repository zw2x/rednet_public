from typing import Union
import numpy as np
import scipy.spatial
import torch

def max_neg_value(tensor: torch.Tensor):
    return torch.finfo(tensor.dtype).min

def exists(val):
    return val is not None

def _sample_cat(probs: torch.Tensor, rng=None, replacement=True, num_samples=1) -> torch.LongTensor:
    _shape = probs.shape
    flat_probs = probs.view(-1, _shape[-1])
    sample = probs.new_empty(_shape[:-1] + (num_samples,), dtype=torch.long)
    torch.multinomial(
        flat_probs, replacement=replacement, num_samples=num_samples, generator=rng, out=sample.view(-1, num_samples)
    )
    return sample


def sample_multinomial(
    logits: torch.Tensor,
    top_k=None,
    top_p=None,
    rng=None,
    return_sample: bool = True,
    num_samples: int = 1,
    **kwargs,
):
    if exists(top_p) and top_p > 0:
        assert top_p < 1, "top_p must be less than 1"
        _sorted_logits, _sorted_ids = logits.sort(dim=-1, descending=False)
        if exists(top_k) and top_k > 0:
            assert top_k <= logits.shape[-1], "top_k must be less than the number of classes"
            _sorted_mask = _sorted_logits < _sorted_logits[..., -top_k].unsqueeze(-1)
            _sorted_logits.masked_fill_(_sorted_mask, -torch.inf)
        _sorted_mask = _sorted_logits.softmax(dim=-1).cumsum_(dim=-1) < (1 - top_p)
        _sorted_mask[..., -1] = False  # ensure at least one token is included
        logits_mask = _sorted_mask.scatter(-1, _sorted_ids, _sorted_mask)
    elif exists(top_k) and top_k > 0:
        assert top_k <= logits.shape[-1], "top_k must be less than the number of classes"
        logits_mask = logits < logits.topk(top_k, largest=True, sorted=False, dim=-1).values.amin(dim=-1, keepdim=True)
    else:
        logits_mask = None

    # post-process logits
    if exists(logits_mask):
        logits = logits.masked_fill(logits_mask, -torch.inf)
    if return_sample:
        probs = logits.softmax(dim=-1)
        sample = _sample_cat(probs, rng=rng, num_samples=num_samples, **kwargs)

    if return_sample:
        return sample, logits
    else:
        return logits

def redesign_mask(tokenizer, sample, design_chain_id: int | None = None, mask_ratio: float | None = None):
    if design_chain_id is None:
        design_mask = torch.ones_like(sample["chain_index"], dtype=torch.bool)
    else:
        design_mask = sample_design_mask(design_chain_id, sample["chain_index"], sample["entity_index"])
    masked_tokens, pred_mask = mask_residue_for_redesign(
        sample["res_type"], design_mask, mask_id=tokenizer.mask_id, num_cls=len(tokenizer)
    )
    sample["gt_res_type"] = sample["res_type"].clone()
    sample["enc_res_type"] = masked_tokens  # sample["res_type"].masked_fill(design_mask, tokenizer.mask_id)
    sample["dsn_mask"] = design_mask
    sample["pred_mask"] = pred_mask
    return sample


def sample_design_mask(chain_id, chain_index, entity_index, res_type=None, res_index=None):
    chain_mask = chain_index == chain_id
    entity_id = entity_index[chain_mask][0]
    entity_mask = entity_index == entity_id
    return entity_mask


def mask_residue_for_redesign(
    res_type: torch.LongTensor,
    design_mask: torch.BoolTensor,
    *,
    mask_id: int,
    num_cls: int,
    mask_ratio: float | None = None,
    dsn_mask_ratio: float | None = None,
    mutate_ratio: float = 0.1,
    keep_ratio: float = 0.1,
    profile_ratio: float | None = None,
    profile_probs: torch.Tensor | None = None,
):
    """
    Args:
        design_mask: positions to be redesigned.
    """
    if mask_ratio is not None or dsn_mask_ratio is not None:
        pred_mask = torch.zeros_like(design_mask, dtype=torch.bool)
        # mask design regions
        has_dsn_tokens = design_mask.any()
        has_pred_tokens = (~design_mask).any()
        if dsn_mask_ratio is not None and has_dsn_tokens:
            res_type, _pred_mask = _mask_tokens(
                res_type,
                design_mask,
                dsn_mask_ratio,
                mask_id,
                num_cls,
                mutate_ratio=mutate_ratio,
                keep_ratio=keep_ratio,
                profile_ratio=profile_ratio,
                profile_probs=profile_probs,
            )
            pred_mask += _pred_mask
        if mask_ratio is not None and has_pred_tokens:
            res_type, _pred_mask = _mask_tokens(
                res_type,
                ~design_mask,
                mask_ratio,
                mask_id,
                num_cls,
                mutate_ratio=mutate_ratio,
                keep_ratio=keep_ratio,
                profile_ratio=profile_ratio,
                profile_probs=profile_probs,
            )
            pred_mask += _pred_mask
        # mask everything in prediction regions
        masked_tokens = res_type.masked_fill(pred_mask, mask_id)
    else:
        masked_tokens = res_type.masked_fill(design_mask, mask_id)
        pred_mask = torch.ones_like(design_mask, dtype=torch.bool)

    return masked_tokens, pred_mask


def _mask_tokens(
    tokens: torch.LongTensor,
    mask: torch.BoolTensor,
    mask_ratio: float,
    mask_id: int,
    num_cls: int,
    *,
    mutate_ratio: float = 0.1,
    keep_ratio: float = 0.1,
    profile_ratio: float | None = None,
    profile_probs: torch.Tensor | None = None,
):
    pred_mask = (torch.rand_like(mask, dtype=torch.float) < mask_ratio) * mask
    _prob = torch.rand_like(mask, dtype=torch.float)
    masked_tokens = tokens.masked_fill(pred_mask, mask_id)

    use_profile = profile_probs is not None and profile_ratio is not None
    _ratio = mutate_ratio + keep_ratio + (profile_ratio if use_profile else 0)
    if _ratio == 0:
        return masked_tokens, pred_mask

    mutate_mask = (_prob < mutate_ratio) * pred_mask
    keep_mask = ((_prob >= mutate_ratio) * _prob < (keep_ratio + mutate_ratio)) * pred_mask
    # unmask positions below the ratio in the prediction mask
    masked_tokens.masked_fill_((_prob < _ratio) * pred_mask, 0)
    if keep_ratio > 0:
        masked_tokens += tokens.masked_fill(~keep_mask, 0)

    if mutate_ratio > 0:
        mutate_probs = torch.full_like(pred_mask, 1 / num_cls, dtype=torch.float)
        mutated_tokens, _ = sample_multinomial(mutate_probs)
        masked_tokens += mutated_tokens.squeeze(-1).masked_fill(~mutate_mask, 0)

    if use_profile:
        profile_mask = ((_prob >= keep_ratio + mutate_ratio) and _prob < _ratio) * pred_mask
        profile_tokens, _ = sample_multinomial(profile_probs)
        masked_tokens += profile_tokens.squeeze(-1).masked_fill(~profile_mask, 0)

    return masked_tokens, pred_mask

Points = Union[np.ndarray, scipy.spatial.KDTree, torch.Tensor]
Index = Union[np.ndarray, torch.LongTensor]


def _to_kd_tree(points: Points) -> scipy.spatial.KDTree:
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(points, np.ndarray):
        _tree = scipy.spatial.KDTree(points)
    else:
        assert isinstance(points, scipy.spatial.KDTree)
        _tree = points
    return _tree


def query_points(qry_points: Points, tgt_points: Points, radius=10.0) -> list[list[int]]:
    _qry_tree, _tgt_tree = map(_to_kd_tree, (qry_points, tgt_points))
    result = _tgt_tree.query_ball_tree(_qry_tree, r=radius)
    return result


def get_pairs(tgt_to_qry: list[list[int]], return_tensor=False, device=None) -> tuple[Index, Index]:
    row = [np.full_like(c, i) for i, c in enumerate(tgt_to_qry)]
    row, col = map(lambda x: np.concatenate(x, axis=0), (row, tgt_to_qry))
    if return_tensor:
        row, col = map(lambda x: torch.from_numpy(x).to(device=device, dtype=torch.long), (row, col))
    return row, col


def compute_dist(pos: np.ndarray, pos_mask: np.ndarray):
    d = np.linalg.norm(pos[..., :, None, :] - pos[..., None, :, :], axis=-1)
    pair_mask = (pos_mask[..., :, None] * pos_mask[..., None, :]).astype(d.dtype)
    d = d * pair_mask + 1e6 * (1 - pair_mask)
    return d


def find_interface_site(
    pos: torch.Tensor, pos_mask, chain_index, radius: float, return_pairs=False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Find interfacial residues: residues that are within a certain distance (radius) to residues from other chains.
    Args:
        return_pairs: bool. If True, return the indices of interfacial pairs
    """
    chain_ids = torch.unique(chain_index)
    # assert len(chain_ids) > 1, "There should be at least two chains"
    interface_mask = torch.zeros_like(chain_index, dtype=torch.bool)
    if len(chain_ids) == 1:
        return interface_mask
    inds = torch.arange(len(chain_index), device=chain_index.device)
    rows, cols = [], []
    for c in chain_ids:
        _chain_mask, _other_chain_mask = chain_index == c, (chain_index != c) * pos_mask
        _result = query_points(pos[_other_chain_mask], pos[_chain_mask], radius)
        _mask = torch.tensor([len(r) > 0 for r in _result], dtype=torch.bool)
        interface_mask[_chain_mask] = _mask.to(interface_mask)
        if return_pairs:
            row, col = get_pairs(_result, return_tensor=True, device=chain_index.device)
            rows.append(inds[_chain_mask][row])
            cols.append(inds[_other_chain_mask][col])
    interface_mask = interface_mask & pos_mask
    if return_pairs:
        row, col = torch.cat(rows, dim=0), torch.cat(cols, dim=0)
        return interface_mask, torch.stack([row, col], dim=0)
    else:
        return interface_mask
    
def calculate_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray):
    """
    Calculate dihedral angle between four points.
    
    Args:
        p1, p2, p3, p4: 3D coordinates as numpy arrays [x, y, z]
        
    Returns:
        Dihedral angle in degrees (-180 to 180)
    """
    # Vectors along the bonds
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    # Normal vectors to the planes
    n1 = np.cross(b1, b2, axis=-1)
    n2 = np.cross(b2, b3, axis=-1)
    
    # Normalize the normal vectors
    n1 = n1 / np.linalg.norm(n1, axis=-1, keepdims=True)
    n2 = n2 / np.linalg.norm(n2, axis=-1, keepdims=True)
    
    # Calculate the dihedral angle
    cos_angle = dotprod(n1, n2)
    sin_angle = dotprod(np.cross(n1, n2, axis=-1), b2 / np.linalg.norm(b2, axis=-1, keepdims=True))
    di = np.concatenate([cos_angle[..., None], sin_angle[..., None]], axis=-1)
    
    return di
   
def dotprod(n1, n2, dim=-1):
    return (n1 * n2).sum(dim)