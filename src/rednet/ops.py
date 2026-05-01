import torch
import torch.nn.functional as F

from einops import repeat


def exists(val):
    return val is not None


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


def to_pairwise_mask(x, ops="and"):
    if ops == "and":
        return x.unsqueeze(-2) * x.unsqueeze(-1)  # (..., N, 1) * (..., 1, N) -> (..., N, N)
    elif ops == "eq":
        return x.unsqueeze(-2) == x.unsqueeze(-1)  # (..., N, 1) == (..., 1, N) -> (..., N, N)
    elif ops == "diff":
        return x.unsqueeze(-2) - x.unsqueeze(-1)  # (..., N, 1) - (..., 1, N) -> (..., N, N)
    else:
        raise ValueError(f"Unsupported operation: {ops}. Supported operations are 'and' and 'eq'.")


def to_one_hot(x: torch.Tensor, num_cls, dtype=torch.float32, device=None):
    device = device or x.device
    return F.one_hot(x.long(), num_classes=num_cls).to(dtype=dtype, device=device)


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    if edges.ndim == 4:
        neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    else:
        neighbors = neighbor_idx
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # nodes (bsz, n, c)
    # neighbors_flat (bsz, t * k, c)
    neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def sample_order(mask, dsn_mask=None, randn=None, decoding_order=None):
    if exists(decoding_order):
        return decoding_order
    if not exists(randn):
        randn = torch.randn_like(mask, dtype=torch.float32)
    # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
    _mask = dsn_mask if exists(dsn_mask) else mask
    decoding_order = torch.argsort((_mask.float() + 0.0001) * (torch.abs(randn)))
    return decoding_order


def scatter_add_edge(pair_repr, edge_repr, edge_index, pair_mask, to_rowwise=True, to_colwise=True):
    assert to_rowwise or to_colwise, "At least one of to_rowwise or to_colwise must be True."
    edge_index = edge_index.long().clamp(min=0, max=pair_repr.shape[2] - 1)
    edge_index = repeat(edge_index, "... -> ... d", d=pair_repr.shape[-1])
    if to_rowwise:
        pair_repr = torch.scatter_add(pair_repr, 2, edge_index, edge_repr)
    if to_colwise:
        pair_repr = torch.scatter_add(pair_repr.transpose(1, 2), 2, edge_index, edge_repr).transpose(1, 2)
    if pair_mask is not None:
        pair_repr = pair_repr.masked_fill(~pair_mask.unsqueeze(-1), 0.0)
    return pair_repr


def make_causal_mask(order, group_index=None, edge_index=None, return_dense_masks=False):
    """
    Args:
        * group_index: (bsz, seqlen) LongTensor, positions that occur simultaneously have the same index.
    """
    bsz, seqlen = order.shape
    rank = torch.zeros_like(order)
    rank = rank.scatter(-1, order.long(), repeat(torch.arange(seqlen), "n -> b n", b=bsz).to(order))
    mask = rank[..., None] > rank[..., None, :]  # [bsz, seqlen, seqlen]
    if exists(group_index):
        same_grp = group_index.unsqueeze(-1) == group_index.unsqueeze(-2)
    else:
        same_grp = repeat(torch.eye(seqlen, device=rank.device, dtype=torch.bool), "... -> b ...", b=bsz)
    mask_include_self = mask.masked_fill(same_grp, True)
    mask_exclude_self = mask.masked_fill(same_grp, False)
    # [bsz, seqlen, k]
    if exists(edge_index):
        edge_mask_include_self = torch.gather(mask_include_self, -1, edge_index.long()).bool()
        edge_mask_exclude_self = torch.gather(mask_exclude_self, -1, edge_index.long()).bool()

    if return_dense_masks and exists(edge_index):
        return edge_mask_include_self, edge_mask_exclude_self, mask_include_self, mask_exclude_self
    elif exists(edge_index):
        return edge_mask_include_self, edge_mask_exclude_self
    return mask_include_self, mask_exclude_self
