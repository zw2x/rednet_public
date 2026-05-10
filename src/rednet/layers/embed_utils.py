from typing import Tuple, Union
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common_utils import to_one_hot, to_pairwise_mask, exists


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, dim, max_positions=10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        half_d = dim // 2
        freqs = torch.exp(-math.log(max_positions) * torch.arange(start=0, end=half_d, dtype=torch.float32) / half_d)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # p = t.float()[:, None] * self.freqs[None, :]
        p = t.float().unsqueeze(-1) * self.freqs.unsqueeze(-2)
        p = torch.cat((p.sin(), p.cos()), dim=-1)
        if self.dim % 2 == 1:
            p = torch.cat((p, torch.zeros_like(p[..., :1])), dim=-1)
        return p


class FourierEmbedding(nn.Module):
    def __init__(self, dim, freqs_scale=16.0, use_bias=False, bias_scale=1.0, only_cosine=False):
        super().__init__()
        self.dim = dim
        _d = dim // 2 if not only_cosine else dim
        freqs = torch.randn(_d) * freqs_scale
        self.register_buffer("freqs", freqs)

        # alphafold3 add a bias term and only use cosine
        if use_bias:
            bias = torch.randn(_d) * bias_scale
            self.register_buffer("bias", bias)
        self.use_bias = use_bias
        self.only_cosine = only_cosine

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # p = t.float()[:, None] * self.freqs[None, :]
        p = t.float().unsqueeze(-1) * self.freqs.unsqueeze(-2)
        if self.use_bias:
            p += self.bias[None, :]
        p *= 2 * math.pi
        if self.only_cosine:
            p = p.cos()
        else:
            p = torch.cat([p.sin(), p.cos()], dim=-1)
            if self.dim % 2 == 1:
                p = torch.cat((p, torch.zeros_like(p[..., :1])), dim=-1)
        return p


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, out_dim, freq_embed_dim=256, embed_type: str = "fourier", **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, out_dim, bias=True),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )
        if embed_type == "positional":  # used in ddpm++ and adm
            self.timestep_embedding = PositionalEmbedding(freq_embed_dim, **kwargs)
        else:  # used in af3 and ncsn++
            self.timestep_embedding = FourierEmbedding(freq_embed_dim, **kwargs)

    def forward(self, t: torch.Tensor):
        assert t.ndim == 1, "Input tensor must be 1-dimensional"
        p = self.timestep_embedding(t)
        p_emb = self.mlp(p)
        return p_emb


def relpos_embed(index: torch.LongTensor, mask: torch.BoolTensor, cutoff: int = 64, shift=True) -> torch.Tensor:
    """
    relpos_ij = index_j - index_i
    Args:
        cutoff: int, the cutoff (inclusive)
    """
    num_cls = cutoff * 2 + 1 if shift else cutoff + 1
    relpos = (index[..., None, :] - index[..., :, None]).clamp(-cutoff, cutoff)
    if not shift:
        relpos.abs_()
    else:
        relpos += cutoff
    relpos_v = to_one_hot(relpos, num_cls=num_cls, dtype=torch.float32)
    if exists(mask):
        pair_mask = to_pairwise_mask(mask)
        relpos_v.masked_fill_(~pair_mask[..., None], 0.0)
    return relpos_v


class SeqPosEmbedding(nn.Module):
    def __init__(self, dim: int, out_dim: int | None = None):
        super().__init__()
        self.seq_pos_embedding = PositionalEmbedding(dim, max_positions=10000)
        self.chain_index_embedding = PositionalEmbedding(dim, max_positions=100)
        self.out_dim = out_dim or dim
        self.linear = nn.Linear(dim + dim, self.out_dim, bias=False)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, residue_index, chain_index):
        pos_embed = self.seq_pos_embedding(residue_index)
        c_embed = self.chain_index_embedding(chain_index)
        pos_embed = torch.cat([pos_embed, c_embed], dim=-1)
        pos_embed = self.norm(self.linear(pos_embed))
        return pos_embed


class RelPosEmbedding(nn.Module):
    def __init__(self, out_dim, cutoff: int = 64, add_post_norm=True, embed_residue=True, embed_entity=True):
        super().__init__()
        self.res_cutoff = cutoff
        self.out_dim = out_dim
        self.embed_residue = embed_residue
        self.embed_entity = embed_entity
        pair_dim = 2 + 2 * int(embed_entity) + (cutoff * 2 + 1) * int(embed_residue)
        self.linear = nn.Linear(pair_dim, self.out_dim, bias=False)

        self.add_post_norm = add_post_norm
        if self.add_post_norm:
            self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, *, chain_index, residue_index=None, entity_index=None, mask=None):
        same_chain = chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)
        embeds = [to_one_hot(same_chain.long(), num_cls=2, dtype=torch.float32)]
        if self.embed_entity:
            same_entity = entity_index.unsqueeze(-1) == entity_index.unsqueeze(-2)
            rel_ent_embed = to_one_hot(same_entity.long(), num_cls=2, dtype=torch.float32)
            embeds.append(rel_ent_embed)
        if self.embed_residue:
            rel_pos_embed = relpos_embed(residue_index, mask, cutoff=self.res_cutoff)
            rel_pos_embed.masked_fill_(~same_chain[..., None], 0.0)
            embeds.append(rel_pos_embed)
        pair_pos = torch.cat(embeds, dim=-1)
        if exists(mask):
            _pair_mask = to_pairwise_mask(mask)
            pair_pos.masked_fill_(~_pair_mask[..., None], 0.0)
        pair_embed = self.linear(pair_pos)

        if self.add_post_norm:
            pair_embed = self.norm(pair_embed)

        return pair_embed


@functools.cache
def get_position_ids_1d(batch_size, L, device):
    # [batch_size, L]
    return torch.arange(L, device=device).unsqueeze(0).repeat(batch_size, 1)


@functools.cache
def get_position_ids(batch_size, patch_nums, device, si=-1, m_maskgit=None):
    # [batch_size, L]
    all_position_ids = []
    largest_patch_num = patch_nums[-1]
    if si == -1:
        pns = patch_nums
    else:
        pns = patch_nums[si : si + 1]
    for level_idx in range(len(pns)):
        patch_num = pns[level_idx]
        _x = torch.arange(patch_num, device=device)
        _y = torch.arange(patch_num, device=device)
        # [pn, pn, 2]
        cartesian = torch.stack(torch.meshgrid(_x, _y, indexing="ij"), dim=-1)
        # normalize to the size in the largest feature map
        coords = cartesian / patch_num * largest_patch_num
        # [pn * pn, 2]
        coords = coords.reshape(-1, 2)
        all_position_ids.append(coords)
    # [batch_size, L, 2]
    pos_ids = torch.cat(all_position_ids, 0).unsqueeze(0).repeat(batch_size, 1, 1)
    if m_maskgit is None:
        return pos_ids
    pos_ids = pos_ids[m_maskgit]
    return pos_ids.reshape(batch_size, -1, pos_ids.shape[-1])


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# unsqueeze_dim=2 because by default our qk has shape [batch_size, seq_len, heads, head_dim]
def apply_rotary_pos_emb(q, k, cos: torch.Tensor, sin: torch.Tensor, position_ids=None, unsqueeze_dim=2):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding1D(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # position_ids: [bs, seq_len]
        # inv_freq: [head_size // 2]
        # inv_freq_expanded: [bs, head_size // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded: [bs, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs: [bs, seq_len, head_size // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [bs, seq_len, head_size]
            cos = emb.cos()
            sin = emb.sin()
        # [bs, seq_len, head_size]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 4, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # position_ids: [bs, seq_len]
        # inv_freq: [head_size // 4]
        # inv_freq_expanded: [bs, head_size // 4, 1, 1]
        inv_freq_expanded = (
            self.inv_freq[None, :, None, None].float().expand(position_ids.shape[0], -1, 1, 1).repeat(1, 1, 1, 2)
        )
        # position_ids_expanded: [bs, 1, seq_len, 2]
        position_ids_expanded = position_ids[:, None, :].float()
        inv_freq_expanded = inv_freq_expanded.permute(0, 3, 1, 2).contiguous()
        position_ids_expanded = position_ids_expanded.permute(0, 3, 1, 2).contiguous()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs: [bs, 2, seq_len, head_size // 4]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [bs, 2, seq_len, head_size // 2]
            cos = emb.cos()
            sin = emb.sin()
            # [bs, seq_len, 2, head_size // 2]
            cos = cos.transpose(2, 1).contiguous()
            sin = sin.transpose(2, 1).contiguous()
            cos = cos.reshape(cos.size(0), cos.size(1), -1)
            sin = sin.reshape(sin.size(0), sin.size(1), -1)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# from hart
class FusedRoPEFunc(torch.autograd.Function):
    """
    Function for FusedRoPE

    This implementation assumes the input tensor to be in `sbhd`, `bshd` or `thd` format and
    the RoPE tensor to be of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid
    the expensive `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
        # cu_seqlens: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if freqs.dtype != torch.float32:
            freqs = freqs.float()
        if tensor_format == "sbhd":
            output = hart_backend.fused_kernels.fused_rope_forward_func(t, freqs, False)
        elif tensor_format == "bshd":
            output = hart_backend.fused_kernels.fused_rope_forward_func(t.transpose(0, 1), freqs, True).transpose(0, 1)
        else:
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
        ctx.save_for_backward(freqs)
        ctx.tensor_format = tensor_format

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        (freqs,) = ctx.saved_tensors
        if ctx.tensor_format == "sbhd":
            grad_input = hart_backend.fused_kernels.fused_rope_backward_func(grad_output, freqs, False)
        elif ctx.tensor_format == "bshd":
            grad_input = hart_backend.fused_kernels.fused_rope_backward_func(
                grad_output.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        else:
            raise ValueError(f"Unsupported tensor_format: {ctx.tensor_format}.")

        return grad_input, None, None


class FusedRoPEFuncWithPos(torch.autograd.Function):
    """
    Function for FusedRoPE

    This implementation assumes the input tensor to be in `sbhd`, `bshd` or `thd` format and
    the RoPE tensor to be of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid
    the expensive `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,  # [B, S, D]
        tensor_format: str = "sbhd",
        # cu_seqlens: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if freqs.dtype != torch.float32:
            freqs = freqs.float()
        if tensor_format == "sbhd":
            output = hart_backend.fused_kernels.fused_rope_with_pos_forward_func(t, freqs, False)
        elif tensor_format == "bshd":
            output = hart_backend.fused_kernels.fused_rope_with_pos_forward_func(
                t.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        else:
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
        ctx.save_for_backward(freqs)
        ctx.tensor_format = tensor_format

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        raise NotImplementedError("Not implemented yet")
        # freqs, = ctx.saved_tensors
        # if ctx.tensor_format == "sbhd":
        #     grad_input = hart_backend.fused_kernels.fused_rope_backward_func(grad_output, freqs, False)
        # elif ctx.tensor_format == "bshd":
        #     grad_input = hart_backend.fused_kernels.fused_rope_backward_func(grad_output.transpose(0, 1), freqs, True).transpose(0, 1)
        # else:
        #     raise ValueError(f"Unsupported tensor_format: {ctx.tensor_format}.")

        # return grad_input, None, None


class FusedLlamaRotaryEmbedding1D(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        seq = torch.arange(max_position_embeddings, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i , j -> i j", seq, inv_freq)
        freqs = freqs.reshape(freqs.shape[0], 1, 1, -1)
        self.embs = torch.cat((freqs, freqs), dim=-1)

    def forward(self, x, seq_len=None, tensor_format="bshd"):
        self.embs = self.embs.to(x.device)
        # print(self.embs)
        # print(self.embs.shape)
        # exit()
        return FusedRoPEFunc.apply(x, self.embs[:seq_len], tensor_format)


class FusedLlamaRotaryEmbedding1DWithPos(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        seq = torch.arange(max_position_embeddings, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i , j -> i j", seq, inv_freq)
        freqs = freqs.reshape(freqs.shape[0], 1, 1, -1)

        self.embs = torch.cat((freqs, freqs), dim=-1)

    def forward(self, x, seq_len=None, position_ids=None, tensor_format="bshd"):
        if position_ids is not None:
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            # print(self.embs.shape)
            # print(context_position_ids.shape)
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            embs = torch.cat((freqs, freqs), dim=-1)  # [B, S, D]
            return FusedRoPEFuncWithPos.apply(x, embs, tensor_format)
        else:  # Original impl
            self.embs = self.embs.to(x.device)
            return FusedRoPEFunc.apply(x, self.embs[:seq_len], tensor_format)


class FusedLlamaRotaryEmbedding2DWithPos(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Haotian: now we have two directions x and y so inv_freq has a stride 4

        # NOTE: Shang: freq stride is 4 rather than 2. While freq is normalized by dim.
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 4, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, seq_len=None, position_ids=None, tensor_format="bshd"):
        if position_ids is not None:
            inv_freq_expanded = (
                self.inv_freq[None, :, None, None].float().expand(position_ids.shape[0], -1, 1, 1).repeat(1, 1, 1, 2)
            )
            # position_ids_expanded: [bs, 1, seq_len, 2]
            position_ids_expanded = position_ids[:, None, :].float()
            inv_freq_expanded = inv_freq_expanded.permute(0, 3, 1, 2).contiguous()
            position_ids_expanded = position_ids_expanded.permute(0, 3, 1, 2).contiguous()

            device_type = x.device.type
            device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                # freqs: [bs, 2, seq_len, head_size // 4]
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
                embs = torch.cat((freqs, freqs), dim=-1)

                embs = embs.transpose(2, 1).contiguous()
                embs = embs.reshape(embs.size(0), embs.size(1), -1)

            return FusedRoPEFuncWithPos.apply(x, embs, tensor_format)

        else:  # Original impl
            raise NotImplementedError("Not implemented yet")
            # self.embs = self.embs.to(x.device)
            # return FusedRoPEFunc.apply(x, self.embs[:seq_len], tensor_format)


# legacy embedding helpers (from core.encoding_utils.py)
def compute_rpe(index: torch.LongTensor, cutoff: int, mask=None, to_onehot=False) -> torch.Tensor:
    """Compute relative position encodings
    Args:
        index: [bsz, num_res]
        cutoff: cutoff distance
        mask: [bsz, num_res]
        to_onehot: whether to convert to one-hot encoding
    """
    rel_idx = index[..., :, None] - index[..., None, :]
    rel_idx = rel_idx.clamp(min=-cutoff, max=cutoff) + cutoff
    if mask is not None:
        pair_mask = mask[..., :, None] * mask[..., None, :]
        rel_idx.masked_fill_(~pair_mask, cutoff * 2 + 1)
    if to_onehot:
        rel_idx = F.one_hot(rel_idx, num_classes=cutoff * 2 + 2).float()
    return rel_idx


def compute_same_index(index: torch.LongTensor, mask=None, to_onehot=False) -> torch.Tensor:
    """Compute same index encodings
    Args:
        index: [bsz, num_res]
        mask: [bsz, num_res]
        to_onehot: whether to convert to one-hot encoding
    Returns:
        same_idx: [bsz, num_res, num_res, 3] or [bsz, num_res, num_res]. 0 for different, 1 for same, 2 for masked
    """
    same_idx = (index[..., :, None] == index[..., None, :]).long()
    if mask is not None:
        pair_mask = mask[..., :, None] * mask[..., None, :]
        same_idx.masked_fill_(~pair_mask, 2)
    if to_onehot:
        same_idx = F.one_hot(same_idx, num_classes=3).float()
    return same_idx


def compute_edge_encodings(
    inputs,
    mask=None,
    edge_index=None,
    add_rpe: bool = True,
    res_index_cutoff=64,
    use_chain_index=True,
    use_entity_index=True,
):
    from asimov.core import layout_utils

    all_acts = []
    raw_encodings = {}
    # rpe
    if add_rpe:
        res_index = inputs["res_index"]
        res_rpe = compute_rpe(
            res_index, cutoff=res_index_cutoff, mask=mask, to_onehot=True
        )  # [bsz, num_res, num_res, 2 * cutoff + 2]
        if use_chain_index:
            chain_index = inputs["chain_index"]
            chain_rpe = compute_same_index(chain_index, mask=mask, to_onehot=True)  # [bsz, num_res, num_res, 3]
        if use_entity_index:
            entity_index = inputs["entity_index"]
            entity_rpe = compute_same_index(entity_index, mask=mask, to_onehot=True)  # [bsz, num_res, num_res, 3]
        rpe = torch.cat([res_rpe, chain_rpe, entity_rpe], dim=-1)  # [bsz, num_res, num_res, dim]
        raw_encodings["res_rpe"] = res_rpe
        raw_encodings["chain_rpe"] = chain_rpe
        raw_encodings["entity_rpe"] = entity_rpe
        if edge_index is not None:
            rpe = layout_utils.gather_pair(rpe, edge_index)  # [bsz, num_res, knn, dim]
        all_acts.append(rpe)

    # TODO: cross attention and hybrid attention ape - aligned position encodings
    # TODO: spe - spatial position encodings

    assert len(all_acts) > 0, "no edge encodings provided"
    edge_acts = torch.cat(all_acts, dim=-1)
    return edge_acts, raw_encodings
