# MIT License
#
# Copyright (c) 2021 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn, einsum
from einops import rearrange, repeat

from .mlp import Mlp

# from .rotary_embedding import RotaryEmbedding, apply_rotary_emb
from ..common_utils import to_pairwise_mask, exists, default


def max_neg_value(x):
    max_neg_value = -torch.finfo(x.dtype).max
    return max_neg_value


class GraphAttention(nn.Module):
    def __init__(
        self,
        dim,
        pos_emb=None,
        dim_head=64,
        heads=8,
        edge_dim=None,
        bias=False,
        use_gate_proj=True,
        zero_init=False,
        kvdim=None,
    ):
        super().__init__()
        edge_dim = default(edge_dim, dim)

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.pos_emb = pos_emb

        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        kvdim = default(kvdim, dim)
        self.to_kv = nn.Linear(kvdim, inner_dim * 2, bias=bias)
        self.edges_proj = nn.Linear(edge_dim, heads, bias=bias)

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.use_gate_proj = use_gate_proj
        if use_gate_proj:
            self.gate_proj = nn.Linear(dim, inner_dim, bias=False)

        self.init_weights(zero_init=zero_init)

    def init_weights(self, zero_init=False):
        nn.init.xavier_uniform_(self.to_q.weight, gain=1.0)
        nn.init.xavier_uniform_(self.to_kv.weight, gain=1.0)
        nn.init.xavier_uniform_(self.edges_proj.weight, gain=1.0)
        if self.use_gate_proj and zero_init:
            nn.init.zeros_(self.gate_proj.weight)
            if exists(self.gate_proj.bias):
                nn.init.constant_(self.gate_proj.bias, -2.0)
        # zero init
        if zero_init:
            nn.init.zeros_(self.to_out.weight)
        else:
            nn.init.xavier_uniform_(self.to_out.weight, gain=1.0)
        if exists(self.to_out.bias):
            nn.init.zeros_(self.to_out.bias)

    def forward(self, nodes, edges, key=None, mask=None, attn_mask=None, use_cache=False, cache=None, static_kv=True):
        h = self.heads

        q = self.to_q(nodes)

        key = default(key, nodes)
        k, v = self.to_kv(key).chunk(2, dim=-1)
        if not static_kv and use_cache:
            k = cache["k_cache"].update(seqdim=1, value=k)
            v = cache["v_cache"].update(seqdim=1, value=v)

        e = self.edges_proj(edges)

        q, k, v, e = map(lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h), (q, k, v, e))
        if exists(self.pos_emb):
            freqs = self.pos_emb(torch.arange(nodes.shape[1], device=nodes.device))
            freqs = rearrange(freqs, "n d -> () () n d")
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        # ek, ev = e_kv, e_kv

        # k, v = map(lambda t: rearrange(t, "b j d -> b () j d "), (k, v))
        # k = k + ek
        # v = v + ev
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale + e.squeeze(-1)

        if exists(mask):
            mask = rearrange(mask, "b i -> b i ()") & rearrange(mask, "b j -> b () j")
            mask = repeat(mask, "b i j -> b h i j", h=h)
            sim = sim.masked_fill(~mask, max_neg_value(sim))

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask[:, None], max_neg_value(sim))

        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        if self.use_gate_proj:
            out = out * torch.sigmoid(self.gate_proj(nodes))
        out = self.to_out(out)
        return out


class GraphTransformer(nn.Module):
    def __init__(
        self,
        dim,
        pdim,
        *,
        kvdim=None,
        depth=1,
        dim_head=64,
        heads=8,
        use_rot_emb=False,
        accept_adjacency_matrix=False,
        dropout_rate=0.1,
        zero_init=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.adj_emb = nn.Embedding(2, pdim) if accept_adjacency_matrix else None
        # pos_emb = RotaryEmbedding(dim_head) if use_rot_emb else None
        pos_emb = None
        self.dropout_rate = dropout_rate

        def _build_layer():
            prenorm_attn = nn.ModuleList(
                [
                    nn.LayerNorm(dim),
                    nn.LayerNorm(pdim),
                    GraphAttention(
                        dim,
                        pos_emb=pos_emb,
                        edge_dim=pdim,
                        dim_head=dim_head,
                        heads=heads,
                        zero_init=zero_init,
                        kvdim=kvdim,
                    ),
                    nn.Dropout(self.dropout_rate),
                ]
            )
            prenorm_mlp = nn.ModuleList(
                [nn.LayerNorm(dim), Mlp(dim, zero_init=zero_init), nn.Dropout(self.dropout_rate)]
            )
            return nn.ModuleList([prenorm_attn, prenorm_mlp])

        for _ in range(depth):
            self.layers.append(_build_layer())

    def forward(
        self,
        nodes,
        edges,
        key=None,
        attn_mask=None,
        adj_mat=None,
        mask=None,
        use_cache=False,
        cache=None,
        static_kv=False,
        mask_empty_positions=True,
    ):
        batch, seq, _ = nodes.shape

        if exists(adj_mat):
            assert adj_mat.shape == (batch, seq, seq)
            assert exists(self.adj_emb), "accept_adjacency_matrix must be set to True"
            adj_mat = self.adj_emb(adj_mat.long())
            edges = edges + adj_mat

        for attn_block, ff_block in self.layers:
            attn_norm, edge_norm, attn, dropout = attn_block
            normed_nodes = attn_norm(nodes)
            normed_edges = edge_norm(edges)
            dx = attn(
                normed_nodes,
                normed_edges,
                key=key,
                mask=mask,
                attn_mask=attn_mask,
                use_cache=use_cache,
                cache=cache,
                static_kv=static_kv,
            )
            if mask_empty_positions and exists(attn_mask):
                _mask = attn_mask.any(dim=-1)
                dx = dx.masked_fill(~_mask.unsqueeze(-1), 0.0)
            nodes = dropout(dx) + nodes

            ffn_norm, ffn, dropout = ff_block
            normed_nodes = ffn_norm(nodes)
            nodes = dropout(ffn(normed_nodes)) + nodes

        return nodes
