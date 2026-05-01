import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .layers import Mlp, PairwiseDropout
from .common_utils import exists
from .ops import gather_nodes


class AtomDecoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        pdim,
        *,
        dropout=0.1,
        num_heads=16,
        head_dim=64,
        add_global=True,
        add_gat=False,
        msg_dim=None,
        msg_depth=2,
        mlp_depth=2,
        expansion_factor=4,
    ):
        super().__init__()
        self.dropout_p = dropout
        msg_dim = msg_dim or pdim

        self.edge_msg_lin = nn.Linear(pdim, msg_dim, bias=False)
        self.node_src_lin = nn.Linear(dim, msg_dim, bias=False)
        self.node_tgt_lin = nn.Linear(dim, msg_dim, bias=False)

        self.msg_mlp = Mlp(msg_dim, depth=msg_depth, expansion_factor=expansion_factor, apply_prenorm=True)

        self.add_global = add_global
        if self.add_global:
            self.gating_lin = nn.Linear(dim, msg_dim, bias=True)
            self.out_lin = nn.Linear(msg_dim, dim, bias=False)

        self.add_gat = add_gat

        if self.add_gat:
            self.num_heads = num_heads
            self.msg_lin = nn.Linear(msg_dim, msg_dim, bias=False)
            self.act_fn = nn.LeakyReLU()
            self.attn_bias_lin = nn.Linear(msg_dim, num_heads, bias=False)
            self.gat_value_lin = nn.Linear(msg_dim, num_heads * head_dim, bias=False)
            self.gat_gating_lin = nn.Linear(dim, num_heads * head_dim, bias=True)
            self.gat_out_lin = nn.Linear(num_heads * head_dim, dim, bias=False)

        self.node_mlp = Mlp(dim, expansion_factor=expansion_factor, depth=mlp_depth, apply_prenorm=True)
        self.init_weights()

    def init_weights(self):
        if self.add_global:
            nn.init.zeros_(self.gating_lin.weight)
            nn.init.constant_(self.gating_lin.bias, -1)

        if self.add_gat:
            # zero init gating
            nn.init.zeros_(self.gat_gating_lin.weight)
            nn.init.constant_(self.gat_gating_lin.bias, -1)

    def forward(
        self,
        node_repr,
        edge_repr,
        edge_index,
        edge_mask,
        *,
        mask=None,
        return_extra=False,
        mask_bw=None,
        cache=None,
        timestep=None,
    ):
        """Parallel computation of full transformer layer"""
        extra = {}

        def _msg_to_edge(_node, _edge, lin, tgt_lin):
            assert mask_bw is not None, "mask_bw must be provided for edge-wise message passing"
            _node_msg = lin(_node)
            if cache is not None and timestep is not None:
                _node_msg = cache["node_msg"].update(_node_msg, seq_dim=1, timestep=timestep)
            _edge = _edge + gather_nodes(_node_msg, edge_index) * mask_bw
            _edge = _edge + repeat(tgt_lin(_node), "b n c -> b n k c", k=edge_index.size(-1))
            return _edge

        msg = _msg_to_edge(node_repr, self.edge_msg_lin(edge_repr), self.node_src_lin, self.node_tgt_lin)
        msg = self.msg_mlp(msg).masked_fill(~edge_mask.bool().unsqueeze(-1), 0.0)

        dh = 0

        if self.add_global:
            edge_mask = edge_mask.to(node_repr)
            o = (msg * edge_mask[..., None]).sum(-2).div(edge_mask.sum(-1, keepdim=True) + 1e-6)
            o = self.gating_lin(node_repr).sigmoid() * o
            dh = dh + self.out_lin(o)

        if self.add_gat:
            attn_bias = rearrange(self.attn_bias_lin(self.act_fn(self.msg_lin(msg))), "b n k h -> b h n k")
            attn_bias = attn_bias.masked_fill(~edge_mask.bool().unsqueeze(1), -torch.finfo(attn_bias.dtype).max)
            attn_weights = torch.softmax(attn_bias, dim=-1)
            v = rearrange(self.gat_value_lin(msg), "b n k (h d) -> b h n k d", h=self.num_heads)
            o = torch.einsum("bhnk, bhnkd -> bhnd", attn_weights, v)
            o = rearrange(o, "b h n d -> b n (h d)")
            # (b n d)
            o = self.gat_gating_lin(node_repr).sigmoid() * o
            dh = dh + self.gat_out_lin(o)
            if return_extra:
                extra["gat_attn_map"] = attn_weights.detach().cpu().numpy()

        node_repr = node_repr + F.dropout(dh, p=self.dropout_p, training=self.training)

        dh = self.node_mlp(node_repr)
        node_repr = node_repr + F.dropout(dh, p=self.dropout_p, training=self.training)

        if exists(mask):
            node_repr = node_repr.masked_fill(~mask.unsqueeze(-1).bool(), 0.0)

        if return_extra:
            return node_repr, edge_repr, extra

        return node_repr, edge_repr
