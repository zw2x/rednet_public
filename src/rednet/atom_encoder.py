import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .layers import Mlp, PairwiseDropout
from .common_utils import exists
from .ops import gather_nodes


class AtomEncoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        pdim,
        # num_hidden,
        # num_in,
        *,
        dropout=0.1,
        num_heads=16,
        head_dim=64,
        add_global=True,
        add_gat=False,
        attn_dropout_p=0.1,
        msg_dim=None,
        msg_depth=1,
        node_mlp_depth=1,
        edge_mlp_depth=1,
        expansion_factor=2,
    ):
        super().__init__()
        self.dropout_p = dropout
        self.attn_dropout_p = attn_dropout_p
        self.num_heads = num_heads
        self.head_dim = head_dim
        msg_dim = msg_dim or pdim

        # node update
        self.node_src_lin = nn.Linear(dim, msg_dim, bias=False)
        self.node_tgt_lin = nn.Linear(dim, msg_dim, bias=False)
        self.edge_msg_lin = nn.Linear(pdim, msg_dim, bias=False)
        self.msg_mlp = Mlp(msg_dim, depth=msg_depth, expansion_factor=expansion_factor, apply_prenorm=True)

        self.node_mlp = Mlp(dim, depth=node_mlp_depth, expansion_factor=expansion_factor, apply_prenorm=True)

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

        # edge update
        self.out_node_src_lin = nn.Linear(dim, pdim, bias=False)
        self.out_node_tgt_lin = nn.Linear(dim, pdim, bias=False)
        self.edge_mlp = Mlp(pdim, depth=edge_mlp_depth, expansion_factor=expansion_factor, apply_prenorm=True)

        # masks are shared across columns (i.e. drop entire rows)
        self.edgewise_dropout = PairwiseDropout(p=self.dropout_p, orientation="col")

        self.init_weights()

    def init_weights(self):
        if self.add_global:
            nn.init.zeros_(self.gating_lin.weight)
            nn.init.constant_(self.gating_lin.bias, -1)

        if self.add_gat:
            # zero init gating
            nn.init.zeros_(self.gat_gating_lin.weight)
            nn.init.constant_(self.gat_gating_lin.bias, -1)

    def forward(self, node_repr, edge_repr, edge_index, edge_mask, mask=None):
        """Parallel computation of full transformer layer"""

        def _msg_to_edge(_node, _edge, node_lin, tgt_lin):
            _edge = _edge + gather_nodes(node_lin(_node), edge_index)
            _edge = _edge + repeat(tgt_lin(_node), "b n c -> b n k c", k=edge_index.size(-1))
            return _edge

        msg = _msg_to_edge(node_repr, self.edge_msg_lin(edge_repr), self.node_src_lin, self.node_tgt_lin)
        msg = self.msg_mlp(msg).masked_fill(~edge_mask.unsqueeze(-1).bool(), 0.0)

        dh = 0
        if self.add_global:
            edge_mask = edge_mask.to(msg)
            o = (msg * edge_mask.unsqueeze(-1)).sum(-2).div(edge_mask.sum(-1, keepdim=True) + 1e-6)
            o = self.gating_lin(node_repr).sigmoid() * o
            dh = dh + self.out_lin(o)

        if self.add_gat:
            attn_bias = rearrange(self.attn_bias_lin(self.act_fn(self.msg_lin(msg))), "b n k h -> b h n k")
            attn_bias = attn_bias.masked_fill(~edge_mask.unsqueeze(1).bool(), -torch.finfo(attn_bias.dtype).max)
            attn_weights = torch.softmax(attn_bias, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)
            v = rearrange(self.gat_value_lin(msg), "b n k (h d) -> b h n k d", h=self.num_heads)
            o = torch.einsum("bhnk, bhnkd -> bhnd", attn_weights, v)
            o = rearrange(o, "b h n d -> b n (h d)")
            # (b n d)
            o = self.gat_gating_lin(node_repr).sigmoid() * o
            dh = dh + self.gat_out_lin(o)

        node_repr = node_repr + F.dropout(dh, p=self.dropout_p, training=self.training)

        node_repr = node_repr + F.dropout(self.node_mlp(node_repr), p=self.dropout_p, training=self.training)
        if exists(mask):
            node_repr = node_repr.masked_fill(~mask.unsqueeze(-1).bool(), 0.0)

        edge_msg = _msg_to_edge(node_repr, edge_repr, self.out_node_src_lin, self.out_node_tgt_lin)
        edge_repr = edge_repr + self.edgewise_dropout(self.edge_mlp(edge_msg))

        return node_repr, edge_repr
