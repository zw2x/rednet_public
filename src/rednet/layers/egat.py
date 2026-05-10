import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, knn, radius
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax

from einops import rearrange, repeat


def _flat(x, mask) -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
    bsz, n, _ = x.shape
    idx = mask.bool().flatten().nonzero(as_tuple=True)[0]
    x = rearrange(x, "bsz n d -> (bsz n) d")[idx]
    b = torch.arange(bsz, device=x.device).repeat_interleave(n)[idx]
    return x, b, idx


def make_knn_graph(
    x: torch.Tensor, y: torch.Tensor, mask_x: torch.Tensor, mask_y: torch.Tensor, k: int = 64
) -> torch.LongTensor:
    """Construct k-NN graph from coordinates x to y.

    Args:
        x: [bsz, q, 3] query coordinates
        y: [bsz, k, 3] key coordinates
        mask_x: [bsz, q] boolean mask for x. True indicates valid points.
        mask_y: [bsz, k] boolean mask for y. True indicates valid points.
    """
    _x, b_x, idx_x = _flat(x, mask_x)
    _y, b_y, idx_y = _flat(y, mask_y)
    _max = min(k, mask_y.sum(-1).min().item())
    row, col = knn(_y, _x, k=_max, batch_x=b_y, batch_y=b_x)
    # maps back to original indices
    row = idx_x[row]
    col = idx_y[col]
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def make_radius_graph(
    x: torch.Tensor,
    y: torch.Tensor,
    mask_x: torch.Tensor,
    mask_y: torch.Tensor,
    r: float = 10.0,
    max_num_neighbors: int = 128,
) -> torch.LongTensor:
    _x, b_x, idx_x = _flat(x, mask_x)
    _y, b_y, idx_y = _flat(y, mask_y)
    _max = min(max_num_neighbors, _x.size(0), _y.size(0))
    row, col = radius(_y, _x, r=r, batch_x=b_y, batch_y=b_x, max_num_neighbors=_max)
    # assert len(row) > 0, 'x: {}, y: {}, r: {}'.format(_x, _y, r)
    # maps back to original indices
    row = idx_x[row]
    col = idx_y[col]
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


class EGATLayer(MessagePassing):
    """Single EGNN layer that updates both node features and coordinates."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        *,
        key_dim: int | None = None,
        hidden_dim: int = 64,
        n_heads: int = 8,
        head_dim: int = 32,
        n_output_points: int = 1,
        use_equivariant_updates: bool = True,
        skip_point_updates: bool = True,
    ):
        super().__init__(aggr="add", flow="target_to_source")
        self.edge_dim = edge_dim
        key_dim = key_dim or node_dim
        edge_input_dim = node_dim + key_dim + 1 + self.edge_dim  # h_i, h_j, ||x_i - x_j||^2, edge_attr
        self.edge_proj = nn.Linear(edge_input_dim, hidden_dim, bias=False)

        self.n_heads = n_heads
        self.att_src = nn.Linear(node_dim, hidden_dim, bias=False)
        self.att_dst = nn.Linear(key_dim, hidden_dim, bias=False)
        self.att_edge = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.attn_head = nn.Linear(hidden_dim, n_heads, bias=False)

        # update output coords
        self.skip_point_updates = skip_point_updates
        self.n_output_points = n_output_points
        self.use_equivariant_updates = use_equivariant_updates
        if not use_equivariant_updates:
            self.x_out_mlp = nn.Sequential(
                nn.Linear(node_dim + hidden_dim + 3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, n_output_points * 3, bias=False),
            )
        else:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, n_heads, bias=False),
            )
            self.weight_average = nn.Linear(n_heads, n_output_points, bias=False)

        # update node features
        inner_dim = n_heads * head_dim
        self.head_dim = head_dim
        self.inner_dim = inner_dim
        self.value_proj = nn.Linear(edge_input_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(node_dim + inner_dim, node_dim, bias=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, x: torch.Tensor, y: torch.Tensor, edge_index: Adj, edge_attr: OptTensor
    ):
        """
        Args:
            q: query node features [num_queries, node_dim]
            k: key node features [num_keys, key_dim]
            x (torch.Tensor): query coordinates [num_queries, 3]
            y (torch.Tensor): key coordinates [num_keys, 3]
            edge_index (torch.Tensor): Edge indices [2, E]. First row: source nodes, second row: target nodes
            edge_attr (torch.Tensor): Edge features [E, edge_dim]
        Returns:
            h_out: updated node features [num_queries, node_dim]
            x_out: updated coordinates [num_queries, n_output_points, 3]
        """
        row, col = edge_index
        num_nodes = q.size(0)
        # Compute relative positions and distances
        z = y[col] - x[row]  # [E, 3]
        dist_sq = (z**2).sum(dim=-1, keepdim=True)  # [E, 1]

        # Build edge inputs
        edge = torch.cat([q[row], k[col], dist_sq, edge_attr], dim=-1)

        # Compute edge messages
        m_ij = self.edge_proj(edge)

        # Compute attention scores
        alpha = self.att_src(q)[row] + self.att_dst(k)[col] + self.att_edge(m_ij)
        alpha = self.attn_head(self.leaky_relu(alpha))
        alpha = softmax(alpha, index=row, num_nodes=num_nodes)
        # Update node features
        v_ij = rearrange(self.value_proj(edge), " E (h d) -> E h d ", h=self.n_heads, d=self.head_dim)
        v_ij = rearrange(v_ij * alpha[..., None], "E h d -> E (h d)")
        o_i = torch.zeros(q.size(0), self.inner_dim, device=q.device)
        o_i.scatter_add_(0, repeat(row, "E -> E d", d=self.inner_dim), v_ij)

        h_out = self.out_proj(torch.cat([q, o_i], dim=-1))

        # Update coordinates
        if self.skip_point_updates:
            x_out = x[..., None]
        elif self.use_equivariant_updates:
            z = self.coord_mlp(m_ij)[..., None] * z[:, None]  # [E, n_heads, 3]
            z = alpha[..., None] * z  # [E, n_heads, 3]
            x_update = torch.zeros(num_nodes, self.n_heads, 3, device=q.device)
            x_update.scatter_add_(0, repeat(row, "E -> E h d", h=self.n_heads, d=3), z)
            x_update = self.weight_average(x_update.transpose(1, 2))
            x_out = rearrange(x[..., None] + x_update, "n d h -> n h d")
        else:
            x_update = self.x_out_mlp(torch.cat([q, o_i, x]))
            x_out = x[..., None] + rearrange(x_update, "n (h d) -> n h d", h=self.n_output_points)

        return h_out, x_out
