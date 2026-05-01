import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .rigid_utils import Rigid
from .ops import gather_edges, exists, to_pairwise_mask
from .layers import Mlp, RelPosEmbedding, TimestepEmbedder
from .layers.egat import EGATLayer, make_knn_graph, make_radius_graph
from .aux_utils import infer_atom_type, transform_atom_from_fa37, flatten_atoms, ATOM_TYPES


@torch.autocast(enabled=False, device_type="cuda")
def compute_dist(pos, pos_mask, eps=1e-6, tgt_pos=None, tgt_pos_mask=None):
    pos = pos.float()
    tgt_pos = tgt_pos.float() if exists(tgt_pos) else pos
    tgt_pos_mask = tgt_pos_mask.bool() if exists(tgt_pos_mask) else pos_mask.bool()
    # pos: [bsz, n, k, 3]
    d = rearrange(pos, "b n k d -> b n () k () d") - rearrange(tgt_pos, "b n k d -> b () n () k d")
    d = (d.square().sum(-1) + eps).sqrt()
    dmask = rearrange(pos_mask, "b n k -> b n () k ()") * rearrange(tgt_pos_mask, "b n k -> b () n () k")
    d.masked_fill_(~dmask, 1e4)  # mask out invalid distances
    d = rearrange(d, "b n m k l -> b n m (k l)")
    return d


@torch.autocast(enabled=False, device_type="cuda")
def infer_cb_pos(pos, pos_mask):
    pos = pos.float()
    b = pos[:, :, 1, :] - pos[:, :, 0, :]  # n->ca
    c = pos[:, :, 2, :] - pos[:, :, 1, :]  # ca->c
    a = torch.cross(b, c, dim=-1)
    cb_pos = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + pos[:, :, 1, :]
    cb_mask = torch.prod(pos_mask[:, :, :3].bool(), dim=-1)  # mask for Cb
    return cb_pos, cb_mask


class AtomFeaturizer(nn.Module):
    def __init__(
        self,
        dim,
        pdim,
        *,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=48,
        augment_eps=0.0,
        version="0.0.1",
        add_sc_embedding=False,
        add_frame_shifts=False,
        add_enc_res_type=False,
        tokenizer=None,
        use_out_mlp=False,
        add_recycle_node=False,
        add_pred_tokens=False,
        add_pred_sc=False,
        **unused,
    ):
        super().__init__()
        self.dim = dim
        self.pdim = pdim
        self.edge_features = pdim
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.add_enc_res_type = add_enc_res_type
        self.is_deprecated = not version
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)

        rbf_bins = [2, 22, 16]
        num_core_atoms = 5  # N, CA, C, O, pseudo-CB
        self.register_buffer("rbf_mu", torch.linspace(*rbf_bins))
        # self.register_buffer('rbf_scale', torch.tensor((rbf_bins[1] - rbf_bins[0]) / rbf_bins[2], device='cuda'))
        self.relpos_embedding = RelPosEmbedding(pdim, embed_entity=False, add_post_norm=False)
        num_atom_pairs = num_core_atoms**2
        # self.dist_mlp = Mlp(_in_dim, out_dim=pdim, expansion_factor=1, apply_prenorm=False, embed_dim=pdim)
        self.dist_lin = nn.Linear((rbf_bins[-1] + 1) * num_atom_pairs, pdim, bias=False)
        self.add_frame_shifts = add_frame_shifts
        if self.add_frame_shifts:
            _in_dim = num_core_atoms * 3
            self.shift_lin = nn.Linear(_in_dim, pdim, bias=False)
        self.add_sc_embedding = add_sc_embedding
        if self.add_sc_embedding:
            num_atoms = 32
            _in_dim = 1 * num_atoms  # cb to sidechain
            self.sc_dist_lin = nn.Linear(_in_dim * rbf_bins[-1], pdim, bias=False)
        if self.add_enc_res_type:
            self.enc_res_type_embedding = nn.Embedding(self.vocab_size, dim)

        self.add_pred_tokens = add_pred_tokens
        if self.add_pred_tokens:
            self.pred_token_embedding = nn.Embedding(self.vocab_size, dim)
            self.pred_token_gating = nn.Linear(dim, dim, bias=True)

        self.add_recycle_node = add_recycle_node
        if self.add_recycle_node:
            self.recycle_node_lin = nn.Linear(dim, dim, bias=False)
            self.recycle_node_gating = nn.Linear(dim, dim, bias=True)

        self.add_pred_sc = add_pred_sc
        self.use_recycle = add_recycle_node or add_pred_sc or add_pred_tokens
        if self.use_recycle:
            self.recycle_num_embedder = TimestepEmbedder(dim, embed_type="fourier")

        self.use_out_mlp = use_out_mlp
        if self.use_out_mlp:
            self.out_mlp = Mlp(pdim, apply_prenorm=True)

        self.init_weights()

    def init_weights(self):
        if not self.is_deprecated:
            if self.add_pred_tokens:
                nn.init.zeros_(self.pred_token_gating.weight)
                nn.init.constant_(self.pred_token_gating.bias, -2.0)  # initially closed
            if self.add_recycle_node:
                nn.init.zeros_(self.recycle_node_gating.weight)
                nn.init.constant_(self.recycle_node_gating.bias, -2.0)

    @torch.autocast(enabled=False, device_type="cuda")
    def compute_rbf(self, d):
        # sigma = 1
        d = d.float()
        rbf = torch.exp(-(d[..., None] - self.rbf_mu).square())
        return rbf

    def make_multigraph(self, pos, pos_mask, *, eps=1e-6):
        pair_mask = to_pairwise_mask(pos_mask[..., 1])
        d = ((pos[..., 1, :].unsqueeze(-2) - pos[..., 1, :].unsqueeze(-3)).square().sum(-1) + eps).sqrt()
        d.masked_fill_(~pair_mask, 1e4)
        _, edge_index = torch.topk(d, min(self.top_k, pos.shape[1]), dim=-1, largest=False)
        return edge_index

    def forward(self, inputs):
        pos = inputs["atom_positions"]
        atom_mask = inputs["atom_mask"]
        mask = inputs["mask"]
        res_index = inputs["res_index"]
        chain_index = inputs["chain_index"]
        bsz, seqlen = mask.shape
        if self.augment_eps > 0 and self.training:
            pos = pos + self.augment_eps * torch.randn_like(pos)

        # cb_pos, cb_mask = infer_cb_pos(pos, atom_mask)
        # in_pos = torch.cat([pos[..., :4, :], cb_pos.unsqueeze(-2)], dim=-2)  # [bsz, num_res, 5, 3]. bb + pseudo-cb
        # in_mask = torch.cat([atom_mask[..., :4], cb_mask.unsqueeze(-1)], dim=-1).bool()  # [bsz, num_res, 5]
        in_pos, in_mask = infer_positions(pos, atom_mask)
        inputs["input_pos"] = pos
        inputs["inferred_pos"] = in_pos
        inputs["inferred_mask"] = in_mask
        # backbone distances
        edge_index = self.make_multigraph(in_pos, in_mask)
        d = compute_dist(in_pos, in_mask)
        rbf_feats = rearrange(self.compute_rbf(d), "... k d -> ... (k d)")
        inv_d = 1 / (1 + d)
        pair_feats = self.dist_lin(torch.cat([rbf_feats, inv_d], dim=-1))

        # relative position embeddings
        relpos_feats = self.relpos_embedding(chain_index=chain_index, residue_index=res_index, mask=mask)
        pair_feats += relpos_feats

        # encode frame shifts
        if self.add_frame_shifts:
            # compute frames
            # frame_mask: [bsz, num_frames, 1]
            frame_mask = torch.prod(atom_mask[:, :, [0, 1, 2]], dim=-1, keepdim=True)
            frame = Rigid.from_points(
                origin=pos[:, :, 1], x_axis=pos[:, :, 0], xy_plane=pos[:, :, 2], mask=frame_mask.squeeze(-1)
            )

            shift = frame[..., None, None].inverse_apply_to_point(in_pos[..., None, :, :, :])
            _pair_mask = frame_mask[..., None].bool() * in_mask[:, None, :, :].bool()
            shift.masked_fill_(~_pair_mask[..., None], 0.0)
            shift = rearrange(shift, "... n c -> ... (n c)")
            pair_feats += self.shift_lin(shift)

        # encode tokens
        if self.add_sc_embedding:
            # cb to sidechain distances (bsz, num_res, num_res, k * tgt_k)
            sc_dist = compute_dist(
                in_pos[..., -1:, :], in_mask[..., -1:], tgt_pos=pos[..., 5:, :], tgt_pos_mask=atom_mask[..., 5:]
            )
            same_chain = chain_index[:, :, None] == chain_index[:, None, :]
            sc_dist.masked_fill_(inputs["dsn_mask"][:, None, :, None], 1e4)  # mask sidechains in the design regions
            sc_dist.masked_fill_(same_chain[..., None], 1e4)
            sc_rbf_feats = rearrange(self.compute_rbf(sc_dist), "... k d -> ... (k d)")
            # inv_sc_dist = 1 / (1 + sc_dist)
            sc_embed = self.sc_dist_lin(sc_rbf_feats)
            pair_feats += sc_embed

        node_feats = torch.zeros((bsz, seqlen, self.dim), device=pair_feats.device, dtype=pair_feats.dtype)

        if self.use_out_mlp:
            pair_feats = self.out_mlp(pair_feats)

        edge_feats = gather_edges(pair_feats, edge_index)

        extra = {}
        return edge_feats, edge_index, pair_feats, node_feats, extra


def infer_positions(atom_positions, atom_mask):
    cb_pos, cb_mask = infer_cb_pos(atom_positions, atom_mask)
    # [bsz, num_res, 5, 3]. bb + pseudo-cb
    in_pos = torch.cat([atom_positions[..., :4, :], cb_pos.unsqueeze(-2)], dim=-2)
    # [bsz, num_res, 5]
    in_mask = torch.cat([atom_mask[..., :4], cb_mask.unsqueeze(-1)], dim=-1).bool()
    return in_pos, in_mask


def get_topk_edges(pos, pos_mask, *, top_k=64, eps=1e-6):
    pair_mask = to_pairwise_mask(pos_mask[..., 1])
    d = ((pos[..., 1, :].unsqueeze(-2) - pos[..., 1, :].unsqueeze(-3)).square().sum(-1) + eps).sqrt()
    d.masked_fill_(~pair_mask, 1e4)
    _, edge_index = torch.topk(d, min(top_k, pos.shape[1]), dim=-1, largest=False)
    return edge_index


def infer_flatten_atom_features(
    res_type, res_index, chain_index, atom_pos, atom_pos_mask, num_atom_classes, num_residue_classes, pad_id
):
    atom_type, atom_exists, dense_atom_type = infer_atom_type(res_type, return_dense_atom_type=True)
    flat_atom_pos, flat_atom_pos_mask = transform_atom_from_fa37(res_type, atom_pos, atom_pos_mask, pad_id)
    _atom_type = rearrange(dense_atom_type, "b r a -> b (r a)")
    flat_atom_pos = rearrange(flat_atom_pos, "b r a c -> b (r a) c")
    flat_atom_pos_mask = rearrange(flat_atom_pos_mask, "b r a -> b (r a)")
    flat_atom_pos = flatten_atoms(flat_atom_pos, _atom_type, pad_value=0)
    flat_atom_pos_mask = flatten_atoms(flat_atom_pos_mask, _atom_type, pad_value=0).bool()
    assert flat_atom_pos_mask.shape == atom_exists.shape
    assert flat_atom_pos.shape[:-1] == atom_exists.shape, f"{flat_atom_pos.shape} vs {atom_exists.shape}"
    _res_index = repeat(res_index, "b r -> b (r a)", a=dense_atom_type.size(-1))
    flat_res_index = flatten_atoms(_res_index, _atom_type)
    _chain_index = repeat(chain_index, "b r -> b (r a)", a=dense_atom_type.size(-1))
    flat_chain_index = flatten_atoms(_chain_index, _atom_type)
    # build flatten atom features
    atom_type_one_hot = F.one_hot(atom_type, num_classes=num_atom_classes).float()
    _res_type = repeat(res_type, "b r -> b (r a)", a=dense_atom_type.size(-1))
    flat_res_type = flatten_atoms(_res_type, _atom_type, pad_value=num_residue_classes - 1)
    res_type_one_hot = F.one_hot(flat_res_type, num_classes=num_residue_classes).float()
    atom_f = torch.cat([atom_type_one_hot, res_type_one_hot, atom_exists[..., None].float()], dim=-1)
    return atom_f, flat_atom_pos, flat_atom_pos_mask, flat_res_index, flat_chain_index, atom_type


def _flatten_batch(x):
    return rearrange(x, "b r ... -> (b r) ...")


class FullAtomStructureFeaturizer(AtomFeaturizer):
    def __init__(
        self,
        dim,
        pair_dim,
        *,
        atom_feat_dim=64,
        cent_hid_dim=64,
        cent_edge_dim=64,
        pairwise_dropout_rate: float = 0.2,
        nodewise_dropout_rate: float = 0.2,
        top_k_centroid_to_atom: int = 96,
        centroid_to_atom_radius: float = 15.0,
        pred_backbone_positions=False,
        use_radius_graph=False,
        **kwargs,
    ):
        super().__init__(dim, pair_dim, **kwargs)

        _in_dim = len(ATOM_TYPES) + 1 + self.vocab_size + 1  # atom type one-hot + residue type one-hot + atom exists
        self.atom_feat_proj = nn.Linear(_in_dim, atom_feat_dim, bias=False)

        self.cent_radius = centroid_to_atom_radius
        self.use_radius_graph = use_radius_graph
        self.cent_to_atom_offset = 32
        _in_dim = self.num_rbf + 1 + (2 * self.cent_to_atom_offset + 1) + 1
        self.cent_edge_feat_proj = nn.Linear(_in_dim, cent_edge_dim, bias=False)
        self.pred_backbone_positions = pred_backbone_positions
        n_output_points = 4 if pred_backbone_positions else 1
        self.cent_attention = EGATLayer(
            dim,
            edge_dim=cent_edge_dim,
            hidden_dim=cent_hid_dim,
            key_dim=atom_feat_dim,
            n_output_points=n_output_points,
            skip_point_updates=not pred_backbone_positions,
        )

        self.final_node_norm = nn.LayerNorm(self.dim)
        expanded_dim = dim
        self.colwise_mlp = Mlp(self.dim, out_dim=expanded_dim, expansion_factor=2, apply_prenorm=False)
        self.rowwise_mlp = Mlp(self.dim, out_dim=expanded_dim, expansion_factor=2, apply_prenorm=False)
        self.pair_mlp = Mlp(expanded_dim, out_dim=pair_dim, expansion_factor=2, apply_prenorm=True)
        self.pair_dropout_rate = pairwise_dropout_rate
        self.node_dropout_rate = nodewise_dropout_rate
        self.num_atom_classes = len(ATOM_TYPES) + 1  # including padding
        self.top_k_centroid_to_atom = top_k_centroid_to_atom
        self.pad_id = self.tokenizer.pad_id
        self.mask_id = self.tokenizer.mask_id

    def forward(self, inputs):
        edge_f, edge_inds, pair_f, node_f, _ = super().forward(inputs)
        pos = inputs["input_pos"][..., 1, :]  # [bsz, num_res, 3]
        pos_mask = inputs["atom_mask"][..., 1]  # [bsz, num_res]
        # atom graph
        # [bsz, num_atom, d], [bsz, num_atom, 3], [bsz, num_atom]
        input_res_type = inputs["res_type"].masked_fill(inputs["dsn_mask"], self.mask_id)
        res_index = inputs["res_index"]
        chain_index = inputs["chain_index"]
        atom_f, atom_pos, atom_mask, flat_res_index, flat_chain_index, _atom_type = infer_flatten_atom_features(
            input_res_type,
            res_index,
            chain_index,
            atom_pos=inputs["input_pos"],
            atom_pos_mask=inputs["atom_mask"],
            num_atom_classes=self.num_atom_classes,
            num_residue_classes=self.vocab_size,
            pad_id=self.pad_id,
        )
        # atom_f: [bsz, num_atom, atom_feat_dim]
        atom_f = self.atom_feat_proj(atom_f)
        # build centroid-to-atom features
        if self.use_radius_graph:
            cent_to_atom_edge_inds = make_radius_graph(
                pos, atom_pos, pos_mask, atom_mask, r=self.cent_radius, max_num_neighbors=self.top_k_centroid_to_atom
            )
        else:
            cent_to_atom_edge_inds = make_knn_graph(pos, atom_pos, pos_mask, atom_mask, k=self.top_k_centroid_to_atom)
        # [E, d]
        cent_feats = {"pos": pos, "pos_mask": pos_mask, "res_index": res_index, "chain_index": chain_index}
        cent_feats = {k: _flatten_batch(v) for k, v in cent_feats.items()}
        atom_feats = {
            "pos": atom_pos,
            "pos_mask": atom_mask,
            "res_index": flat_res_index,
            "chain_index": flat_chain_index,
        }
        atom_feats = {k: _flatten_batch(v) for k, v in atom_feats.items()}
        # [E, cent_edge_dim]
        cent_to_atom_f = self.make_centroid_to_atom_features(
            cent_to_atom_edge_inds, cent_feats=cent_feats, atom_feats=atom_feats
        )
        cent_to_atom_f = self.cent_edge_feat_proj(cent_to_atom_f)

        # centroid from atom
        _node_f, _atom_f, _pos, _atom_pos = map(_flatten_batch, (node_f, atom_f, pos, atom_pos))
        _node_repr, _node_pos = self.cent_attention(
            q=_node_f, k=_atom_f, x=_pos, y=_atom_pos, edge_attr=cent_to_atom_f, edge_index=cent_to_atom_edge_inds
        )
        node_pos = rearrange(_node_pos, "(b r) n d -> b r n d", b=pos.shape[0])
        _node_repr = rearrange(_node_repr, "(b r) d -> b r d", b=pos.shape[0])
        node_f = node_f + F.dropout(_node_repr, p=self.node_dropout_rate, training=self.training)

        # add pairwise transition
        node_f = self.final_node_norm(node_f)
        _pair_f = self.pair_mlp(self.colwise_mlp(node_f)[..., :, None, :] + self.rowwise_mlp(node_f)[..., None, :, :])
        pair_f = pair_f + F.dropout(_pair_f, p=self.pair_dropout_rate, training=self.training)

        edge_f = gather_edges(pair_f, edge_inds)

        extra = {
            "atom_feat": atom_f,
            "cent_to_atom_edge_index": cent_to_atom_edge_inds,
            "cent_to_atom_edge_feat": cent_to_atom_f,
            "node_pos": node_pos,
            "flat_centroid_pos": _flatten_batch(node_pos[..., 1, :]),  # CA
            "flat_atom_pos": _atom_pos,
        }
        return edge_f, edge_inds, pair_f, node_f, extra

    @torch.autocast(enabled=False, device_type="cuda")
    def make_centroid_to_atom_features(self, edge_inds, cent_feats, atom_feats):
        src_inds, tgt_inds = edge_inds
        src_pos = cent_feats["pos"][src_inds]  # [E, 3]
        tgt_pos = atom_feats["pos"][tgt_inds]  # [E, 3]

        dist = torch.norm(tgt_pos - src_pos, dim=-1)  # [E]

        edge_mask = cent_feats["pos_mask"][src_inds] & atom_feats["pos_mask"][tgt_inds]  # [E]
        pad_mask = ~edge_mask.bool()
        dist = dist.masked_fill(pad_mask, 0.0)

        edge_rbf = self.compute_rbf(dist.masked_fill(pad_mask, 1e6))  # [E, num_rbf]

        src_res_inds = cent_feats["res_index"][src_inds]
        tgt_res_inds = atom_feats["res_index"][tgt_inds]
        _offset = self.cent_to_atom_offset
        res_ind_offset = torch.clamp(tgt_res_inds - src_res_inds, min=-_offset, max=_offset) + _offset
        res_ind_offset_one_hot = F.one_hot(res_ind_offset, num_classes=2 * _offset + 1).float()  # [E, 2*offset+1]

        src_chain_inds = cent_feats["chain_index"][src_inds]
        tgt_res_inds = atom_feats["chain_index"][tgt_inds]
        same_chain = (src_chain_inds == tgt_res_inds).unsqueeze(-1).float()  # [E, 1]

        edge_features = torch.cat([edge_rbf, dist.unsqueeze(-1), res_ind_offset_one_hot, same_chain], dim=-1)  # [E, D]

        return edge_features

    def update_atom_repr(self, node_f, pos, pos_mask, atom_pos, atom_mask, cent_to_atom_f):
        atom_pair_repr = None  # [bsz, num_atom, num_atom, atom_pair_dim]
        atom_edge_inds = get_topk_edges(atom_pos)  # [bsz, num_atom, top_k]
        # atom-to-centroid graph
        atom_to_cent_edge_inds = get_topk_edges(atom_pos, tgt_pos=pos)
        atom_to_cent_edge_repr = gather_edges(cent_to_atom_f.transpose(1, 2), atom_to_cent_edge_inds)
        # atom from centroid
        atom_points, _atom_repr = self.atom_point_update(
            atom_f,
            node_f,
            points=atom_pos,
            key_points=pos,
            point_mask=atom_mask,
            key_point_mask=pos_mask,
            edge_repr=atom_to_cent_edge_repr,
            edge_index=atom_to_cent_edge_inds,
        )
        atom_f = atom_f + _atom_repr

        # windowed attention
        _atom_repr = self.atom_point_attention(
            atom_f, pair_repr=atom_pair_repr, points=atom_points, point_mask=atom_mask
        )
        atom_f = atom_f + _atom_repr

        # graph attention
        _atom_repr = self.atom_graph_attention(atom_f, pair_repr=atom_pair_repr, edge_index=atom_edge_inds)
        atom_f = atom_f + _atom_repr
