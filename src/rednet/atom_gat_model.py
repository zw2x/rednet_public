import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .layers import GraphTransformer, Mlp
from .layers.cache_utils import _gather, CachedState
from .ops import *
from .atom_featurizer import AtomFeaturizer, FullAtomStructureFeaturizer
from .atom_encoder import AtomEncoderLayer
from .atom_decoder import AtomDecoderLayer


class AtomGraphTransformerModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        model_cfg = config.model

        noise_scale = config.noise_scale

        self.dim = model_cfg.node_features
        self.pdim = model_cfg.edge_features
        self.token_dim = model_cfg.get("token_dim", self.dim)
        self.tokenizer = tokenizer
        # hidden_dim = self.dim
        self.vocab_size = len(tokenizer)
        dropout = model_cfg.dropout
        self.enc_depth = model_cfg.num_encoder_layers
        self.dec_depth = model_cfg.num_decoder_layers

        self.use_node_feats = model_cfg.get("use_node_feats", False)
        feat_cfg = model_cfg.get("featurizer", {})
        featurizer_type = feat_cfg.get("type", "default")
        self.pred_backbone_positions = model_cfg.get("pred_backbone_positions", False)
        featurizer_cls = AtomFeaturizer if featurizer_type == "default" else FullAtomStructureFeaturizer
        self.features = featurizer_cls(
            self.dim,
            self.pdim,
            top_k=model_cfg.subset_size,
            augment_eps=noise_scale,
            version=feat_cfg.get("version"),
            add_sc_embedding=feat_cfg.get("add_sc_embedding", False),
            add_frame_shifts=feat_cfg.get("add_frame_shifts", False),
            # add_enc_res_type=feat_cfg.get("add_enc_res_type", False),
            use_out_mlp=feat_cfg.get("use_out_mlp", False),
            tokenizer=tokenizer,
            atom_feat_dim=feat_cfg.get("atom_feat_dim", 128),
            cent_hid_dim=feat_cfg.get("cent_hid_dim", 64),
            cent_edge_dim=feat_cfg.get("cent_edge_dim", 64),
            top_k_centroid_to_atom=feat_cfg.get("top_k_centroid_to_atom", 96),
            pred_backbone_positions=self.pred_backbone_positions,
            use_radius_graph=feat_cfg.get("use_radius_graph", False),
            centroid_to_atom_radius=feat_cfg.get("centroid_to_atom_radius", 15.0),
        )

        self.token_embedding = nn.Embedding(self.vocab_size, self.dim)

        # Encoder layers
        def build_enc_layer():
            enc_cfg = model_cfg.get("encoder", {})
            layer = AtomEncoderLayer(
                self.dim,
                self.pdim,
                dropout=dropout,
                add_gat=enc_cfg.get("add_gat", False),
                add_global=enc_cfg.get("add_global", True),
                num_heads=enc_cfg.get("num_heads", 8),
                head_dim=enc_cfg.get("head_dim", 64),
                msg_dim=enc_cfg.get("msg_dim", None),
                msg_depth=enc_cfg.get("msg_depth", 1),
                node_mlp_depth=enc_cfg.get("node_mlp_depth", 1),
                edge_mlp_depth=enc_cfg.get("edge_mlp_depth", 1),
                expansion_factor=enc_cfg.get("expansion_factor", 4),
            )
            return layer

        self.encoder_layers = nn.ModuleList([build_enc_layer() for _ in range(self.enc_depth)])

        # Decoder layers
        dec_cfg = model_cfg.get("decoder", {})
        self.use_token_attn = dec_cfg.get("use_token_attn", False)
        if self.use_token_attn:
            self.token_attn = GraphTransformer(
                self.dim,
                self.pdim,
                dropout_rate=dec_cfg.get("token_attn_dropout_rate", 0.2),
                zero_init=False,
                kvdim=self.token_dim,
            )
        self.use_self_attn = dec_cfg.get("use_self_attn", False)

        def build_dec_layer():
            layer = AtomDecoderLayer(
                self.dim,
                self.pdim,
                dropout=dropout,
                add_gat=dec_cfg.get("add_gat", True),
                add_global=dec_cfg.get("add_global", True),
                num_heads=dec_cfg.get("num_heads", 8),
                head_dim=dec_cfg.get("head_dim", 64),
                msg_depth=dec_cfg.get("msg_depth", 1),
                mlp_depth=dec_cfg.get("mlp_depth", 1),
                expansion_factor=dec_cfg.get("expansion_factor", 4),
            )
            if self.use_self_attn:
                self_attn = GraphTransformer(
                    self.dim,
                    self.pdim,
                    dropout_rate=dec_cfg.get("self_attn_dropout_rate", 0.2),
                    zero_init=False,
                )
                layer = nn.ModuleList([self_attn, layer])
            return layer

        self.token_to_edge_lin = nn.Linear(self.token_dim, self.pdim, bias=False)
        self.enc_node_to_edge_lin = nn.Linear(self.dim, self.pdim, bias=False)

        self.decoder_layers = nn.ModuleList([build_dec_layer() for _ in range(self.dec_depth)])

        # output heads
        self.logits_head = nn.Linear(self.dim, self.vocab_size, bias=True)

        self.pred_edgewise_logits = model_cfg.get("pred_edgewise_logits", False)
        if self.pred_edgewise_logits:
            self.use_enc_edge_to_edgwise_logits = model_cfg.get("use_enc_edge_to_edgwise_logits", True)
            self.edgewise_logits_head = Mlp(
                self.pdim, out_dim=self.vocab_size**2, embed_dim=self.pdim, expansion_factor=2, apply_prenorm=True
            )

    def train_step(self, train_loss, inputs, i, epoch_num=None, **kwargs):
        inputs["input_tokens"] = inputs["res_type"]
        mask = inputs["mask"].bool()
        gt_tokens = inputs["gt_res_type"]

        preds = self(inputs)

        # prepare targets
        edge_index = preds["edge_index"].long()

        tgts = {
            "gt_tokens": gt_tokens,
            "mask": mask,
            "pred_mask": inputs["pred_mask"],
            "dsn_mask": inputs["dsn_mask"],
            "resolved_mask": inputs["resolved_mask"] if "resolved_mask" in inputs else mask,
            "site_mask": inputs["site_mask"] if "site_mask" in inputs else mask,
            "edge_index": edge_index,
            "atom_mask": inputs["atom_mask"],
            "atom_positions": inputs["atom_positions"],
        }
        tgts["dsn_site_mask"] = inputs["dsn_mask"] * inputs["site_mask"]

        if self.pred_edgewise_logits:
            assert "edgewise_logits" in preds, "Edgewise logits are not predicted, but expected for training."

            # gt_tokens (b, n), edge_index: (b, n, k)
            def _gather_x(x):
                n = edge_index.shape[1]
                if x.ndim == 3:
                    x = torch.gather(x, 1, repeat(edge_index, "b n k -> b (n k) d", d=x.shape[-1]))
                    return rearrange(x, "b (n k) d -> b n k d", n=n)
                else:
                    x = torch.gather(x, 1, rearrange(edge_index, "b n k -> b (n k)"))
                    return rearrange(x, "b (n k) -> b n k", n=n)

            tgts["edgewise_gt_tokens"] = _gather_x(gt_tokens) + gt_tokens.unsqueeze(-1) * self.vocab_size
            pred_mask = inputs["pred_mask"]
            tgts["edgewise_pred_mask"] = _gather_x(pred_mask) * pred_mask.unsqueeze(-1)

            # pos, pos_mask = inputs["atom_positions"][..., 1, :], inputs["atom_mask"][..., 1]
            # # ca-ca distance <= dist_cutoff
            # dist_cutoff = 25
            # d = torch.norm(pos[..., None, :] - _gather_x(pos), dim=-1)
            # edgewise_mask = d <= dist_cutoff
            # # seq distance > short_range_cutoff
            # same_chain = to_pairwise_mask(inputs["chain_index"], ops="eq")
            # short_range_cutoff = 4
            # short_range_pair_mask = to_pairwise_mask(inputs["res_index"], ops="diff").abs() <= short_range_cutoff
            # edgewise_pred_mask = edgewise_mask * ~(gather_edges(same_chain * short_range_pair_mask, edge_index))

        loss, log_out = train_loss(preds, tgts, epoch_num)
        return preds, tgts, loss, log_out

    def encode(self, feature_dict):
        mask = feature_dict["mask"]

        edge_repr, edge_index, pair_repr, node_repr, extra = self.features(feature_dict)

        edge_mask = gather_nodes(mask.unsqueeze(-1), edge_index).squeeze(-1)
        for layer in self.encoder_layers:
            node_repr, edge_repr = layer(node_repr, edge_repr, edge_index, edge_mask, mask=mask)

        return node_repr, edge_repr, edge_index, pair_repr, extra

    def forward(self, inputs, return_extra=False, **kwargs):
        input_tokens = inputs["input_tokens"]
        mask = inputs["mask"]
        enc_node, enc_edge, edge_index, enc_pair, enc_extra = self.encode(inputs)
        edge_mask = gather_nodes(mask.unsqueeze(-1), edge_index).squeeze(-1)

        order = sample_order(
            mask,
            dsn_mask=inputs.get("dsn_mask"),
            decoding_order=inputs.get("decoding_order"),
            randn=inputs.get("decoding_order_randn"),
        )
        mask_bw_include_self, mask_bw_exclude_self, attn_mask_include_self, attn_mask_exclude_self = make_causal_mask(
            order, edge_index=edge_index, return_dense_masks=True
        )
        token_acts = self.token_embedding(input_tokens)
        # print('mask_bw_include_self', mask_bw_include_self[0], edge_index[0])
        # add to edge_repr
        edge_repr = enc_edge + self.enc_node_to_edge_lin(gather_nodes(enc_node, edge_index))
        edge_repr = edge_repr.masked_fill(~edge_mask.unsqueeze(-1), 0.0)
        _tok_repr = gather_nodes(token_acts, edge_index)
        _tok_repr = self.token_to_edge_lin(_tok_repr).masked_fill(~mask_bw_exclude_self.unsqueeze(-1), 0.0)
        edge_repr = edge_repr + _tok_repr
        node_repr = enc_node
        if self.use_token_attn:
            node_repr = self.token_attn(
                node_repr,
                edges=enc_pair,
                key=token_acts,
                attn_mask=attn_mask_exclude_self,
                mask_empty_positions=True,
            )
        for i, layer in enumerate(self.decoder_layers):
            _return_extra = return_extra and i == self.dec_depth - 1
            if self.use_self_attn:
                self_attn, layer = layer
                node_repr = self_attn(node_repr, edges=enc_pair, attn_mask=attn_mask_include_self)
            out = layer(
                node_repr,
                edge_repr,
                edge_index,
                edge_mask,
                mask=mask,
                return_extra=_return_extra,
                mask_bw=mask_bw_include_self.unsqueeze(-1),
            )
            if _return_extra:
                node_repr, dec_edge, extra = out
            else:
                node_repr, dec_edge = out

        edge_repr = dec_edge

        logits = self.logits_head(node_repr)
        log_probs = F.log_softmax(logits, dim=-1)

        out = {
            "pred_tokens": torch.argmax(logits, dim=-1),
            "log_probs": log_probs,
            "pred_logits": logits,
            "decoding_order": order,
            "edge_index": edge_index,
            "node_acts": node_repr,
            "edge_acts": edge_repr,
            "enc_node": enc_node,
            "enc_edge": enc_edge,
        }

        # prediction heads
        if self.pred_edgewise_logits:
            _edge = enc_edge
            out["edgewise_logits"] = self.edgewise_logits_head(_edge)

        if self.pred_backbone_positions:
            out["pred_bb_pos"] = enc_extra["node_pos"]

        if return_extra:
            out.update(extra)

        return out

    @torch.no_grad()
    def decode_step(
        self,
        t,
        cache,
        *,
        enc_node,
        mask_t,
        edge_t,
        edge_index_t,
        mask_bw_t,
        edge_mask_t,
        enc_pair=None,
        attn_mask_exclude_self_t=None,
    ):

        _node = enc_node
        if self.use_token_attn:
            _node_t = self.token_attn(
                _gather(_node, t),
                edges=_gather(enc_pair, t),
                attn_mask=attn_mask_exclude_self_t,
                key=cache["token_acts"].value,
                mask_empty_positions=True,
                static_kv=True,
            )
            _node = cache["token_attn"]["node_cache"].update(seq_dim=1, value=_node_t, timestep=t)
        for l, layer in enumerate(self.decoder_layers):
            # print(_node.shape, t.shape, _gather(_node, t).shape)
            _node_t, _dec_edge_t = layer(
                _gather(_node, t),
                edge_t,
                edge_index_t,
                edge_mask=edge_mask_t,
                mask=mask_t,
                mask_bw=mask_bw_t,
                timestep=t,
                cache=cache[f"dec.{l}"],
            )
            _node = cache[f"dec.{l}"]["node_cache"].update(_node_t, seq_dim=1, timestep=t)
        logits = self.logits_head(_node_t)

        return _node, logits

    @torch.no_grad()
    def sample(self, feature_dict, use_tqdm=False):
        self.eval()

        mask = feature_dict["mask"]
        fixed_tokens = feature_dict.get("fixed_tokens")
        input_tokens = feature_dict.get("input_tokens")
        bias = feature_dict.get("bias")
        temperature = feature_dict.get("temperature", 1e-3)
        dsn_mask = feature_dict.get("dsn_mask", None)
        enc_node, enc_edge, edge_index, enc_pair, extra = self.encode(feature_dict)
        edge_mask = gather_nodes(mask.unsqueeze(-1), edge_index).squeeze(-1)
        decoding_order = sample_order(
            mask,
            decoding_order=feature_dict.get("decoding_order"),
            dsn_mask=dsn_mask,
            randn=feature_dict.get("decoding_order_randn"),
        )
        mask_bw_include_self, mask_bw_exclude_self, attn_mask_include_self, attn_mask_exclude_self = make_causal_mask(
            decoding_order, edge_index=edge_index, return_dense_masks=True
        )

        cache = self.prepare_cached_states(mask)

        token_acts = cache["token_acts"].value

        edge_repr = enc_edge + self.enc_node_to_edge_lin(gather_nodes(enc_node, edge_index))

        timesteps = decoding_order.unbind(dim=-1)
        for i, t in enumerate(timesteps):
            # t = decoding_order[:, t_]  # [B]
            edge_t, mask_bw_t, edge_index_t, mask_t, edge_mask_t, attn_mask_exclude_self_t = map(
                lambda x: _gather(x, t),
                [edge_repr, mask_bw_exclude_self, edge_index, mask, edge_mask, attn_mask_exclude_self],
            )

            # TODO: cache token_to_edge_lin in 'token_to_edge'
            tok_repr_t = gather_nodes(token_acts, edge_index_t)
            tok_repr_t = self.token_to_edge_lin(tok_repr_t).masked_fill(~mask_bw_t.unsqueeze(-1), 0.0)
            edge_t = edge_t + tok_repr_t
            mask_bw_t = _gather(mask_bw_include_self.unsqueeze(-1), t)

            node_repr, logits = self.decode_step(
                t,
                cache,
                edge_mask_t=edge_mask_t,
                enc_node=enc_node,
                mask_t=mask_t,
                edge_index_t=edge_index_t,
                mask_bw_t=mask_bw_t,
                edge_t=edge_t,
                attn_mask_exclude_self_t=attn_mask_exclude_self_t,
                enc_pair=enc_pair,
            )
            bias_t = _gather(bias, t) if exists(bias) else 0.0
            probs = F.softmax((logits + bias_t) / temperature, dim=-1)
            tok_t = sample_multinomial(probs).squeeze(-1)

            if exists(fixed_tokens):
                tok_t = torch.gather(fixed_tokens, 1, t[:, None]).long()

            if exists(input_tokens) and exists(dsn_mask):
                in_tok_t = _gather(input_tokens, t).long()
                dsn_mask_t = _gather(dsn_mask, t).bool()
                tok_t = tok_t * dsn_mask_t + in_tok_t * ~dsn_mask_t

            token_acts_t = self.token_embedding(tok_t)
            token_acts = cache["token_acts"].update(token_acts_t, seq_dim=1, timestep=t)

            cache["pred_tokens"].update(tok_t, seq_dim=1, timestep=t)
            cache["pred_logits"].update(logits, seq_dim=1, timestep=t)

        log_probs = F.log_softmax(cache["pred_logits"].value, dim=-1)
        out = {
            "pred_tokens": cache["pred_tokens"].value,
            "decoding_order": decoding_order,
            "edge_index": edge_index,
            "pred_logits": cache["pred_logits"].value,
            "node_acts": node_repr,
            "token_acts": token_acts,
            "log_probs": log_probs,
            "enc_node": enc_node,
            "enc_edge": enc_edge,
        }
        return out

    def prepare_cached_states(self, mask) -> dict[str, CachedState]:
        device, (bsz, seqlen) = mask.device, mask.shape

        def _init_dec_cache():
            cache = {"node_cache": CachedState(cache_type="dynamic_subset")}
            cache["node_msg"] = CachedState(cache_type="dynamic_subset")
            # if self.use_self_attn:
            #     cache["sa_cache"] = CachedState(cache_type="dynamic_subset")
            return cache

        cached_states = {
            **{f"dec.{l}": _init_dec_cache() for l in range(self.dec_depth)},
            "token_acts": CachedState(cache_type="dynamic_subset"),
            "pred_logits": CachedState(cache_type="dynamic_subset"),
            "pred_tokens": CachedState(cache_type="dynamic_subset"),
            "modified_logits": CachedState(cache_type="dynamic_subset"),
        }
        if self.use_token_attn:
            cached_states["token_attn"] = {"node_cache": CachedState(cache_type="dynamic_subset")}
        for key, state in cached_states.items():
            if key.startswith("dec."):
                state["node_cache"].alloc(shape=(bsz, seqlen, self.dim), device=device, dtype=torch.float32)
                if "node_msg" in state:
                    state["node_msg"].alloc(shape=(bsz, seqlen, self.pdim), device=device, dtype=torch.float32)
                # if self.use_self_attn:
                #     state["sa_cache"].alloc(shape=(bsz, seqlen, self.dim), device=device, dtype=torch.float32)
            elif key == "token_acts":
                state.alloc(shape=(bsz, seqlen, self.dim), device=device, dtype=torch.float32)
            elif key in {"pred_logits", "modified_logits"}:
                state.alloc(shape=(bsz, seqlen, self.vocab_size), device=device, dtype=torch.float32)
            elif key == "pred_tokens":
                state.alloc(shape=(bsz, seqlen), device=device, dtype=torch.long)
            elif key == "token_attn":
                state["node_cache"].alloc(shape=(bsz, seqlen, self.dim), device=device, dtype=torch.float32)

        return cached_states

    @torch.no_grad()
    def decode_prefix(self, inputs, prefix_timesteps, decoding_order):

        mask = inputs["mask"]
        dsn_mask = inputs["dsn_mask"]
        enc_node, enc_edge, edge_index, enc_pair, extra = self.encode(inputs)
        edge_mask = gather_nodes(mask.unsqueeze(-1), edge_index).squeeze(-1)
        mask_bw_include_self, mask_bw_exclude_self, attn_mask_include_self, attn_mask_exclude_self = make_causal_mask(
            decoding_order, edge_index=edge_index, return_dense_masks=True
        )

        cache = self.prepare_cached_states(mask)

        # get encoder edge_repr
        edge_repr = enc_edge + self.enc_node_to_edge_lin(gather_nodes(enc_node, edge_index))

        enc_feats = {
            "enc_node": enc_node,
            "enc_pair": enc_pair,
            "edge_repr": edge_repr,
            "edge_mask": edge_mask,
            "edge_index": edge_index,
        }
        mask_feats = {
            "mask": mask,
            "mask_bw_exclude_self": mask_bw_exclude_self,
            "mask_bw_include_self": mask_bw_include_self,
            "attn_mask_exclude_self": attn_mask_exclude_self,
        }
        if not exists(prefix_timesteps):
            return cache, enc_feats, mask_feats

        # embed prefix tokens
        if exists(inputs.get("input_tokens")) and exists(prefix_timesteps):
            tok_t = torch.gather(inputs["input_tokens"], 1, prefix_timesteps).long()
            token_acts_t = self.token_embedding(tok_t)
            token_acts = cache["token_acts"].update(token_acts_t, seq_dim=1, timestep=prefix_timesteps)

        token_acts = cache["token_acts"].value

        # get feature from t
        edge_t, mask_bw_t, edge_index_t, mask_t, edge_mask_t, attn_mask_exclude_self_t = map(
            lambda x: _gather(x, prefix_timesteps),
            [edge_repr, mask_bw_exclude_self, edge_index, mask, edge_mask, attn_mask_exclude_self],
        )

        # TODO: cache token_to_edge_lin in 'token_to_edge'
        # token_acts_t = _gather(token_acts, prefix_timesteps)
        # print("token_acts_t", token_acts_t.shape, prefix_timesteps.shape)
        tok_repr_t = gather_nodes(token_acts, edge_index_t)
        tok_repr_t = self.token_to_edge_lin(tok_repr_t).masked_fill(~mask_bw_t.unsqueeze(-1), 0.0)
        edge_t = edge_t + tok_repr_t
        mask_bw_t = _gather(mask_bw_include_self.unsqueeze(-1), prefix_timesteps)

        node_repr, logits_t = self.decode_step(
            prefix_timesteps,
            cache,
            edge_mask_t=edge_mask_t,
            enc_node=enc_node,
            mask_t=mask_t,
            edge_index_t=edge_index_t,
            mask_bw_t=mask_bw_t,
            edge_t=edge_t,
            attn_mask_exclude_self_t=attn_mask_exclude_self_t,
            enc_pair=enc_pair,
        )
        # bias_t = _gather(bias, t) if exists(bias) else 0.0
        # probs = F.softmax((logits + bias_t) / temperature, dim=-1)
        # tok_t = sample_multinomial(probs).squeeze(-1)

        # if exists(fixed_tokens):
        #     tok_t = torch.gather(fixed_tokens, 1, t[:, None]).long()

        # if exists(input_tokens) and exists(dsn_mask):
        #     in_tok_t = _gather(input_tokens, t).long()
        #     dsn_mask_t = _gather(dsn_mask, t).bool()
        #     tok_t = tok_t * dsn_mask_t + in_tok_t * ~dsn_mask_t

        # token_acts_t = self.token_embedding(tok_t)
        # token_acts = cache["token_acts"].update(token_acts_t, seq_dim=1, timestep=t)

        cache["pred_tokens"].update(tok_t, seq_dim=1, timestep=prefix_timesteps)
        cache["pred_logits"].update(logits_t, seq_dim=1, timestep=prefix_timesteps)

        return cache, enc_feats, mask_feats

    def _sample_step(
        self,
        t,
        cache: dict[str, CachedState],
        *,
        enc_feats: dict[str, torch.Tensor],
        mask_feats: dict[str, torch.Tensor],
    ):
        # t = decoding_order[:, t_]  # [B]
        token_acts = cache["token_acts"].value
        # get features
        _keys = ["edge_repr", "edge_mask", "edge_index", "enc_node", "enc_pair"]
        edge_repr, edge_mask, edge_index, enc_node, enc_pair = map(lambda k: enc_feats[k], _keys)
        _mask_keys = ["mask", "mask_bw_exclude_self", "mask_bw_include_self", "attn_mask_exclude_self"]
        mask, mask_bw_exc_self, mask_bw_inc_self, attn_mask_exc_self = map(lambda k: mask_feats[k], _mask_keys)
        # gather timewise features
        edge_t, mask_bw_t, edge_index_t, mask_t, edge_mask_t, attn_mask_exclude_self_t = map(
            lambda x: _gather(x, t), [edge_repr, mask_bw_exc_self, edge_index, mask, edge_mask, attn_mask_exc_self]
        )

        # TODO: cache token_to_edge_lin in 'token_to_edge'
        tok_repr_t = gather_nodes(token_acts, edge_index_t)
        tok_repr_t = self.token_to_edge_lin(tok_repr_t).masked_fill(~mask_bw_t.unsqueeze(-1), 0.0)
        edge_t = edge_t + tok_repr_t
        mask_bw_t = _gather(mask_bw_inc_self.unsqueeze(-1), t)

        node_repr, logits_t = self.decode_step(
            t,
            cache,
            edge_mask_t=edge_mask_t,
            enc_node=enc_node,
            mask_t=mask_t,
            edge_index_t=edge_index_t,
            mask_bw_t=mask_bw_t,
            edge_t=edge_t,
            attn_mask_exclude_self_t=attn_mask_exclude_self_t,
            enc_pair=enc_pair,
        )

        cache["pred_logits"].update(logits_t, seq_dim=1, timestep=t)

        return logits_t

    def update_tokens_(self, cache: dict[str, CachedState], t: torch.Tensor, tok_t: torch.Tensor):
        token_acts_t = self.token_embedding(tok_t)
        cache["token_acts"].update(token_acts_t, seq_dim=1, timestep=t)
        cache["pred_tokens"].update(tok_t, seq_dim=1, timestep=t)
