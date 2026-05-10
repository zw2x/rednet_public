"""RedNet modules for structure-based protein redesign"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_utils import get_logger, exists, mask_mean, to_one_hot

from .loss_utils import RedsnLoss

from .atom_gat_model import AtomGraphTransformerModel

# Composing pretrained pgat and other models for protein redesign
log = get_logger(__name__)

__all__ = [
    "RedNetModel",
]


class RedNetModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.stage = self.config.get("stage", "pretrain")
        self.model_type = self.config.get("model_type", "atom_gat")
        # if self.model_type == "pmpnn":
        #     self.model: ProteinMPNNModel = ProteinMPNNModel.from_config(config.model)
        # elif self.model_type == "esmif":
        #     self.model = GVPTransformerModel.from_config(config.model)
        # else:
        self.model: AtomGraphTransformerModel = AtomGraphTransformerModel(config, tokenizer)
        # pretrain
        if self.stage == "pretrain":
            self.pretrain_loss = RedsnLoss(config=config.pretrain_loss, tokenizer=tokenizer)
        # inference
        self._cached_data = {}  # cache for inference, e.g., for contrastive decoding
        # sft
        # rlft
        # retrieval

    def prepare_targets(self, inputs):
        if hasattr(self.model, "prepare_targets"):
            tgts = self.model.prepare_targets(inputs)
        else:
            tgts = inputs
        return tgts

    def _train_step(self, inputs, i, epoch_num=None):
        self.train()
        if self.stage == "pretrain":
            preds, tgts, loss, log_out = self.model.train_step(self.pretrain_loss, inputs, i, epoch_num=epoch_num)
        else:
            raise NotImplementedError(f"Stage {self.stage} not implemented for RedNetModel")
        return loss, log_out

    def forward(self, inputs, **kwargs):
        inputs["input_tokens"] = inputs.get("input_tokens", inputs["res_type"])
        return self.model(inputs, **kwargs)

    @torch.no_grad()
    def sample(self, inputs, hparams=None, **kwargs):
        """
        Generate tokens using the model.
        """
        # inputs["input_tokens"] = inputs.get("input_tokens", inputs["res_type"])
        if "input_tokens" not in inputs and "dsn_mask" in inputs and "res_type" in inputs:
            inputs["input_tokens"] = inputs["res_type"].masked_fill(inputs["dsn_mask"], 0)
        out = self.model.sample(inputs)
        return out

    @torch.no_grad()
    def encode(self, inputs, prefix_len: int = 0, cached_key: str = "root", reset=True):
        if reset or not exists(self._cached_data.get(cached_key)):
            self._cached_data[cached_key] = {}
        self.model.eval()
        if prefix_len > 0:
            inputs["input_tokens"] = inputs["res_type"]
        fw_out = self.model.encode(inputs, prefix_len=prefix_len, init_decoder=True)
        self._cached_data[cached_key]["timesteps"] = fw_out["timesteps"]
        self._cached_data[cached_key]["static_data"] = fw_out["static_data"]
        self._cached_data[cached_key]["cached_states"] = fw_out["cached_states"]

    def get_timesteps(self, cached_key) -> tuple[torch.Tensor, ...]:
        return self._cached_data[cached_key]["timesteps"]

    @torch.no_grad()
    def pred_logits_step(self, idx, t=None, cached_key: str = "root"):
        self.model.eval()
        _cached_dict = self._cached_data[cached_key]
        if t is None:
            t = _cached_dict["timesteps"][idx]
        _static_data = _cached_dict["static_data"]
        _states = _cached_dict["cached_states"]
        prev_step = _cached_dict.get("prev_step")
        pred_tokens = _states["pred_tokens"].value
        _out = self.model.decoder.decode_step(
            {**_static_data, "prev_step": prev_step, "pred_tokens": pred_tokens}, t, _states
        )
        logits_t = _out["logits"]
        return logits_t

    @torch.no_grad()
    def update_step(self, idx, tokens_t, logits_t, t=None, cached_key="root"):
        _cached_dict = self._cached_data[cached_key]
        _states = _cached_dict["cached_states"]
        if t is None:
            t = _cached_dict["timesteps"][idx]
        _states["pred_tokens"].update(tokens_t, timestep=t)
        _states["pred_logits"].update(logits_t, timestep=t)
        _cached_dict["prev_step"] = t

    def finalize_output(self, cached_key: str = "root"):
        # finalize outputs
        states = self._cached_data[cached_key]["cached_states"]
        log_probs = F.log_softmax(states["pred_logits"].value, dim=-1)  # [bsz, seqlen, vocab_size]
        out = {
            "pred_tokens": states["pred_tokens"].value,
            "pred_logits": states["pred_logits"].value,
            "log_probs": log_probs,
        }

        return out

    @torch.no_grad()
    def score(self, batch, reduction="per_token", **kwargs):
        self.eval()
        out = self(batch, **kwargs)
        tokens = batch["gt_res_type"]
        log_probs = F.log_softmax(out["pred_logits"], dim=-1)
        ce = (log_probs * to_one_hot(tokens, num_cls=log_probs.shape[-1])).sum(-1)
        if reduction == "per_token":
            out["global_ll"] = mask_mean(ce, batch["mask"]).item()
            out["ll"] = mask_mean(ce, batch["dsn_mask"]).item()
        else:
            out["global_ll"] = mask_mean(ce, batch["mask"], dim=-1)
            out["ll"] = mask_mean(ce, batch["dsn_mask"], dim=-1)
        return out
