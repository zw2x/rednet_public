import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .ce_utils import compute_cross_entropy, _reduce_losses, _compute_acc
from .rigid_utils import weighted_align
from .common_utils import mask_mean


class RedsnLoss(nn.Module):
    def __init__(
        self,
        config,
        tokenizer,
        logging_mask_names: tuple[str] = ("dsn_site_mask", "dsn_mask", "pred_mask"),
        term_weights: dict[str, float] = {"dsn_mask": 1.0, "pred_mask": 0.1},
        normalize_weights: bool = False,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.reduction = config.get("reduction", "per_token")
        assert self.reduction in {"per_sample", "per_token"}, f"reduction {self.reduction} not supported"
        self.label_smoothing = config.get("label_smoothing", 0.0)

        self.logging_mask_names = tuple(logging_mask_names)

        # self.mask_names = tuple(mask_names)
        # assert len(self.mask_names) == len(term_weights), f"{self.mask_names} not match weights {term_weights}"
        term_weights = config.get("term_weights", term_weights)
        mask_names, term_weights = list(zip(*term_weights.items()))
        term_weights = torch.tensor(term_weights, dtype=torch.float32)
        if normalize_weights:
            assert torch.all(term_weights >= 0), f" term weights {term_weights} should be all be non-negative"
            term_weights = term_weights / term_weights.sum()
        self.term_weights = term_weights
        self.mask_names = tuple(mask_names)

        self.pred_edgewise_loss = config.get("pred_edgewise_loss", False)
        self.edgewise_weight = config.get("edgewise_weight", 1.0)

        self.pred_backbone_positions = config.get("pred_backbone_positions", False)
        self.backbone_weight = config.get("backbone_weight", 0.1)

    def forward(self, inputs, targets, epoch_num=None, **unused):
        loss, log_dict = self.compute_nodewise_loss(inputs, targets)
        log_dict["nodewise_loss"] = loss.item()

        if self.pred_edgewise_loss:
            edge_loss, edge_log_dict = self.compute_edgewise_loss(inputs, targets)
            log_dict["edgewise_loss"] = edge_loss.item()
            if self.edgewise_weight > 0:
                loss = loss + edge_loss * self.edgewise_weight
            log_dict.update(edge_log_dict)

        if self.pred_backbone_positions:
            bb_pos = rearrange(targets["atom_positions"][..., :4, :], "b n c d -> b (n c) d")
            bb_mask = rearrange(targets["atom_mask"][..., :4].bool(), "b n c -> b (n c)")
            pred_pos = rearrange(inputs["pred_bb_pos"], "b n c d -> b (n c) d")
            aligned_gt_pos = weighted_align(pred_pos, bb_pos, bb_mask)
            bb_loss = ((pred_pos - aligned_gt_pos) ** 2).sum(dim=-1).clamp(max=25)  # b n
            bb_loss = mask_mean(bb_loss, bb_mask, dim=1)  # b
            bb_loss = bb_loss.mean()
            log_dict["backbone_loss"] = bb_loss.item()
            if self.backbone_weight > 0:
                loss = loss + bb_loss * self.backbone_weight

        log_dict["loss"] = loss.item()

        return loss, log_dict

    def compute_nodewise_loss(self, inputs, targets):

        mask = targets["mask"].bool()  # valid positions
        pred_logits = inputs["pred_logits"]
        num_cls = pred_logits.shape[-1]  # number of classes
        gt_tokens = targets["gt_tokens"].clamp(min=0, max=num_cls - 1)
        ce_losses = compute_cross_entropy(
            pred_logits,
            gt_tokens,
            mask,
            label_smoothing=self.label_smoothing,
            # weight=self.weight,  # class weights
        )
        # ppl = torch.exp(ce_losses.float())
        is_cor = pred_logits.argmax(dim=-1) == gt_tokens

        loss, log_dict = 0, {}
        for i, _mask_name in enumerate(self.mask_names):
            prefix = _mask_name.replace("_mask", "")
            w = self.term_weights[i]
            _mask = targets[_mask_name].bool() * mask
            _loss = _reduce_losses(ce_losses, _mask, reduction=self.reduction)
            loss += w * _loss
            log_dict[f"{prefix}_loss"] = _loss.item()
            # log_dict.update({**_compute_acc(prefix, is_cor, _mask), **_compute_ppl(prefix, ppl, _mask)})
            log_dict.update(_compute_acc(prefix, is_cor, _mask))

        for _mask_name in self.logging_mask_names:
            if _mask_name not in self.mask_names:
                prefix = _mask_name.replace("_mask", "")
                _mask = targets[_mask_name].bool() * mask
                log_dict.update(_compute_acc(prefix, is_cor, _mask))

        return loss, log_dict

    def compute_edgewise_loss(self, inputs, targets):

        _reshape = lambda x: rearrange(x, "b n k ... -> b (n k) ... ")
        logits = _reshape(inputs["edgewise_logits"])  # num_cls 32 * 32
        gt_tokens = _reshape(targets["edgewise_gt_tokens"])  # row * 32 + col
        pred_mask = _reshape(targets["edgewise_pred_mask"].bool())

        is_cor = logits.argmax(dim=-1) == gt_tokens
        _losses = compute_cross_entropy(logits, gt_tokens, pred_mask, label_smoothing=self.label_smoothing)

        log_dict = {}
        loss = _reduce_losses(_losses, pred_mask, reduction=self.reduction)
        log_dict.update(_compute_acc("edgewise", is_cor, pred_mask))

        return loss, log_dict


class FreeEnergyLoss(nn.Module):
    def __init__(self):
        super().__init__()
