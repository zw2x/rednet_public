import torch
import torch.nn.functional as F
from einops import rearrange

from .common_utils import mask_mean


def compute_masked_values(value, mask):
    # value: (bsz, seqlen)
    # mask: (bsz, seqlen)
    bsz, seqlen = mask.shape
    assert mask.shape == value.shape, f"mask shape {mask.shape} not match value shape {value.shape}"
    mask = mask.to(value)
    v_per_token = mask_mean(value, mask, dim=(-1, -2)).item()
    v_per_sample = mask_mean(value, mask, dim=-1).mean().item()
    return v_per_token, v_per_sample


def _compute_acc(prefix: str, is_correct: torch.BoolTensor, mask):
    acc_per_token, acc_per_sample = compute_masked_values(is_correct, mask)
    return {f"acc_per_{prefix}_token": acc_per_token, f"acc_per_{prefix}_sample": acc_per_sample}


def _compute_ppl(prefix: str, perp: torch.Tensor, mask):
    perp_per_token, perp_per_sample = compute_masked_values(perp, mask)
    return {f"ppl_per_{prefix}_token": perp_per_token, f"ppl_per_{prefix}_sample": perp_per_sample}


def _reduce_losses(losses, mask, reduction: str = "per_token", max_num_tokens: int | None = None):
    """
    Args:
        max_num_tokens: maximum number of tokens for each sample
    """
    if reduction == "per_token":
        loss = mask_mean(losses, mask, eps=1e-6)
    elif reduction == "per_sample":
        if max_num_tokens:
            w = mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0, max=max_num_tokens) / max_num_tokens
        else:
            w = 1
        loss = (mask_mean(losses, mask, dim=-1) * w).mean()
    else:
        raise ValueError(f"loss reduction {reduction} not supported")
    return loss


def compute_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.BoolTensor,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    reduction: str = "none",
    is_label: bool = True,
    sample_weights: torch.Tensor | None = None,
    return_unweighted_loss: bool = False,
) -> torch.Tensor:
    """
    Args:
        * mask, valid positions
        * reduction: "none", "per_sample", "per_token"
        * is_label: if True, target is a token tensor (shape: bsz, seqlen).
    """
    assert reduction in {"none", "per_sample", "per_token"}, f"reduction {reduction} not supported"

    bsz, seqlen, num_cls = logits.shape
    # flatten inputs
    _logits = rearrange(logits, "b t c -> (b t) c")
    _mask = rearrange(mask, "b t -> (b t)").to(_logits)
    if is_label:
        target = target.long()
        _target = rearrange(target, "b t -> (b t)")
        assert _target.shape == _mask.shape, f"target shape {_target.shape} mask shape {_mask.shape}"
    else:
        _target = rearrange(target, "b t c -> (b t) c")
        assert _target.shape[0] == _mask.shape[0], f"target shape {_target.shape} mask shape {_mask.shape}"
    # assert torch.sum(_target != ignore_index) > 0, f"no valid target, {target}"
    losses = F.cross_entropy(_logits, _target, reduction="none", weight=weight, label_smoothing=label_smoothing)
    losses = rearrange(losses, "(b t) -> b t", b=bsz, t=seqlen)

    if return_unweighted_loss:
        unweighted_loss = _reduce_losses(losses, mask, reduction=reduction)

    if sample_weights is not None:
        assert sample_weights.shape == (bsz,), f"sample_weights shape {sample_weights.shape} not match bsz {bsz}"
        losses = losses * sample_weights[..., None]

    if reduction == "none":
        return losses

    loss = _reduce_losses(losses, mask, reduction=reduction)

    if return_unweighted_loss:
        return loss, unweighted_loss

    return loss
