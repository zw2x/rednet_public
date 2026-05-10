from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


def sample_tokens(
    t: Tensor,
    logits_t: Tensor,
    con_logits_t: Tensor | None,
    cst_logits_t: Tensor | None,
    alpha=0.0,
    beta=0.0,
    fixed_mask: Tensor | None = None,
    fixed_tokens: Tensor | None = None,
    temperature=1e-3,
):
    if fixed_mask is not None:
        mask_t = torch.gather(fixed_mask, 1, t[:, None]).long()
        if mask_t.sum() > 0:
            tok_t = torch.gather(fixed_tokens, 1, t[:, None]).long()
            return tok_t[0]

    if alpha > 0:
        logits_t = (1 + alpha) * logits_t - alpha * con_logits_t

    if beta > 0:
        cst_probs_t = F.softmax(cst_logits_t.float(), dim=-1)
        cst_mask_t = cst_probs_t > beta * cst_probs_t.max(dim=-1, keepdim=True).values
        logits_t.masked_fill_(~cst_mask_t, -1e9)

    tok_t = sample_cate(logits_t[0, 0], temperature=temperature).long()[None]

    return tok_t


def sample_cate(logits: Tensor, top_k=None, top_p=None, temperature=1e-3):
    assert logits.ndim == 1, f"logits should be a 1D tensor, {logits.shape}"
    logits = logits.float() / temperature
    # logits = logits - logits.max()
    probs = F.softmax(logits, dim=-1)
    assert not torch.isnan(probs).any(), "NaN in probs"
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    if top_k is not None and top_k > 0:
        top_k = min(top_k, sorted_probs.size(-1))
        sorted_probs[top_k:] = 0

    if top_p is not None and 0 < top_p < 1:
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum > top_p
        # Shift mask right to keep first token exceeding threshold
        mask[1:] = mask[:-1].clone()
        mask[0] = False
        sorted_probs[mask] = 0

    sorted_probs = sorted_probs / sorted_probs.sum()
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)

    return sorted_indices[sampled_sorted_idx].squeeze()


@dataclass
class TStepDict:
    prefix_tsteps: torch.Tensor | None
    design_tsteps: torch.Tensor
    tsteps: torch.Tensor
    design_order: torch.Tensor


def sample_order(dsn_mask: torch.Tensor, dsn_order=None):
    # dsn_mask: [seqlen]
    dsn_mask = dsn_mask.bool()
    assert dsn_mask.ndim == 1, "dsn_mask should be a 1D tensor"
    seqlen, device = dsn_mask.shape[0], dsn_mask.device
    # dsn_mask: 1 if the residue is designed, 0 otherwise. put fixed residues at the beginning
    order = torch.arange(seqlen, device=device)
    fixed_positions = order[~dsn_mask]
    if len(fixed_positions) == 0:
        fixed_positions = None

    # get design timesteps
    dsn_positions = order[dsn_mask]
    if dsn_order is None:
        dsn_order = torch.argsort(torch.randn_like(dsn_positions.float()))
    else:
        assert len(dsn_order) == len(dsn_positions), "dsn_order must match the length of dsn_positions"
    dsn_positions = dsn_positions[dsn_order]

    # concatenate prefix and design timesteps
    if fixed_positions is None:
        timesteps = dsn_positions
    else:
        timesteps = torch.cat([fixed_positions, dsn_positions], dim=0)

    return TStepDict(
        prefix_tsteps=fixed_positions,
        design_tsteps=dsn_positions,
        tsteps=timesteps,
        design_order=dsn_order,
    )


def contrast_decode_(
    ts,
    model,
    bd_inputs,
    bd_cache,
    sampling_cfg,
    con_cache=None,
    use_contrast=False,
    fixed_inputs=None,
    con_inputs=None,
    con_model=None,
    cst_cache=None,
    use_cst_model=False,
    cst_inputs=None,
    cst_model=None,
):
    if fixed_inputs is not None:
        fixed_masks, fixed_tokens = fixed_inputs
    else:
        fixed_masks, fixed_tokens = None, None

    t = ts[0]
    bd_enc_feats, bd_masks = bd_inputs
    logits_t = model._sample_step(t, bd_cache, enc_feats=bd_enc_feats, mask_feats=bd_masks)
    con_logits_t = None
    if use_contrast:
        _t = ts[1]
        _enc_feats, _masks = con_inputs
        if con_model is None:
            con_model = model
        con_logits_t = con_model._sample_step(_t, con_cache, enc_feats=_enc_feats, mask_feats=_masks)

    cst_logits_t = logits_t
    if use_cst_model:
        _cst_t = ts[-1]
        _enc_feats, _masks = cst_inputs
        cst_logits_t = cst_model._sample_step(_cst_t, cst_cache, enc_feats=_enc_feats, mask_feats=_masks)

    tok_t = sample_tokens(
        t,
        logits_t,
        con_logits_t,
        cst_logits_t,
        alpha=sampling_cfg.alpha,
        beta=sampling_cfg.beta,
        fixed_mask=fixed_masks,
        fixed_tokens=fixed_tokens,
        temperature=sampling_cfg.temperature,
    )
    model.update_tokens_(bd_cache, t, tok_t[None])
    if use_contrast:
        con_model.update_tokens_(con_cache, _t, tok_t[None])
    if use_cst_model:
        cst_model.update_tokens_(cst_cache, _cst_t, tok_t[None])


def contrast_decode_batch(batch, model, sampling_cfg, con_batch=None, check_fixed=False, use_con=False, verbose=False):
    _unbind = lambda x: x[None].unbind(-1)

    assert batch["dsn_mask"].shape[0] == 1, "Only batch size 1 is supported"
    # bound structures
    batch["input_tokens"] = batch["res_type"]
    bd_tstep_dict = sample_order(batch["dsn_mask"][0])
    bd_cache, *bd_inputs = model.decode_prefix(batch, bd_tstep_dict.prefix_tsteps[None], bd_tstep_dict.tsteps[None])
    fixed_tokens, fixed_mask = None, None
    if check_fixed:
        fixed_tokens, fixed_mask = batch["res_type"], torch.ones_like(batch["dsn_mask"])
    pbar = [_unbind(bd_tstep_dict.design_tsteps)]

    # contrast structures
    if use_con:
        dsn_order = bd_tstep_dict.design_order
        con_tstep_dict = sample_order(con_batch["dsn_mask"][0], dsn_order)
        con_batch["input_tokens"] = con_batch["res_type"]
        _prefix = con_tstep_dict.prefix_tsteps[None] if con_tstep_dict.prefix_tsteps is not None else None
        con_cache, *con_inputs = model.decode_prefix(con_batch, _prefix, con_tstep_dict.tsteps[None])
        pbar.append(_unbind(con_tstep_dict.design_tsteps))

    pbar = zip(*pbar, strict=True)

    for ts in pbar:
        contrast_decode_(
            ts,
            model,
            bd_inputs,
            bd_cache,
            sampling_cfg,
            use_contrast=use_con,
            con_cache=con_cache if use_con else None,
            con_inputs=con_inputs if use_con else None,
            fixed_inputs=(fixed_mask, fixed_tokens) if check_fixed else None,
        )
    if check_fixed:
        batch["decoding_order"] = bd_tstep_dict.tsteps[None]
        _out = model(batch)
        _sampled_logits = bd_cache["pred_logits"].value
        assert torch.allclose(_out["pred_logits"], _sampled_logits, atol=1e-4)

    pred_tokens = bd_cache["pred_tokens"].value[0]
    con_pred_tokens = None
    if use_con:
        con_pred_tokens = con_cache["pred_tokens"].value[0]
        if verbose:
            print("Bound decoded tokens:", pred_tokens)
            print("Contrast decoded tokens:", con_pred_tokens)

    return pred_tokens, con_pred_tokens
