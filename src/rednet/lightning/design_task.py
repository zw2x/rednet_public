"""Main task class for atomic motif scaffolding"""

from copy import deepcopy

import torch
import torch.nn.functional as F


from .base_task import BaseTask
from ..common_utils import get_logger, mask_mean, to_one_hot

log = get_logger(__name__)


class MotifScaffoldingTask(BaseTask):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

    # def _log_extra(self, batch, model_out, model_in):
    #     """Log extra metrics for template embedder"""
    #     _log_out = {}
    #     return _log_out

    def training_step(self, batch, batch_idx, verbose=False):
        loss, log_out = self.model._train_step(batch, batch_idx, epoch_num=self.current_epoch)
        self._log_metrics("train", log_out)
        if verbose:
            print(batch_idx, log_out)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()
        try:
            loss, log_out = self.model._train_step(batch, batch_idx)
        except torch.cuda.OutOfMemoryError:
            log.warning(f"OOM during validation step {batch_idx}. Skipping this batch.")
            torch.cuda.empty_cache()
            loss, log_out = self.model._train_step(batch, batch_idx)

        self._log_metrics("val", log_out)

    @torch.no_grad()
    def sample(self, batch, hparams=None, eval_tokens=True, eval_ll=True, **kwargs):

        self.model.eval()
        batch = deepcopy(batch)
        out = self.model.sample(batch, hparams=hparams, **kwargs)
        # # # print_batch(out)
        if hasattr(self.model, "prepare_targets"):
            tgts = self.model.prepare_targets(batch)
            if "gt_tokens" not in tgts:
                gt_res_type = tgts["gt_res_type"]
            else:
                gt_res_type = tgts["gt_tokens"]
        else:
            gt_res_type = batch["gt_res_type"]

        if eval_tokens:
            # TODO: move to gdit
            # out["pred_tokens"] = out["node_tokens"]
            pred_tokens = out["pred_tokens"]
            log_probs = out["log_probs"]
            is_correct = (gt_res_type == pred_tokens).float()
            nsr = mask_mean(is_correct, batch["mask"]).item()
            dsn_nsr = mask_mean(is_correct, batch["dsn_mask"] * batch["mask"]).item()
            dsn_site_nsr = mask_mean(is_correct, batch["site_mask"] * batch["dsn_mask"] * batch["mask"]).item()
            out["nsr"] = nsr
            out["dsn_nsr"] = dsn_nsr
            out["dsn_site_nsr"] = dsn_site_nsr
            if eval_ll:
                ce = (log_probs * to_one_hot(pred_tokens, num_cls=log_probs.shape[-1])).sum(-1)
                out["ll"] = mask_mean(ce, batch["dsn_mask"]).item()
            # out["dsn_nsr"] = dsn_nsr
        return out

    @torch.no_grad()
    def score(self, batch, reduction="per_token", **kwargs):
        self.eval()
        out = self.model.score(batch, reduction=reduction, **kwargs)
        return out
