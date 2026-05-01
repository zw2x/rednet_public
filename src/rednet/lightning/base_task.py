"""Base lightning module"""

from __future__ import annotations
import rich
import hydra
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

import torch
import torch.nn as nn

from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import MetricCollection

from faust.tokenizer import Tokenizer

from ..common_utils import get_logger

log = get_logger(__name__)


def instantiate_module(config_dict, tokenizer=None, cond_embedder=None) -> nn.Module | None:
    if not config_dict:
        return
    if config_dict.get("requires_tokenizer", False) and tokenizer is None:
        raise ValueError("Tokenizer is required for this module")
    # add extra arguments for instantiation
    kwargs = {"tokenizer": tokenizer}
    if config_dict["config"].get("use_cond_embedder", False) and cond_embedder is not None:
        kwargs["cond_embedder"] = cond_embedder

    _keys = ["_target_", "config"]
    mod = hydra.utils.instantiate({k: config_dict[k] for k in _keys}, **kwargs, _recursive_=False)
    return mod


def build_task(ckpt_file, extra_cfg=None):

    assert Path(ckpt_file).exists(), f"Checkpoint file {ckpt_file} does not exist"
    ckpt = torch.load(ckpt_file, weights_only=False, map_location="cpu")
    # rich.print(ckpt.keys())
    cfg = OmegaConf.create(ckpt["hyper_parameters"])
    if extra_cfg:
        cfg = OmegaConf.merge(
            cfg, (OmegaConf.create(extra_cfg) if not isinstance(extra_cfg, DictConfig) else extra_cfg)
        )
    task = BaseTask.from_hydra(cfg, prepare_datamodule=False)
    task.load_state_dict(ckpt["state_dict"], strict=False)
    return task


class BaseTask(LightningModule):
    @staticmethod
    def from_hydra(config: DictConfig, **kwargs) -> BaseTask:
        config.task._target_ = "rednet.lightning.MotifScaffoldingTask"
        config.model._target_ = "rednet.RedNetModel"
        task = hydra.utils.instantiate(config.task, cfg=config, _recursive_=False, **kwargs)
        return task

    @staticmethod
    def build_from_ckpt_file(
        ckpt_file, strict=True, check_non_frozen_params=True, weights_only=False, load_weights=True
    ) -> BaseTask:
        assert Path(ckpt_file).exists(), f"Checkpoint file {ckpt_file} does not exist"
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=weights_only)
        # rich.print(ckpt.keys())
        cfg = OmegaConf.create(ckpt["hyper_parameters"])
        task = BaseTask.from_hydra(cfg, prepare_datamodule=False)
        if not strict and check_non_frozen_params:
            non_frozen_keys = {}
            for k, p in task.named_parameters():
                if p.requires_grad:
                    non_frozen_keys[k] = p.shape
            assert all(
                s == ckpt["state_dict"][k].shape for k, s in non_frozen_keys.items()
            ), f"Checkpoint {ckpt_file} does not match model parameters. Non-frozen parameters: {non_frozen_keys}"
        if load_weights:
            task.load_state_dict(ckpt["state_dict"], strict=False)
        return task

    def __init__(self, cfg: DictConfig, model_cfg: DictConfig = None, prepare_datamodule: bool = True):
        super(BaseTask, self).__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg or self.cfg.model

        self.save_hyperparameters(self.cfg)
        self.instantiate_model()

        if self.cfg.train.get("loss_fn"):
            log.info(f"Instantiating loss function: {self.cfg.train.get('loss_fn')}")
            self.instantiate_loss()

        self._datamodule: LightningDataModule | None = None
        if self.cfg.get("datamodule") and prepare_datamodule:
            self.instantiate_datamodule()

        self.instantiate_metrics()

    def instantiate_metrics(self):
        # instantiate metrics
        # use separate metric instance for train, val and test step to ensure a proper reduction over the epoch
        if "eval" in self.cfg and "metrics" in self.cfg.eval:
            metrics_cfg = self.cfg.eval.metrics
            metrics = MetricCollection({name: hydra.utils.instantiate(cfg) for name, cfg in metrics_cfg.items()})
            self.train_metrics = metrics.clone(prefix="train/")
            self.val_metrics = metrics.clone(prefix="val/")
            self.test_metrics = metrics.clone(prefix="test/")

    def instantiate_datamodule(self):
        # logger.info(f"Instantiating datamodule <{self.cfg.datamodule._target_}>")
        # Calling this self.datamodule will mess with PL since it also assigns self.datamodule
        self._datamodule: LightningDataModule = hydra.utils.instantiate(self.cfg.datamodule, _recursive_=False)
        self._datamodule.prepare_data()  # this is usually not necessary
        self._datamodule.setup()
        OmegaConf.clear_resolver("datamodule")
        OmegaConf.register_new_resolver("datamodule", lambda attr: getattr(self._datamodule, attr))

    def instantiate_model(self):
        log.info(f"Instantiating model <{self.model_cfg._target_}>")
        log.info(self.model_cfg)
        # self.tokenizer = make_tokenizer(self.model_cfg.get("tokenizer_type"))
        self.tokenizer = Tokenizer()
        self.model = instantiate_module(self.model_cfg, tokenizer=self.tokenizer)

    def instantiate_loss(self):
        loss_cfg = self.cfg.train.loss_fn
        self.loss_fn = instantiate_module(loss_cfg, tokenizer=self.tokenizer)
        self.val_loss_fn = instantiate_module(self.cfg.train.get("val_loss_fn", loss_cfg), tokenizer=self.tokenizer)

    # log metrics
    def _log_metrics(
        self, phase: str, log_out: dict[str, torch.Tensor], log_on_step: bool = False, model_outputs=None, targets=None
    ):
        _on_step = True if phase == "train" else log_on_step
        self.log_dict(
            {f"{phase}/{k}": v for k, v in log_out.items()},
            on_step=_on_step,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        metrics = getattr(self, f"{phase}_metrics", None)
        if metrics:
            # update metrics and return values
            metrics(model_outputs, targets)
            # log metrics object, not the metric result
            self.log_dict(metrics, on_step=_on_step, on_epoch=True, prog_bar=True, sync_dist=True)

    # optimization
    def configure_optimizers(self):
        if "optimizer_param_grouping" in self.cfg.train:  # Set zero weight decay for some params
            raise NotImplementedError("optimizer_param_grouping is not implemented in BaseTask")
            # from asimov.trainer.optim.optim_utils import group_parameters_for_optimizer

            # parameters = group_parameters_for_optimizer(
            #     self.model, self.cfg.train.optimizer, **self.cfg.train.optimizer_param_grouping
            # )
        else:
            parameters = self.parameters()
        optimizer = hydra.utils.instantiate(self.cfg.train.optimizer, parameters)

        # Log optimizer info
        for i, g in enumerate(optimizer.param_groups):
            ntensors = len(g["params"])
            nparams = sum(p.numel() for p in g["params"])
            hparams = {k: v for k, v in g.items() if k != "params"}
            rich.print(f"Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}")

        if "scheduler" not in self.cfg.train:
            return optimizer
        else:
            # lr_scheduler should be called either every step (default) or every epoch
            lr_scheduler = hydra.utils.instantiate(self.cfg.train.scheduler, optimizer)
            return [optimizer], {
                "scheduler": lr_scheduler,
                "interval": self.cfg.train.get("scheduler_interval", "step"),
                "monitor": self.cfg.train.get("scheduler_monitor", "val/loss"),
            }

    # def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
    #     # https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#set-grads-to-none
    #     # TD [2022-04-30]: DeepSpeed optimizer uses the kwarg set_grad_to_none instead of set_to_none
    #     if 'set_to_none' in inspect.signature(optimizer.zero_grad).parameters:
    #         optimizer.zero_grad(set_to_none=True)
    #     else:
    #         optimizer.zero_grad()

    # save checkpoints
    def on_save_checkpoint(self, checkpoint):
        # remove freezed parameters
        # since we use self.parameters() in configure_optimizers,  there will be a "model." prefix
        self.remove_freezed_params_(checkpoint["state_dict"])

    def remove_freezed_params_(self, state_dict: dict):
        # rich.print(state_dict.keys())
        for key, param in self.named_parameters():
            if not param.requires_grad:
                state_dict.pop(key)

    def training_step(self, batch, batch_idx):
        loss, log_out = self.model.training_step(batch, batch_idx)
        self._log_metrics("train", log_out)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, log_out = self.model.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        self._log_metrics("val", log_out, log_on_step=False)

    @torch.no_grad()
    def generate(self, inputs, **kwargs):
        raise NotImplementedError("generate method not implemented in BaseTask")

    @torch.no_grad()
    def evaluate(self, inputs, **kwargs):
        raise NotImplementedError("evaluate method not implemented in BaseTask")

    @property
    def use_input_embedder(self):
        return getattr(self.model, "input_embedder", None) is not None

    @property
    def use_encoder(self):
        return getattr(self.model, "encoder", None) is not None

    @property
    def use_cond_embedder(self):
        return getattr(self.model, "cond_embedder", None) is not None

    def get_datamodule(self):
        return self._datamodule

    def on_train_epoch_end(self) -> None:
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        torch.cuda.empty_cache()
