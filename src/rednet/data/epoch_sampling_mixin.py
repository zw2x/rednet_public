"""Mixin for data sampling"""

from typing import Callable, Optional
import sys
import random
import functools

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from rednet.common_utils import get_logger, collate

log = get_logger(__name__)


def sample_ids(
    df: pd.DataFrame,
    epoch: int,
    num_samples: int = None,
    seed: int = None,
) -> list[int]:
    _seed = (seed or random.randrange(0, sys.maxsize)) + epoch
    rng = np.random.default_rng(_seed)
    # rich.print(clusters)
    weights = df["weight"].values
    probs = weights / np.sum(weights)
    counts = rng.multinomial(num_samples, probs)
    selected_indices = []
    for i, count in enumerate(counts):
        selected_indices.extend([i] * count)
    rng.shuffle(selected_indices)

    return [int(i) for i in selected_indices]


class EpochSamplingMixin:
    def sample_epoch_indices(
        self,
        epoch: int,
        seed: int = None,
        deterministic: bool = False,
        num_samples: int = None,
        num_clusters: int = None,
        order_by_size: bool = False,
        reverse_order: bool = False,
    ) -> tuple[int]:

        # sample cluster ids
        if deterministic:
            _ids = list(range(len(self.schedule_df)))
        else:
            _ids = sample_ids(self.schedule_df, epoch, num_samples=num_samples or len(self.schedule_df), seed=seed)
        if num_clusters and num_clusters > 0:
            _ids = _ids[:num_clusters]

        # t0 = time.time()
        clus = self.schedule_df.loc[list(_ids), self.pdb_cluster_key].values
        # print(f"Sampling clusters in {time.time() - t0:.4f}s")
        selected_ids = []
        # t1 = time.time()
        for clu_id in clus:
            items = self.clusters[str(clu_id)]
            i = 0 if deterministic else np.random.randint(0, len(items))
            sel_item = items[int(i)]
            selected_ids.append(self.sample_ids[str(sel_item["sample_id"])])
        # print(f"Sampling ids in {time.time() - t1:.4f}s")
        # return tuple(sorted(selected_ids))
        if order_by_size:
            selected_ids = sorted(selected_ids, key=lambda x: self.sample_sizes[x])

        if reverse_order:
            selected_ids.reverse()
        return tuple(selected_ids)

    def get_dataloader(
        self,
        *,
        epoch: int = 0,
        deterministic_sampling: bool = False,
        batch_size=1,
        num_workers: int = 0,
        shuffle: bool = False,
        num_replicas: Optional[int] = 1,
        rank: Optional[int] = 0,
        order_by_size: bool = True,
        use_bucketing: bool = True,
        reverse_order: bool = False,
        convert_fn: Optional[Callable] = None,
        return_sampler: bool = False,
        **sampler_kwargs,
    ):
        log.info(
            f"Epoch: {epoch}, deterministic_sampling: {deterministic_sampling}, shuffle: {shuffle}, "
            f"batch_size: {batch_size}, num_workers: {num_workers}"
        )
        order_by_size = order_by_size or use_bucketing
        sampler = functools.partial(
            self.sample_epoch_indices,
            deterministic=deterministic_sampling,
            num_samples=self.config.get("max_num_samples"),
            order_by_size=order_by_size,
            reverse_order=reverse_order,
            **sampler_kwargs,
        )
        if return_sampler:
            return sampler(0)

        dataloader = DataLoader(
            self,
            sampler=sampler(0),
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=functools.partial(self.collate_fn, tokenizer=self.tokenizer, convert_fn=convert_fn),
            batch_size=batch_size,
        )
        return dataloader

    @staticmethod
    def collate_fn(samples, tokenizer, convert_fn=None):
        mask_id = tokenizer.mask_id
        batch = collate(
            samples,
            ignored_keys=("interface_dist", "interface_chains"),
            pad_values={"res_type": mask_id, "gt_res_type": mask_id, "masked_tokens": mask_id, "input_tokens": mask_id},
        )
        if convert_fn is not None:
            return convert_fn(batch)
        return batch
