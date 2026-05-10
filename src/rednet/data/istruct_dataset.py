import json
import collections
from typing import Any, Mapping
from typing_extensions import Self
from pathlib import Path
from dataclasses import dataclass
from omegaconf import DictConfig
import time
import functools
import pandas as pd

from torch.utils.data import Dataset

from .tokenizer import Tokenizer
from .structure_pipeline import StructurePipeline
from .epoch_sampling_mixin import EpochSamplingMixin
from ..common_utils import get_logger, exists

log = get_logger(__name__)


def get_entry_id(sample_id: str) -> str:
    # sample_id: [pdb_id|asm_id]_[chain_id1:chain_id2:...]_[suffix]
    _ids = str(sample_id).split("_")
    entry_id = _ids[0]

    return entry_id


def get_chain_ids(sample_id: str) -> tuple[str] | None:
    _ids = str(sample_id).split("_")
    chain_ids = None if len(_ids) == 1 else tuple(_ids[1].split(":"))
    return chain_ids


@dataclass(frozen=True, kw_only=True, slots=True)
class StructureEntry:
    key: str
    structure_path: Path
    entry: dict[str, Any]

    @classmethod
    def from_pdb_entry(cls, key, structure_dir: Path, entry) -> Self | None:
        structure_path = Path(structure_dir) / f"{key}.lz4"
        if not structure_path.exists():
            return
        return cls(entry=entry, key=key, structure_path=structure_path)

    @classmethod
    def from_ted_entry(cls, key, structure_dir: Path, entry) -> Self | None:
        structure_path = Path(structure_dir) / f"{key}.lz4"
        if not structure_path.exists():
            return
        return cls(entry=entry, key=key, structure_path=structure_path)

    @property
    def chains(self) -> dict[str, dict[str, Any]]:
        """Return chains from the entry"""
        return self.entry["chains"]

    @property
    def hits(self) -> dict[str, list[Any]]:
        """Return hits from the entry
        each chain hits is a list of [first_hit_id, num_hits, evalue, fident, alnlen, qlen]
        """
        return self.entry.get("hits", {})

    def get_evalue(self, chain_id: str) -> float | None:
        if self.hits and self.hits.get(chain_id):
            evalue = float(self.hits[chain_id][2])
            return evalue
        return

    def get_seq_fident(self, chain_id: str) -> float | None:
        if self.hits and self.hits.get(chain_id):
            fident = float(self.hits[chain_id][3])
            return fident
        return None


def make_pdb_entries(structure_dir, entries, max_n=None, filter_fn=None) -> Mapping[str, StructureEntry]:
    _dir = Path(structure_dir)
    assert _dir.exists(), f"Structure directory {_dir} does not exist"
    _max_n = max_n or len(entries)
    _keys = list(entries.keys())[:_max_n]
    # seq_info = _load_seq_info(seq_info_file) if exists(seq_info_file) else None
    pdb_entries = collections.OrderedDict()
    for k in _keys:
        entry = entries[k]
        if not isinstance(entry, StructureEntry):
            entry = StructureEntry.from_pdb_entry(k, _dir, entries[k])
        if exists(entry):
            if exists(filter_fn) and not filter_fn(entry):
                continue
            pdb_entries[k] = entry
    assert len(pdb_entries) > 0, f"No valid PDB entries found in {structure_dir} with the given filters"
    return pdb_entries


class IStructDataset(Dataset, EpochSamplingMixin):
    """Intergrative structure datasets"""

    def __init__(
        self,
        config: DictConfig,
        pdb_cfg=None,
        afdb_cfg=None,
        max_num_entries: int = None,
        deterministic: bool = False,
        filter_hits: bool = True,
        min_num_chains: int = None,
        max_num_chains: int = None,
        add_metadata_to_sample: bool = False,
        evalue_threshold: float | None = None,
        use_chain_cluster: bool = False,
        keep_interface_types: None | set[str] = None,
        min_seqlen: int | None = None,
        max_total_seqlen: int | None = None,
        min_sample_size: int | None = None,
        max_sample_size: int | None = None,
        selected_ids: list[str] | None = None,
        selected_sample_ids: list[str] | None = None,
    ):
        super().__init__()

        self.config = config
        self.clusters = {}  # cluster id
        self.metadata = {}
        self.ids = []

        self.tokenizer = Tokenizer()

        self.deterministic = deterministic

        # epoch sampling config
        self.feat_cfg = config.get("feature_config")
        self.sampling_cfg = config.get("sampling_config")

        # load dataset set config
        self.pdb_cfg = pdb_cfg or config.get("pdb_config")
        self.afdb_cfg = afdb_cfg or config.get("afdb_config")
        assert self.pdb_cfg is not None or self.afdb_cfg is not None, "PDB config or AFDB config must be provided"

        # load records
        self.use_structure = True
        if self.pdb_cfg:
            self._prefetch_pdb_records(
                max_num_entries=max_num_entries,
                filter_hits=filter_hits,
                min_num_chains=min_num_chains,
                max_num_chains=max_num_chains,
                evalue_threshold=evalue_threshold,
                use_chain_cluster=use_chain_cluster,
                keep_interface_types=keep_interface_types,
                min_seqlen=min_seqlen,
                max_total_seqlen=max_total_seqlen,
                min_sample_size=min_sample_size,
                max_sample_size=max_sample_size,
                selected_ids=selected_ids,
                selected_sample_ids=selected_sample_ids,
            )

        if self.afdb_cfg:
            self._prefetch_afdb_records()

        self._add_metadata_to_sample = add_metadata_to_sample

    @functools.cached_property
    def entries(self) -> Mapping[str, StructureEntry]:
        entries = collections.OrderedDict()
        if self.pdb_cfg:
            assert self.pdb_entries is not None, "PDB entries are not loaded"
            entries.update(self.pdb_entries)
        return entries

    @functools.cached_property
    def sample_ids(self) -> dict[str, int]:
        _sample_ids = {}
        for i, sample_id in enumerate(self.ids):
            _sample_ids[sample_id] = i
        return _sample_ids

    def get_sample_info(self, sample_id: str):
        return self.metadata[sample_id]

    def get_entry(self, sample_id: str) -> StructureEntry:
        key = get_entry_id(sample_id)
        if key in self.entries:
            return self.entries[key]
        else:
            raise KeyError(f"Sample ID {sample_id} not found in entries")

    @functools.cached_property
    def feature_pipeline(self):
        pipeline = StructurePipeline(config=self.feat_cfg, deterministic=self.deterministic, tokenizer=self.tokenizer)
        return pipeline

    def _prefetch_pdb_records(
        self,
        max_num_entries: int = None,
        min_num_chains: int = None,
        max_num_chains: int = None,
        filter_hits: bool = True,
        evalue_threshold: float = None,
        use_chain_cluster: bool = False,
        keep_interface_types: None | set[str] = None,
        min_seqlen: int | None = None,
        max_total_seqlen: int | None = None,
        min_sample_size: int | None = None,
        max_sample_size: int | None = None,
        selected_ids: list[str] | None = None,
        selected_sample_ids: list[str] | None = None,
    ):
        log.info(
            f"Filtering PDB entries with the following parameters: filter_hits={filter_hits}, "
            f"min_num_chains={min_num_chains}, use_chain_cluster={use_chain_cluster}"
        )

        # load records
        t0 = time.time()
        entry_file = Path(self.pdb_cfg.entry_file)
        assert entry_file.exists(), f"Entry file {entry_file} does not exist"
        entries = json.load(open(entry_file))

        def _keep_chain(cid, chain, entry: StructureEntry):
            if min_seqlen and len(chain["seq"]) < min_seqlen:
                return False
            if not filter_hits:
                return True
            _evalue = entry.get_evalue(cid)
            if not exists(_evalue):
                assert not chain.get(
                    "has_hits", False
                ), f"Chain {cid} in entry {entry.key} has hits but no evalue found"
                return True
            _has_hit = True
            if exists(evalue_threshold):
                _has_hit = _evalue <= evalue_threshold
            return not _has_hit

        def _keep_entry(entry: StructureEntry):
            chains = entry.chains
            if max_num_chains and len(chains) > max_num_chains:
                return False
            if min_num_chains and len(chains) < min_num_chains:
                return False

            if filter_hits:
                if not any([_keep_chain(cid, c, entry) for cid, c in chains.items()]):
                    return False
            return True

        # seq_info_file = self.pdb_cfg.get("seq_info_file")
        _num_init_entries = len(entries)
        entries = make_pdb_entries(self.pdb_cfg.structure_dir, entries, max_num_entries, _keep_entry)
        assert len(entries) > 0, f"No valid PDB entries found in {entry_file} with the given filters"
        log.info(f"Load entries from {entry_file} in {time.time() - t0:.4f}s ({len(entries)} / {_num_init_entries}).")

        # build self.ids
        clu_file = Path(self.pdb_cfg.cluster_file)
        sample_df = pd.read_csv(clu_file)

        sample_df["ignored"] = sample_df["sample_id"].apply(lambda x: get_entry_id(x) not in entries)
        log.info(f"Number of ignored samples: {sample_df['ignored'].sum()}")
        # print(pdb_cluster_df[pdb_cluster_df["ignored"] == True])
        sample_df = sample_df[sample_df["ignored"] == False]

        clu_key = self.pdb_cfg.get("cluster_key", "clu_id")

        t0 = time.time()
        col_keys = set(["interface_type", clu_key])
        # for sample_id, grp in pdb_cluster_df.groupby("sample_id"):
        pdb_clus = collections.defaultdict(list)
        for _, row in sample_df.iterrows():
            init_sample_id = str(row["sample_id"])
            entry_id = get_entry_id(init_sample_id)
            chain_ids = get_chain_ids(init_sample_id)
            entry = entries[entry_id]
            chains = entry.chains
            row = row.to_dict()
            interface_type = row.get("interface_type", "none")
            if keep_interface_types is not None and interface_type not in keep_interface_types:
                continue
            total_len = sum([len(chain["seq"]) for chain in chains.values()])
            if max_total_seqlen and total_len > max_total_seqlen:
                continue
            # expand sample_id to include target chains
            sample_size = sum([len(chains[c]["seq"]) for c in chain_ids])
            for tc in chain_ids:
                if not _keep_chain(tc, chains[tc], entry):
                    continue
                item = {
                    **{k: str(row[k]) for k in col_keys if k in row},
                    "sample_id": init_sample_id + f"_{tc}",
                    "evalue": entry.get_evalue(tc),
                    "seqid": entry.get_seq_fident(tc),
                    "target_chain_ids": (tc,),
                    "chain_ids": tuple(chain_ids),
                    "sample_size": sample_size,
                    "interface_type": interface_type,
                }
                if selected_ids is not None and item["sample_id"].split("_")[0] not in selected_ids:
                    continue
                if selected_sample_ids is not None and item["sample_id"] not in selected_sample_ids:
                    continue
                if max_sample_size and sample_size > max_sample_size:
                    continue
                if min_sample_size and sample_size < min_sample_size:
                    continue
                if use_chain_cluster:
                    _chain_clu_id = entry.entry["cluster_ids"][tc]["clu_id"]
                    # change cluster id
                    item[clu_key] = f"{interface_type}:{_chain_clu_id}"
                    # pdb_clus[_chain_clu_id].append(item)
                    # raise NotImplementedError("Chain clustering is not implemented yet")
                pdb_clus[item[clu_key]].append(item)
                self.metadata[item["sample_id"]] = item
        assert len(pdb_clus) > 0, f"No valid PDB clusters found in {clu_file} with the given filters"
        self.ids = sorted(self.metadata.keys())
        self.sample_sizes = [self.metadata[i]["sample_size"] for i in self.ids]
        self.clusters = collections.OrderedDict(pdb_clus)
        self.pdb_entries = entries
        self.pdb_cluster_key = clu_key

        self.schedule_df = self.resolve_schedule_(self.clusters, [clu_key])
        log.info(f"Load PDB clusters from {clu_file} in {time.time() - t0:.4f}s: number of clusters {len(pdb_clus)}")

    def get_sample_ids(self, pdb_id):
        for i, sample_id in enumerate(self.ids):
            if sample_id.startswith(pdb_id):
                yield i, sample_id

    @staticmethod
    def resolve_schedule_(clusters: dict[str, list[Any]], added_keys) -> pd.DataFrame:
        # assign weights to each cluster
        _interface_type_weights = {"hetero": 1, "homo": 1, "none": 1}
        # clus_df["weight"] = clus_df.apply(lambda x: _interface_type_weights[x["interface_type"]], axis=1)
        schedule_df = collections.defaultdict(list)
        for clu_id, clus in clusters.items():
            item = clus[0]  # use the first item as representative
            for k in added_keys:
                schedule_df[k].append(item[k])

            if "interface_type" not in added_keys:
                interface_type = item.get("interface_type", "none")
                schedule_df["interface_type"].append(interface_type)
            else:
                interface_type = item["interface_type"]

            if "weight" not in added_keys:
                schedule_df["weight"].append(_interface_type_weights[interface_type])

            if "clu_size" not in added_keys:
                schedule_df["clu_size"].append(len(clus))

        return pd.DataFrame(schedule_df)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        assert isinstance(idx, int), f"Index must be an integer, but got {type(idx)}"
        sample_id = self.ids[idx]
        sample = self.parse_feats(sample_id, idx)
        sample["index"] = idx
        if self._add_metadata_to_sample:
            sample["metadata"] = self.metadata[sample_id]

        return sample

    def parse_feats(self, sample_id, idx, transform=True):
        entry = self.get_entry(sample_id=sample_id)
        raw_sample = {}
        sample = {"sample_id": sample_id}
        t0 = time.time()
        if self.use_structure:
            chain_ids = self.metadata[sample_id]["chain_ids"]
            target_chain_ids = self.metadata[sample_id]["target_chain_ids"]
            _structure = self.feature_pipeline.load_parsed_structure(entry.structure_path, chain_ids, target_chain_ids)
            raw_sample.update(_structure)
            sample["load_structure_time"] = time.time() - t0
        t1 = time.time()
        if not transform:
            sample.update(raw_sample)
        else:
            sample.update(self.feature_pipeline.transform(raw_sample, index=idx))
        sample["transform_time"] = time.time() - t1
        return sample
