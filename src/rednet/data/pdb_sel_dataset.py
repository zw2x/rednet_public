import dataclasses
import collections
from typing import Any, Mapping
from typing_extensions import Self
from pathlib import Path
import joblib
import pandas as pd

from torch.utils.data import Dataset

from .tokenizer import Tokenizer
from .structure_pipeline import StructurePipeline
from .epoch_sampling_mixin import EpochSamplingMixin
from ..common_utils import get_logger, exists, collate

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


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
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


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class FilterHparams:
    filter_hits: bool = True
    evalue_threshold: float | None = None
    min_seqlen: int | None = None
    max_total_seqlen: int | None = None
    min_sample_size: int | None = None  # min num of residues of selected chains
    max_sample_size: int | None = None
    max_num_entries: int | None = None
    min_num_chains: int | None = None
    max_num_chains: int | None = None


def _keep_chain(cid, chain, entry: StructureEntry, hp: FilterHparams):
    if hp.min_seqlen and len(chain["seq"]) < hp.min_seqlen:
        return False
    if not hp.filter_hits:
        return True
    _evalue = entry.get_evalue(cid)
    if not exists(_evalue):
        assert not chain.get("has_hits", False), f"Chain {cid} in entry {entry.key} has hits but no evalue found"
        return True
    _has_hit = True
    if exists(hp.evalue_threshold):
        _has_hit = _evalue <= hp.evalue_threshold
    return not _has_hit


def _keep_entry(entry: StructureEntry, hp: FilterHparams):
    chains = entry.chains
    if hp.max_num_chains and len(chains) > hp.max_num_chains:
        return False
    if hp.min_num_chains and len(chains) < hp.min_num_chains:
        return False

    if hp.filter_hits:
        if not any([_keep_chain(cid, c, entry, hp) for cid, c in chains.items()]):
            return False
    return True


def make_pdb_entries(structure_dir, entries, max_n=None, filter_fn=None) -> Mapping[str, StructureEntry]:
    _dir = Path(structure_dir)
    assert _dir.exists(), f"Structure directory {_dir} does not exist"
    _max_n = max_n or len(entries)
    _keys = list(entries.keys())[:_max_n]
    # seq_info = _load_seq_info(seq_info_file) if exists(seq_info_file) else None
    pdb_entries = collections.OrderedDict()
    filtered = {}
    for k in _keys:
        entry = entries[k]
        if not isinstance(entry, StructureEntry):
            entry = StructureEntry.from_pdb_entry(k, _dir, entries[k])
        if exists(entry):
            if exists(filter_fn) and not filter_fn(entry):
                filtered[k] = entry
                continue
            pdb_entries[k] = entry

    assert len(pdb_entries) > 0, f"No valid PDB entries found in {structure_dir} with the given filters"

    return pdb_entries


class PdbSelDataset(Dataset, EpochSamplingMixin):
    def __init__(self, cfg, use_pfeats: bool = False, selected_ids=None, **unused):
        super().__init__()
        self.config = cfg
        self.pdb_cfg = cfg.pdb_config

        self.use_pfeats = use_pfeats
        self.selected_ids = selected_ids

        self.feat_cfg = cfg.feature_config
        self.structure_dir = Path(self.pdb_cfg.structure_dir)
        self.tokenizer = Tokenizer()
        self.pipeline = StructurePipeline(config=self.feat_cfg, deterministic=True, tokenizer=self.tokenizer)
        self.root = Path(self.pdb_cfg.parsed_dir)
        self._prefetch_records(self.pdb_cfg)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        feats = self.load_feats(sample_id)
        return feats

    def _prefetch_records(self, cfg):
        d = Path(cfg.parsed_dir)
        data_df = pd.read_csv(cfg.dataset_file)
        results = {}
        feat_paths = {}

        def _parse_row(row):
            row = row.to_dict()
            _id = f"{row['tgt_lig_id']}:{row['tgt_rec_id']}+{row['off_tgt_lig_id']}:{row['off_tgt_rec_id']}"
            if self.selected_ids is not None:
                # _pair_id = f'{row["tgt_lig_id"].split("_")[0]}:{row["off_tgt_lig_id"].split("_")[0]}'
                # if row['tgt_lig_id'].startswith('2nts'):
                #     print(row['tgt_lig_id'], row['off_tgt_lig_id'], _pair_id)
                if _id not in self.selected_ids:
                    return
            res_dir = d / _id
            results[_id] = row
            # paired features
            pfeat_path = res_dir / "pfeats.lz4"
            assert pfeat_path.exists(), f"Paired feature file {pfeat_path} does not exist!"
            feat_paths[_id] = pfeat_path

        for lig_clu, grp in data_df.groupby("lig_clu"):
            for i, row in grp.iterrows():
                _parse_row(row)

        self.records = results

        self.ids = sorted(feat_paths.keys())
        self.feat_paths = feat_paths
        log.info(f"Loaded {len(self.records)} records")

    def load_feats(self, sample_id):
        path = self.feat_paths[sample_id]
        feats = joblib.load(path)
        # there is a bug in preprocessed features. so swap it here. TODO: fix the bug in preprocessing
        on_target_feat = feats["off_target_feat"]
        off_target_feat = feats["on_target_feat"]
        out = {"pair_id": sample_id, **on_target_feat, **{"off_" + k: v for k, v in off_target_feat.items()}}
        if self.use_pfeats:
            out.update({k: v for k, v in feats.items() if not k.endswith("_target_feat")})
        return out

    @staticmethod
    def collate_fn(samples, tokenizer, convert_fn=None):
        mask_id = tokenizer.mask_id
        # fmt:off
        batch = collate(
            samples,
            ignored_keys=("interface_dist", "interface_chains"),
            pad_values={
                "res_type": mask_id, "gt_res_type": mask_id, "masked_tokens": mask_id, "input_tokens": mask_id,
                "off_res_type": mask_id, "off_gt_res_type": mask_id, "off_masked_tokens": mask_id, "off_input_tokens": mask_id
            },
        )
        # fmt:on
        if convert_fn is not None:
            return convert_fn(batch)
        return batch

    def sample_epoch_indices(self, epoch: int, inds: list[int] | None = None, **unused) -> tuple[int]:
        if inds is not None:
            return tuple(inds)
        return tuple(range(len(self)))
