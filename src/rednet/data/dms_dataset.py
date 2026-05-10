from pathlib import Path
import pandas as pd
import numpy as np
import collections
import dataclasses
import re
import json
from typing import Mapping, Any

import torch
from torch.utils.data import Dataset


from .utils import redesign_mask
from .tokenizer import Tokenizer
from .pdb_parsing import parse_pdb_string
from .epoch_sampling_mixin import EpochSamplingMixin

import rednet.residue_constants as rc
from rednet.common_utils import get_logger

log = get_logger(__name__)


def _translate_sequence(sequence: str) -> torch.Tensor:
    res_type = torch.tensor([rc.RESTYPE_ORDER_WITH_X[w] for w in str(sequence)]).long()
    return res_type


_DTYPES = {
    "atom_positions": torch.float32,
    "atom_mask": torch.bool,
    "b_factors": torch.float32,
}


def _parse_structure(pdb_dir: Path, file_id, seq=None, protein=None):
    # load protein structure from pdb file
    if not exists(protein):
        pdb_file = pdb_dir / f"{file_id}.pdb"
        with open(pdb_file, "r") as f:
            pdb_str = f.read()
        protein = parse_pdb_string(pdb_str)

    # align wt sequence to pdb structure
    pdb_res_type = protein.aatype
    if exists(seq):
        # protein = protein.replace_res_type(seq)
        assert len(protein.aatype) == len(seq), f"Length mismatch between pdb and wt seq for {file_id}"
        dataclasses.replace(protein, aatype=_translate_sequence(seq))

    def _cast(k, v):
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
            if k in _DTYPES:
                v = v.to(_DTYPES[k])
        return v

    protein = {k: _cast(k, v) for k, v in protein.to_dict().items()}
    return protein, pdb_res_type


# helpers


def exists(x):
    return x is not None


class DmsDataset(Dataset, EpochSamplingMixin):
    def __init__(self, config, prefetch_assay_data=False, **kwargs):
        self.config = config
        # data: file_id -> assay_data
        self.data, self.ids, self.clusters = None, None, None
        self.tokenizer = Tokenizer()
        self.prefetch_records(prefetch_assay_data=prefetch_assay_data, **kwargs)
        self.use_mut_chain_id = kwargs.get("use_mut_chain_id", False)

    # @staticmethod
    # def make_dataset(config, dms_type="skempi", **kwargs):
    #     if dms_type == "skempi":
    #         return SkempiDataset(config, **kwargs)

    def prefetch_records(self, prefetch_assay_data=False, max_num_clusters=None, **unused):
        self.data = {}
        self.ids = []
        self.clusters = collections.defaultdict(list)

        split_file = self.config.split_file
        if isinstance(split_file, (str, Path)):
            self.split_data = json.load(open(split_file))
        else:
            self.split_data = {}
            for split_file in self.config.split_file:
                self.split_data.update(json.load(open(split_file)))

        if self.config.get("assay_dir"):
            self.assay_dir = Path(self.config.assay_dir)

        self.pdb_dir = Path(self.config.pdb_dir)

        assay_data = None
        if prefetch_assay_data:
            assay_file = Path(self.config.assay_file)
            assert assay_file.exists(), f"Assay file {assay_file} does not exist"
            assay_data = pd.read_csv(assay_file)
            assay_data = self.preprocess_assay_data(assay_data)

        # load assays
        ignored = []
        for clu_id, file_ids in self.split_data.items():
            for file_id in file_ids:
                assay_result = self._parse_assay(file_id, assay_data=assay_data)
                if not exists(assay_result):
                    ignored.append(file_id)
                    continue

                ass_data = assay_result["assay_data"]
                # filter out varaints
                if "is_filtered" in ass_data.columns:
                    ass_data = ass_data[ass_data["is_filtered"] == False]

                # reset index
                ass_data = ass_data.reset_index(drop=True)

                self.data[file_id] = assay_result
                # inclusive at both start_idx and end_idx
                self.data[file_id]["start_idx"] = len(self.ids)
                self.ids += [(file_id, i) for i in range(len(ass_data))]
                self.data[file_id]["end_idx"] = len(self.ids) - 1
                self.clusters[clu_id].append(file_id)
            if exists(max_num_clusters) and len(self.clusters) >= max_num_clusters:
                break

    @property
    def file_ids(self) -> tuple[str]:
        file_ids = sorted(self.data.keys())
        return tuple(file_ids)

    def get_index_range(self, file_id):
        start_idx, end_idx = self.data[file_id]["start_idx"], self.data[file_id]["end_idx"]
        return start_idx, end_idx

    def __len__(self):
        assert self.ids is not None, "IDs not set. Call prefetch_records() first."
        return len(self.ids)

    def __getitem__(self, idx):
        file_id, i = self.ids[idx]
        row = self.data[file_id]["assay_data"].iloc[i].to_dict()
        structure = self.data[file_id]["structure"]
        var = {**structure, "aatype": _translate_sequence(row["aa_seq"])}
        var["pdb_res_type"] = structure["aatype"].clone()
        var["design_type"] = self.data[file_id]["design_type"]
        var["res_index"] = var.pop("residue_index")
        var["res_type"] = var.pop("aatype")
        var["mask"] = torch.ones_like(var["res_type"], dtype=torch.bool)
        var["resolved_mask"] = var["atom_mask"][:, 0].bool()
        var["site_mask"] = torch.ones_like(var["res_type"], dtype=torch.bool)
        var["res_conf"] = torch.ones_like(var["res_type"], dtype=torch.float32)
        # design_chain_id = var["chain_index"][0].item()  # single domain
        if self.use_mut_chain_id:
            mut_chain_id = row["mut_chain_id"]
            chain_mask = var["chain_index"] == mut_chain_id
            assert chain_mask.any(), f"Mut chain {mut_chain_id} not found in {file_id} {i}"
        else:
            mut_chain_id = None
        # add redesign mask
        var = redesign_mask(self.tokenizer, var, mut_chain_id)
        var["file_id"] = file_id
        var["index"] = idx
        var["df_index"] = i
        var.update(self._get_scores(row))
        return var

    def _get_scores(self, row: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        return {"score": torch.tensor(row["score"]).float()}

    def sample_epoch_indices(
        self,
        epoch: int,
        seed: int = None,
        deterministic: bool = False,
        num_samples: int | None = None,
        num_clusters: int | None = None,
        num_samples_per_file: int | None = None,
        only_wildtype: bool = False,
        sel_file_ids: tuple[str] | None = None,
        **unused,
    ) -> tuple[int]:
        if only_wildtype:
            _ids = []
            file_ids = sorted(self.data.keys())
            for file_id in file_ids:
                start_idx, _ = self.get_index_range(file_id)
                _ids.append(start_idx)
            return sorted(_ids)

        if exists(sel_file_ids):
            ret_ids = []
            if not deterministic:
                rng = np.random.default_rng(seed)
                sel_file_ids = rng.permutation(sel_file_ids)
            for file_id in sel_file_ids:
                start_idx, end_idx = self.get_index_range(file_id)  # both ends are inclusive
                _ids = list(range(start_idx, end_idx + 1))
                if not deterministic:
                    _ids = rng.permutation(_ids)
                if exists(num_samples_per_file):
                    _ids = _ids[:num_samples_per_file]
                ret_ids.extend(_ids)
            return tuple(ret_ids)

        if deterministic:
            return tuple(range(len(self.ids)))

        else:
            rng = np.random.default_rng(seed)
            clu_ids = sorted(self.clusters.keys())
            if num_clusters:
                clu_ids = rng.choice(clu_ids, min(len(clu_ids), num_clusters), replace=False)

            ids = []
            for clu_id in clu_ids:
                file_ids = self.clusters[clu_id]
                if len(file_ids) == 0:
                    continue
                else:
                    file_id = rng.choice(file_ids, 1)[0]
                    start_idx, end_idx = self.get_index_range(file_id)  # both ends are inclusive
                    _ids = list(range(start_idx, end_idx + 1))
                    _ids = rng.permutation(_ids)[:num_samples_per_file]
            return tuple(ids)


class MegascaleDataset(DmsDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config, prefetch_assay_data=False, **kwargs)

    def _get_scores(self, row):
        try:
            score = float(row["dG_ML"])
        except ValueError:
            if row["dG_ML"] == "<-1":
                score = -1.0
            elif row["dG_ML"] == ">5":
                score = 5.0
            else:
                raise ValueError(f"Invalid dG_ML value: {row['dG_ML']}")

        return {
            "score": -torch.tensor(score).float(),
            # "score": torch.tensor(np.log10(row["kd_mut"])).float(),
            # "score_wt": torch.tensor(np.log10(row["kd_wt"])).float(),
        }

    def _parse_assay(self, file_id: str, protein=None, assay_data=None) -> Mapping[str, Any] | None:
        pdb_file_id = file_id

        # infer design_type, pdb file id, assay file id
        if "_rd" in file_id:
            # denovo design
            ass_file_id = "_".join(["denovo", file_id.split("_")[0], file_id.split("_rd")[-1]])
            design_type = "denovo"
        elif file_id.endswith("TrROS_Hall"):
            # trrosetta design
            ass_file_id = "trrosetta" + "_" + file_id.split("_TrROS_Hall")[0]
            design_type = "trrosetta"
        elif len(file_id) == 4:  # assume pdb id
            design_type = "natural"
            ass_file_id = file_id

        # read assay data

        if not exists(assay_data):
            assay_file = self.assay_dir / f"{ass_file_id}.csv"
            if not assay_file.exists():
                return

            assert assay_file.exists(), f"Assay file {assay_file} does not exist"
            assay_data = pd.read_csv(assay_file)

        wt_seq = assay_data.iloc[0]["aa_seq"]
        # replace pdb seq with wt seq
        protein, wt_res_type = _parse_structure(self.pdb_dir, pdb_file_id, seq=wt_seq)
        # parse indel data
        assay_data["has_indel"] = assay_data.apply(lambda x: len(wt_seq) != len(x["aa_seq"]), axis=1)
        if self.config.get("ignore_indels", True):
            assay_data = assay_data[assay_data["has_indel"] == False]

        assay_data["is_invalid"] = assay_data.apply(lambda x: x["dG_ML"] == "-", axis=1)
        if self.config.get("remove_invalid", True):
            assay_data = assay_data[assay_data["is_invalid"] == False]
        return {
            "assay_data": assay_data,
            "structure": protein,
            "pdb_res_type": torch.from_numpy(wt_res_type),
            "num_variants": len(assay_data),
            "design_type": design_type,
        }


class SkempiDataset(DmsDataset):
    _IGNORED_SKEMPI_PDB_IDS = ["1NCA", "1NMB"]  # has insertion codes

    def __init__(self, config, **kwargs):
        self.minimum_rows = config.get("minimum_rows", 1)
        super().__init__(config, prefetch_assay_data=True, **kwargs)

    def _get_scores(self, row):
        return {
            "score": torch.tensor(np.log(row["kd_mut"])).float(),
            "score_wt": torch.tensor(np.log(row["kd_wt"])).float(),
        }

    def preprocess_assay_data(self, assay_data: pd.DataFrame, add_wt=False):
        assay_data["file_id"] = assay_data["pdb_id"].apply(lambda x: x.split("_")[0])
        _mask = assay_data["file_id"].isin(self._IGNORED_SKEMPI_PDB_IDS)
        assay_data.drop(assay_data[_mask].index, inplace=True)
        # row["kd_wt"]
        if add_wt:
            groups = []
            for file_id, grp in assay_data.groupby("file_id"):
                first_row = grp.iloc[0]
                kd_mut = first_row["kd_wt"]
                kd_wt = first_row["kd_wt"]
                placeholder_row = pd.DataFrame(
                    {
                        "pdb_id": [first_row["pdb_id"]],
                        "mut_pdb": ["-"],
                        "mut_pdb_clean": ["-"],
                        "mut_loc": ["-"],
                        "kd_mut": [kd_mut],
                        "kd_wt": [kd_wt],
                        "file_id": [file_id],
                    }
                )

                merged_group = pd.concat([placeholder_row, grp], ignore_index=True)
                groups.append(merged_group)
            assay_data = pd.concat(groups, ignore_index=True)

        return assay_data

    def _parse_assay(self, file_id, *, assay_data) -> Mapping[str, Any] | None:
        ass_data = assay_data[assay_data["file_id"] == file_id]
        if len(ass_data) < self.minimum_rows:
            return

        pdb_file_id = file_id
        protein, pdb_res_type = _parse_structure(self.pdb_dir, pdb_file_id)

        sel_ass = collections.defaultdict(list)

        def _to_seq(res_type):
            return "".join([rc.PROTEIN_TYPES_WITH_X[i] for i in res_type])

        def _to_kd_float(kd_str: str) -> float | None:
            try:
                if kd_str.startswith((">", "<", "~")):
                    _kd_value = float(kd_str[1:])
                elif kd_str in {"n.b", "n.b.", "unf"}:
                    return
                else:
                    _kd_value = float(kd_str)
            except ValueError as e:
                log.warning(str(e) + f"\n  {kd_str}")
                return
            return _kd_value

        for _, row in ass_data.iterrows():
            mut_pdb_clean = row["mut_pdb_clean"] # positions in pdb clean
            mut_res_type = protein["aatype"].clone()
            if mut_pdb_clean == "-":  # wildtype placeholder
                chain_id = ""
            else:
                for _loc in mut_pdb_clean.split(","):
                    _loc = _loc.strip()
                    m = re.match(r'([A-Z])([A-Za-z]+)(\d+)([A-Z])', _loc)
                    # wtaa, auth_chain_id, mtaa = _loc[0], _loc[1], _loc[-1]
                    # mut_pos = int(_loc[2:-1])
                    wtaa, auth_chain_id, mut_pos, mtaa = m.group(1), m.group(2), int(m.group(3)), m.group(4)
                    chain_id = protein["chain_id_mapping"][auth_chain_id]
                    _sel_mask = (protein["chain_index"] == chain_id) & (protein["residue_index"] == mut_pos)
                    # check wildtype amino acid
                    assert wtaa == rc.PROTEIN_TYPES_WITH_X[int(protein["aatype"][_sel_mask][0])], row
                    mut_res_type[_sel_mask] = rc.RESTYPE_ORDER_WITH_X[mtaa]
            num_mut = (mut_res_type != protein["aatype"]).sum().item()
            aa_seq = _to_seq(mut_res_type)
            kd_mut = _to_kd_float(row["kd_mut"])
            kd_wt = _to_kd_float(row["kd_wt"])
            if exists(kd_mut) and exists(kd_wt):
                sel_ass["aa_seq"].append(aa_seq)
                sel_ass["kd_mut"].append(kd_mut)
                sel_ass["kd_wt"].append(kd_wt)
                sel_ass["mut_chain_id"].append(chain_id)
                sel_ass['num_mut'].append(num_mut)
        if len(sel_ass) == 0:
            log.warning(f"empty assay for {file_id}")
            return
        sel_ass = pd.DataFrame(sel_ass)
        first_seq = str(sel_ass["aa_seq"].iloc[0])
        # parse indel data
        sel_ass["has_indel"] = sel_ass.apply(lambda x: len(first_seq) != len(x["aa_seq"]), axis=1)
        if self.config.get("ignore_indels", True):
            sel_ass = sel_ass[sel_ass["has_indel"] == False]
        return {
            "assay_data": sel_ass,
            "structure": protein,
            "pdb_res_type": torch.from_numpy(pdb_res_type).long(),
            "num_variants": len(sel_ass),
            "design_type": "skempi",
        }
