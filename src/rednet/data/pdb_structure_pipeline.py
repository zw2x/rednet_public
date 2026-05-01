import numpy as np
import pandas as pd
from enum import Enum
from pathlib import Path
import joblib

import torch

from rednet.common_utils import get_logger, crop_by_mask
from .tokenizer import Tokenizer
from .utils import redesign_mask, find_interface_site, compute_dist

log = get_logger(__name__)


class FeatureType(Enum):
    INTERFACE_REDESIGN = "interface_redesign"


class StructurePipeline:
    def __init__(self, config, tokenizer: Tokenizer, deterministic: bool = False):
        super().__init__()
        self.config = config
        self.use_structure = config.get("use_structure", True)
        self.feature_type = FeatureType(config.get("feature_type", "interface_redesign"))
        self.radius = config.get("radius", 10.0)
        self.tokenizer = tokenizer
        self.deterministic = deterministic

    def transform(self, sample, index: int | None = None):
        # ca positions
        pos, pos_mask = sample["atom_positions"][:, 1, :], sample["atom_mask"][:, 1]
        site_mask = find_interface_site(pos, pos_mask, sample["chain_index"], radius=self.radius)
        sample["interface_site_mask"] = site_mask
        sample["site_mask"] = site_mask.clone()
        sample["resolved_mask"] = pos_mask.clone()
        chain_id_mapping = sample["chain_id_mapping"]  # mapping[str, int]

        if self.feature_type == FeatureType.INTERFACE_REDESIGN:
            target_chain_ids = sample["target_chain_ids"]  # list[str]
            _exist_chain_ids = [i.item() for i in sample["chain_index"].long().unique()]
            if not target_chain_ids:
                target_ids = _exist_chain_ids
            else:
                target_ids = [chain_id_mapping[c] for c in target_chain_ids if c in chain_id_mapping]
                if len(target_ids) == 0:
                    target_ids = _exist_chain_ids
            # print("Target chain ids:", target_chain_ids, "Exist chain ids:", _exist_chain_ids, "Selected chain ids:", target_ids, chain_id_mapping)
            assert len(target_ids) > 0, "No target ids found"

            _sel_id = 0 if (self.deterministic or len(target_ids) == 1) else np.random.randint(len(target_ids))
            sample = redesign_mask(self.tokenizer, sample, target_ids[_sel_id])

        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")

        # add residue-wise confidence
        if "res_conf" not in sample:
            sample["res_conf"] = torch.ones_like(sample["res_index"], dtype=torch.float32)

        return sample

    @staticmethod
    def load_parsed_structure(
        structure_path: Path,
        chain_ids: tuple[str] | None = None,
        target_chain_ids: tuple[str] | None = None,
        use_resolved_positions: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Load parsed structure from disk"""
        structure_path = Path(structure_path)
        assert structure_path.exists(), f"Structure file {structure_path} does not exist"
        try:
            sample = joblib.load(structure_path)
        except Exception as e:
            print(f"Error loading {structure_path}: {e}")
            raise e
        # remove keys
        _removed_keys = ["bioassembly", "entry"]
        [sample.pop(k, None) for k in _removed_keys if k in sample]
        # transform sample
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                if k in {"atom_positions", "b_factors"}:
                    v = v.float()
            sample[k] = v
        sample["res_type"] = sample.pop("aatype").long()
        sample["res_index"] = sample.pop("residue_index").long()
        sample["mask"] = torch.ones(sample["res_index"].shape, dtype=torch.bool)
        sample["atom_mask"] = sample["atom_mask"].bool()

        chain_id_mapping = sample["chain_id_mapping"]
        if chain_ids:
            chain_id_mapping = {c: chain_id_mapping[c] for c in chain_ids if c in chain_id_mapping}

        # select chains
        selected_chain_mask = torch.zeros_like(sample["res_index"], dtype=torch.bool)
        for c, i in chain_id_mapping.items():
            _chain_mask = sample["chain_index"] == i  # chain_id_mapping[chain_id]
            assert _chain_mask.any(), f"Chain {(c, i)} not found in {structure_path}"
            selected_chain_mask += _chain_mask

        if use_resolved_positions:
            resolved_mask = selected_chain_mask * sample["atom_mask"].any(dim=-1)
            if resolved_mask.any():
                selected_chain_mask = resolved_mask

        sample = crop_by_mask(sample, selected_chain_mask)
        # remove chain ids
        _ids = list(torch.unique(sample["chain_index"].long()))
        chain_id_mapping = {c: i for c, i in chain_id_mapping.items() if i in _ids}

        assert len(chain_id_mapping) > 0, f"No chains found in {structure_path} with the given chain ids {_ids}"

        sample["chain_id_mapping"] = chain_id_mapping
        sample["target_chain_ids"] = list(target_chain_ids or chain_id_mapping.keys())

        assert (
            len(sample["target_chain_ids"]) > 0
        ), f"No target chains found in {structure_path} with the given chain ids {target_chain_ids}"

        return sample

    @staticmethod
    def check_features(sample):
        # import torch.nn.functional as F
        aatype = sample["aatype"]
        # row * 32 + col
        aatype_pair = aatype[:, None] * 32 + aatype[None, :]
        atom_pos = sample["atom_positions"]
        atom_mask = sample["atom_mask"]
        d = compute_dist(atom_pos[..., 1, :], atom_mask[..., 1])
        res_index = sample["residue_index"]
        res_index_offset = res_index[:, None] - res_index[None, :]
        chain_index = sample["chain_index"]
        same_chain = chain_index[:, None] == chain_index[None, :]
        entity_index = sample["entity_index"]
        same_entity = entity_index[:, None] == entity_index[None, :]
        df = pd.DataFrame(
            {
                "aatype_pair": aatype_pair.flatten(),
                "dist": d.flatten(),
                "res_index_offset": res_index_offset.flatten(),
                "same_chain": same_chain.flatten(),
                "same_entity": same_entity.flatten(),
            }
        )
        _short_range_df = df[(df["dist"] <= 8) & (df["same_chain"] == False)]
        if len(_short_range_df) == 0:
            return
        joint_distr = np.zeros(32 * 32)
        for p in _short_range_df["aatype_pair"].values:
            joint_distr[p] += 1
        joint_distr = joint_distr / joint_distr.sum()
        x_distr, y_distr = np.zeros(32), np.zeros(32)
        for p in aatype:
            x_distr[p] += 1
            y_distr[p] += 1
        x_distr = x_distr / x_distr.sum()
        y_distr = y_distr / y_distr.sum()

        # compute mutual information and entropy
        def _log_clip(x):
            return np.log(np.clip(x, 1e-10, 1))

        x_site_ent = -np.sum(x_distr * _log_clip(x_distr))
        y_site_ent = -np.sum(y_distr * _log_clip(y_distr))
        log_indep_distr = (_log_clip(x_distr[:, None]) + _log_clip(y_distr[None, :])).reshape(-1)
        mi = np.sum(joint_distr * (_log_clip(joint_distr) - log_indep_distr))
        nmi = 2 * mi / (x_site_ent + y_site_ent)
        print("Mutual Information (nats):", mi, "Entropy (nats):", x_site_ent, y_site_ent, "normalized MI:", nmi)
        return mi

    @staticmethod
    def save_parsed_structure(sample: dict[str, torch.Tensor], save_path: Path):
        # save parsed structure as pdbs
        save_path = Path(save_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        _key_mapping = {
            "residue_index": "res_index",
            "chain_index": "chain_index",
            "entity_index": "entity_index",
            "aatype": "res_type",
            "atom_positions": "atom_positions",
            "atom_mask": "atom_mask",
            "b_factors": "b_factors",
        }
        _dict = {"chain_id_mapping": sample["chain_id_mapping"]}
        for t_k, k in _key_mapping.items():
            _dict[t_k] = sample[k].detach().cpu().numpy()
        protein_array = ProteinArray(**_dict)
        pdb_str = to_pdb(protein_array)
