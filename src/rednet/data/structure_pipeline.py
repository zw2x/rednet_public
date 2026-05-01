import os
from pathlib import Path
import torch

from .tokenizer import Tokenizer
from .pdb_structure_pipeline import StructurePipeline as PdbStructurePipeline

from rednet.common_utils import get_logger

log = get_logger(__name__)


class StructurePipeline:
    def __init__(self, config, tokenizer: Tokenizer, deterministic: bool = False):
        super().__init__()

        self.pipeline = PdbStructurePipeline(config, tokenizer, deterministic=deterministic)

        self.config = config
        self.radius = config.get("radius", 10.0)
        self.use_crop = config.get("use_crop", True)
        self.reorder_chains = config.get("reorder_chains", False)
        if self.use_crop:
            self._crop_size = config.crop_size
        self.deterministic = deterministic
        self.tokenizer = tokenizer

        # self.force_mode = config.get("force_mode", 'SPATIAL_RESOLVED')
        self.force_mode = config.get("force_mode")

    @staticmethod
    def load_parsed_structure(
        structure_path: Path,
        chain_ids: tuple[str] | None = None,
        target_chain_ids: tuple[str] | None = None,
        use_resolved_positions: bool = True,
    ):
        return PdbStructurePipeline.load_parsed_structure(
            structure_path,
            chain_ids=chain_ids,
            target_chain_ids=target_chain_ids,
            use_resolved_positions=use_resolved_positions,
        )

    def transform(self, sample, index: int):
        sample = self.pipeline.transform(sample, index)
        # if self.use_crop:
        #     sample, crop_mask = self.crop(sample, index, deterministic=self.deterministic, force_mode=self.force_mode)

        if "interface_chains" in sample:
            # convert set to tuple for tree
            sample["interface_chains"] = tuple(sample["interface_chains"])

        # if self.reorder_chains:
        #     sample = self.reorder_design_chains(sample)

        return sample

    def reorder_design_chains(self, sample):
        dsn_mask = sample["dsn_mask"]

        # Create indices for reordering: False values first, then True values
        false_indices = torch.where(~dsn_mask)[0]  # indices where dsn_mask is False
        if len(false_indices) == 0:
            return sample

        true_indices = torch.where(dsn_mask)[0]  # indices where dsn_mask is True

        # Combine indices: False first, then True
        reorder_indices = torch.cat([false_indices, true_indices])

        # Create new dictionary with reordered data
        reordered_dict = {}

        # Reorder the fields in the sample dictionary
        # fmt: off
        fields_to_reorder = [
            'atom_positions', 'atom_mask', 'chain_index', 'b_factors', 'entity_index', 'interface_dist', 'res_type', 'res_index', 'mask', 'interface_site_mask', 'site_mask', 'resolved_mask', 'gt_res_type', 'enc_res_type', 'masked_tokens', 'dsn_mask', 'pred_mask', 'res_conf'
        ]
        # fmt: on
        for k in sample:
            if k in fields_to_reorder:
                reordered_dict[k] = sample[k][reorder_indices]
            else:
                reordered_dict[k] = sample[k]
        return reordered_dict


# def _get_batch(names, crop_size=64, dataset_dir=None):
#     from omegaconf import OmegaConf
# 
#     dataset_dir = Path(dataset_dir or os.environ["PDB_DATASET_DIR"])
#     # tokenizer = make_tokenizer("default")
#     tokenizer = Tokenizer()
#     batch = []
#     for name in names:
#         structure_path = dataset_dir / "parsed_structures" / f"{name}.lz4"
#         assert structure_path.exists(), f"Structure file {structure_path} does not exist"
#         sample = StructurePipeline.load_parsed_structure(structure_path)
#         feat_cfg = {
#             "use_structure": True,
#             "use_crop": True,
#             "crop_size": crop_size,
#             "radius": 10.0,
#             "reorder_chains": True,
#         }
#         pipeline = StructurePipeline(config=OmegaConf.create(feat_cfg), deterministic=True, tokenizer=tokenizer)
#         sample = pipeline.transform(sample, index=0)
#         batch.append(sample)
#     batch = IStructDataset.collate_fn(batch, tokenizer)
#     # print_batch(batch)
#     return batch
# 