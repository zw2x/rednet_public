# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import numpy as np
import collections
from typing import Optional
from Bio.PDB import Structure, PDBParser
import dataclasses

import rednet.residue_constants as rc

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.

@dataclasses.dataclass(kw_only=True, frozen=True)
class ProteinArray:
    # num_atom = 37
    atom_positions: np.ndarray  # [num_res, num_atom, 3]
    atom_mask: np.ndarray  # [num_res, num_atom]
    aatype: np.ndarray
    residue_index: np.ndarray
    chain_index: np.ndarray
    b_factors: np.ndarray  # [num_res, num_atoms]
    entity_index: np.ndarray  # entity index for each chain
    chain_id_mapping: dict[str, int]  # label_asym_id -> chain_index

    def __post_init__(self):
        num_res, num_atom_type, _ = self.atom_positions.shape
        assert self.atom_mask.shape == (num_res, num_atom_type)
        assert self.aatype.shape == (num_res,)
        assert self.residue_index.shape == (num_res,)
        assert self.chain_index.shape == (num_res,)
        assert self.b_factors.shape == (num_res, num_atom_type)
        assert self.entity_index.shape == (num_res,)
        num_chains = len(self.chain_id_mapping)
        assert (
            len(np.unique(self.chain_index)) == num_chains
        ), f"Number of chains {num_chains} does not match {len(np.unique(self.chain_index))}"

    @property
    def num_chains(self):
        return len(self.chain_id_mapping)
    
    @classmethod
    def merge(cls, arrays):
        assert len(arrays) > 0, "No arrays to merge"
        if len(arrays) == 1:
            return arrays[0]
        _inputs = {
            k.name: np.concatenate([getattr(a, k.name) for a in arrays], axis=0)
            for k in dataclasses.fields(cls)
            if k.name != "chain_id_mapping"
        }
        chain_id_mapping = {k: v for a in arrays for k, v in a.chain_id_mapping.items()}
        _inputs["chain_id_mapping"] = chain_id_mapping
        return cls(**_inputs)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def parse_pdb_string(pdb_str: str, chain_id=None, **kwargs) -> ProteinArray:
    """Parse a PDB file and return a structure object."""
    with io.StringIO(pdb_str) as pdb_fh:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(id="none", file=pdb_fh)
    return _from_bio_structure(structure, chain_id=chain_id, **kwargs)


def _from_bio_structure(
    structure: Structure,
    chain_id: Optional[str] = None,
    ignore_hetatom: bool = True,
    raise_insertion_error: bool = True,
    keep_insertion_code: bool = False,
) -> ProteinArray:
    """Takes a Biopython structure and creates a `Protein` instance.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      structure: Structure from the Biopython library.
      chain_id: If chain_id is specified (e.g. A), then only that chain is parsed.
        Otherwise all chains are parsed.

    Returns:
      A new `Protein` created from the structure contents.

    Raises:
      ValueError: If the number of models included in the structure is not 1.
      ValueError: If insertion code is detected at a residue.
    """
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError("Only single model PDBs/mmCIFs are supported. Found" f" {len(models)} models.")
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    entity_seqs = collections.defaultdict(list)
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        _seq = []
        for res in chain:
            if res.id[2] != " " and not keep_insertion_code:
                if raise_insertion_error:
                    raise ValueError(
                        f"PDB/mmCIF contains an insertion code at chain {chain.id} and residue index {res.id[1]}. "
                        "These are not supported."
                    )
                else:
                    continue
            if ignore_hetatom and res.id[0] != " ":
                continue
            res_shortname = rc.RESTYPE_3TO1.get(res.resname, "X")
            restype_idx = rc.RESTYPE_ORDER.get(res_shortname, rc.RESTYPE_NUM)
            pos = np.zeros((rc.ATOM_TYPE_NUM, 3))
            mask = np.zeros((rc.ATOM_TYPE_NUM,))
            res_b_factors = np.zeros((rc.ATOM_TYPE_NUM,))
            for atom in res:
                if atom.name not in rc.ATOM_TYPES:
                    continue
                pos[rc.ATOM_ORDER[atom.name]] = atom.coord
                mask[rc.ATOM_ORDER[atom.name]] = 1.0
                res_b_factors[rc.ATOM_ORDER[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)
            _seq.append(res_shortname)
        _seq = "".join(_seq)
        entity_seqs[_seq].append(chain.id)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])
    entity_id_mapping = {}
    for ent_id, (seq, cids) in enumerate(entity_seqs.items()):
        for cid in cids:
            entity_id_mapping[cid] = ent_id
    entity_index = np.array([entity_id_mapping[cid] for cid in chain_ids])

    return ProteinArray(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
        chain_id_mapping=chain_id_mapping,  # author_chain_id to chain_index
        entity_index=entity_index,
    )


def to_pdb(prot: ProteinArray) -> str:
    """Converts a `ProteinArray` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = rc.PROTEIN_TYPES_WITH_X
    res_1to3 = lambda r: rc.RESTYPE_1TO3.get(restypes[r], "UNK")
    atom_types = rc.ATOM_TYPES

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > rc.RESTYPE_NUM):
        raise ValueError("Invalid aatypes.")

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(f"The PDB format supports at most {PDB_MAX_CHAINS} chains.")
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append("MODEL     1")
    atom_index = 1
    last_chain_index = chain_index[0]
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    res_1to3(aatype[i - 1]),
                    chain_ids[chain_index[i - 1]],
                    residue_index[i - 1],
                )
            )
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(
        _chain_end(
            atom_index,
            res_1to3(aatype[-1]),
            chain_ids[chain_index[-1]],
            residue_index[-1],
        )
    )
    pdb_lines.append("ENDMDL")
    pdb_lines.append("END")

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # Add terminating newline.


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = "TER"
    return f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} " f"{chain_name:>1}{residue_index:>4}"
