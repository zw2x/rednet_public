import torch
import torch.nn.functional as F

from einops import rearrange, repeat

ATOM_TYPES = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]

PROTEIN_TYPES = ("A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V")
RESTYPE_1TO3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

FA14_MAPPING = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
    "UNK": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
}


def get_indices(atom_mapping):
    atom_to_idx = {name: idx for idx, name in enumerate(ATOM_TYPES)}

    indices = []
    for t in PROTEIN_TYPES + ("X",):
        mapping = atom_mapping[RESTYPE_1TO3.get(t, "UNK")]
        indices.append([atom_to_idx.get(atom, len(ATOM_TYPES)) for atom in mapping])  # use last index for missing atoms

    return torch.tensor(indices, dtype=torch.long)


FA14_INDICES = get_indices(FA14_MAPPING)  # (21, 14)


def flatten_atoms(x, atom_type, unk_idx=len(ATOM_TYPES), pad_value=-1):
    mask = atom_type != unk_idx
    bsz = x.size(0)
    _flatten = []
    for _x, _mask in zip(x, mask):
        _flatten.append(_x[_mask])
    _padded = []
    max_len = max([v.size(0) for v in _flatten])
    for i in range(bsz):
        pad_size = max_len - _flatten[i].size(0)
        pad_dims = (0, 0) * (_flatten[i].dim() - 1) + (0, pad_size)
        _padded.append(F.pad(_flatten[i], pad_dims, value=pad_value))
    _padded = torch.stack(_padded, dim=0)
    return _padded


def infer_atom_type(
    res_type: torch.LongTensor, layout_indices: torch.LongTensor = FA14_INDICES, return_dense_atom_type: bool = False
):
    num_res = res_type.size(1)
    res_type = res_type.clamp(max=len(PROTEIN_TYPES))
    # atom_type: [bsz, num_atom]
    num_layout_atoms = layout_indices.size(1)
    indices = layout_indices.to(res_type.device)
    atom_type = indices[res_type]  # [bsz, num_res, num_layout_atoms]
    atom_type = rearrange(atom_type, "b r a -> b (r a)")
    padded_atom_type = flatten_atoms(atom_type, atom_type, pad_value=len(ATOM_TYPES))

    padded_atom_exists = padded_atom_type != len(ATOM_TYPES)

    if return_dense_atom_type:
        dense_atom_type = rearrange(atom_type, "b (r a) -> b r a", r=num_res, a=num_layout_atoms)
        return padded_atom_type, padded_atom_exists, dense_atom_type
    else:
        return padded_atom_type, padded_atom_exists


def transform_atom_from_fa37(restype, pos, mask, pad_id, atom_mapping_indices=FA14_INDICES):
    # pos: (bsz, seqlen, 37, 3)
    # mask: (bsz, seqlen, 37)
    # Gather indices for each residue type
    assert pos.shape[-2] == 37, f"Expected 37 atoms in input positions, got {pos.shape[-2]}"
    pad_mask = restype == pad_id  # (bsz, seqlen)
    restype = restype.clamp(max=len(PROTEIN_TYPES))  # turn padding residues to 'X' type
    gather_indices = atom_mapping_indices.to(restype)[restype]  # (bsz, seqlen, k)

    # Gather positions and masks
    bsz, seqlen, _, _ = pos.shape
    pad_pos = torch.zeros([bsz, seqlen, 1, 3]).to(pos)
    padded_pos = torch.cat([pos, pad_pos], dim=-2)
    padded_mask = torch.cat([mask, torch.zeros([bsz, seqlen, 1]).to(mask)], dim=-1)

    fa_mask = torch.gather(padded_mask, 2, gather_indices)  # (bsz, seqlen, k)
    fa_mask = fa_mask.masked_fill(pad_mask.unsqueeze(-1), 0.0).bool()

    # Expand for 3D coordinates
    gather_indices_pos = repeat(gather_indices, "b s a -> b s a d", d=3)

    fa_pos = torch.gather(padded_pos, 2, gather_indices_pos)  # (bsz, seqlen, k, 3)

    return fa_pos, fa_mask
