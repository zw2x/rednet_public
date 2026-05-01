
from typing import Mapping
import torch
import numpy as np

import rednet.residue_constants as rc

# fmt:off
# restype_with_x + special tokens (32)
PROTEIN_ALPHABET = rc.PROTEIN_TYPES_WITH_X + ('B', 'U', 'Z', 'O', '.', '-', '<bos>', '<eos>', '<mask>', '<pad>', '<cls>')
# rna, dna, common atoms, common metals, other common chemicals more special tokens (see gemmi, 400)
extended_alphabet = []
# fmt:on


def exists(val):
    return val is not None

class Tokenizer:
    @classmethod
    def make_tokenizer(cls, tokenizer_type: str = "default", **kwargs):
        if tokenizer_type == "default":
            return cls(**kwargs)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Available types: 'default'.")

    def __init__(
        self,
        alphabet=PROTEIN_ALPHABET,
        *,
        append_eos=False,
        prepend_bos=False,
        bos_tok: str = "<bos>",
        eos_tok: str = "<eos>",
        mask_tok: str = "<mask>",
        pad_tok: str = "<pad>",
    ):
        """Biomolecule tokenizer"""
        self.alphabet = alphabet
        self.tok_to_int = {k: i for i, k in enumerate(self.alphabet)}
        # specify special tokens
        self._pad_tok = pad_tok
        self._pad_id = self.tok_to_int[self._pad_tok]
        self._eos_tok = eos_tok or self._pad_tok
        self._eos_id = self.tok_to_int[self._eos_tok]
        self._bos_tok = bos_tok or self._pad_tok
        self._bos_id = self.tok_to_int[self._bos_tok]
        self._mask_tok = mask_tok or self._pad_tok
        self._mask_id = self.tok_to_int[self._mask_tok]

        self._append_eos = append_eos
        self._prepend_bos = prepend_bos

    def add_bos(self, tokens: torch.LongTensor) -> torch.LongTensor:
        """add bos token to the beginning of the sequence"""
        if self.prepend_bos:
            _shape = list(tokens.shape)
            _shape[-1] = 1
            tokens = torch.cat([tokens.new_full(_shape, self._bos_id), tokens], dim=-1)
        return tokens

    def tokenize(self, text: str) -> list[str]:
        return list(text)

    def encode(self, text: str, *args, **kwargs) -> torch.LongTensor:
        """encode string to integer tokens"""
        raise NotImplementedError

    def decode(
        self, tokens: torch.LongTensor, chain_index: torch.LongTensor = None, *args, **kwargs
    ) -> str | Mapping[int, str]:
        """decode integer tokens to string"""
        if exists(chain_index):
            chain_ids = chain_index.unique()
            seqs = {int(i): self.decode(tokens[chain_index == i]) for i in chain_ids}
            return seqs
        seq = "".join([self.alphabet[i] for i in tokens.tolist()])
        return seq

    def translate(self, index: np.ndarray, to_scop=False, **kwargs) -> np.ndarray:
        """translate tokens from source to target"""
        index = index.astype(np.int64)
        if to_scop:
            translated_tokens = np.array([rc.RESTYPE_1TO3.get(self.alphabet[i], "UNK") for i in index.tolist()])
        else:
            translated_tokens = np.array([self.alphabet[i] for i in index.tolist()])
        return translated_tokens

    def __len__(self):
        return len(self.alphabet)

    def __contains__(self, tok: str):
        return tok in self.tok_to_int

    @property
    def mask_id(self) -> int:
        return self._mask_id

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def bos_tok(self) -> str:
        return self._bos_tok

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def padding_idx(self) -> int:  # alias for pad_id
        return self._pad_id

    @property
    def append_eos(self) -> bool:
        return self._append_eos

    @property
    def prepend_bos(self) -> bool:
        return self._prepend_bos

    def get_idx(self, tok: str) -> int:
        return self.tok_to_int[tok]
