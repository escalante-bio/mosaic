# Log-likelihood losses for proteinMPNN
# 1. BoltzProteinMPNNLoss: Average log-likelihood of soft binder sequence given Boltz-predicted complex structure
# 2. FixedChainInverseFoldingLL: Average log-likelihood of fixed monomer sequence given fixed monomer structure

import gemmi
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float

from ..common import TOKENS, LossTerm
from ..proteinmpnn.mpnn import MPNN_ALPHABET, ProteinMPNN


def boltz_to_mpnn_matrix():
    """Converts from standard tokenization to ProteinMPNN tokenization"""
    T = np.zeros((len(TOKENS), len(MPNN_ALPHABET)))
    for i, tok in enumerate(TOKENS):
        mpnn_idx = MPNN_ALPHABET.index(tok)
        T[i, mpnn_idx] = 1
    return T


class FixedChainInverseFoldingLL(LossTerm):
    mpnn: ProteinMPNN
    encoded_state: tuple
    name: str

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        *,
        key,
    ):
        binder_length = sequence.shape[0]
        # assert self.coords.shape[0] == self.encoded_state.shape[1], "Sequence length mismatch"

        sequence_mpnn = sequence @ boltz_to_mpnn_matrix()
        mpnn_mask = jnp.ones(binder_length, dtype=jnp.int32)

        # generate a decoding order
        decoding_order = jax.random.uniform(key, shape=(binder_length,))
        logits = self.mpnn.decode(
            S=sequence_mpnn,
            h_V=self.encoded_state[0],
            h_E=self.encoded_state[1],
            E_idx=self.encoded_state[2],
            mask=mpnn_mask,
            decoding_order=decoding_order,
        )[0]

        ll = (logits * sequence_mpnn).sum(-1).mean()

        return -ll, {f"{self.name}_ll": ll}

    @staticmethod
    def from_structure(
        st: gemmi.Structure,
        mpnn: ProteinMPNN,
    ):
        st = st.clone()
        st.remove_ligands_and_waters()
        st.remove_alternative_conformations()
        st.remove_empty_chains()
        model = st[0]
        if len(model) != 1:
            print(
                f"Structure {st.name} has {len(model)} chains, expected 1. Using first chain!"
            )
        chain = model[0]
        coords = np.zeros((len(chain), 4, 3))

        def _set_coords(idx: int, atom_idx: int, atom_name: str):
            try:
                atom = chain[idx].sole_atom(atom_name)
                pos = atom.pos
                coords[idx, atom_idx, 0] = pos.x
                coords[idx, atom_idx, 1] = pos.y
                coords[idx, atom_idx, 2] = pos.z
            except Exception:
                print(f"Failed to get {atom_name} for residue {chain[idx].name}")
                coords[idx, atom_idx] = np.nan

        for idx in range(len(chain)):
            _set_coords(idx, 0, "N")
            _set_coords(idx, 1, "CA")
            _set_coords(idx, 2, "C")
            _set_coords(idx, 3, "O")

        # encode the structure
        h_V, h_E, E_idx = mpnn.encode(
            X=coords,
            mask=jnp.ones(len(chain), dtype=jnp.int32),
            residue_idx=jnp.arange(len(chain)),
            chain_encoding_all=jnp.zeros(len(chain), dtype=jnp.int32),
        )

        return FixedChainInverseFoldingLL(
            mpnn=mpnn,
            encoded_state=(h_V, h_E, E_idx),
            name=st.name,
        )
