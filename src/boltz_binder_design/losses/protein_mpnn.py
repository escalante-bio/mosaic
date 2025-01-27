# Log-likelihood losses for proteinMPNN
# 1. BoltzProteinMPNNLoss: Average log-likelihood of soft binder sequence given Boltz-predicted complex structure
# 2. FixedChainInverseFoldingLL: Average log-likelihood of fixed monomer sequence given fixed monomer structure

import gemmi
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float, PyTree

from ..common import LossTerm, TOKENS
from ..proteinmpnn.mpnn import MPNN_ALPHABET, ProteinMPNN
from .boltz import StructureLoss
from joltz import StructureModuleOutputs, TrunkOutputs
from boltz.data.const import ref_atoms


def boltz_to_mpnn_matrix():
    """Converts from standard tokenization to ProteinMPNN tokenization"""
    T = np.zeros((len(TOKENS), len(MPNN_ALPHABET)))
    for i, tok in enumerate(TOKENS):
        mpnn_idx = MPNN_ALPHABET.index(tok)
        T[i, mpnn_idx] = 1
    return T


class BoltzProteinMPNNLoss(StructureLoss):
    """Average log-likelihood of binder sequence given Boltz-predicted complex structure

    Args:

        mpnn: ProteinMPNN
        num_samples: int
        stop_grad: bool = True : Whether to stop gradient through the structure module output

    """

    mpnn: ProteinMPNN
    num_samples: int
    stop_grad: bool = True

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        trunk_output: TrunkOutputs,
        structure_output: StructureModuleOutputs,
        *,
        key,
    ):
        if self.stop_grad:  
            structure_output = jax.tree.map(jax.lax.stop_gradient, structure_output)

        features = jax.tree_map(lambda x: x[0], features)

        binder_length = sequence.shape[0]

        total_length = features["res_type"].shape[0]
        # Get the atoms required for proteinMPNN:
        # In order these are N, C-alpha, C, O
        assert ref_atoms["UNK"][:4] == ["N", "CA", "C", "O"]
        # first step, which is a bit cryptic is to get the first atom for each token
        first_atom_idx = jax.vmap(lambda atoms: jnp.nonzero(atoms, size=1)[0][0])(
            features["atom_to_token"].T
        )
        # NOTE: this is completely fail if any tokens are non-protein!
        all_atom_coords = structure_output.sample_atom_coords
        coords = jnp.stack([all_atom_coords[first_atom_idx + i] for i in range(4)], -2)

        full_sequence = jnp.concatenate(
            [sequence, features["res_type"][binder_length:, 2:22]], 0
        )
        sequence_mpnn = full_sequence @ boltz_to_mpnn_matrix()
        mpnn_mask = jnp.ones(total_length, dtype=jnp.int32)
        # adjust residue idx by chain
        asym_id = features["asym_id"]
        # hardcode max number of chains = 16
        chain_lengths = (asym_id[:, None] == np.arange(16)[None]).sum(-2)
        # vector of length 16 with length of each chain
        res_idx_adjustment = jnp.cumsum(chain_lengths, -1) - chain_lengths
        # now add res_idx_adjustment to each chain
        residue_idx = (
            features["residue_index"]
            + (asym_id[:, None] == np.arange(16)[None]) @ res_idx_adjustment
        )
        # this is why I dislike vectorized code
        # add 100 residue gap to match proteinmpnn
        residue_idx += 100 * asym_id

        # alright, we have all our features.
        # encode the fixed structure
        h_V, h_E, E_idx = self.mpnn.encode(
            X=coords,
            mask=mpnn_mask,
            residue_idx=residue_idx,
            chain_encoding_all=asym_id,
        )

        def decoder_LL(key):
            # MPNN is cheap, let's call the decoder a few times to average over random decoding order
            # generate a decoding order
            # this should be random but end with the binder
            decoding_order = (
                jax.random.uniform(key, shape=(total_length,))
                .at[:binder_length]
                .add(2.0)
            )

            logits = self.mpnn.decode(
                S=sequence_mpnn,
                h_V=h_V,
                h_E=h_E,
                E_idx=E_idx,
                mask=mpnn_mask,
                decoding_order=decoding_order,
            )[0]

            return (
                (logits[:binder_length] * (sequence @ boltz_to_mpnn_matrix()))
                .sum(-1)
                .mean()
            )

        binder_ll = (
            jax.vmap(decoder_LL)(jax.random.split(key, self.num_samples))
        ).mean()

        return -binder_ll, {"protein_mpnn_ll": binder_ll}


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
            print(f"Structure {st.name} has {len(model)} chains, expected 1. Using first chain!")
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
