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
from .structure_prediction import AbstractStructureOutput


def boltz_to_mpnn_matrix():
    """Converts from standard tokenization to ProteinMPNN tokenization"""
    T = np.zeros((len(TOKENS), len(MPNN_ALPHABET)))
    for i, tok in enumerate(TOKENS):
        mpnn_idx = MPNN_ALPHABET.index(tok)
        T[i, mpnn_idx] = 1
    return T


def load_chain(chain: gemmi.Chain) -> tuple[str, Float[Array, "N 4 3"]]:
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

    return gemmi.one_letter_code([r.name for r in chain]), coords


class FixedStructureInverseFoldingLL(LossTerm):
    sequence_boltz: Float[Array, "N 20"]
    mpnn: ProteinMPNN
    encoded_state: tuple
    name: str
    stop_grad: bool = False

    def __call__(
        self,
        binder_sequence: Float[Array, "N 20"],
        *,
        key,
    ):
        binder_length = binder_sequence.shape[0]
        complex_length = self.sequence_boltz.shape[0]
        # assert self.coords.shape[0] == self.encoded_state.shape[1], "Sequence length mismatch"

        # replace binder sequence
        sequence = self.sequence_boltz.at[:binder_length].set(binder_sequence)

        sequence_mpnn = sequence @ boltz_to_mpnn_matrix()
        mpnn_mask = jnp.ones(complex_length, dtype=jnp.int32)

        # generate a decoding order that ends with binder
        decoding_order = jax.random.uniform(key, shape=(complex_length,))
        decoding_order = decoding_order.at[:binder_length].add(2.0)
        logits = self.mpnn.decode(
            S=sequence_mpnn,
            h_V=self.encoded_state[0],
            h_E=self.encoded_state[1],
            E_idx=self.encoded_state[2],
            mask=mpnn_mask,
            decoding_order=decoding_order,
        )[0]
        if self.stop_grad:
            logits = jax.lax.stop_gradient(logits)

        ll = (logits * sequence_mpnn).sum(-1)[:binder_length].mean()

        return -ll, {f"{self.name}_ll": ll}

    @staticmethod
    def from_structure(
        st: gemmi.Structure,
        mpnn: ProteinMPNN,
        stop_grad: bool = False,
    ):
        st = st.clone()
        st.remove_ligands_and_waters()
        st.remove_alternative_conformations()
        st.remove_empty_chains()
        model = st[0]

        sequences_and_coords = [load_chain(c) for c in model]

        residue_idx = np.concatenate(
            [
                np.arange(len(s)) + chain_idx * 100
                for (chain_idx, (s, _)) in enumerate(sequences_and_coords)
            ]
        )

        chain_encoding = np.concatenate(
            [
                np.ones(len(s)) * chain_idx
                for (chain_idx, (s, _)) in enumerate(sequences_and_coords)
            ]
        )
        coords = np.concatenate([c for (_, c) in sequences_and_coords])
        # encode the structure
        h_V, h_E, E_idx = mpnn.encode(
            X=coords,
            mask=jnp.ones(coords.shape[0], dtype=jnp.int32),
            residue_idx=residue_idx,  # jnp.arange(len(chain)),
            chain_encoding_all=chain_encoding,  # jnp.zeros(len(chain), dtype=jnp.int32),
            key = jax.random.key(np.random.randint(1000000))
        )
        # one hot sequence
        full_sequence = "".join(s for (s, _) in sequences_and_coords)

        return FixedStructureInverseFoldingLL(
            sequence_boltz=jax.nn.one_hot(
                [TOKENS.index(AA) if AA in TOKENS else 0 for AA in full_sequence ], 20
            ),
            mpnn=mpnn,
            encoded_state=(h_V, h_E, E_idx),
            name=st.name,
            stop_grad=stop_grad,
        )


class ProteinMPNNLoss(LossTerm):
    """Average log-likelihood of binder sequence given predicted complex structure

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
        output: AbstractStructureOutput,
        key,
    ):
        
        # Get the atoms required for proteinMPNN:
        # In order these are N, C-alpha, C, O
        coords = output.backbone_coordinates
        if self.stop_grad:
            coords = jax.lax.stop_gradient(coords)


        binder_length = sequence.shape[0]

      
        # NOTE: this will completely fail if any tokens are non-protein!
        # all_atom_coords = structure_output.sample_atom_coords
        # coords = jnp.stack([all_atom_coords[first_atom_idx + i] for i in range(4)], -2)
        full_sequence = output.full_sequence.at[:binder_length].set(sequence)
        total_length = full_sequence.shape[0]

        sequence_mpnn = full_sequence @ boltz_to_mpnn_matrix()
        mpnn_mask = jnp.ones(total_length, dtype=jnp.int32)
        # adjust residue idx by chain
        asym_id = output.asym_id
        # hardcode max number of chains = 16
        chain_lengths = (asym_id[:, None] == np.arange(16)[None]).sum(-2)
        # vector of length 16 with length of each chain
        res_idx_adjustment = jnp.cumsum(chain_lengths, -1) - chain_lengths
        # now add res_idx_adjustment to each chain
        residue_idx = (
            output.residue_idx
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
            key = key
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
