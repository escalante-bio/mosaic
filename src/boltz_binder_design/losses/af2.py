# TODO: This is fairly rough, especially how we inject soft sequences into alphafold. Clean this up!

import jax
import jax.numpy as jnp
from jax import tree
from jaxtyping import Array, Float, PyTree

from ..common import LossTerm, LinearCombination, TOKENS
from ..proteinmpnn.mpnn import MPNN_ALPHABET, ProteinMPNN
from ..af2.featurization import AFFeatures
from ..af2.alphafold2 import AFOutput
from .boltz import contact_cross_entropy

import numpy as np


def af2_to_mpnn_matrix():
    """Converts from standard tokenization to ProteinMPNN tokenization"""
    T = np.zeros((len(TOKENS), len(MPNN_ALPHABET)))
    for i, tok in enumerate(TOKENS):
        mpnn_idx = MPNN_ALPHABET.index(tok)
        T[i, mpnn_idx] = 1
    return T


class AFLoss(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: AFFeatures,
        output: AFOutput,
        key=None,
    ):
        raise NotImplementedError


class AlphaFold(LossTerm):
    forward: callable
    stacked_params: PyTree
    features: AFFeatures
    losses: LinearCombination
    name: str

    def __call__(self, soft_sequence: Float[Array, "N 20"], *, key):
        # pick a random model
        model_idx = jax.random.randint(key=key, shape=(), minval=0, maxval=5)
        params = tree.map(lambda v: v[model_idx], self.stacked_params)
        # build full soft sequence
        full_sequence = jax.nn.one_hot(self.features.aatype, 21)
        # set binder sequence
        full_sequence = full_sequence.at[: soft_sequence.shape[0], :20].set(
            soft_sequence
        )
        # run the model
        output = self.forward(
            params,
            jax.random.fold_in(key, 1),
            features=self.features,
            initial_guess=None,
            replace_target_feat=full_sequence,
        )

        v, aux = self.losses(
            sequence=soft_sequence, features=self.features, output=output, key=key
        )

        return v, {f"{self.name}/{k}": v for k, v in aux.items()} | {
            f"{self.name}/model_idx": model_idx,
            f"{self.name}/loss": v,
        }


class WithinBinderContact(AFLoss):
    max_contact_distance: float = 14.0
    min_sequence_separation: int = 8
    num_contacts_per_residue: int = 25

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: AFFeatures,
        output: AFOutput,
        key=None,
    ):
        binder_len = sequence.shape[0]
        log_contact_intra = contact_cross_entropy(
            output.distogram.logits[:binder_len, :binder_len],
            self.max_contact_distance,
            output.distogram.bin_edges[0],
            output.distogram.bin_edges[-1],
        )
        # only count binder-binder contacts with sequence sep > min_sequence_separation
        within_binder_mask = (
            jnp.abs(jnp.arange(binder_len)[:, None] - jnp.arange(binder_len)[None, :])
            > self.min_sequence_separation
        )
        # for each position in binder find positions most likely to make contact
        binder_binder_max_p, _ = jax.vmap(
            lambda lcp: jax.lax.top_k(lcp, self.num_contacts_per_residue)
        )(log_contact_intra + (1 - within_binder_mask) * -30)
        average_log_prob = binder_binder_max_p.mean()

        return -average_log_prob, {"intra_contact": average_log_prob}


class PLDDTLoss(AFLoss):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: AFFeatures,
        output: AFOutput,
        key=None,
    ):
        binder_len = sequence.shape[0]
        plddt = output.plddt[:binder_len].mean()
        return -plddt, {"plddt": plddt}


class BinderTargetContact(AFLoss):
    """Encourages contacts between binder and target."""

    paratope_idx: list[int] | None = None
    paratope_size: int | None = None
    contact_distance: float = 20.0

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: AFFeatures,
        output: AFOutput,
        key=None,
    ):
        binder_len = sequence.shape[0]
        log_contact_inter = contact_cross_entropy(
            output.distogram.logits[:binder_len, binder_len:],
            self.contact_distance,
            output.distogram.bin_edges[0],
            output.distogram.bin_edges[-1],
        )

        # binder_target_max_p = log_contact_inter[:binder_len, binder_len:].max(-1)
        binder_target_max_p = jax.vmap(lambda v: jax.lax.top_k(v, 3)[0])(
            log_contact_inter
        ).mean(-1)
        # log probability of contacting target for each position in binder

        if self.paratope_idx is not None:
            binder_target_max_p = binder_target_max_p[self.paratope_idx]
        if self.paratope_size is not None:
            binder_target_max_p = jax.lax.top_k(
                binder_target_max_p, self.paratope_size
            )[0]

        average_log_prob = binder_target_max_p.mean()
        return -average_log_prob, {"target_contact": average_log_prob}


class DistogramCE(AFLoss):
    f: Float[Array, "... B"]
    name: str

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: AFFeatures,
        output: AFOutput,
        key=None,
    ):
        binder_len = sequence.shape[0]
        # expand dims so self.f is broadcastable to network_output["pdistogram"] of size (N, N, B)
        f = jnp.expand_dims(
            self.f, [i for i in range(output.distogram.logits.ndim - self.f.ndim)]
        )

        ce = -jnp.fill_diagonal(
            (
                jax.nn.log_softmax(output.distogram.logits)[:binder_len, :binder_len]
                * f
            ).sum(-1),
            0,
            inplace=False,
        ).mean()

        return ce, {self.name: ce}


class BinderPAE(AFLoss):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: AFFeatures,
        output: AFOutput,
        key=None,
    ):
        binder_len = sequence.shape[0]

        pae_within = jnp.fill_diagonal(
            output.predicted_aligned_error[:binder_len, :binder_len], 0, inplace=False
        ).mean()

        return pae_within, {"pae_within": pae_within}


class BinderTargetPAE(AFLoss):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: AFFeatures,
        output: AFOutput,
        key=None,
    ):
        binder_len = sequence.shape[0]

        p = output.predicted_aligned_error[:binder_len, binder_len:].mean()

        return p, {"bt_pae": p}


class TargetBinderPAE(AFLoss):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: AFFeatures,
        output: AFOutput,
        key=None,
    ):
        binder_len = sequence.shape[0]

        p = output.predicted_aligned_error[binder_len:, :binder_len].mean()

        return p, {"tb_pae": p}


class IPTM(AFLoss):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: AFFeatures,
        output: AFOutput,
        key=None,
    ):
        return output.iptm, {"iptm": output.iptm}


class RadiusOfGyration(AFLoss):
    target_radius: float

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: AFFeatures,
        output: AFOutput,
        key=None,
    ):
        binder_len = sequence.shape[0]
        first_atom_coords = output.structure_module.final_atom_positions[:binder_len, 0]
        rg = jnp.sqrt(
            ((first_atom_coords - first_atom_coords.mean(0)) ** 2).sum(-1).mean()
        )

        return jax.nn.elu(rg - self.target_radius), {"actual_rg": rg}


class AFProteinMPNNLoss(AFLoss):
    """Average log-likelihood of binder sequence given AF2-predicted complex structure

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
        features: AFFeatures,
        output: AFOutput,
        *,
        key,
    ):
        if self.stop_grad:
            output = jax.tree.map(jax.lax.stop_gradient, output)

        binder_length = sequence.shape[0]

        # total_length = features["res_type"].shape[0]
        total_length = features.aatype.shape[0]
        # Get the atoms required for proteinMPNN:
        # In order these are N, C-alpha, C, O
        coords = output.structure_module.final_atom_positions[:, :4]

        full_sequence = jnp.concatenate(
            [sequence, jax.nn.one_hot(features.aatype[binder_length:], 20)], 0
        )
        sequence_mpnn = full_sequence @ af2_to_mpnn_matrix()
        mpnn_mask = jnp.ones(total_length, dtype=jnp.int32)
        # adjust residue idx by chain
        # asym_id = features["asym_id"]
        asym_id = features.asym_id
        # hardcode max number of chains = 16
        chain_lengths = (asym_id[:, None] == np.arange(16)[None]).sum(-2)
        # vector of length 16 with length of each chain
        res_idx_adjustment = jnp.cumsum(chain_lengths, -1) - chain_lengths
        # now add res_idx_adjustment to each chain
        residue_idx = (
            features.residue_index
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
                (logits[:binder_length] * (sequence @ af2_to_mpnn_matrix()))
                .sum(-1)
                .mean()
            )

        binder_ll = (
            jax.vmap(decoder_LL)(jax.random.split(key, self.num_samples))
        ).mean()

        return -binder_ll, {"protein_mpnn_ll": binder_ll}
