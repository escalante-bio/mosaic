"""Mosaic loss integration for OpenFold3.

Provides:
- Token vocabulary mapping between mosaic's 20-AA and OF3's 32-dim restype
- Binder sequence injection into OF3 Batch
- OF3FromTrunkOutput: lazy AbstractStructureOutput wrapping trunk state
- MultiSampleOF3Loss: trunk-once, vmap-over-samples loss term
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from mosaic.common import LinearCombination, LossTerm
from mosaic.losses.structure_prediction import AbstractStructureOutput

from jopenfold3.batch import Batch
from jopenfold3.heads.head_modules import ConfidenceOutput
from jopenfold3.model import InitialEmbedding, OpenFold3, TrunkEmbedding


# OF3's first 20 indices are standard AAs in identical order to mosaic's TOKENS:
#   A=0, R=1, N=2, D=3, C=4, Q=5, E=6, G=7, H=8, I=9, L=10, K=11,
#   M=12, F=13, P=14, S=15, T=16, W=17, Y=18, V=19
# Indices 20-31 are UNK, RNA(5), DNA(5), GAP.
_MOSAIC_TO_OF3 = np.zeros((20, 32), dtype=np.float32)
_MOSAIC_TO_OF3[:20, :20] = np.eye(20)

# OF3 vocab: 0-19 protein, 20 UNK, 21-25 RNA, 26-30 DNA, 31 GAP
GAP_IDX = 31




def set_binder_sequence(new_sequence: Float[Array, "N 20"], batch: Batch,
                        lut=None) -> Batch:
    """Inject a soft sequence (PSSM) into the binder positions of an OF3 Batch.

    Assumes the binder occupies the first ``new_sequence.shape[0]`` tokens.
    Always updates ``restype``, ``profile``, ``msa`` (soft, differentiable).

    """

    binder_len = new_sequence.shape[0]
    of3_sequence = new_sequence @ jnp.array(_MOSAIC_TO_OF3)  # [N, 32]

    new_restype = batch.restype.at[0, :binder_len, :].set(of3_sequence)

    new_profile = batch.profile.at[0, :binder_len].set(of3_sequence)
    # Set query row in MSA
    new_msa = batch.msa.at[0, 0, :binder_len].set(of3_sequence)
    # Set the rest of MSA for binder to GAP (no hits)
    new_msa = new_msa.at[0, 1:, :binder_len,].set(jax.nn.one_hot(GAP_IDX, 32))

    return eqx.tree_at(
        lambda b: (b.restype, b.profile, b.msa),
        batch,
        (new_restype, new_profile, new_msa),
    )

    # # --- Atom feature update + compaction (for prediction with poly-W) ---
    # # Requires lut (AtomLookup) and a 14-stride binder layout.
    # # Commented out: with poly-X featurization the binder has 6 atoms/token
    # # (UNK layout), so this 14-stride compaction code doesn't apply.
    # binder_slots = binder_len * 14
    # hard_idx = new_sequence.argmax(-1)
    #
    # new_ref_pos = jax.lax.stop_gradient(lut.ref_pos[hard_idx].reshape(binder_slots, 3))
    # new_ref_element = jax.lax.stop_gradient(lut.ref_element[hard_idx].reshape(binder_slots, 119))
    # new_ref_charge = jax.lax.stop_gradient(lut.ref_charge[hard_idx].reshape(binder_slots))
    # new_ref_mask = jax.lax.stop_gradient(lut.ref_mask[hard_idx].reshape(binder_slots))
    # new_atom_name = jax.lax.stop_gradient(lut.ref_atom_name_chars[hard_idx].reshape(binder_slots, 4, 64))
    # new_atom_mask = jax.lax.stop_gradient(lut.atom_mask[hard_idx].reshape(binder_slots))
    #
    # padded_ref_pos = batch.ref_pos.at[0, :binder_slots].set(new_ref_pos)
    # padded_ref_element = batch.ref_element.astype(jnp.float32).at[0, :binder_slots].set(new_ref_element)
    # padded_ref_charge = batch.ref_charge.at[0, :binder_slots].set(new_ref_charge)
    # padded_ref_mask = batch.ref_mask.astype(jnp.float32).at[0, :binder_slots].set(new_ref_mask)
    # padded_atom_name = batch.ref_atom_name_chars.astype(jnp.float32).at[0, :binder_slots].set(new_atom_name)
    # padded_atom_mask = batch.atom_mask.at[0, :binder_slots].set(new_atom_mask)
    #
    # aa_counts = lut.atom_mask.sum(-1).astype(jnp.int32)
    # counts = aa_counts[hard_idx]
    # cumsum_right = jnp.cumsum(counts)
    # cumsum_left = jnp.concatenate([jnp.array([0], dtype=jnp.int32), cumsum_right[:-1]])
    # total_compact = cumsum_right[-1]
    #
    # Na = batch.atom_mask.shape[-1]
    # n_target = Na - binder_slots
    # d = jnp.arange(Na, dtype=jnp.int32)
    #
    # token_for_d = jnp.searchsorted(cumsum_right, d, side='right')
    # token_for_d = jnp.clip(token_for_d, 0, binder_len - 1)
    # local_off = d - cumsum_left[token_for_d]
    # binder_src = 14 * token_for_d + local_off
    # target_src = binder_slots + (d - total_compact)
    #
    # is_binder = d < total_compact
    # is_target = (d >= total_compact) & (d < total_compact + n_target)
    # gather = jnp.where(is_binder, binder_src, jnp.where(is_target, target_src, 0))
    # gather = jnp.clip(gather, 0, Na - 1)
    #
    # def _gather(field):
    #     return field[0][gather][None]
    #
    # c_ref_pos = _gather(padded_ref_pos)
    # c_ref_element = _gather(padded_ref_element)
    # c_ref_charge = _gather(padded_ref_charge)
    # c_ref_mask = _gather(padded_ref_mask)
    # c_atom_name = _gather(padded_atom_name)
    # c_atom_mask = _gather(padded_atom_mask)
    # c_atom_to_token = _gather(batch.atom_to_token_index)
    # c_ref_space_uid = _gather(batch.ref_space_uid)
    #
    # tail_mask = (d < total_compact + n_target).astype(jnp.float32)
    # c_atom_mask = c_atom_mask * tail_mask[None]
    # c_ref_mask = c_ref_mask * tail_mask[None]
    #
    # shift = total_compact - binder_slots
    # c_start = batch.start_atom_index[0].at[:binder_len].set(cumsum_left)
    # c_start = c_start.at[binder_len:].add(shift)
    # c_start = c_start[None]
    #
    # c_napt = batch.num_atoms_per_token[0].at[:binder_len].set(counts)[None]
    #
    # cb_offset = jnp.where(hard_idx == 7, 1, 4).astype(jnp.int32)
    # new_rep_binder = cumsum_left + cb_offset
    # rep_target = batch.representative_atom_index[0, binder_len:] + shift
    # c_rep_idx = jnp.concatenate([new_rep_binder, rep_target])[None]
    #
    # rep_mask_binder = jnp.ones(binder_len, dtype=jnp.float32)
    # c_rep_mask = batch.representative_atom_mask[0].at[:binder_len].set(rep_mask_binder)[None]
    #
    # return eqx.tree_at(
    #     lambda b: (b.restype, b.profile, b.msa,
    #                b.ref_pos, b.ref_element, b.ref_charge,
    #                b.ref_mask, b.ref_atom_name_chars, b.atom_mask,
    #                b.atom_to_token_index, b.ref_space_uid,
    #                b.start_atom_index, b.num_atoms_per_token,
    #                b.representative_atom_index, b.representative_atom_mask),
    #     batch,
    #     (new_restype, new_profile, new_msa,
    #      c_ref_pos, c_ref_element, c_ref_charge,
    #      c_ref_mask, c_atom_name, c_atom_mask,
    #      c_atom_to_token, c_ref_space_uid,
    #      c_start, c_napt,
    #      c_rep_idx, c_rep_mask),
    # )




def _bin_centers(bin_min: float, bin_max: float, num_bins: int) -> np.ndarray:
    width = (bin_max - bin_min) / num_bins
    return np.linspace(bin_min + 0.5 * width, bin_max - 0.5 * width, num_bins)


# Distogram: 64 bins, 2.0–22.0 Å
DISTOGRAM_BINS = _bin_centers(2.0, 22.0, 64)
# PAE: 64 bins, 0.0–32.0 Å
PAE_BINS = _bin_centers(0.0, 32.0, 64)
# pLDDT: 50 bins, 0.0–1.0
PLDDT_BINS = _bin_centers(0.0, 1.0, 50)


# ---------------------------------------------------------------------------
# OF3FromTrunkOutput — lazy AbstractStructureOutput
# ---------------------------------------------------------------------------


@dataclass
class OF3FromTrunkOutput(AbstractStructureOutput):
    """Lazy structure output from a frozen trunk state.

    Expensive computations (diffusion sampling, confidence heads) are
    deferred to first property access via @cached_property, so multiple
    loss terms sharing the same output don't recompute.
    """

    model: OpenFold3
    batch: Batch
    init_emb: InitialEmbedding
    trunk_emb: TrunkEmbedding
    sampling_steps: int
    key: jax.Array

    @property
    def full_sequence(self) -> Float[Array, "N 20"]:
        return self.batch.restype[0].astype(jnp.float32) @ jnp.array(_MOSAIC_TO_OF3).T

    @property
    def asym_id(self) -> Float[Array, "N"]:
        return self.batch.asym_id[0]

    @property
    def residue_idx(self) -> Int[Array, "N"]:
        return self.batch.residue_index[0]

    @property
    def distogram_bins(self) -> Float[Array, "64"]:
        return DISTOGRAM_BINS

    @property
    def pae_bins(self) -> Float[Array, "64"]:
        return PAE_BINS


    @property
    def distogram_logits(self) -> Float[Array, "N N 64"]:
        z = self.trunk_emb.z[:, None, ...]
        return self.model.aux_heads.distogram(z=z)[0, 0]


    @cached_property
    def structure_coordinates(self) -> Array:
        """[B, S, N_atom, 3] diffusion-sampled atom coordinates."""
        return self.model.sample_structures(
            self.init_emb, self.trunk_emb, self.batch,
            self.sampling_steps, num_samples=1,
            key=self.key,
        )

    @cached_property
    def _confidence(self) -> ConfidenceOutput:
        return self.model.confidence_metrics(
            self.init_emb, self.trunk_emb, self.batch,
            self.structure_coordinates, key=self.key,
        )


    @property
    def plddt(self) -> Float[Array, "N"]:
        """Token-level pLDDT normalized to [0, 1]."""
        # plddt_logits: [B, S, N_atom, 50] — select representative atom per token
        logits = self._confidence.plddt_logits[0, 0]  # [N_atom, 50]
        rep_idx = self.batch.representative_atom_index[0]  # [N_token]
        token_logits = logits[rep_idx]  # [N_token, 50]
        probs = jax.nn.softmax(token_logits, axis=-1)
        return (probs * jnp.array(PLDDT_BINS)[None, :]).sum(-1)

    @property
    def pae(self) -> Float[Array, "N N"]:
        logits = self.pae_logits  # [N, N, 64]
        probs = jax.nn.softmax(logits, axis=-1)
        return (probs * jnp.array(PAE_BINS)[None, None, :]).sum(-1)

    @property
    def pae_logits(self) -> Float[Array, "N N 64"]:
        return self._confidence.pae_logits[0, 0]

    @property
    def backbone_coordinates(self) -> Float[Array, "N 4 3"]:
        """N, CA, C, O coordinates per token."""
        start = self.batch.start_atom_index[0].astype(jnp.int32)  # [N_token]
        coords = self.structure_coordinates[0, 0]  # [N_atom, 3]
        # First 4 atoms per standard protein residue are N, CA, C, O
        return jnp.stack([coords[start + i] for i in range(4)], axis=-2)


class MultiSampleOF3Loss(LossTerm):
    """Run trunk once, vmap the diffusion/confidence/loss over multiple samples.

    vmap over samples consumes ``num_samples × memory``. For
    memory-constrained cases, swap ``jax.vmap`` for ``jax.lax.map``.
    """

    model: OpenFold3
    batch: Batch
    loss: LossTerm | LinearCombination
    num_cycles: int = 4
    sampling_steps: int = 20
    num_samples: int = 4
    reduction: any = jnp.mean
    atom_lookup: any = None

    def __call__(self, sequence: Float[Array, "N 20"], key):
        batch = set_binder_sequence(sequence, self.batch)

        init_emb, trunk_emb = self.model.run_trunk(
            batch, self.num_cycles, key=key,
        )

        def single_sample(key):
            output = OF3FromTrunkOutput(
                model=self.model,
                batch=batch,
                init_emb=init_emb,
                trunk_emb=trunk_emb,
                sampling_steps=self.sampling_steps,
                key=key,
            )
            return self.loss(sequence=sequence, output=output, key=key)

        vs, auxs = jax.vmap(single_sample)(jax.random.split(key, self.num_samples))
        sortperm = jnp.argsort(vs)
        return self.reduction(vs), jax.tree.map(lambda v: list(v[sortperm]), auxs)
