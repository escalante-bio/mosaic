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
