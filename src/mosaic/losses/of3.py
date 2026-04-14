"""Mosaic loss integration for OpenFold3.

Provides:
- Token vocabulary mapping between mosaic's 20-AA and OF3's 32-dim restype
- Binder sequence injection into OF3 Batch
- of3_forward_from_trunk: eager forward returning StructureModelOutput
- MultiSampleOF3Loss: trunk-once, vmap-over-samples loss term
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from mosaic.common import LinearCombination, LossTerm
from mosaic.losses.structure_prediction import PAE_BINS, StructureModelOutput

import equinox as eqx
from jopenfold3.batch import Batch
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
# pLDDT: 50 bins, 0.0–1.0
PLDDT_BINS = _bin_centers(0.0, 1.0, 50)


# ---------------------------------------------------------------------------
# of3_forward_from_trunk — eager forward returning StructureModelOutput
# ---------------------------------------------------------------------------


def of3_forward_from_trunk(
    model: OpenFold3,
    batch: Batch,
    init_emb: InitialEmbedding,
    trunk_emb: TrunkEmbedding,
    sampling_steps: int,
    key: jax.Array,
) -> StructureModelOutput:
    """Run distogram, structure, and confidence from pre-computed trunk state."""
    z = trunk_emb.z[:, None, ...]
    distogram_logits = model.aux_heads.distogram(z=z)[0, 0]

    structure_coordinates = model.sample_structures(
        init_emb, trunk_emb, batch,
        sampling_steps, num_samples=1,
        key=key,
    )

    confidence = model.confidence_metrics(
        init_emb, trunk_emb, batch,
        structure_coordinates, key=key,
    )

    # pLDDT normalized to [0, 1]
    logits = confidence.plddt_logits[0, 0]  # [N_atom, 50]
    rep_idx = batch.representative_atom_index[0]  # [N_token]
    token_logits = logits[rep_idx]  # [N_token, 50]
    probs = jax.nn.softmax(token_logits, axis=-1)
    plddt = (probs * jnp.array(PLDDT_BINS)[None, :]).sum(-1)

    # PAE
    pae_logits = confidence.pae_logits[0, 0]
    pae_probs = jax.nn.softmax(pae_logits, axis=-1)
    pae = (pae_probs * jnp.array(PAE_BINS)[None, None, :]).sum(-1)

    # Backbone coordinates (N, CA, C, O)
    start = batch.start_atom_index[0].astype(jnp.int32)
    coords = structure_coordinates[0, 0]  # [N_atom, 3]
    backbone_coordinates = jnp.stack([coords[start + i] for i in range(4)], axis=-2)

    return StructureModelOutput(
        distogram_logits=distogram_logits,
        distogram_bins=DISTOGRAM_BINS,
        plddt=plddt,
        pae=pae,
        pae_logits=pae_logits,
        pae_bins=PAE_BINS,
        structure_coordinates=structure_coordinates,
        backbone_coordinates=backbone_coordinates,
        full_sequence=batch.restype[0].astype(jnp.float32) @ jnp.array(_MOSAIC_TO_OF3).T,
        asym_id=batch.asym_id[0],
        residue_idx=batch.residue_index[0],
    )


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

    def __call__(self, sequence: Float[Array, "N 20"], key):
        batch = set_binder_sequence(sequence, self.batch)

        init_emb, trunk_emb = self.model.run_trunk(
            batch, self.num_cycles, key=key,
        )

        def single_sample(key):
            output = of3_forward_from_trunk(
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
