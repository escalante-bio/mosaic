"""Shared interface for trunk + diffusion structure models.

Each model implements `structure_prediction.DiffusionStructurePredictionModel`. The
recycling loop, decoding, `model_output`, `predict` and the design loss are
defined once, below, as functions of the model.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from mosaic.common import LinearCombination, LossTerm
from mosaic.losses.structure_prediction import (
    PAE_BINS,
    IPTMLoss,
    StructureModelOutput,
    reduce_samples,
)
from mosaic.sampling import Sampler, sample
from mosaic.structure_prediction import (
    Confidence,
    DiffusionStructurePredictionModel,
    StructurePrediction,
    TrunkState,
    Writer,
)

# ---------------------------------------------------------------------------
# Shared logic
# ---------------------------------------------------------------------------


def _pass_keys(key: PRNGKeyArray, recycling_steps: int):
    k_init, k_passes = jax.random.split(key)
    return k_init, jax.random.split(k_passes, recycling_steps)


def run_trunk(
    model: DiffusionStructurePredictionModel,
    features,
    inputs,
    *,
    recycling_steps: int,
    key: PRNGKeyArray,
) -> TrunkState:
    """Run `recycling_steps` trunk passes. Gradients flow through every pass if
    `model.backprop_all_passes`, otherwise through the last only."""
    if recycling_steps < 1:
        raise ValueError(f"recycling_steps must be at least 1, got {recycling_steps}")
    k_init, keys = _pass_keys(key, recycling_steps)
    state = model.init_state(features, inputs, key=k_init)

    def one_pass(state, k):
        return model.step(features, inputs, state, key=k), None

    if model.backprop_all_passes:
        return jax.lax.scan(one_pass, state, keys)[0]
    if recycling_steps > 1:
        state = jax.lax.scan(one_pass, state, keys[:-1])[0]
    return model.step(features, inputs, jax.lax.stop_gradient(state), key=keys[-1])


_BACKBONE_ATOM37 = (0, 1, 2, 4)  # N, CA, C, O


def decode(
    model: DiffusionStructurePredictionModel,
    features,
    inputs,
    state: TrunkState,
    *,
    sampler: Sampler,
    key: PRNGKeyArray,
) -> StructureModelOutput:
    """Sample one structure from a trunk state and evaluate the heads."""
    denoise, atom_mask = model.structure_denoiser(features, inputs, state)
    coords = sample(denoise, atom_mask, sampler, key=key)
    tokens = model.tokens(features)
    distogram = model.distogram(features, inputs, state)
    confidence = model.confidence(features, inputs, state, coords)
    if confidence is None:  # no confidence head
        n = tokens.sequence.shape[0]
        confidence = Confidence(
            plddt=jnp.full((n,), jnp.nan),
            pae_logits=jnp.full((n, n, len(PAE_BINS)), jnp.nan),
            pae_bins=jnp.asarray(PAE_BINS, dtype=jnp.float32),
        )
    atoms = model.atom37(features, coords)
    pae = jnp.sum(jax.nn.softmax(confidence.pae_logits, axis=-1) * confidence.pae_bins, axis=-1)
    return StructureModelOutput(
        distogram_logits=distogram.logits,
        distogram_bins=distogram.bins,
        plddt=confidence.plddt,
        pae=pae,
        pae_logits=confidence.pae_logits,
        pae_bins=confidence.pae_bins,
        structure_coordinates=coords,
        backbone_coordinates=atoms.coords[:, _BACKBONE_ATOM37],
        full_sequence=tokens.sequence,
        asym_id=tokens.asym_id,
        residue_idx=tokens.residue_index,
        atom37_coords=atoms.coords,
        atom37_mask=atoms.mask,
    )


def prepare(model: DiffusionStructurePredictionModel, features, sequence, *, key: PRNGKeyArray):
    """Set the binder sequence (if given) and embed."""
    k_sequence, k_embed = jax.random.split(key)
    if sequence is not None:
        features = model.set_sequence(features, sequence, key=k_sequence)
    return features, model.embed(features, key=k_embed)


@eqx.filter_jit
def model_output(
    model: DiffusionStructurePredictionModel,
    features,
    *,
    sequence: Float[Array, "L 20"] | None = None,
    recycling_steps: int | None = None,
    sampler: Sampler | None = None,
    key: PRNGKeyArray,
) -> StructureModelOutput:
    """Predict one structure, with design settings by default."""
    if recycling_steps is None:
        recycling_steps = model.design_recycling_steps
    sampler = model.sampler() if sampler is None else sampler
    k_prepare, k_trunk, k_sample = jax.random.split(key, 3)
    features, inputs = prepare(model, features, sequence, key=k_prepare)
    state = run_trunk(model, features, inputs, recycling_steps=recycling_steps, key=k_trunk)
    return decode(model, features, inputs, state, sampler=sampler, key=k_sample)


def predict(
    model: DiffusionStructurePredictionModel,
    features,
    writer: Writer,
    *,
    sequence: Float[Array, "L 20"] | None = None,
    recycling_steps: int | None = None,
    sampler: Sampler | None = None,
    key: PRNGKeyArray,
) -> StructurePrediction:
    """Predict one structure, with inference settings by default."""
    output = model_output(
        model,
        features,
        sequence=sequence,
        recycling_steps=(
            model.inference_recycling_steps if recycling_steps is None else recycling_steps
        ),
        sampler=model.sampler() if sampler is None else sampler,
        key=key,
    )
    seq = sequence if sequence is not None else jnp.zeros((0, 20))
    iptm = -IPTMLoss()(seq, output, key=jax.random.key(0))[0]
    return StructurePrediction(
        st=model.to_structure(features, writer, output, sequence),
        plddt=output.plddt,
        pae=output.pae,
        iptm=iptm,
        model_output=output,
    )


class MultiSampleLoss(LossTerm):
    """Run the trunk once and reduce `loss` over `num_samples` diffusion samples."""

    model: DiffusionStructurePredictionModel
    features: Any
    loss: LossTerm | LinearCombination
    sampler: Sampler
    recycling_steps: int = eqx.field(static=True)
    num_samples: int = eqx.field(static=True, default=1)
    reduction: Callable = eqx.field(static=True, default=jnp.mean)

    def __call__(self, sequence: Float[Array, "L 20"], *, key: PRNGKeyArray):
        k_prepare, k_trunk, k_samples = jax.random.split(key, 3)
        features, inputs = prepare(self.model, self.features, sequence, key=k_prepare)
        state = run_trunk(
            self.model, features, inputs, recycling_steps=self.recycling_steps, key=k_trunk
        )

        def one_sample(k):
            k_decode, k_loss = jax.random.split(k)
            output = decode(self.model, features, inputs, state, sampler=self.sampler, key=k_decode)
            return self.loss(sequence, output, key=k_loss)

        values, aux = jax.vmap(one_sample)(jax.random.split(k_samples, self.num_samples))
        return reduce_samples(values, aux, self.reduction, self.num_samples)


def build_loss(
    model: DiffusionStructurePredictionModel,
    *,
    loss: LossTerm | LinearCombination,
    features,
    recycling_steps: int | None = None,
    num_samples: int = 1,
    reduction: Callable = jnp.mean,
    sampler: Sampler | None = None,
) -> MultiSampleLoss:
    """A `MultiSampleLoss` with the model's design settings by default."""
    return MultiSampleLoss(
        model=model,
        features=features,
        loss=loss,
        sampler=model.sampler() if sampler is None else sampler,
        recycling_steps=(
            model.design_recycling_steps if recycling_steps is None else recycling_steps
        ),
        num_samples=num_samples,
        reduction=reduction,
    )
