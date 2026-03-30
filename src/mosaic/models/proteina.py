"""Beam search over jproteina-complexa flow matching trajectories."""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Bool, Int, Key
from mosaic.util import fold_in

from jproteina_complexa.flow_matching import (
    DenoiseState,
    denoise_steps,
    init_noise,
    SamplingConfig,
    PRODUCTION_SAMPLING,
)
from jproteina_complexa.types import (
    DecoderBatch,
    DecoderOutput,
    TargetCond,
)
from jproteina_complexa.constants import AA_CODES as _JPC_ORDER
from jproteina_complexa.nn.models import DecoderTransformer, LocalLatentsTransformer
from mosaic.common import LossTerm, TOKENS as _MOSAIC_ORDER

# Permutation: decoder logits (jpc order) → mosaic token order.
_JPC_TO_MOSAIC = jnp.array([_JPC_ORDER.index(aa) for aa in _MOSAIC_ORDER])
_LATENT_DIM = 8


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ScoredDesign(eqx.Module):
    """A decoded + scored design candidate."""
    decoder_output: DecoderOutput
    sequence: Int[Array, "N"]             # hard sequence (mosaic token order)
    loss: Float[Array, ""]               # loss (lower = better)
    aux: dict                             # full auxiliary output from loss_fn
    bb: Float[Array, "N 3"]              # backbone CA (Angstroms)
    lat: Float[Array, "N D"]             # latents



# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _denoise(
    model: LocalLatentsTransformer,
    state: DenoiseState,
    key: Key[Array, ""],
    mask: Bool[Array, "N"],
    cfg: SamplingConfig,
    ts_bb: Float[Array, "S+1"],
    ts_lat: Float[Array, "S+1"],
    start_step: Int[Array, ""],
    end_step: Int[Array, ""],
    target: TargetCond,
) -> DenoiseState:
    """Denoise candidates ``[C, N, ...]`` from *start_step* to *end_step*.

    Vmaps the shared :func:`denoise_steps` over the candidate axis.
    """
    def _one(state, key):
        s = DenoiseState(bb=state.bb, lat=state.lat,
                         sc_bb=state.sc_bb, sc_lat=state.sc_lat, key=key)
        s = denoise_steps(model, s, mask, cfg, ts_bb, ts_lat,
                          start_step, end_step, target)
        return s
    keys = jax.random.split(key, state.bb.shape[0])
    return jax.vmap(_one)(state, keys)


@eqx.filter_jit
def _score_batch(
    decoder: DecoderTransformer,
    loss_fn: LossTerm,
    bbs: Float[Array, "C N 3"],
    lats: Float[Array, "C N D"],
    mask: Bool[Array, "N"],
    keys: Key[Array, "C 2"],
) -> ScoredDesign:
    """Decode + score candidates.  Returns batched ``ScoredDesign [C, ...]``."""
    def _one(bb, lat, key):
        out = decoder(DecoderBatch(z_latent=lat, ca_coors=bb * 10.0, mask=mask))
        seq = out.seq_logits[..., _JPC_TO_MOSAIC].argmax(-1)
        seq_hard = jax.nn.one_hot(seq, 20)
        loss, aux = loss_fn(seq_hard, key=key)
        return ScoredDesign(decoder_output=out, sequence=seq, loss=loss, aux=aux, bb=bb * 10.0, lat=lat)
    return jax.vmap(_one)(bbs, lats, keys)


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


def beam_search(
    model: LocalLatentsTransformer,
    decoder: DecoderTransformer,
    loss_fn: LossTerm,
    mask: Bool[Array, "N"],
    key: Key[Array, ""],
    target: TargetCond,
    *,
    cfg: SamplingConfig | None = None,
    step_checkpoints: list[int],
    beam_width: int = 4,
    n_branch: int = 4,
) -> list[ScoredDesign]:
    """Beam search over flow matching denoising trajectories.

    Args:
        model: denoiser (LocalLatentsTransformer).
        decoder: DecoderTransformer.
        loss_fn: mosaic LossTerm --
                 ``(Float[N, 20], *, key) -> (scalar, dict)``.
        mask: ``[n_residues]`` boolean residue mask.
        key: PRNG key.
        target: TargetCond for conditional generation.
        cfg: SamplingConfig (defaults to PRODUCTION_SAMPLING).
        step_checkpoints: e.g. ``[0, 100, 200, 300, 400]``.
        beam_width: beams kept after each pruning.
        n_branch: branches explored per beam per checkpoint.

    Returns:
        List of :class:`ScoredDesign` batches, one ``[W*Br]`` per interval.
    """
    cfg = cfg or PRODUCTION_SAMPLING
    nsteps = step_checkpoints[-1]
    (n_residues,) = mask.shape
    bw, br = beam_width, n_branch

    ts_bb  = cfg.bb_ca.time_schedule(nsteps)
    ts_lat = cfg.local_latents.time_schedule(nsteps)

    # ── initialise [W] beams ──────────────────────────────────────────
    key, k_noise = jax.random.split(key)
    states = jax.vmap(lambda k: init_noise(k, _LATENT_DIM, mask, cfg))(
        jax.random.split(k_noise, bw)
    )

    # ── main search loop ──────────────────────────────────────────────
    n_intervals = len(step_checkpoints) - 1
    designs: list[ScoredDesign] = []
    nsteps_jax = jnp.int32(nsteps)

    for iv in range(n_intervals):
        key = fold_in(key, "iter")
        start, end = step_checkpoints[iv], step_checkpoints[iv + 1]

        # Branch: [W] -> [W*Br]
        cands = jax.tree.map(lambda x: jnp.repeat(x, br, axis=0), states)

        # partial denoise each branched candidate
        cands = _denoise(model, cands, fold_in(key, "denoise"), mask, cfg, ts_bb, ts_lat,
                         jnp.int32(start), jnp.int32(end), target)

        # look-ahead (denoise all the way)
        la = _denoise(model, cands, fold_in(key, "rollout"), mask, cfg, ts_bb, ts_lat,
                      jnp.int32(end), nsteps_jax, target)

        # score using mosaic loss
        scored = _score_batch(
            decoder, loss_fn, la.bb, la.lat, mask,
            jax.random.split(fold_in(key, "score"), bw * br),
        )

        # unpack batched ScoredDesign into individual items
        designs.extend(
            jax.tree.map(lambda x: np.array(x[i]), scored) for i in range(bw * br)
        )

        # Top-k selection
        _, top_idx = jax.lax.top_k(-scored.loss, bw)
        states = jax.tree.map(lambda a: a[top_idx], cands)

    return designs
