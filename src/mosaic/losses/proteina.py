"""Proteina-Complexa inverse-folding sequence-recovery loss.

Direct analog of `InverseFoldingSequenceRecovery` from
`mosaic.losses.protein_mpnn`, powered by jproteina-complexa.

Reads the per-residue atom37 view that mosaic's `StructureModelOutput`
provides (`atom37_coords`, `atom37_mask`) plus the `full_sequence` field —
so the loss is fully model-agnostic. Each call rebuilds a fresh
`TargetCond` from the predicted bound complex, with the binder CA and
target coords both centered on the target's CA COM (proteina's training
convention).

Public API
    `ProteinaInverseFoldingRecovery`  — the LossTerm
    `inverse_fold`                    — the proteina denoise+decode helper
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PyTree

from jproteina_complexa.flow_matching import (
    DenoiseState,
    init_noise,
    PRODUCTION_SAMPLING,
    predict_x1_from_v,
)
from jproteina_complexa.target_features import (
    CHI_ANGLES_MASK as _CHI_ANGLES_MASK_NP,
    CHI_ATOM_INDICES as _CHI_ATOM_INDICES_NP,
)
from jproteina_complexa.types import (
    DecoderBatch,
    DecoderOutput,
    DenoiserBatch,
    NoisyState,
    Timesteps,
    TargetCond,
)

from ..common import LossTerm
from .atom37 import ATOM37_INDEX
from .structure_prediction import StructureModelOutput


ATOM37_N = ATOM37_INDEX["N"]
ATOM37_CA = ATOM37_INDEX["CA"]
ATOM37_C = ATOM37_INDEX["C"]


# ── dihedral / binning helpers (JAX) ────────────────────────────────────────

def _dihedral(
    p0: Float[Array, "... 3"],
    p1: Float[Array, "... 3"],
    p2: Float[Array, "... 3"],
    p3: Float[Array, "... 3"],
) -> Float[Array, "..."]:
    """Signed dihedral between four points; matches jproteina target_features."""
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n1 = jnp.cross(b0, b1)
    n2 = jnp.cross(b1, b2)
    n1 = n1 / (jnp.linalg.norm(n1, axis=-1, keepdims=True) + 1e-8)
    n2 = n2 / (jnp.linalg.norm(n2, axis=-1, keepdims=True) + 1e-8)
    cross = jnp.cross(n1, n2)
    return jnp.arctan2(
        jnp.sign((cross * b1).sum(-1)) * jnp.sqrt((cross ** 2).sum(-1) + 1e-16),
        (n1 * n2).sum(-1),
    )


def _bin_one_hot(
    values: Float[Array, "..."],
    n_bins: int = 20,
) -> Float[Array, "... n_bins_plus_1"]:
    """Binned one-hot of values in `[-π, π]` over `n_bins`+1 cells."""
    bins = jnp.linspace(-jnp.pi, jnp.pi, n_bins)
    return jax.nn.one_hot(jnp.searchsorted(bins, values), n_bins + 1, dtype=jnp.float32)


def _torsion_feat(
    coords: Float[Array, "T 37 3"],
) -> Float[Array, "T 63"]:
    """Backbone (psi, omega, phi) features from atom37 coords."""
    T = coords.shape[0]
    N, CA, C = coords[:, ATOM37_N], coords[:, ATOM37_CA], coords[:, ATOM37_C]
    z1 = jnp.zeros((1,))
    psi = jnp.concatenate([_dihedral(N[:-1], CA[:-1], C[:-1], N[1:]), z1])
    omega = jnp.concatenate([_dihedral(CA[:-1], C[:-1], N[1:], CA[1:]), z1])
    phi = jnp.concatenate([_dihedral(C[:-1], N[1:], CA[1:], C[1:]), z1])
    return _bin_one_hot(jnp.stack([psi, omega, phi], axis=-1)).reshape(T, -1)


def _sidechain_feat(
    coords: Float[Array, "T 37 3"],
    atom_mask: Float[Array, "T 37"],
    residue_types: Int[Array, "T"],
) -> Float[Array, "T 88"]:
    """Chi-angle one-hot features from atom37 coords.

    `residue_types` are mosaic 20-letter indices (matches jproteina AA_CODES).
    Missing-atom positions are masked out via `chi_mask`.
    """
    T = coords.shape[0]
    chi_atom_indices = jnp.asarray(_CHI_ATOM_INDICES_NP, dtype=jnp.int32)  # [20, 4, 4]
    chi_angles_mask = jnp.asarray(_CHI_ANGLES_MASK_NP, dtype=jnp.float32)  # [20, 4]
    chi_atoms = chi_atom_indices[residue_types]
    chi_mask = chi_angles_mask[residue_types]
    rows = jnp.arange(T)[:, None, None]
    p = coords[rows, chi_atoms]
    chi = _dihedral(p[..., 0, :], p[..., 1, :], p[..., 2, :], p[..., 3, :])
    all_present = jnp.all(atom_mask[rows, chi_atoms] > 0.5, axis=-1).astype(jnp.float32)
    final_mask = chi_mask * all_present
    binned = _bin_one_hot(chi) * final_mask[..., None]
    return jnp.concatenate([binned.reshape(T, -1), final_mask], axis=-1)


# ── Proteina inverse fold (verbatim from mosaic/examples/proteina.py) ──────

@eqx.filter_jit
def inverse_fold(
    denoiser: PyTree,
    decoder: PyTree,
    bb_ca: Float[Array, "N 3"],
    mask: Bool[Array, "N"],
    target: TargetCond,
    key: jax.Array,
) -> DecoderOutput:
    """Denoise latents from noise with `bb_ca` fixed, then decode."""
    bb_ca_nm = bb_ca / 10.0
    cfg = PRODUCTION_SAMPLING
    ts_lat = cfg.local_latents.time_schedule(cfg.nsteps)
    mask_f = mask.astype(jnp.float32)

    k_noise, k_run = jax.random.split(key)
    state = init_noise(k_noise, 8, mask, cfg)
    state = DenoiseState(
        bb=bb_ca_nm, lat=state.lat, sc_bb=bb_ca_nm, sc_lat=state.sc_lat, key=k_run,
    )

    def body(carry):
        state, key, i = carry
        t_lat = ts_lat[i]
        dt_lat = ts_lat[i + 1] - t_lat
        out = denoiser(
            DenoiserBatch(
                x_t=NoisyState(bb_ca=bb_ca_nm, local_latents=state.lat),
                t=Timesteps(bb_ca=jnp.array(1.00), local_latents=t_lat),
                mask=mask,
                x_sc=NoisyState(bb_ca=state.sc_bb, local_latents=state.sc_lat),
                target=target,
            )
        )
        sc_lat = predict_x1_from_v(state.lat, out.local_latents, t_lat)
        key, k_step = jax.random.split(key)
        lat = cfg.local_latents.step(
            state.lat, out.local_latents, t_lat, dt_lat, mask_f, k_step,
        )
        return (
            DenoiseState(bb=bb_ca_nm, lat=lat, sc_bb=bb_ca_nm, sc_lat=sc_lat, key=key),
            key, i + 1,
        )

    state, _, _ = jax.lax.fori_loop(
        0, cfg.nsteps, lambda i, carry: body(carry), (state, k_run, jnp.int32(0)),
    )
    return decoder(
        DecoderBatch(z_latent=state.lat, ca_coors=state.bb * 10.0, mask=mask)
    )


# ── Loss term ───────────────────────────────────────────────────────────────

class ProteinaInverseFoldingRecovery(LossTerm):
    """`-mean_n (avg_proteina_sequence · sequence)`.

    Mirror of `InverseFoldingSequenceRecovery`. Slices the target portion
    out of `output.atom37_coords` / `atom37_mask` / `full_sequence` and
    runs proteina inverse-folding on the binder backbone conditioned on
    that target.
    """

    denoiser: PyTree
    decoder: PyTree
    hotspots: tuple[int, ...] | None = None
    random_hotspots: int | None = None
    num_samples: int = 1
    name: str = "proteina_recovery"

    def __check_init__(self):
        if self.hotspots is not None and self.random_hotspots is not None:
            raise ValueError(
                "Specify `hotspots` or `random_hotspots`, not both."
            )

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key: jax.Array,
    ):
        binder_length = sequence.shape[0]

        target_coords = output.atom37_coords[binder_length:]
        target_atom_mask = output.atom37_mask[binder_length:]
        target_residue_types = output.full_sequence[binder_length:].argmax(-1)

        # Center on target CA COM (proteina training convention).
        ca_present = target_atom_mask[:, ATOM37_CA]
        com = (target_coords[:, ATOM37_CA, :] * ca_present[:, None]).sum(0) / jnp.maximum(
            ca_present.sum(), 1.0,
        )
        target_coords = target_coords - com[None, None, :]
        binder_ca = output.backbone_coordinates[:binder_length, 1, :] - com[None, :]

        target_len = target_coords.shape[0]
        hotspot_mask = None
        if self.hotspots is not None:
            hotspot_mask = (
                jnp.zeros(target_len, dtype=bool)
                .at[jnp.asarray(self.hotspots)]
                .set(True)
            )
        elif self.random_hotspots is not None:
            key, k_h = jax.random.split(key)
            target_ca = target_coords[:, ATOM37_CA, :]
            d = jnp.linalg.norm(
                target_ca[:, None, :] - binder_ca[None, :, :], axis=-1
            ).min(-1)
            d = jnp.where(ca_present > 0.5, d, jnp.inf)
            ranks = jnp.argsort(jnp.argsort(d))
            k = jax.random.randint(k_h, (), 0, self.random_hotspots + 1)
            hotspot_mask = ranks < k

        target = TargetCond(
            coords=target_coords,
            atom_mask=target_atom_mask,
            seq=target_residue_types,
            hotspot_mask=hotspot_mask,
            sidechain_feat=_sidechain_feat(target_coords, target_atom_mask, target_residue_types),
            torsion_feat=_torsion_feat(target_coords),
        )

        mask = jnp.ones(binder_length, dtype=bool)

        def single(k):
            inv = inverse_fold(self.denoiser, self.decoder, binder_ca, mask, target, k)
            return jax.nn.one_hot(inv.aatype, 20)

        sequences = jax.vmap(single)(jax.random.split(key, self.num_samples))
        average = jax.lax.stop_gradient(sequences.mean(0))
        ip = (average * sequence).sum(-1).mean()
        return -ip, {"proteina_recovery": ip}
