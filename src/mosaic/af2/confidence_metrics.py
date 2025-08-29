### Confidence metrics on GPU
# These are translated from alphafold.common.confidence
import jax
import jax.numpy as jnp
import numpy as onp


def plddt(logits: jnp.ndarray) -> jnp.ndarray:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = onp.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
    probs = jax.nn.softmax(logits, -1)  # scipy.special.softmax(logits, axis=-1)
    predicted_lddt_ca = jnp.sum(probs * bin_centers[None, :], axis=-1)
    return predicted_lddt_ca * 100


def predicted_aligned_error(
    logits: jnp.ndarray, breaks: jnp.ndarray
) -> dict[str, jnp.ndarray]:
    aligned_confidence_probs = jax.nn.softmax(logits, axis=-1)
    return _calculate_expected_aligned_error(
        alignment_confidence_breaks=breaks,
        aligned_distance_error_probs=aligned_confidence_probs,
    )


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: jnp.ndarray, aligned_distance_error_probs: jnp.ndarray
) -> jnp.ndarray:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)

    return jnp.sum(aligned_distance_error_probs * bin_centers, axis=-1)


def _calculate_bin_centers(breaks: jnp.ndarray):
    step = breaks[1] - breaks[0]

    # Add half-step to get the center
    bin_centers = breaks + step / 2
    # Add a catch-all bin at the end.
    return jnp.append(bin_centers, bin_centers[-2:-1] + step, axis=0)


def interaction_prediction_score(
    logits: jnp.ndarray,
    bin_centers: jnp.ndarray,
    asym_id: jnp.ndarray | None = None,
    interface: bool = False,
    pae_cutoff: float = 15.0,
) -> jnp.ndarray:

    probs = jax.nn.softmax(logits, axis=-1)
    pae = jnp.sum(probs * bin_centers, axis=-1)

    pair_mask = jnp.ones_like(pae, dtype=bool)
    if interface:
        pair_mask *= asym_id[:, None] != asym_id[None, :]

    # only include residue pairs below the pae_cutoff
    pair_mask *= (pae < pae_cutoff)
    n_residues = jnp.sum(pair_mask, axis=-1, keepdims=True)

    # Compute adjusted d_0(num_res) per residue  as defined by eqn. (15) in 
    # Dunbrack, R., "What's wrong with AlphaFold’s ipTM score and how to fix it."
    # 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC11844409/
    d0 = 1.24 * (jnp.clip(n_residues, min=27) - 15) ** (1.0 / 3) - 1.8

    tm_per_bin = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0))
    predicted_tm_term = jnp.sum(probs * tm_per_bin, axis=-1)

    normed_residue_mask = pair_mask / (
        1e-8 + n_residues
    )
    per_alignment = jnp.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return per_alignment[per_alignment.argmax()]


def predicted_tm_score(
    logits: jnp.ndarray,
    bin_centers: jnp.ndarray,
    asym_id: jnp.ndarray | None = None,
    interface: bool = False,
) -> jnp.ndarray:

    num_res = logits.shape[0]
    # Clip num_res to avoid negative/undefined d0.
    clipped_num_res = max(num_res, 19)

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    # Convert logits to probs.
    probs = jax.nn.softmax(logits, axis=-1)

    # TM-Score term for every bin.
    tm_per_bin = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0))
    # E_distances tm(distance).
    predicted_tm_term = jnp.sum(probs * tm_per_bin, axis=-1)

    pair_mask = jnp.ones(shape=(num_res, num_res), dtype=bool)
    if interface:
        pair_mask *= asym_id[:, None] != asym_id[None, :]

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask
    normed_residue_mask = pair_residue_weights / (
        1e-8 + jnp.sum(pair_residue_weights, axis=-1, keepdims=True)
    )
    per_alignment = jnp.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return per_alignment[per_alignment.argmax()]


def confidence_metrics(prediction_result: dict[str, any]):
    return {
        "plddt": plddt(prediction_result["predicted_lddt"]["logits"]),
        "predicted_aligned_error": jnp.squeeze(
            predicted_aligned_error(
                logits=prediction_result["predicted_aligned_error"]["logits"],
                breaks=prediction_result["predicted_aligned_error"]["breaks"],
            )
        ),
        "iptm": predicted_tm_score(
            logits=prediction_result["predicted_aligned_error"]["logits"],
            breaks=prediction_result["predicted_aligned_error"]["breaks"],
            asym_id=prediction_result["predicted_aligned_error"]["asym_id"],
            interface=True,
        ),
    }
