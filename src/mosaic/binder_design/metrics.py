"""AF2 confidence metrics for binder design.

These reproduce the definitions ColabDesign's binder protocol reports, so that
numbers coming out of the native-Mosaic pipeline can be compared directly with
the ones DdCraft produces (and with the thresholds in existing filter configs).

Two conventions are easy to get wrong and are handled explicitly here:

* PAE is symmetrised and divided by 62 (i.e. ``(pae + pae.T) / 2 / 31``) so it
  lands on ColabDesign's 0-1 scale.
* ``i_ptm`` is scored with exactly two asym groups, binder versus everything
  else.  Mosaic gives every chain its own ``asym_id``, which would wrongly count
  target-target pairs as interface pairs for a multi-chain target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from mosaic.losses import structure_prediction as sp

__all__ = ["ConfidenceMetrics", "normalized_symmetric_pae", "complex_metrics", "monomer_metrics"]


def normalized_symmetric_pae(pae) -> np.ndarray:
    """Symmetrise PAE and rescale to ColabDesign's 0-1 range."""
    pae = np.asarray(pae)
    return (pae + np.swapaxes(pae, -1, -2)) / 62.0


@dataclass
class ConfidenceMetrics:
    plddt: float
    ptm: float
    pae: float
    i_ptm: float | None = None
    i_pae: float | None = None
    ipsae: float | None = None

    def as_dict(self, prefix: str = "") -> dict[str, float | None]:
        out = {
            f"{prefix}pLDDT": self.plddt,
            f"{prefix}pTM": self.ptm,
            f"{prefix}pAE": self.pae,
        }
        if self.i_ptm is not None:
            out[f"{prefix}i_pTM"] = self.i_ptm
        if self.i_pae is not None:
            out[f"{prefix}i_pAE"] = self.i_pae
        if self.ipsae is not None:
            out[f"{prefix}ipSAE"] = self.ipsae
        return out


def _masked_mean(values, mask) -> float:
    mask = np.asarray(mask, dtype=np.float64)
    return float((np.asarray(values) * mask).sum() / (mask.sum() + 1e-8))


def _binder_asym_id(total_length: int, binder_length: int):
    return jnp.concatenate(
        (jnp.zeros(binder_length), jnp.ones(total_length - binder_length))
    ).astype(jnp.int32)


def complex_metrics(
    prediction,
    *,
    binder_length: int,
    binder_mask: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
    ipsae_cutoff: float | None = None,
) -> ConfidenceMetrics:
    """Confidence metrics for a binder/target complex prediction.

    ``binder_mask`` / ``target_mask`` restrict the averages to the residues the
    design actually optimises (DdCraft excludes fixed positions).
    """
    model_output = prediction.model_output
    pae = normalized_symmetric_pae(prediction.pae)
    total_length = pae.shape[0]
    target_length = total_length - binder_length

    if binder_mask is None:
        binder_mask = np.ones(binder_length, dtype=np.float32)
    if target_mask is None:
        target_mask = np.ones(target_length, dtype=np.float32)

    plddt = _masked_mean(np.asarray(prediction.plddt)[:binder_length], binder_mask)

    ptm = float(
        sp.predicted_tm_score(
            logits=model_output.pae_logits,
            bin_centers=model_output.pae_bins,
            pair_mask=jnp.ones(pae.shape, dtype=bool),
        ).max()
    )

    asym_id = _binder_asym_id(total_length, binder_length)
    i_ptm = float(
        sp.predicted_tm_score(
            logits=model_output.pae_logits,
            bin_centers=model_output.pae_bins,
            pair_mask=asym_id[:, None] != asym_id[None, :],
        ).max()
    )

    intra = _masked_mean(
        pae[:binder_length], binder_mask[:, None] * np.ones((binder_length, total_length))
    )
    inter = _masked_mean(
        pae[:binder_length, binder_length:], binder_mask[:, None] * target_mask[None, :]
    )

    ipsae = None
    if ipsae_cutoff is not None:
        ipsae = float(
            jnp.max(
                sp.interaction_prediction_score(
                    logits=model_output.pae_logits,
                    bin_centers=model_output.pae_bins,
                    asym_id=asym_id,
                    pae_cutoff=float(ipsae_cutoff),
                )
            )
        )

    return ConfidenceMetrics(
        plddt=plddt, ptm=ptm, pae=intra, i_ptm=i_ptm, i_pae=inter, ipsae=ipsae
    )


def monomer_metrics(prediction) -> ConfidenceMetrics:
    """Confidence metrics for a binder-alone prediction."""
    model_output = prediction.model_output
    pae = normalized_symmetric_pae(prediction.pae)
    ptm = float(
        sp.predicted_tm_score(
            logits=model_output.pae_logits,
            bin_centers=model_output.pae_bins,
            pair_mask=jnp.ones(pae.shape, dtype=bool),
        ).max()
    )
    return ConfidenceMetrics(
        plddt=float(np.asarray(prediction.plddt).mean()),
        ptm=ptm,
        pae=float(pae.mean()),
    )
