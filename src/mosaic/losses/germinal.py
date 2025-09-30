import jax
import jax.numpy as jnp
import numpy as np

from ..common import LossTerm
from .structure_prediction import (
    AbstractStructureOutput,
    contact_log_probability,
    contact_cross_entropy,
)


class CDRHelixSuppression(LossTerm):
    """Penalize helix-like geometry within selected CDR positions using distogram.

    Implements a binary/categorical cross-entropy penalty for i,i+3 pairs whose
    predicted distances fall inside the helix window [2.0, 6.2] Å. Restrict to
    provided binder indices (CDR positions). Lower is better.
    """

    cdr_positions: tuple[int, ...]
    cutoff_lower: float = 2.0
    cutoff_upper: float = 6.2

    def __call__(self, *args, key, **kwds):
        sequence, output = args[:2]
        binder_len = int(sequence.shape[0])
        dgram = output.distogram_logits[:binder_len, :binder_len]
        bins = output.distogram_bins

        # Build mask for helix window: penalize mass inside [cutoff_lower, cutoff_upper]
        bins_mask = jnp.logical_or(bins > float(self.cutoff_upper), bins < float(self.cutoff_lower))
        px = jax.nn.softmax(dgram, axis=-1)
        px_cut = jax.nn.softmax(dgram - 1e7 * (1.0 - bins_mask[None, None, :]), axis=-1)
        # Cross entropy like in Germinal
        con_loss_cat_ent = -(px_cut * jax.nn.log_softmax(dgram, axis=-1)).sum(-1)
        con_loss_bin_ent = -jnp.log((bins_mask[None, None, :] * px + 1e-8).sum(-1))
        x = con_loss_bin_ent  # binary version

        # Restrict to CDR positions on diagonal offset +3
        cdr_mask_1d = jnp.zeros((binder_len,), dtype=jnp.float32).at[jnp.asarray(self.cdr_positions, dtype=jnp.int32)].set(1.0)
        cdr_mask_2d = jnp.outer(cdr_mask_1d, cdr_mask_1d)
        val = jnp.diagonal(x * cdr_mask_2d, 3)
        denom = jnp.maximum(jnp.diagonal(cdr_mask_2d, 3).sum(), 1.0)
        loss = jnp.sum(val) / denom
        return loss, {"cdr_helix_suppress": loss}


class CDRBetaSuppression(LossTerm):
    """Penalize beta-strand–like geometry within/near CDRs using distogram.

    Follows Germinal: penalize i,i+3 distances in a beta window [9.75, 11.5] Å,
    and average the top-K diagonal(+3) penalties where K ~= |cdr_positions|/3.
    """

    cdr_positions: tuple[int, ...]
    cutoff_lower: float = 9.75
    cutoff_upper: float = 11.5

    def __call__(self, *args, key, **kwds):
        sequence, output = args[:2]
        binder_len = int(sequence.shape[0])
        dgram = output.distogram_logits[:binder_len, :binder_len]
        bins = output.distogram_bins

        bins_mask = jnp.logical_or(bins > float(self.cutoff_upper), bins < float(self.cutoff_lower))
        px = jax.nn.softmax(dgram, axis=-1)
        px_cut = jax.nn.softmax(dgram - 1e7 * (1.0 - bins_mask[None, None, :]), axis=-1)
        con_loss_cat_ent = -(px_cut * jax.nn.log_softmax(dgram, axis=-1)).sum(-1)
        con_loss_bin_ent = -jnp.log((bins_mask[None, None, :] * px + 1e-8).sum(-1))
        x = con_loss_bin_ent

        # Include neighbors just outside CDRs (i-1,i+1) like Germinal
        cdr_mask_1d = jnp.zeros((binder_len,), dtype=jnp.float32).at[jnp.asarray(self.cdr_positions, dtype=jnp.int32)].set(1.0)
        # expand by 1 residue on each side
        cdr_mask_1d = jnp.clip(cdr_mask_1d + jnp.roll(cdr_mask_1d, 1) + jnp.roll(cdr_mask_1d, -1), 0.0, 1.0)
        cdr_mask_2d = jnp.outer(cdr_mask_1d, cdr_mask_1d)
        diag_vals = jnp.diagonal(x * cdr_mask_2d, 3)

        k = max(1, len(self.cdr_positions) // 3)
        topk = jax.lax.top_k(diag_vals, k)[0]
        loss = jnp.mean(topk)
        return loss, {"cdr_beta_suppress": loss}


class FrameworkContactProbability(LossTerm):
    """Penalty for framework-target contacts within a distance threshold.

    Computes mean contact probability (under distogram) between framework binder positions
    and target residues. Minimizing discourages framework-driven interfaces.
    """

    framework_positions: tuple[int, ...]
    contact_distance: float = 6.0
    epitope_idx: tuple[int, ...] | None = None
    offset: float = 0.0

    def __call__(self, *args, key, **kwds):
        sequence, output = args[:2]
        binder_len = int(sequence.shape[0])
        # inter-chain binder (rows) x target (cols)
        logits_bt = output.distogram_logits[:binder_len, binder_len:]
        bins = output.distogram_bins
        if self.epitope_idx is not None and len(self.epitope_idx) > 0:
            logits_bt = logits_bt[:, jnp.asarray(self.epitope_idx, dtype=jnp.int32)]

        # log P(D < d0)
        logp = contact_log_probability(logits_bt, self.contact_distance, bins)
        p = jnp.exp(logp)
        # select framework rows
        fw_rows = jnp.asarray(self.framework_positions, dtype=jnp.int32)
        fw_p = p[fw_rows]
        loss = fw_p.mean() - float(self.offset)
        return loss, {"framework_contact_p": loss}


class EpitopeCDRContactCCE(LossTerm):
    """Encourage CDR-epitope contacts via distogram cross-entropy under a contact threshold.

    Wraps contact_cross_entropy on inter-chain blocks, restricted to epitope columns and CDR rows.
    """

    cdr_positions: tuple[int, ...]
    epitope_idx: tuple[int, ...]
    contact_distance: float = 6.0
    top_k: int = 3

    def __call__(self, *args, key, **kwds):
        sequence, output = args[:2]
        binder_len = int(sequence.shape[0])
        logits_bt = output.distogram_logits[:binder_len, binder_len:]
        ce = contact_cross_entropy(logits_bt, self.contact_distance, output.distogram_bins)
        # restrict cols to epitope and rows to CDR
        ce = ce[jnp.asarray(self.cdr_positions, dtype=jnp.int32)[:, None], jnp.asarray(self.epitope_idx, dtype=jnp.int32)[None, :]]
        # take top-k over epitope per CDR row, then average
        topk = jax.vmap(lambda v: jax.lax.top_k(v, min(self.top_k, v.shape[0]))[0])(ce)
        avg = topk.mean()
        # return negative average log-prob as loss (consistent with CE sign)
        loss = -avg
        return loss, {"cdr_epitope_ce": -loss}


