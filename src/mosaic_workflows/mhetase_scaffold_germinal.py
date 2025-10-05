from __future__ import annotations

from typing import Any, Iterable, Sequence, cast

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.common import LinearCombination, LossTerm
from mosaic.models.af2 import AlphaFold2 as AF2Model
from mosaic.structure_prediction import TargetChain
from mosaic.losses.structure_prediction import (
    PLDDTLoss,
    WithinBinderContact,
    DistogramRadiusOfGyration,
)
from .mhetase_scaffold import (
    MotifDistogramCCE,
    MotifRMSDCA,
    AF2SidechainRMSD,
)
from .transforms import (
    temperature_on_logits,
    e_soft_on_logits,
    gradient_normalizer,
    record_probs_max_mean,
    germinal_softmax_convergence,
    record_logits_in_ctx,
    record_probs_in_ctx,
    zymctrl_pcgrad_merge,
    per_position_allowed_tokens,
    per_position_allowed_probs,
    position_mask,
)
from .optimizers import jacobian_descent_adapter, jd_pcgrad_aggregator, semi_greedy_adapter
from .analyzers import log_inline
from functools import partial


def _as_tuple(x: Iterable[int] | Sequence[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    if x is None:
        return ()
    return tuple(int(i) for i in x)


def make_workflow(
    *,
    binder_len: int,
    # Optional motif specification
    motif_positions: tuple[int, ...] | None = None,
    motif_template_ca: np.ndarray | None = None,
    # Supervision / enforcement
    supervised_positions: tuple[int, ...] | None = None,
    fix_supervised_identities: str | None = None,
    freeze_supervised_positions: bool = False,
    motif_roles: tuple[str, ...] | None = None,
    # Optional PDB inputs (used by caller shim to build motif_template_ca)
    motif_pdb_path: str | None = None,
    motif_chain_id: str | None = None,
    motif_resnums: tuple[int, ...] | None = None,
    # AF2 params
    af2_params_dir: str | None = None,
    af2_num_recycles: int = 3,
    # Loss weights
    w_plddt: float = 0.5,
    w_intra_con: float = 0.1,
    w_rg: float = 0.0,
    w_motif_lddt: float = 0.2,
    w_motif_orient: float = 0.0,
    w_sc_rmsd: float = 0.1,
    # Phases
    steps_logits: int = 65,
    steps_softmax: int = 35,
    steps_semigreedy: int = 10,
    e_soft_logits: float = 1.0,
    e_soft_softmax: float = 0.8,
    temp_init: float = 1.0,
    # Optimizer
    lr: float = 0.1,
    grad_merge_method: str = "pcgrad",
    guidance_every_k: int = 1,
    # ZymCTRL
    ec_label: str = "3.5.5.1",
):
    binder_len = int(binder_len)
    motif_positions = _as_tuple(motif_positions)

    # AF2 single-chain features (binder only)
    model = AF2Model(data_dir=af2_params_dir or ".")
    feats, _ = model.binder_features(int(binder_len), chains=[TargetChain(sequence="", use_msa=False)])

    # Structural terms
    terms: list[LossTerm] = [
        float(w_plddt) * PLDDTLoss(),
        float(w_intra_con) * WithinBinderContact(
            max_contact_distance=14.0, min_sequence_separation=9, num_contacts_per_residue=2
        ),
    ]
    if float(w_rg) != 0.0:
        terms.append(float(w_rg) * DistogramRadiusOfGyration())

    # Motif geometry shaping (optional)
    if motif_template_ca is not None and len(motif_positions) > 0:
        mca = jnp.asarray(motif_template_ca)
        # Use scaffold-style motif losses: Distogram CCE and motif RMSD
        if float(w_motif_lddt) != 0.0:
            terms.append(
                float(w_motif_lddt)
                * MotifDistogramCCE(
                    motif_positions=tuple(motif_positions), motif_template_ca=cast(Any, mca)
                )
            )
        if float(w_motif_orient) != 0.0:
            terms.append(
                float(w_motif_orient)
                * MotifRMSDCA(
                    motif_positions=tuple(motif_positions), motif_template_ca=cast(Any, mca)
                )
            )
        if float(w_sc_rmsd) != 0.0:
            # AF2 side-chain RMSD over motif positions; returns 0.0 if templates are unavailable
            terms.append(
                float(w_sc_rmsd) * AF2SidechainRMSD(positions=tuple(motif_positions))
            )

    # Combine
    assert len(terms) > 0
    combined: LossTerm | LinearCombination = terms[0]
    for t in terms[1:]:
        combined = combined + t
    # Freeze AF2 model id across phases/iters for kernel/cache reuse
    _rng = np.random.default_rng(int(0))
    _model_idx_fixed = int(_rng.integers(0, 5))

    # Build per-phase losses with configurable recycling steps
    def _build_loss_with_recycles(recycles: int):
        return model.build_loss(
            loss=combined,
            features=feats,
            recycling_steps=int(recycles),
            model_idx_fixed=_model_idx_fixed,
        )

    # Transforms
    pre_logits = [record_logits_in_ctx(), temperature_on_logits(), e_soft_on_logits()]
    pre_probs: list[Any] = [record_probs_in_ctx()]
    grad_chain_warm = [gradient_normalizer(mode="l2_effL")]
    grad_chain_late = [gradient_normalizer(mode="l2_effL")]
    post_logits: list[Any] = []

    # Enforce identities at supervised positions if provided
    if fix_supervised_identities and supervised_positions is not None and len(supervised_positions) > 0:
        vocab = "ARNDCQEGHILKMFPSTWYV"
        allowed = np.ones((int(binder_len), 20), dtype=np.float32)
        ids = [s.strip().upper() for s in str(fix_supervised_identities).split(',') if s.strip()]
        for i, sup_pos in enumerate(tuple(int(x) for x in supervised_positions)):
            if i < len(ids) and ids[i] in vocab and 0 <= int(sup_pos) < int(binder_len):
                allowed[int(sup_pos), :] = 0.0
                allowed[int(sup_pos), vocab.index(ids[i])] = 1.0
        post_logits.append(per_position_allowed_tokens(allowed))
        pre_probs.append(per_position_allowed_probs(allowed))

    # Optionally freeze gradients at supervised positions after warmup
    if bool(freeze_supervised_positions) and supervised_positions is not None and len(supervised_positions) > 0:
        mask = np.ones(int(binder_len), dtype=np.float32)
        for p in supervised_positions:
            if 0 <= int(p) < int(binder_len):
                mask[int(p)] = 0.0
        grad_chain_late = [position_mask(mask)] + grad_chain_late

    # Record softmax convergence metric (no early-stop gating)
    conv_mask = np.ones((int(binder_len),), dtype=np.float32)
    pre_probs.append(record_probs_max_mean(mask=conv_mask, key="probs_max_mean"))
    # Softmax-phase convergence mask excluding supervised (often fixed) positions
    softmax_conv_mask = np.ones((int(binder_len),), dtype=np.float32)
    if supervised_positions is not None and len(supervised_positions) > 0:
        for p in supervised_positions:
            if 0 <= int(p) < int(binder_len):
                softmax_conv_mask[int(p)] = 0.0

    def phase(name: str, n_steps: int, temperature: float, e_soft: float, *, use_semi_greedy: bool = False, zym_scale: float | tuple[float, float] = 0.0, late: bool = False, add_conv: bool = False, recycles: int = 1):
        return {
            "name": name,
            # Build AF2 loss lazily per phase with chosen recycling steps
            "build_loss": (lambda recycles=recycles: _build_loss_with_recycles(int(recycles))),
            "optimizer": (semi_greedy_adapter if use_semi_greedy else partial(jacobian_descent_adapter, aggregator=jd_pcgrad_aggregator)),
            "steps": int(n_steps),
            "schedule": (
                lambda g, p: {
                    "lr": float(lr),
                    "stepsize": 0.1 * float(jnp.sqrt(jnp.maximum(1, binder_len))),
                    "scale": 1.0,
                    "temperature": float(temperature),
                    "e_soft": float(e_soft),
                    # Weight for ZymCTRL guidance (also mirror under iglm_scale for compatibility)
                    "zymctrl_scale": (
                        (float(zym_scale[0]) + (float(zym_scale[1]) - float(zym_scale[0])) * (float(p) / max(1.0, float(n_steps - 1))))
                        if isinstance(zym_scale, tuple)
                        else float(zym_scale)
                    ),
                    "iglm_scale": (
                        (float(zym_scale[0]) + (float(zym_scale[1]) - float(zym_scale[0])) * (float(p) / max(1.0, float(n_steps - 1))))
                        if isinstance(zym_scale, tuple)
                        else float(zym_scale)
                    ),
                    # Apply guidance transforms every k steps (default 1)
                    "guidance_every_k": int(guidance_every_k),
                    "grad_merge_method": str(grad_merge_method),
                    "min_stop_step": 5,
                    "proposals_per_step": (int(np.ceil(0.05 * binder_len)) if use_semi_greedy else 5),
                }
            ),
            "transforms": {
                "pre_logits": list(pre_logits),
                "pre_probs": (
                    list(pre_probs)
                    + ([germinal_softmax_convergence(mask=softmax_conv_mask, threshold=0.10, key="probs_max_mean")] if bool(add_conv) else [])
                ),
                "grad": (
                    (list(grad_chain_late) if late else list(grad_chain_warm))
                    + ([zymctrl_pcgrad_merge(ec_label=ec_label)] if not use_semi_greedy else [])
                ),
                "post_logits": list(post_logits),
            },
            "analyzers": [log_inline],
            "analyze_every": 1,
        }

    phases = [

        {**phase("logits", steps_logits, temp_init, e_soft_logits, use_semi_greedy=False, zym_scale=(0.2, 0.4), late=False, add_conv=False, recycles=3)},
        {**phase("softmax", steps_softmax, temp_init, e_soft_softmax, use_semi_greedy=False, zym_scale=0.4, late=bool(freeze_supervised_positions), add_conv=True, recycles=3)},
        {**phase("semi_greedy", steps_semigreedy, temp_init, 1.0, use_semi_greedy=True, zym_scale=1.0, late=bool(freeze_supervised_positions), add_conv=False, recycles=int(af2_num_recycles))},
    ]

    return {"phases": phases, "binder_len": binder_len, "seed": 0}