from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
from jax import config as jax_config
import numpy as np


from mosaic.common import LinearCombination, LossTerm
from mosaic.models.af2 import AlphaFold2 as AF2Model
from mosaic.structure_prediction import TargetChain
from mosaic.losses.structure_prediction import (
    PLDDTLoss,
    IPTMLoss,
    BinderTargetPAE,
    WithinBinderPAE,
    WithinBinderContact,
    BinderTargetContact,
    DistogramRadiusOfGyration,
)
from mosaic.losses.germinal import (
    CDRHelixSuppression,
    CDRBetaSuppression,
    FrameworkContactProbability,
    EpitopeCDRContactCCE,
)

from .transforms import (
    temperature_on_logits,
    e_soft_on_logits,
    gradient_normalizer,
    per_position_allowed_tokens,
    per_position_allowed_probs,
    germinal_softmax_convergence,
    framework_sequence_bias_on_logits,
    record_logits_in_ctx,
    record_probs_in_ctx,
)
from .analyzers import log_inline, jsonl_stream
from .iglm_jax import IgLMLoss
from .optimizers import jacobian_descent_adapter, jd_pcgrad_aggregator, semi_greedy_adapter
from functools import partial


def _as_tuple(x: Iterable[int] | Sequence[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    if x is None:
        return ()
    return tuple(int(i) for i in x)


def _read_chain_sequence(pdb_path: str | Path, chain_id: str) -> str:
    import importlib
    gemmi = importlib.import_module("gemmi")
    st = gemmi.read_structure(str(pdb_path))
    ch = st[0][str(chain_id)]
    return gemmi.one_letter_code([r.name for r in ch])


def make_workflow(
    *,
    binder_len: int,
    # Target spec
    target_pdb_path: str | Path,
    target_chain_id: str,
    target_hotspots: tuple[int, ...] | None = None,
    # Antibody masking
    cdr_positions: tuple[int, ...],
    framework_positions: tuple[int, ...],
    framework_sequence: str,
    # AF2 params
    af2_params_dir: str | None = None,
    af2_num_recycles: int = 3,
    # Loss weights
    w_plddt: float = 0.5,
    w_iptm: float = 1.0,
    w_pae_bt: float = 1.0,
    w_intra_con: float = 0.1,
    w_inter_con: float = 0.0,
    w_rg: float = 0.0,
    w_pae_intra: float = 0.0,
    # Paratope/geometry
    w_dgram_cce: float = 0.01,
    w_fw_penalty: float = 0.5,
    w_cdr_helix_suppress: float = 0.2,
    w_cdr_beta_suppress: float = 0.2,
    # Schedules / phases
    steps_logits: int = 15,
    steps_softmax: int = 35,
    steps_semigreedy: int = 10,
    e_soft_logits: float = 1.0,
    e_soft_softmax: float = 0.8,
    temp_init: float = 1.0,
    # Framework bias (soft preservation of FR)
    framework_bias: float = 10.0,
    framework_contact_offset: float = 1.0,
    # Optimizer
    optimizer=None,
    lr: float = 0.1,
    plddt_thr: float = 0.82,
    iptm_thr: float = 0.68,
    ipae_thr: float = 0.27,
    seq_entropy_thr: float = 0.10,
    grad_merge_method: str = "pcgrad",
    omit_aas: str = "C",
    seq_init_mode: str = "gumbel",
    # IgLM guidance cadence: 1=every step; >1 runs every k steps; 0 disables
    iglm_every: int = 1,
    # AF2 recycling per phase (logits/softmax/semi_greedy)
    logits_recycles: int = 1,
    softmax_recycles: int = 2,
    semi_recycles: int = 0,
):
    """Build a 3-phase Germinal workflow (logits → softmax → semi-greedy) on AF2.

    Notes:
      - Uses AF2 binder+target features for interface-aware losses.
      - IgLM guidance can be added later as an additional LossTerm in the LC.
    """

    binder_len = int(binder_len)
    cdr_positions = _as_tuple(cdr_positions)
    framework_positions = _as_tuple(framework_positions)
    epitope_idx = _as_tuple(target_hotspots)

    # AF2 features with binder + target chain
    model = AF2Model(data_dir=af2_params_dir or ".")
    target_seq = _read_chain_sequence(target_pdb_path, target_chain_id)
    feats, _ = model.binder_features(
        int(binder_len),
        chains=[TargetChain(sequence=target_seq, use_msa=False)],
    )

    # Core structural terms
    terms: list[LossTerm] = [
        w_plddt * PLDDTLoss(),
        w_iptm * IPTMLoss(),
        w_pae_bt * BinderTargetPAE(),
        w_intra_con * WithinBinderContact(
            max_contact_distance=14.0, min_sequence_separation=9, num_contacts_per_residue=2
        ),
    ]
    if float(w_rg) != 0.0:
        terms.append(w_rg * DistogramRadiusOfGyration())
    if float(w_pae_intra) != 0.0:
        terms.append(w_pae_intra * WithinBinderPAE())

    # Paratope + geometry
    if len(cdr_positions) > 0:
        if epitope_idx:
            # Germinal-style epitope contact encouragement via distogram CCE (top-k per CDR)
            terms.append(
                w_dgram_cce
                * EpitopeCDRContactCCE(
                    cdr_positions=tuple(cdr_positions),
                    epitope_idx=tuple(epitope_idx),
                    contact_distance=6.0,
                    top_k=3,
                )
            )
        terms.append(
            w_cdr_helix_suppress
            * CDRHelixSuppression(cdr_positions=tuple(cdr_positions))
        )
        terms.append(
            w_cdr_beta_suppress
            * CDRBetaSuppression(cdr_positions=tuple(cdr_positions))
        )
    if len(framework_positions) > 0:
        terms.append(
            w_fw_penalty
            * FrameworkContactProbability(
                framework_positions=tuple(framework_positions),
                contact_distance=6.0,
                epitope_idx=tuple(epitope_idx) if epitope_idx else None,
                offset=float(framework_contact_offset),
            )
        )
    # Inter-chain contact encouragement analogous to Germinal i_con
    if float(w_inter_con) != 0.0:
        terms.append(w_inter_con * BinderTargetContact(contact_distance=6.0))

    # Optional IgLM guidance as a separate task (merged by JD aggregator)
    iglm_idx: int | None = None
    if int(iglm_every) > 0:
        iglm_idx = len(terms)
        terms.append(IgLMLoss(chain_token="[HEAVY]", species="[HUMAN]", temp=0.6))

    # Combine scaled terms
    assert len(terms) > 0
    combined: LossTerm | LinearCombination = terms[0]
    for t in terms[1:]:
        combined = combined + t
    # Build loss per phase to allow varying recycling steps without recompilation churn
    def _build_loss_with_recycles(n_recycles: int):
        return model.build_loss(loss=combined, features=feats, recycling_steps=int(n_recycles))

    # Transforms
    pre_logits = [record_logits_in_ctx(), temperature_on_logits(), e_soft_on_logits()]
    pre_probs: list[Any] = [record_probs_in_ctx()]
    grad_chain_soft = [gradient_normalizer(mode="l2_effL")]
    post_logits = []

    # Convergence metric mask (apply only in softmax phase)
    if len(cdr_positions) > 0:
        conv_mask = jnp.zeros((binder_len,), dtype=jnp.float32)
        conv_mask = conv_mask.at[jnp.asarray(cdr_positions, dtype=jnp.int32)].set(1.0)
    else:
        conv_mask = jnp.ones((binder_len,), dtype=jnp.float32)

    # Numpy view for type-checked transforms
    conv_mask_np = np.asarray(conv_mask)
    _use_conv_mask = bool(conv_mask_np.sum() > 0.0)

    # Omit amino acids globally (e.g., omit_AAs: "C")
    if isinstance(omit_aas, str) and len(omit_aas) > 0:
        vocab = "ARNDCQEGHILKMFPSTWYV"
        allowed = jnp.ones((binder_len, 20), dtype=jnp.float32)
        for ch in omit_aas:
            if ch in vocab:
                allowed = allowed.at[:, vocab.index(ch)].set(0.0)
        allowed_np: np.ndarray = np.asarray(allowed)
        pre_probs.insert(0, per_position_allowed_probs(allowed_np))

    # Framework bias on logits (soft preservation)
    if framework_sequence and len(framework_positions) > 0 and float(framework_bias) != 0.0:
        post_logits.append(
            framework_sequence_bias_on_logits(
                fr_positions=list(framework_positions), framework_sequence=str(framework_sequence), bias=float(framework_bias)
            )
        )

    # Phase builders
    # Minimal per-step analyzer using Loguru (if available). Logs only safe host values.
    def _log_step_analyzer(aux: dict) -> dict:
        try:
            from loguru import logger  # type: ignore
        except Exception:
            logger = None  # type: ignore
        # Prefer fields that are already Python types to avoid device sync
        msg_parts: list[str] = []
        if isinstance(aux, dict):
            # loss as Python float if present (many optimizers already cast to float)
            val = aux.get("loss")
            if isinstance(val, (int, float)):
                msg_parts.append(f"loss={val:.4f}")
            # phase name if provided by runner
            ph = aux.get("phase")
            if isinstance(ph, str):
                msg_parts.append(f"phase={ph}")
            # simple metrics that are plain floats
            mets = aux.get("metrics") if isinstance(aux.get("metrics"), dict) else None
            if isinstance(mets, dict):
                for k, v in mets.items():
                    if isinstance(v, (int, float)):
                        msg_parts.append(f"{k}={float(v):.4f}")
        text = " ".join(msg_parts) if msg_parts else "step"
        if logger is not None:
            logger.info(text)
        else:
            print(text)
        return {}

    def phase(name: str, n_steps: int, temperature: float, e_soft: float, *, use_semi_greedy: bool = False, iglm_scale: float | tuple[float, float] = 0.0, add_conv: bool = False, recycles: int = 1):
        return {
            "name": name,
            "build_loss": (lambda recycles=recycles: _build_loss_with_recycles(int(recycles))),
            "optimizer": (
                semi_greedy_adapter if use_semi_greedy else (optimizer or partial(jacobian_descent_adapter, aggregator=jd_pcgrad_aggregator))
            ),
            "steps": int(n_steps),
            "schedule": (
                lambda g, p: {
                    "lr": float(lr),
                    # Host-side math to avoid device sync in schedules
                    "stepsize": 0.1 * __import__("math").sqrt(max(1, int(binder_len))),
                    "scale": 1.0,
                    "temperature": float(temperature),
                    "e_soft": float(e_soft),
                    # Task scales: set IgLM weight dynamically if present; others at 1.0
                    **({
                        "task_scales": (
                            ([
                                *([1.0] * int(iglm_idx)),
                                (
                                    (float(iglm_scale[0]) + (float(iglm_scale[1]) - float(iglm_scale[0])) * (float(p) / max(1.0, float(n_steps - 1))))
                                    if isinstance(iglm_scale, tuple) else float(iglm_scale)
                                ),
                            ] if (iglm_idx is not None) else None)
                        )
                    }),
                    "grad_merge_method": str(grad_merge_method),
                    # Provide step indices and IgLM cadence to transforms
                    "phase_step": int(p),
                    "global_step": int(g),
                    "min_stop_step": 5,
                    # Semigreedy: default tries per step ~ ceil(0.05 * L)
                    "proposals_per_step": (int(np.ceil(0.05 * binder_len)) if use_semi_greedy else 5),
                }
            ),
            "transforms": {
                "pre_logits": list(pre_logits),
                "pre_probs": (list(pre_probs) + [germinal_softmax_convergence(mask=(conv_mask_np if _use_conv_mask else None), threshold=float(seq_entropy_thr), key="probs_max_mean_cdr")] if bool(add_conv) else list(pre_probs)),
                "grad": list(grad_chain_soft),
                "post_logits": list(post_logits),
            },
            "analyzers": [log_inline, jsonl_stream],
            # Lower analyzer cadence to reduce host sync/log overhead
            "analyze_every": 1,
        }

    # Analyzer to surface AF metrics from aux and pass-through recorded metrics
    def _metrics_analyzer(aux: dict) -> dict:
        m: dict[str, float] = {}
        if isinstance(aux, dict):
            # passthrough metrics recorded by transforms
            if isinstance(aux.get("metrics"), dict):
                for k, v in (aux.get("metrics") or {}).items():
                    if isinstance(v, (int, float)):
                        m[k] = float(v)
            # scan nested aux for structural metrics
            def _scan(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k in ("plddt", "i_ptm", "i_pae") and isinstance(v, (int, float)):
                            m[k] = float(v)
                        else:
                            _scan(v)
                elif isinstance(obj, list):
                    for it in obj:
                        _scan(it)
            _scan(aux)
        return m

    # Determine IgLM scaling schedule based on cadence toggle
    _iglm_enabled = int(iglm_every) > 0
    logits_scale = (0.2, 0.4) if _iglm_enabled else 0.0
    softmax_scale = 0.4 if _iglm_enabled else 0.0
    semi_scale = 1.0 if _iglm_enabled else 0.0

    phases = [
        # When enabled, anneal IgLM weight from 0.2 -> 0.4 during logits
        {**phase("logits", steps_logits, temp_init, e_soft_logits, use_semi_greedy=False, iglm_scale=logits_scale, add_conv=False, recycles=int(logits_recycles))},
        {**phase("softmax", steps_softmax, temp_init, e_soft_softmax, use_semi_greedy=False, iglm_scale=softmax_scale, add_conv=True, recycles=int(softmax_recycles))},
        {**phase("semi_greedy", steps_semigreedy, temp_init, 1.0, use_semi_greedy=True, iglm_scale=semi_scale, recycles=int(semi_recycles))},
    ]

    # Phase-gating callback to mirror Germinal thresholds
    def _gate_cb(ev: dict):
        if not isinstance(ev, dict) or ev.get("event") != "end_phase":
            return
        name = ev.get("phase")
        traj = ev.get("trajectory") or []
        if not traj:
            return
        last = traj[-1]
        metrics = last.get("metrics") or {}
        def _fail_logits():
            # For logits → require pLDDT and iPTM above thresholds
            plddt = float(metrics.get("plddt", -1.0))
            iptm = float(metrics.get("i_ptm", -1.0))
            return not (plddt > float(plddt_thr) and iptm > float(iptm_thr))
        def _fail_softmax():
            plddt = float(metrics.get("plddt", -1.0))
            iptm = float(metrics.get("i_ptm", -1.0))
            ipae = float(metrics.get("i_pae", 1e9))
            seqc = float(metrics.get("probs_max_mean_cdr", 0.0))
            return not (plddt > float(plddt_thr) and iptm > float(iptm_thr) and ipae < float(ipae_thr) and seqc >= float(seq_entropy_thr))

        if name == "logits" and _fail_logits():
            # Zero steps of remaining phases
            for ph in phases:
                if ph.get("name") != "logits":
                    ph["steps"] = 0
        if name == "softmax" and _fail_softmax():
            for ph in phases:
                if ph.get("name") == "semi_greedy":
                    ph["steps"] = 0

    # Initializer: seq_init_mode ["gumbel"|"soft"]
    def _init_logits():
        rng = np.random.default_rng(0)
        logits = np.zeros((binder_len, 20), dtype=np.float32)
        vocab = "ARNDCQEGHILKMFPSTWYV"
        if seq_init_mode == "gumbel":
            # bias framework identities
            if framework_sequence and len(framework_positions) > 0:
                for i in framework_positions:
                    aa = framework_sequence[int(i)] if int(i) < len(framework_sequence) else None
                    if aa in vocab:
                        logits[int(i), vocab.index(aa)] = 5.0
            # Gumbel noise
            U = np.clip(rng.random((binder_len, 20)), 1e-6, 1-1e-6)
            gumbel = -np.log(-np.log(U))
            logits = logits + gumbel.astype(np.float32)
        else:
            # soft: small gaussian
            logits = rng.normal(0.0, 0.1, size=(binder_len, 20)).astype(np.float32)
        return logits

    return {"phases": phases, "binder_len": binder_len, "seed": 0, "callbacks": [_gate_cb], "initial_x": _init_logits()}


