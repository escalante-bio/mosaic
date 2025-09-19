import os
import json
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import time
import json
from pathlib import Path
import numpy as np
from loguru import logger

from transformers import AutoTokenizer, AutoModelForCausalLM

from mosaic.common import LinearCombination
from mosaic.losses.clean import CleanCosineSimilarityLoss, CleanHead, load_clean_head_from_torch
from mosaic.losses.boltz2 import load_boltz2, load_features_and_structure_writer, Boltz2Loss
from mosaic.losses.structure_prediction import PLDDTLoss
from mosaic_workflows.design import run_workflow
from mosaic_workflows.rl import rl_trl_adapter
from mosaic_workflows.rl.trainers import trl_grpo_trainer_ctor


def _build_boltz2_loss(*, binder_len: int, smiles: str) -> Boltz2Loss | None:
    yaml_str = f"""
version: 1
sequences:
  - protein:
      id: [A]
      sequence: {"X" * binder_len}
  - ligand:
      id: [L]
      smiles: '{smiles}'
"""
    try:
        import os as _os
        from pathlib import Path as _Path
        cache_dir = _Path(_os.environ.get("BOLTZ_CACHE", "/root/.boltz")).expanduser()
        features, _writer = load_features_and_structure_writer(yaml_str, cache=cache_dir)
        joltz2 = load_boltz2()
        conf = PLDDTLoss()
        loss = LinearCombination(1.0 * conf)
        return Boltz2Loss(joltz2=joltz2, features=features, loss=loss, name="boltz2")
    except Exception:
        return None


def _make_clean_loss(ec_label: str) -> CleanCosineSimilarityLoss | None:
    try:
        import torch
        # Load CLEAN head weights (LayerNormNet equivalence) trained on split100
        ckpt_path = "/workspace/_external/tiny-clean-test/CLEAN/app/data/pretrained/split100.pth"
        head: CleanHead = load_clean_head_from_torch(ckpt_path)

        # Load cluster center embeddings for EC labels (post-head 128-d vectors)
        centers_path = "/workspace/_external/tiny-clean-test/CLEAN/app/data/pretrained/100.pt"
        centers = torch.load(centers_path, map_location="cpu")
        # centers may be a dict ec->tensor or a tuple; handle dict case
        if isinstance(centers, dict):
            if ec_label not in centers:
                return None
            tgt = centers[ec_label].detach().cpu().numpy()
        else:
            # Unsupported format; skip CLEAN
            return None

        # ESM2 650M produces 1280-d token embeddings; mean-pooled embedding
        from transformers import AutoTokenizer, AutoModel
        esm_name = "facebook/esm2_t33_650M_UR50D"
        _tok = AutoTokenizer.from_pretrained(esm_name, do_lower_case=False)
        _esm = AutoModel.from_pretrained(esm_name)
        _esm.eval()

        def embed_fn(seq: str):
            with torch.no_grad():
                toks = _tok(seq, return_tensors="pt", add_special_tokens=False)
                out = _esm(**toks)
                h = out.last_hidden_state  # [1, L, 1280]
                v = h.mean(dim=1).squeeze(0)
                return v.detach().cpu().numpy()

        return CleanCosineSimilarityLoss(
            clean_head=head,
            target_embedding=tgt,
            embed_fn=embed_fn,
            name="clean",
            differentiable=False,
        )
    except Exception:
        return None


def build_reward_loss(*, binder_len: int, smiles: str, ec_label: str):
    boltz_conf = _build_boltz2_loss(binder_len=binder_len, smiles=smiles)
    clean_loss = _make_clean_loss(ec_label)

    def loss_fn(probs, *, key):
        total = 0.0
        aux_all = {}
        if boltz_conf is not None:
            v_conf, aux_conf = boltz_conf(probs, key=key)
            total = total + v_conf
            aux_all.update(aux_conf)
        # Naturalness removed per request
        if clean_loss is not None:
            v_clean, aux_clean = clean_loss(probs, key=key)
            total = total + 0.2 * v_clean
            aux_all.update(aux_clean)
        return total, aux_all

    return loss_fn


def init_zymctrl():
    tok = AutoTokenizer.from_pretrained("AI4PD/ZymCTRL")
    model = AutoModelForCausalLM.from_pretrained("AI4PD/ZymCTRL")
    return model, tok


def main():
    binder_len = 80
    ec_prompt = "3.1.1.102"
    ligand_smiles = "C1=CC(=CC=C1C(=O)O)C(=O)OCCO"

    # Configure output directory and logging
    run_id = f"rl_zymectrl_{int(time.time())}_len{binder_len}"
    out_dir = Path("/results") / run_id
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fallback to local if /results unavailable
        out_dir = Path.cwd() / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(out_dir / "run.log", level="INFO", enqueue=True, backtrace=True, diagnose=True)
    logger.info("Starting ZymCTRL RL run")

    def optimizer(**kw):
        return rl_trl_adapter(
            **kw,
            init_model_and_tokenizer=init_zymctrl,
            trainer_ctor=trl_grpo_trainer_ctor,
            prompt=ec_prompt,
            binder_len=binder_len,
            num_return=8,
            gen_cfg={"top_k": 8, "temperature": 0.9, "max_length": binder_len + 16},
            trainer_args={"logging_steps": 10, "num_train_epochs": 1},
        )

    def build_loss():
        return build_reward_loss(binder_len=binder_len, smiles=ligand_smiles, ec_label=ec_prompt)

    phase = {
        "name": "rl_zymectrl_mhetase",
        "build_loss": build_loss,
        "optimizer": optimizer,
        "steps": 10,
        "schedule": lambda g, p: {},
    }

    wf = {"phases": [phase], "binder_len": binder_len, "seed": 0}
    logger.info("Running workflow…")
    res = run_workflow(wf)

    # Persist best sequence and logits if present
    best_seq = res.get("best_sequence", "") or ""
    (out_dir / "best_sequence.txt").write_text(str(best_seq))
    if res.get("best_x") is not None:
        try:
            np.save(out_dir / "best_x.npy", res.get("best_x"))
        except Exception:
            pass

    # Write trajectory as JSONL with selected metrics and improvements
    traj = res.get("trajectory", []) or []
    best_val = None
    with open(out_dir / "trajectory.jsonl", "w") as f:
        for rec in traj:
            step_i = int(rec.get("step", 0))
            metrics_i = rec.get("metrics", {}) or {}
            # If metrics weren’t surfaced at top-level, check aux.metrics
            if not metrics_i:
                aux_i_try = rec.get("aux", {}) or {}
                if isinstance(aux_i_try, dict) and isinstance(aux_i_try.get("metrics"), dict):
                    metrics_i = aux_i_try.get("metrics", {}) or {}
            loss_val = rec.get("aux", {}).get("loss") if isinstance(rec.get("aux"), dict) else None
            row = {
                "step": step_i,
                "loss": float(loss_val) if loss_val is not None else None,
                "metrics": {},
            }
            # Surface common keys
            for k in ("value_x", "value_y", "gap", "seq_x", "ent_x", "ent_y"):
                if k in metrics_i:
                    try:
                        row["metrics"][k] = float(metrics_i[k]) if hasattr(metrics_i[k], "__float__") else metrics_i[k]
                    except Exception:
                        row["metrics"][k] = metrics_i[k]
            f.write(json.dumps(row) + "\n")

            # Log improvements
            vx = metrics_i.get("value_x")
            if vx is not None:
                try:
                    vx = float(vx)
                    if (best_val is None) or (vx < best_val):
                        logger.info("Improved value_x at step {}: {} -> {}", step_i, best_val, vx)
                        best_val = vx
                except Exception:
                    pass

    logger.info("Completed. Results at {}", str(out_dir))
    print(best_seq)


if __name__ == "__main__":
    main()


