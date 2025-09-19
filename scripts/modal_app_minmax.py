import os
import sys
from pathlib import Path

import modal


image = (
    modal.Image.debian_slim(python_version="3.12.0")
    .apt_install("git", "aria2")
    .env({"BOLTZ_CACHE": "/root/.boltz", "JAX_PLATFORMS": "cuda"})
    .run_commands(
        "python -m pip install -U pip setuptools wheel && "
        # CUDA PyTorch (provides CUDA libs)
        "python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1 && "
        # JAX + CUDA plugin
        "python -m pip install --upgrade jax==0.7.1 && "
        "python -m pip install --upgrade jax-cuda12-plugin==0.7.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && "
        # Equinox for binder_games
        "python -m pip install equinox==0.13.0 && "
        # PTX toolchain for JAX (provides ptxas/nvlink)
        "python -m pip install nvidia-cuda-nvcc-cu12==12.8.93 && "
        # Extras
        "python -m pip install optax==0.2.4 dm-haiku>=0.0.13 flax>=0.10.2 ml-collections>=1.0.0 httpx>=0.28.1 gemmi>=0.6.0 matplotlib>=3.10.0 && "
        # Git-only deps
        "python -m pip install git+https://github.com/escalante-bio/jablang.git && "
        "python -m pip install git+https://github.com/escalante-bio/esmj.git && "
        "python -m pip install git+https://github.com/adaptyvbio/joltz.git && "
        # Boltz models and tooling
        "python -m pip install git+https://github.com/jwohlwend/boltz.git && "
        # Bake repo
        "git clone --depth 1 https://github.com/adaptyvbio/mosaic_workflows.git /repo"
    )
)


app = modal.App("adaptyv-binder-minmax", image=image)

boltz_cache = modal.Volume.from_name("boltz-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("results-binder-minmax", create_if_missing=True)
af2_cache = modal.Volume.from_name("alphafold-cache", create_if_missing=True)
local_src_mount = modal.Mount.from_local_dir("/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/src", remote_path="/workspace/src")


@app.function(gpu="H100", timeout=3 * 60 * 60, volumes={"/root/.boltz": boltz_cache, "/results": results_vol, "/repo/params": af2_cache}, mounts=[local_src_mount], secrets=[modal.Secret.from_name("github-token")])
def run_binder_minmax(*, binder_len: int = 60, steps: int = 600, helix_weight: float = -0.3, seed: int = 42, game: str = "minmax"):
    os.environ.setdefault("BOLTZ_CACHE", "/root/.boltz")
    Path(os.environ["BOLTZ_CACHE"]).mkdir(parents=True, exist_ok=True)

    # Prefer mounted local source (latest edits), fall back to baked repo
    local_src = Path("/workspace/src")
    if local_src.exists():
        sys.path.insert(0, str(local_src))
    repo_src = Path("/repo/src")
    if repo_src.exists():
        sys.path.append(str(repo_src))

    import time
    import json
    import numpy as np
    import jax
    import jax.numpy as jnp
    import mosaic.losses.structure_prediction as sp
    from mosaic.losses.af2 import AlphaFoldLoss, AF2Output
    from mosaic.af2.alphafold2 import AF2
    import gemmi
    from mosaic.common import LinearCombination
    from mosaic_workflows import run_workflow
    from mosaic_workflows.transforms import temperature_on_logits, e_soft_on_logits, gradient_normalizer, zero_disallowed
    from binder_games import build_minmax_phase, build_stackelberg_phase, make_minmax_loss
    from binder_games.analyzers import (
        saddle_gap_estimate,
        decode_sequences_xy,
        value_components,
        probs_entropy_xy,
        kl_divergence_xy,
        sequence_hamming_xy,
        per_position_entropy_xy,
        composition_charge_hydropathy_xy,
        grad_norms_xy,
    )

    # Build AF2 with templates (PDL1) using provided PDB
    # Ensure AF2 params are present (cached in alphafold-cache volume mounted at /repo/params)
    params_dir = Path("/repo") / "params"
    key_file = params_dir / "params_model_1.npz"
    if not key_file.exists():
        import subprocess
        script = Path("/repo") / "download_params.sh"
        if script.exists():
            subprocess.run(["bash", str(script), "/repo"], check=True)
    af2 = AF2(num_recycle=1, data_dir="/repo")
    pdb_path = Path("/workspace/src/PDL1_stable_region.pdb")
    if not pdb_path.exists():
        pdb_path = Path("/repo/src/PDL1_stable_region.pdb")
    st = gemmi.read_pdb(str(pdb_path))
    st.setup_entities()
    target_chain = st[0]["A"]
    # Binder unknown sequence placeholders; AF2 will replace binder positions during loss call
    binder_placeholder = "X" * int(binder_len)
    # Always derive target sequence from the PDL1 template PDB
    target_seq = "".join([gemmi.one_letter_code([r.name]) for r in target_chain])
    features, _ = af2.build_features(
        chains=[binder_placeholder, target_seq],
        template_chains={1: target_chain},
        initial_guess=None,
    )

    def build_loss():
        # Local wrapper to avoid traced random model index inside jitted loop
        class FixedModelAF2Loss(AlphaFoldLoss):
            def __call__(self, soft_sequence, *, key):
                output = self.predict(soft_sequence, key=key, model_idx=0)
                v, aux = self.losses(
                    soft_sequence,
                    AF2Output(
                        features=self.features,
                        output=output,
                    ),
                    key=key,
                )
                return v, {self.name: aux, f"{self.name}/model_idx": 0, f"{self.name}/loss": v}

        # BindCraft-like composite objective on AF2 output
        structure_loss: LinearCombination = (
            1.0 * sp.BinderTargetContact(contact_distance=21.0)
            + 1.0 * sp.WithinBinderContact(max_contact_distance=14.0, num_contacts_per_residue=4, min_sequence_separation=8)
            + 0.05 * sp.TargetBinderPAE()
            + 0.05 * sp.BinderTargetPAE()
            + 0.025 * sp.IPTMLoss()
            + 0.4 * sp.WithinBinderPAE()
            + 0.025 * sp.pTMEnergy()
            + 0.1 * sp.PLDDTLoss()
            + 0.0 * sp.PLDDTPerResidueReport()
        )
        loss_x = FixedModelAF2Loss(
            forward=af2.jitted_apply,
            stacked_params=af2.stacked_model_params,
            features=features,
            losses=structure_loss,
            name="af2_x",
        )
        loss_y = FixedModelAF2Loss(
            forward=af2.jitted_apply,
            stacked_params=af2.stacked_model_params,
            features=features,
            losses=structure_loss,
            name="af2_y",
        )
        # Ranking hinge with anti-collusion (symmetric KL) to prevent collapse
        margin = 0.1
        lambda_hinge = 0.5
        gamma_anti_collusion = 0.05

        def loss_fn(x_probs, y_probs, key=None):
            vx, auxx = loss_x(x_probs, key=key)
            vy, auxy = loss_y(y_probs, key=key)
            # base competitive term (y minimizes its Ly due to ascent on -Ly)
            base = vx - vy
            # hinge to enforce Lx <= Ly - margin
            h = jnp.maximum(0.0, margin + vx - vy)
            # symmetric KL as similarity; subtract to discourage collapse
            eps = 1e-6
            x = jnp.clip(x_probs, eps, 1.0)
            y = jnp.clip(y_probs, eps, 1.0)
            kxy = jnp.sum(x * (jnp.log(x) - jnp.log(y)))
            kyx = jnp.sum(y * (jnp.log(y) - jnp.log(x)))
            sym_kl = kxy + kyx
            v = base + lambda_hinge * h - gamma_anti_collusion * sym_kl
            aux = {
                "value_x": jnp.asarray(vx),
                "value_y": jnp.asarray(vy),
                "x": auxx,
                "y": auxy,
                "ranking_hinge": h,
                "sym_kl_xy": sym_kl,
            }
            return v, aux

        return loss_fn

    game = str(game).lower()
    if game == "stackelberg":
        phase = build_stackelberg_phase(
            name="stackelberg_af2",
            build_loss=build_loss,
            steps=int(steps),
            schedule=lambda g, p: {
                "lr_x": 0.05,
                "lr_y": 0.05,
                "br_steps": 5,
                "reinit_y_each_step": False,
                "temperature": 1.0,
                "y_init": "random",
            },
            transforms={
                "x": {
                    "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
                    "grad": [gradient_normalizer(mode="per_chain", log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"])],
                },
                "y": {
                    "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
                    "grad": [gradient_normalizer(mode="per_chain", log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"])],
                },
            },
            analyzers=[
                saddle_gap_estimate(),
                value_components(),
                decode_sequences_xy(),
                probs_entropy_xy(),
                kl_divergence_xy(),
                sequence_hamming_xy(),
                per_position_entropy_xy(),
                composition_charge_hydropathy_xy(),
                grad_norms_xy(),
            ],
            analyze_every=1,
        )
    else:
        phase = build_minmax_phase(
            name="minmax_af2",
            build_loss=build_loss,
            steps=int(steps),
            schedule=lambda g, p: {"lr_x": 0.05, "lr_y": 0.05, "temperature": 1.0, "y_init": "random"},
            transforms={
                "x": {
                    "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
                    "grad": [gradient_normalizer(mode="per_chain", log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"])],
                },
                "y": {
                    "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
                    "grad": [gradient_normalizer(mode="per_chain", log_norm=True), zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"])],
                },
            },
            analyzers=[
                saddle_gap_estimate(),
                value_components(),
                decode_sequences_xy(),
                probs_entropy_xy(),
                kl_divergence_xy(),
                sequence_hamming_xy(),
                per_position_entropy_xy(),
                composition_charge_hydropathy_xy(),
                grad_norms_xy(),
            ],
            analyze_every=1,
        )

    # Start both players from noise: x from Gaussian logits, y via schedule y_init="random"
    x0 = np.random.randn(binder_len, 20).astype(np.float32) * 0.1
    wf = {"phases": [phase], "binder_len": binder_len, "seed": int(seed), "initial_x": x0}
    out = run_workflow(wf)

    run_id = f"binder_minmax_{int(time.time())}_seed{seed}_len{binder_len}"
    out_dir = Path("/results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "best_sequence.txt").write_text(str(out.get("best_sequence", "")))
    if out.get("best_x") is not None:
        np.save(out_dir / "best_x.npy", out.get("best_x"))

    traj = out.get("trajectory", []) or []
    last = None
    for rec in reversed(traj):
        if rec.get("metrics"):
            last = rec["metrics"]
            break
    if last:
        keys = [k for k in ("gap", "value_x", "value_y", "seq_x", "seq_y", "ent_x", "ent_y", "kl_x_to_y", "kl_y_to_x", "hamming_xy", "identity_xy") if k in last]
        with open(out_dir / "last_metrics.json", "w") as fh:
            json.dump({k: last[k] for k in keys}, fh)

    print({"results_dir": str(out_dir)})

    # Persist full trajectory (trimmed) and sequences, plus analysis plots
    try:
        import json as _json
        # Save trajectory as JSONL with selected scalar fields
        traj = out.get("trajectory", []) or []
        with open(out_dir / "trajectory.jsonl", "w") as f:
            for rec in traj:
                step_i = int(rec.get("step", 0))
                metrics_i = rec.get("metrics", {}) or {}
                aux_i = rec.get("aux", {}) or {}
                loss_i = aux_i.get("loss")
                f.write(_json.dumps({
                    "step": step_i,
                    "loss": loss_i,
                    "metrics": {k: v for k, v in metrics_i.items() if k in (
                        "gap", "value_x", "value_y",
                        "ent_x", "ent_y", "kl_x_to_y", "kl_y_to_x",
                        "hamming_xy", "identity_xy",
                        "pos_ent_mean_x", "pos_ent_mean_y",
                        "pos_ent_min_x", "pos_ent_min_y",
                        "pos_ent_max_x", "pos_ent_max_y",
                        "charge_x", "charge_y", "hydropathy_x", "hydropathy_y",
                        "grad_norm_x", "grad_norm_y",
                    )},
                }) + "\n")

        # Save sequences over time where available
        with open(out_dir / "sequences.jsonl", "w") as f:
            for rec in traj:
                step_i = int(rec.get("step", 0))
                m = rec.get("metrics", {}) or {}
                sx = m.get("seq_x")
                sy = m.get("seq_y")
                if sx is not None or sy is not None:
                    f.write(_json.dumps({"step": step_i, "seq_x": sx, "seq_y": sy}) + "\n")

        # Collect vectors for advanced plots
        try:
            import matplotlib.pyplot as plt
            steps_plot = []
            gaps = []
            vx = []
            vy = []
            ent_x = []
            ent_y = []
            kxy = []
            kyx = []
            ident = []
            pos_ent_x_steps = []
            pos_ent_y_steps = []
            comp_x_steps = []
            comp_y_steps = []
            charge_x_s = []
            charge_y_s = []
            hyd_x_s = []
            hyd_y_s = []
            gnx = []
            gny = []
            for rec in traj:
                if rec.get("metrics"):
                    steps_plot.append(int(rec.get("step", 0)))
                    m = rec["metrics"]
                    gaps.append(m.get("gap"))
                    vx.append(m.get("value_x"))
                    vy.append(m.get("value_y"))
                    ent_x.append(m.get("ent_x"))
                    ent_y.append(m.get("ent_y"))
                    kxy.append(m.get("kl_x_to_y"))
                    kyx.append(m.get("kl_y_to_x"))
                    ident.append(m.get("identity_xy"))
                    if m.get("pos_ent_x") is not None:
                        pos_ent_x_steps.append(m.get("pos_ent_x"))
                    if m.get("pos_ent_y") is not None:
                        pos_ent_y_steps.append(m.get("pos_ent_y"))
                    if m.get("comp_x") is not None:
                        comp_x_steps.append(m.get("comp_x"))
                    if m.get("comp_y") is not None:
                        comp_y_steps.append(m.get("comp_y"))
                    cx = m.get("charge_x"); cy = m.get("charge_y")
                    hx = m.get("hydropathy_x"); hy = m.get("hydropathy_y")
                    gnx.append(m.get("grad_norm_x"))
                    gny.append(m.get("grad_norm_y"))
                    charge_x_s.append(cx)
                    charge_y_s.append(cy)
                    hyd_x_s.append(hx)
                    hyd_y_s.append(hy)

            # Gap plot
            if any(g is not None for g in gaps):
                plt.figure(figsize=(6, 4))
                plt.plot(steps_plot, gaps, label="gap")
                plt.xlabel("step")
                plt.ylabel("gap (value_x - value_y)")
                plt.title("Saddle gap over steps")
                plt.tight_layout()
                plt.savefig(out_dir / "gap.png", dpi=150)
                plt.close()

            # Values plot
            if any(v is not None for v in vx) or any(v is not None for v in vy):
                plt.figure(figsize=(6, 4))
                if any(v is not None for v in vx):
                    plt.plot(steps_plot, vx, label="value_x")
                if any(v is not None for v in vy):
                    plt.plot(steps_plot, vy, label="value_y")
                plt.xlabel("step")
                plt.ylabel("value components")
                plt.legend()
                plt.title("Value components over steps")
                plt.tight_layout()
                plt.savefig(out_dir / "values.png", dpi=150)
                plt.close()

            # Entropy plot
            if any(v is not None for v in ent_x) or any(v is not None for v in ent_y):
                plt.figure(figsize=(6, 4))
                if any(v is not None for v in ent_x):
                    plt.plot(steps_plot, ent_x, label="ent_x")
                if any(v is not None for v in ent_y):
                    plt.plot(steps_plot, ent_y, label="ent_y")
                plt.xlabel("step")
                plt.ylabel("mean entropy")
                plt.legend()
                plt.title("Entropy over steps")
                plt.tight_layout()
                plt.savefig(out_dir / "entropy.png", dpi=150)
                plt.close()

            # KL plot
            if any(v is not None for v in kxy) or any(v is not None for v in kyx):
                plt.figure(figsize=(6, 4))
                if any(v is not None for v in kxy):
                    plt.plot(steps_plot, kxy, label="KL(x||y)")
                if any(v is not None for v in kyx):
                    plt.plot(steps_plot, kyx, label="KL(y||x)")
                plt.xlabel("step")
                plt.ylabel("KL divergence")
                plt.legend()
                plt.title("Symmetric KL over steps")
                plt.tight_layout()
                plt.savefig(out_dir / "kl.png", dpi=150)
                plt.close()

            # Identity plot
            if any(v is not None for v in ident):
                plt.figure(figsize=(6, 4))
                plt.plot(steps_plot, ident, label="identity_xy")
                plt.xlabel("step")
                plt.ylabel("sequence identity")
                plt.title("Sequence identity between x and y")
                plt.tight_layout()
                plt.savefig(out_dir / "identity.png", dpi=150)
                plt.close()

            # Gradient norms
            if any(v is not None for v in gnx) or any(v is not None for v in gny):
                plt.figure(figsize=(6, 4))
                if any(v is not None for v in gnx):
                    plt.plot(steps_plot, gnx, label="grad_norm_x")
                if any(v is not None for v in gny):
                    plt.plot(steps_plot, gny, label="grad_norm_y")
                plt.xlabel("step"); plt.ylabel("grad norm"); plt.legend()
                plt.title("Gradient norms")
                plt.tight_layout(); plt.savefig(out_dir / "grad_norms.png", dpi=150); plt.close()

            # Charge and hydropathy trends
            if any(v is not None for v in charge_x_s) or any(v is not None for v in charge_y_s):
                plt.figure(figsize=(6, 4))
                if any(v is not None for v in charge_x_s):
                    plt.plot(steps_plot, charge_x_s, label="charge_x")
                if any(v is not None for v in charge_y_s):
                    plt.plot(steps_plot, charge_y_s, label="charge_y")
                plt.xlabel("step"); plt.ylabel("mean charge"); plt.legend()
                plt.title("Charge over steps")
                plt.tight_layout(); plt.savefig(out_dir / "charge.png", dpi=150); plt.close()

            if any(v is not None for v in hyd_x_s) or any(v is not None for v in hyd_y_s):
                plt.figure(figsize=(6, 4))
                if any(v is not None for v in hyd_x_s):
                    plt.plot(steps_plot, hyd_x_s, label="hydropathy_x")
                if any(v is not None for v in hyd_y_s):
                    plt.plot(steps_plot, hyd_y_s, label="hydropathy_y")
                plt.xlabel("step"); plt.ylabel("mean hydropathy"); plt.legend()
                plt.title("Hydropathy over steps")
                plt.tight_layout(); plt.savefig(out_dir / "hydropathy.png", dpi=150); plt.close()

            # Phase planes
            if any(v is not None for v in vx) and any(v is not None for v in vy):
                plt.figure(figsize=(5, 5))
                plt.plot(vx, vy, marker=".")
                plt.xlabel("value_x"); plt.ylabel("value_y")
                plt.title("Phase plane: value_x vs value_y")
                plt.tight_layout(); plt.savefig(out_dir / "phase_values.png", dpi=150); plt.close()
            if any(v is not None for v in ent_x) and any(v is not None for v in ent_y):
                plt.figure(figsize=(5, 5))
                plt.plot(ent_x, ent_y, marker=".")
                plt.xlabel("ent_x"); plt.ylabel("ent_y")
                plt.title("Phase plane: ent_x vs ent_y")
                plt.tight_layout(); plt.savefig(out_dir / "phase_entropy.png", dpi=150); plt.close()

            # Per-position entropy heatmaps (if vectors logged)
            import numpy as _np
            if len(pos_ent_x_steps) > 0:
                Hx = _np.stack([_np.array(h) for h in pos_ent_x_steps], axis=0)
                plt.figure(figsize=(8, 4)); plt.imshow(Hx.T, aspect="auto", origin="lower", cmap="magma")
                plt.colorbar(label="entropy"); plt.xlabel("step idx"); plt.ylabel("position")
                plt.title("Per-position entropy (x)")
                plt.tight_layout(); plt.savefig(out_dir / "pos_entropy_x.png", dpi=150); plt.close()
            if len(pos_ent_y_steps) > 0:
                Hy = _np.stack([_np.array(h) for h in pos_ent_y_steps], axis=0)
                plt.figure(figsize=(8, 4)); plt.imshow(Hy.T, aspect="auto", origin="lower", cmap="magma")
                plt.colorbar(label="entropy"); plt.xlabel("step idx"); plt.ylabel("position")
                plt.title("Per-position entropy (y)")
                plt.tight_layout(); plt.savefig(out_dir / "pos_entropy_y.png", dpi=150); plt.close()

            # Autocorrelation of gap
            if any(g is not None for g in gaps):
                gs = _np.array([0.0 if v is None else float(v) for v in gaps], dtype=float)
                gs = gs - gs.mean()
                ac = _np.correlate(gs, gs, mode="full")[len(gs)-1:]
                lags = _np.arange(len(ac))
                plt.figure(figsize=(6, 4)); plt.plot(lags[: min(100, len(lags))], ac[: min(100, len(ac))])
                plt.xlabel("lag"); plt.ylabel("autocorr (unnormalized)")
                plt.title("Gap autocorrelation")
                plt.tight_layout(); plt.savefig(out_dir / "gap_autocorr.png", dpi=150); plt.close()
        except Exception as _e:
            (out_dir / "plot_warn.txt").write_text(str(_e))
    except Exception as _e:
        (out_dir / "save_warn.txt").write_text(str(_e))


@app.local_entrypoint()
def main(binder_len: int = 20, steps: int = 120, seed: int = 42, helix_weight: float = -0.3, game: str = "minmax"):
    run_binder_minmax.remote(
        binder_len=binder_len,
        steps=steps,
        helix_weight=helix_weight,
        seed=seed,
        game=game,
    )


