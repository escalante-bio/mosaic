import os
import sys
from pathlib import Path

import modal


image = (
    modal.Image.debian_slim(python_version="3.12.0")
    .apt_install(
        "git",
        "aria2",
        # For matplotlib usetex support in provided style
        "texlive-latex-recommended",
        "texlive-latex-extra",
        "texlive-fonts-recommended",
        "dvipng",
    )
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
        "python -m pip install optax==0.2.4 dm-haiku>=0.0.13 flax>=0.10.2 ml-collections>=1.0.0 httpx>=0.28.1 gemmi>=0.6.0 matplotlib>=3.10.0 seaborn>=0.13.2 pandas>=2.2.2 && "
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


app = modal.App("adaptyv-binder-compare", image=image)

boltz_cache = modal.Volume.from_name("boltz-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("results-binder-compare", create_if_missing=True)
af2_cache = modal.Volume.from_name("alphafold-cache", create_if_missing=True)
local_src_mount = modal.Mount.from_local_dir(
    "/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/src", remote_path="/workspace/src"
)


def _now_id():
    import time as _time

    return str(int(_time.time()))


def _build_af2_and_features(binder_len: int):
    import gemmi
    from mosaic.af2.alphafold2 import AF2

    # Ensure params exist
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

    binder_placeholder = "X" * int(binder_len)
    target_seq = "".join([gemmi.one_letter_code([r.name]) for r in target_chain])
    features, _ = af2.build_features(
        chains=[binder_placeholder, target_seq],
        template_chains={1: target_chain},
        initial_guess=None,
    )

    return af2, features


def _build_structure_losses(af2, features):
    import mosaic.losses.structure_prediction as sp
    from mosaic.losses.af2 import AlphaFoldLoss, AF2Output
    from mosaic.common import LinearCombination

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

    structure_loss: LinearCombination = (
        1.0 * sp.BinderTargetContact(contact_distance=21.0)
        + 1.0 * sp.WithinBinderContact(
            max_contact_distance=14.0, num_contacts_per_residue=4, min_sequence_separation=8
        )
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

    return loss_x, loss_y


def _flatten_scalars(prefix, obj, out):
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten_scalars(f"{prefix}{k}/", v, out)
        else:
            # record leaf numeric scalars
            import numpy as _np

            if obj is None:
                return
            if isinstance(obj, (int, float)):
                out[prefix[:-1]] = float(obj)
            elif hasattr(obj, "shape") and _np.ndim(_np.array(obj)) == 0:
                out[prefix[:-1]] = float(_np.array(obj))
    except Exception:
        pass


@app.function(
    gpu="H100",
    timeout=3 * 60 * 60,
    volumes={"/root/.boltz": boltz_cache, "/results": results_vol, "/repo/params": af2_cache},
    mounts=[local_src_mount],
    secrets=[modal.Secret.from_name("github-token")],
)
def run_experiment(
    *,
    mode: str,  # "minmax" | "stackelberg" | "simple"
    binder_len: int = 60,
    steps: int = 600,
    seed: int = 42,
    lr_x: float = 0.05,
    lr_y: float | None = None,
    initial_x: list | None = None,  # pass np.ndarray.tolist() to ensure serialization
):
    # Environment and import path preference
    os.environ.setdefault("BOLTZ_CACHE", "/root/.boltz")
    Path(os.environ["BOLTZ_CACHE"]).mkdir(parents=True, exist_ok=True)

    local_src = Path("/workspace/src")
    if local_src.exists():
        sys.path.insert(0, str(local_src))
    repo_src = Path("/repo/src")
    if repo_src.exists():
        sys.path.append(str(repo_src))

    import json
    import time
    import numpy as np
    import jax
    import jax.numpy as jnp

    from binder_games import build_minmax_phase, build_stackelberg_phase
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
    from mosaic_workflows.transforms import (
        temperature_on_logits,
        e_soft_on_logits,
        gradient_normalizer,
        zero_disallowed,
    )

    # Build AF2 and losses
    af2, features = _build_af2_and_features(binder_len)
    loss_x, loss_y = _build_structure_losses(af2, features)

    # Two-player objective (used for minmax/stackelberg)
    margin = 0.1
    lambda_hinge = 0.5
    gamma_anti_collusion = 0.05

    def twoplayer_loss(x_probs, y_probs, key=None):
        vx, auxx = loss_x(x_probs, key=key)
        vy, auxy = loss_y(y_probs, key=key)
        base = vx - vy
        h = jnp.maximum(0.0, margin + vx - vy)
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

    # Schedule and transforms
    def schedule_minmax(_g, _p):
        return {
            "lr_x": float(lr_x),
            "lr_y": float(lr_y if lr_y is not None else lr_x),
            "temperature": 1.0,
            "y_init": "random",
        }

    def schedule_stack(_g, _p):
        return {
            "lr_x": float(lr_x),
            "lr_y": float(lr_y if lr_y is not None else lr_x * 1.6),
            "br_steps": 5,
            "reinit_y_each_step": False,
            "temperature": 1.0,
            "y_init": "random",
        }

    transforms = {
        "x": {
            "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
            "grad": [
                gradient_normalizer(mode="per_chain", log_norm=True),
                zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"]),
            ],
        },
        "y": {
            "pre_logits": [temperature_on_logits(), e_soft_on_logits()],
            "grad": [
                gradient_normalizer(mode="per_chain", log_norm=True),
                zero_disallowed(restrict_to_canon=True, avoid_residues=["CYS"]),
            ],
        },
    }

    analyzers = [
        saddle_gap_estimate(),
        value_components(),
        decode_sequences_xy(),
        probs_entropy_xy(),
        kl_divergence_xy(),
        sequence_hamming_xy(),
        per_position_entropy_xy(),
        composition_charge_hydropathy_xy(),
        grad_norms_xy(),
    ]

    # Initialize x logits from shared input
    if initial_x is None:
        rng = np.random.RandomState(int(seed))
        x0 = rng.randn(int(binder_len), 20).astype(np.float32) * 0.1
    else:
        x0 = np.array(initial_x, dtype=np.float32)

    # Build phase per mode
    from mosaic_workflows import run_workflow

    if mode.lower() == "minmax":
        phase = build_minmax_phase(
            name="minmax_af2",
            build_loss=lambda: twoplayer_loss,
            steps=int(steps),
            schedule=schedule_minmax,
            transforms=transforms,
            analyzers=analyzers,
            analyze_every=1,
        )
        wf = {"phases": [phase], "binder_len": binder_len, "seed": int(seed), "initial_x": x0}
    elif mode.lower() == "stackelberg":
        phase = build_stackelberg_phase(
            name="stackelberg_af2",
            build_loss=lambda: twoplayer_loss,
            steps=int(steps),
            schedule=schedule_stack,
            transforms=transforms,
            analyzers=analyzers,
            analyze_every=1,
        )
        wf = {"phases": [phase], "binder_len": binder_len, "seed": int(seed), "initial_x": x0}
    else:
        # Simple single-player optimization using same transforms for x only
        # Implemented as a Mosaic one-phase with a custom optimizer inline
        import equinox as eqx

        def _center_last_axis(g):
            return g - g.mean(axis=-1, keepdims=True)

        @eqx.filter_jit
        def _value_and_grad_single(logits, key):
            def f(lg, k):
                p = jax.nn.softmax(lg, axis=-1)
                v, aux = loss_x(p, key=k)
                return v, aux

            (v, aux), g = eqx.filter_value_and_grad(lambda lg, k: f(lg, k), has_aux=True)(
                logits, key
            )
            return (v, aux), g

        def _run_simple():
            key = jax.random.key(int(seed))
            x = np.array(x0, dtype=np.float32)
            best_val = np.inf
            best_x = x.copy()
            traj = []
            for step in range(int(steps)):
                ctx = {"schedule": {"temperature": 1.0}}
                x_logits = x
                for fn in transforms["x"]["pre_logits"]:
                    x_logits = fn(x_logits, ctx)
                (value, aux), gx = _value_and_grad_single(x_logits, key)
                key = jax.random.fold_in(key, 0)
                # x_probs for analyzers
                x_probs = jax.nn.softmax(x_logits, axis=-1)
                # grads
                gx = _center_last_axis(gx)
                for fn in transforms["x"]["grad"]:
                    gx = fn(gx, ctx)
                # update
                x = x_logits - float(lr_x) * gx
                for fn in transforms["x"].get("post_logits", []):
                    x = fn(x, ctx)

                # enrich aux for analyzers and logging
                try:
                    aux = dict(aux) if isinstance(aux, dict) else {"aux": aux}
                    aux["value_x"] = float(value)
                    aux.setdefault("x", {})
                    aux["x"]["probs"] = np.array(x_probs)
                except Exception:
                    pass

                # Run analyzers
                metrics = {}
                for an in analyzers:
                    try:
                        metrics.update(an(aux))
                    except Exception:
                        pass
                # Also flatten scalar per-objective signals from aux["x"]
                flat = {}
                _flatten_scalars("x/", aux.get("x", {}), flat)
                for k, v in flat.items():
                    if isinstance(v, (int, float)):
                        metrics[k] = float(v)

                traj.append({"step": step, "metrics": metrics, "aux": {"loss": float(value)}})
                if float(value) < best_val:
                    best_val = float(value)
                    best_x = np.array(x)
            return {"trajectory": traj, "best_x": best_x}

        out = _run_simple()
        run_id = f"binder_compare_{_now_id()}_{mode.lower()}_seed{seed}_len{binder_len}"
        out_dir = Path("/results") / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "best_sequence.txt").write_text(
            str((out.get("trajectory") or [{}])[-1].get("metrics", {}).get("seq_x", ""))
        )
        if out.get("best_x") is not None:
            np.save(out_dir / "best_x.npy", out.get("best_x"))

        # Save compact trajectory jsonl (numbers only)
        import json as _json
        import numpy as _np

        def _only_numeric(d):
            out = {}
            for k, v in (d or {}).items():
                try:
                    if isinstance(v, (int, float)):
                        out[k] = float(v)
                        continue
                    arr = _np.array(v)
                    if arr.ndim == 0:
                        out[k] = float(arr)
                except Exception:
                    pass
            return out

        traj = out.get("trajectory", []) or []
        with open(out_dir / "trajectory.jsonl", "w") as f:
            for rec in traj:
                step_i = int(rec.get("step", 0))
                metrics_i = _only_numeric(rec.get("metrics", {}) or {})
                aux_i = rec.get("aux", {}) or {}
                try:
                    loss_val = float(aux_i.get("loss")) if aux_i.get("loss") is not None else None
                except Exception:
                    loss_val = None
                f.write(_json.dumps({"step": step_i, "loss": loss_val, "metrics": metrics_i}) + "\n")

        print({"results_dir": str(out_dir)})
        return str(out_dir)

    # Two-player modes via Mosaic workflow runner
    phase = phase  # silences lints in branches above
    wf = wf
    out = run_workflow(wf)

    run_id = f"binder_compare_{_now_id()}_{mode.lower()}_seed{seed}_len{binder_len}"
    out_dir = Path("/results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "best_sequence.txt").write_text(str(out.get("best_sequence", "")))
    if out.get("best_x") is not None:
        np.save(out_dir / "best_x.npy", out.get("best_x"))

    # Persist full trajectory with richer metrics
    try:
        import json as _json

        traj = out.get("trajectory", []) or []
        with open(out_dir / "trajectory.jsonl", "w") as f:
            for rec in traj:
                step_i = int(rec.get("step", 0))
                metrics_i = rec.get("metrics", {}) or {}
                aux_i = rec.get("aux", {}) or {}
                # Flatten per-objective scalars from nested x/y for later comparative plots
                flat = {}
                _flatten_scalars("x/", (metrics_i.get("x") or {}), flat)
                _flatten_scalars("y/", (metrics_i.get("y") or {}), flat)
                # Note: analyzers already extracted most top-level series (gap, ent, KL, etc.)
                metrics_out = dict(metrics_i)
                metrics_out.update({k: v for k, v in flat.items() if isinstance(v, (int, float))})
                f.write(
                    _json.dumps(
                        {
                            "step": step_i,
                            "loss": aux_i.get("loss"),
                            "metrics": metrics_out,
                        }
                    )
                    + "\n"
                )
    except Exception as _e:
        (out_dir / "save_warn.txt").write_text(str(_e))

    print({"results_dir": str(out_dir)})
    return str(out_dir)


# ======= Plotting (runs on Modal, pools results, renders side-by-side) =======


@app.function(
    gpu=None,
    timeout=30 * 60,
    volumes={"/results": results_vol},
)
def plot_compare(result_dirs: list[str], out_name: str = "compare"):
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap

    # Style and colors (provided)
    adaptyv_colors = [
        "#8CD2F4",
        "#3E6175",
        "#56A6D4",
        "#8B90DD",
        "#F5A43E",
        "#FFB3BA",
        "#8CD2F4",
        "#3E6175",
        "#56A6D4",
        "#8B90DD",
        "#F5A43E",
        "#FFB3BA",
    ]

    cmap_colors = [adaptyv_colors[1], adaptyv_colors[2], "#ffffff"]
    cmap_colors_corr = ["#F5A43E", "#ffffff", "#8CD2F4"]
    adaptyv_cmap = LinearSegmentedColormap.from_list("adaptyv_cmap", cmap_colors)
    adaptyv_cmap_r = adaptyv_cmap.reversed()
    adaptyv_corr_cmap = LinearSegmentedColormap.from_list("adaptyv_corr_cmap", cmap_colors_corr)

    def set_adaptyv_style():
        plt.style.use("seaborn-v0_8-whitegrid")

        plt.rcParams["figure.figsize"] = (3.5, 2.625)

        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["xtick.major.size"] = 3
        plt.rcParams["xtick.major.width"] = 0.5
        plt.rcParams["xtick.minor.size"] = 1.5
        plt.rcParams["xtick.minor.width"] = 0.5
        plt.rcParams["xtick.minor.visible"] = True
        plt.rcParams["xtick.top"] = True

        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["ytick.major.size"] = 3
        plt.rcParams["ytick.major.width"] = 0.5
        plt.rcParams["ytick.minor.size"] = 1.5
        plt.rcParams["ytick.minor.width"] = 0.5
        plt.rcParams["ytick.minor.visible"] = True
        plt.rcParams["ytick.right"] = True

        plt.rcParams["axes.linewidth"] = 0.5
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.spines.top"] = True
        plt.rcParams["axes.spines.right"] = True

        plt.rcParams["lines.linewidth"] = 1.0

        plt.rcParams["legend.frameon"] = False
        plt.rcParams["legend.edgecolor"] = "black"
        plt.rcParams["legend.loc"] = "center left"
        plt.rcParams["legend.fontsize"] = "small"
        plt.rcParams["legend.framealpha"] = 1.0
        plt.rcParams["legend.borderpad"] = 0.4
        plt.rcParams["legend.borderaxespad"] = 0.5
        plt.rcParams["legend.handlelength"] = 1.0
        plt.rcParams["legend.handleheight"] = 0.7
        plt.rcParams["legend.handletextpad"] = 0.5
        plt.rcParams["legend.columnspacing"] = 1.0
        plt.rcParams["legend.labelspacing"] = 0.4
        plt.rcParams["legend.markerscale"] = 0.8
        plt.rcParams["legend.fancybox"] = False

        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["savefig.pad_inches"] = 0.05

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"

        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"

        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=adaptyv_colors)

        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["figure.frameon"] = True

        plt.rcParams["axes.grid"] = False
        plt.rcParams["figure.dpi"] = 600

    def _read_traj(dirp):
        steps = []
        series = {}
        path = Path(dirp) / "trajectory.jsonl"
        if not path.exists():
            return steps, series
        with open(path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    steps.append(int(rec.get("step", 0)))
                    metrics = rec.get("metrics", {}) or {}
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)) or (hasattr(v, "__float__")):
                            series.setdefault(k, []).append(float(v))
                except Exception:
                    pass
        return steps, series

    # Load all three
    names = []
    data = []
    for d in result_dirs:
        names.append(Path(d).name)
        data.append(_read_traj(d))

    # Determine canonical x-axis
    all_keys = set()
    for steps, series in data:
        all_keys.update(series.keys())

    # Select main metrics we want to mirror previous plots
    main_keys = [
        "gap",
        "value_x",
        "value_y",
        "ent_x",
        "ent_y",
        "kl_x_to_y",
        "kl_y_to_x",
        "identity_xy",
        "grad_norm_x",
        "grad_norm_y",
        "charge_x",
        "charge_y",
        "hydropathy_x",
        "hydropathy_y",
    ]
    # Include any per-objective scalars if present (e.g., x/plddt)
    objective_like = sorted([k for k in all_keys if k.startswith("x/") or k.startswith("y/")])

    set_adaptyv_style()

    out_dir = Path("/results") / (out_name or "compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Helper to render a row of 3 subplots for a given metric key
    def render_row(keys, fname, title):
        fig, axes = plt.subplots(1, len(result_dirs), figsize=(3.5 * len(result_dirs), 2.625))
        if len(result_dirs) == 1:
            axes = [axes]
        for ax, (steps, series), nm in zip(axes, data, names):
            for k in keys:
                if k in series:
                    ax.plot(steps, series[k], label=k)
            ax.set_xlabel("step")
            ax.set_title(title)
            if len(keys) > 1:
                ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=600)
        plt.close(fig)

    # Core plots
    render_row(["gap"], "gap.png", "saddle gap")
    render_row(["value_x", "value_y"], "values.png", "value components")
    render_row(["ent_x", "ent_y"], "entropy.png", "entropy")
    render_row(["kl_x_to_y", "kl_y_to_x"], "kl.png", "symmetric kl")
    render_row(["identity_xy"], "identity.png", "sequence identity")
    render_row(["grad_norm_x", "grad_norm_y"], "grad_norms.png", "gradient norms")
    render_row(["charge_x", "charge_y"], "charge.png", "charge")
    render_row(["hydropathy_x", "hydropathy_y"], "hydropathy.png", "hydropathy")

    # Per-objective overlays (if present) — plot up to a handful
    for k in objective_like[:12]:
        safe = k.replace("/", "_")
        render_row([k], f"{safe}.png", k.replace("_", " "))

    print({"compare_dir": str(out_dir)})
    return str(out_dir)


@app.local_entrypoint()
def main(
    binder_len: int = 20,
    steps: int = 120,
    seed: int = 42,
    lr_x: float = 0.05,
    lr_y: float | None = None,
):
    # Generate shared initial logits deterministically on client and pass to all runs
    import numpy as np

    rng = np.random.RandomState(int(seed))
    x0 = (rng.randn(int(binder_len), 20).astype(np.float32) * 0.1).tolist()

    # Launch three runs in parallel
    f_minmax = run_experiment.spawn(
        mode="minmax", binder_len=binder_len, steps=steps, seed=seed, lr_x=lr_x, lr_y=lr_y, initial_x=x0
    )
    f_stack = run_experiment.spawn(
        mode="stackelberg", binder_len=binder_len, steps=steps, seed=seed, lr_x=lr_x, lr_y=lr_y, initial_x=x0
    )
    f_simple = run_experiment.spawn(
        mode="simple", binder_len=binder_len, steps=steps, seed=seed, lr_x=lr_x, lr_y=lr_y, initial_x=x0
    )

    # Gather results
    d_minmax = f_minmax.get()
    d_stack = f_stack.get()
    d_simple = f_simple.get()

    # Plot comparison on Modal
    plot_compare.remote([d_minmax, d_stack, d_simple], out_name=f"compare_{_now_id()}")


