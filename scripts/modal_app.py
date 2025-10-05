import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict

import modal


# -------- JAX/XLA Configuration Helpers --------

def configure_jax_environment(*, enable_pgle: bool = False, cache_dir: str = "/root/.cache/jax"):
    """Centralized JAX/XLA configuration for optimal GPU performance.

    Args:
        enable_pgle: Enable profile-guided latency estimator (slower compilation, better runtime)
        cache_dir: JAX compilation cache directory
    """
    import os as _os

    # JAX platform and precision
    _os.environ.setdefault("JAX_PLATFORMS", "cuda")
    _os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "tensorfloat32")  # ~3x faster on A100/H100

    # PGLE settings (profile-guided compilation)
    _os.environ["JAX_ENABLE_PGLE"] = "true" if enable_pgle else "false"
    if enable_pgle:
        _os.environ.setdefault("JAX_PGLE_PROFILING_RUNS", "1")
        _os.environ.setdefault("JAX_PGLE_AGGREGATION_PERCENTILE", "85")
    _os.environ.pop("JAX_COMPILATION_CACHE_EXPECT_PGLE", None)

    # Compilation caching
    _os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)

    # TensorFlow logging
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    # XLA optimizations (consolidated and deduplicated)
    xla_flags = []
    xla_flags.append("--xla_gpu_enable_latency_hiding_scheduler=true")
    xla_flags.append("--xla_gpu_enable_triton_gemm=true")
    xla_flags.append("--xla_gpu_enable_cudnn_fmha=true")
    xla_flags.append("--xla_gpu_graph_level=3")
    _os.environ["XLA_FLAGS"] = " ".join(xla_flags)

    # NCCL optimizations for multi-GPU (no-op on single GPU)
    _os.environ.setdefault("NCCL_LL128_BUFFSIZE", "-2")
    _os.environ.setdefault("NCCL_LL_BUFFSIZE", "-2")
    _os.environ.setdefault("NCCL_PROTO", "SIMPLE,LL,LL128")

    # HuggingFace cache
    _os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    _os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface")

    # Ensure cache directories exist
    Path(cache_dir).mkdir(parents=True, exist_ok=True)


# -------- Modal configuration --------

## per-pipeline images configured below


image_bindcraft = (
    modal.Image.debian_slim(python_version="3.12.0")
    .apt_install("git", "aria2")
    .run_commands(
        "python -m pip install -U pip setuptools wheel && "
        "python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1 && "
        "python -m pip uninstall -y jax jaxlib jax-cuda12-plugin || true && "
        "python -m pip install --no-cache-dir jax==0.6.2 jaxlib==0.6.2 jax-cuda12-plugin==0.6.2 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && "
        "python -m pip install numpy==1.26.4 gemmi==0.6.6 matplotlib pandas && "
        "python -m pip install git+https://github.com/sokrypton/ColabDesign.git && "
        "python -m pip install git+https://github.com/openmm/pdbfixer.git && "
        "python -m pip install loguru equinox==0.11.4 && "
        "true"
    )
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/src", "/workspace/src")  # type: ignore[attr-defined]
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/adaptyv_bindcraft/src/BindCraft", "/root/BindCraft")  # type: ignore[attr-defined]
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/adaptyv_bindcraft/utilities", "/root/utilities")  # type: ignore[attr-defined]
)

image_germinal = (
    modal.Image.debian_slim(python_version="3.12.0")
    .apt_install("git", "aria2")
    .run_commands(
        "python -m pip install -U pip setuptools wheel && "
        "python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1 && "
        "python -m pip uninstall -y jax jaxlib jax-cuda12-plugin || true && "
        "python -m pip install --no-cache-dir jax==0.6.2 jaxlib==0.6.2 jax-cuda12-plugin==0.6.2 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && "
        "python -m pip install --no-cache-dir numpy==1.26.4 && "
        "python -m pip install nvidia-cuda-nvcc-cu12==12.8.93 && "
        "python -m pip install optax==0.2.4 dm-haiku>=0.0.13 flax>=0.10.2 ml-collections>=1.0.0 httpx>=0.28.1 gemmi>=0.6.0 && "
        "python -m pip install loguru openmm mdtraj biopython freesasa scipy scikit-learn pyyaml pandas matplotlib && "
        "python -m pip install git+https://github.com/sokrypton/ColabDesign.git && "
        "python -m pip install git+https://github.com/openmm/pdbfixer.git && "
        "git clone --depth 1 https://github.com/adaptyvbio/mosaic_workflows.git /repo && "
        "true"
    )
    # No redundant re-pins; jax/numpy already pinned above
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/src", "/workspace/src")  # type: ignore[attr-defined]
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/adaptyv_bindcraft/src/BindCraft", "/root/BindCraft")  # type: ignore[attr-defined]
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/adaptyv_bindcraft/utilities", "/root/utilities")  # type: ignore[attr-defined]
)

image_mhetase = (
    modal.Image.debian_slim(python_version="3.12.0")
    .apt_install("git")
    .run_commands(
        "python -m pip install -U pip setuptools wheel && "
        "python -m pip install --no-cache-dir jax==0.6.2 jaxlib==0.6.2 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && "
        "python -m pip install numpy==1.26.4 gemmi==0.6.6 pandas tqdm jaxtyping equinox==0.11.4 dm-haiku"
    )
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows", "/workspace/repo")  # type: ignore[attr-defined]  # project root for scripts, params
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/src", "/workspace/src")  # type: ignore[attr-defined]
)


app = modal.App("mosaic-e2e")


germinal_cache_vol = modal.Volume.from_name("germinal-cache", create_if_missing=True)
germinal_outputs_vol = modal.Volume.from_name("germinal-outputs", create_if_missing=True)
bindcraft_cache_vol = modal.Volume.from_name("bindcraft-cache", create_if_missing=True)
bindcraft_outputs_vol = modal.Volume.from_name("bindcraft-outputs", create_if_missing=True)
mhetase_cache_vol = modal.Volume.from_name("mhetase-cache", create_if_missing=True)
mhetase_outputs_vol = modal.Volume.from_name("mhetase-outputs", create_if_missing=True)

def _add_paths(workspace: Path):
    sys.path.append(str(workspace / "src"))


@app.function(
    image=image_bindcraft,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={
        "/bindcraft_cache": bindcraft_cache_vol,
        "/bindcraft_outputs": bindcraft_outputs_vol,
    },
)
def run_bindcraft():
    import os as _os
    from pathlib import Path as _Path
    import subprocess as _sp
    import sys as _sys

    # Configure JAX/XLA with optimized settings and persist caches in volume
    configure_jax_environment(enable_pgle=False, cache_dir="/bindcraft_cache/jax")
    _os.environ.setdefault("XDG_CACHE_HOME", "/bindcraft_cache/xdg")
    _os.environ.setdefault("HF_HOME", "/bindcraft_cache/hf")
    _os.environ.setdefault("TRANSFORMERS_CACHE", "/bindcraft_cache/hf")

    # Ensure local src and baked repo are importable
    local_src = _Path("/workspace/src")
    if local_src.exists():
        _sys.path.insert(0, str(local_src))
    repo_src = _Path("/repo/src")
    if repo_src.exists():
        _sys.path.append(str(repo_src))

    # Add BindCraft and utilities to path (required by bindcraft_compat)
    for p in ("/root/BindCraft", "/root/utilities"):
        if p not in _sys.path and _Path(p).exists():
            _sys.path.insert(0, p)

    # No external clones required for BindCraft runner

    # Build configs
    target_pdb_path = str(_Path("/root/BindCraft") / "pdbs" / "pdl1.pdb")
    # BindCraft E2E wrapper
    from mosaic_workflows.e2e.bindcraft_e2e import run_e2e as run_bindcraft_e2e  # type: ignore

    design_path = "/bindcraft_outputs/bindcraft"
    _Path(design_path).mkdir(parents=True, exist_ok=True)

    # Load PD-L1 defaults from external BindCraft repo
    bc_root = _Path("/root/BindCraft")
    import json as _json
    ts_path = bc_root / "settings_target" / "pdl1.json"
    adv_path = bc_root / "settings_advanced" / "default_4stage_multimer.json"
    filt_path = bc_root / "settings_filters" / "default_filters.json"
    target_settings = _json.loads(ts_path.read_text())
    advanced_settings = _json.loads(adv_path.read_text())
    filters = _json.loads(filt_path.read_text())
    advanced_settings["af_params_dir"] = "/bindcraft_cache/af2_params"
    out = run_bindcraft_e2e(
        design_path=design_path,
        repo_root="/root/BindCraft",
        target_settings=target_settings,
        advanced_settings=advanced_settings,
        filters=filters,
        max_trajectories=10,
        runtime_seed=0,
    )
    print({"results_dir": design_path, "status": "completed"})

@app.function(
    image=image_germinal,
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={
        "/germinal_cache": germinal_cache_vol,
        "/germinal_outputs": germinal_outputs_vol,
    },
)
def run_germinal():
    from pathlib import Path as _Path
    import subprocess as _sp
    import sys as _sys
    import os as _os
    # Centralized JAX/XLA environment
    configure_jax_environment(enable_pgle=False, cache_dir="/germinal_cache/jax")
    _os.environ.setdefault("XDG_CACHE_HOME", "/germinal_cache/xdg")
    _os.environ.setdefault("HF_HOME", "/germinal_cache/hf")
    _os.environ.setdefault("TRANSFORMERS_CACHE", "/germinal_cache/hf")
    # Ensure local source is importable inside container
    local_src = _Path("/workspace/src")
    if local_src.exists() and (str(local_src) not in _sys.path):
        _sys.path.insert(0, str(local_src))
    from mosaic_workflows.e2e.germinal_e2e import run_e2e as run_germinal_e2e  # type: ignore
    g_dir = _Path("/tmp/germinal")
    if not g_dir.exists():
        _sp.run(["git", "clone", "--depth", "1", "https://github.com/SantiagoMille/germinal.git", str(g_dir)], check=True)
    if str(g_dir) not in _sys.path:
        _sys.path.insert(0, str(g_dir))
    import yaml as _yaml  # type: ignore
    with open(g_dir / "configs/run/vhh.yaml", "r") as f:
        vhh = _yaml.safe_load(f) or {}
    with open(g_dir / "configs/target/pdl1.yaml", "r") as f:
        tgt = _yaml.safe_load(f) or {}
    design_path = "/germinal_outputs/germinal"
    _Path(design_path).mkdir(parents=True, exist_ok=True)
    target_settings = {"starting_pdb": str(g_dir / tgt["target_pdb_path"]), "chains": tgt.get("target_chain", "A"), "target_seq": ""}
    af3_settings = {
        "af_params_dir": "/germinal_cache/af2_params",
        "af3_repo_path": "/germinal_cache/af3_repo",
        "af3_sif_path": "/germinal_cache/af3_models/alphafold3.sif",
        "af3_model_dir": "/germinal_cache/af3_models",
        "af3_db_dir": "/germinal_cache/af3_db",
        "msa_db_dir": "/germinal_cache/msa_db",
        "use_metagenomic_db": False,
    }
    out = run_germinal_e2e(
        design_path=design_path,
        target_settings=target_settings,
        vhh_config=vhh,
        af3_settings=af3_settings,
        max_trajectories=10,
        runtime_seed=0,
    )
    print({"results_dir": design_path, "status": "completed"})

@app.function(
    image=image_mhetase,
    gpu="H100",
    timeout=3 * 60 * 60,
    volumes={
        "/mhetase_cache": mhetase_cache_vol,
        "/mhetase_outputs": mhetase_outputs_vol,
    },
    mounts=[
        modal.Mount.from_local_dir(
            "/Users/tudorcotet/Downloads",
            remote_path="/workspace/downloads",
        ),
    ],
)
def run_mhetase(
    *,
    binder_len: int = 100,
    ser: int | None = None,
    asp: int | None = None,
    his: int | None = None,
    gly: int | None = None,
    glu: int | None = None,
    total_steps: int = 100,
    pdb_path: str = "",
    pdb_residues: str = "",
    freeze_supervised_positions: bool = False,
    fix_supervised_identities: str = "",
    use_jd: bool = False,
):
    import os as _os
    from pathlib import Path as _Path
    import sys as _sys
    import json as _json
    # Ensure local source is importable inside container
    local_src = _Path("/workspace/src")
    if local_src.exists() and (str(local_src) not in _sys.path):
        _sys.path.insert(0, str(local_src))
    # Ensure repo root (for download_params.sh) is available and cwd points there
    repo_root = _Path("/workspace/repo")
    if repo_root.exists() and (str(repo_root) not in _sys.path):
        _sys.path.insert(0, str(repo_root))
    try:
        _os.chdir(str(repo_root))
    except Exception:
        pass
    from mosaic_workflows.e2e.mhetase_e2e import run_e2e as run_mhetase_e2e  # type: ignore
    configure_jax_environment(enable_pgle=False, cache_dir="/mhetase_cache/jax")
    design_path = "/mhetase_outputs/mhetase"
    # Build scaffold kwargs from CLI
    # Supervised positions and roles (in the order provided)
    roles_and_vals = [("ser", ser), ("asp", asp), ("his", his), ("gly", gly), ("glu", glu)]
    supervised_positions = tuple(int(v) for (r, v) in roles_and_vals if v is not None)
    motif_roles = tuple(r for (r, v) in roles_and_vals if v is not None)

    # PDB path mapping (mounts host Downloads to /workspace/downloads)
    remote_pdb_path = None
    if pdb_path:
        remote_pdb_path = str(_Path("/workspace/downloads") / _Path(pdb_path).name)
    else:
        # fallback to repo external copy if present
        candidate = repo_root / "_external" / "6QZ4.pdb"
        if candidate.exists():
            remote_pdb_path = str(candidate)

    # Parse residue list
    motif_resnums = None
    if pdb_residues:
        try:
            motif_resnums = tuple(int(x.strip()) for x in str(pdb_residues).split(",") if x.strip())
        except Exception:
            motif_resnums = None

    # Auto-detect chain id if not specified by inspecting PDB and residue list
    motif_chain_id = None
    if remote_pdb_path and motif_resnums:
        try:
            import gemmi as _gemmi  # type: ignore
            st = _gemmi.read_structure(remote_pdb_path)
            for ch in st[0]:
                res_ids = {r.seqid.num for r in ch}
                if all((int(rn) in res_ids) for rn in motif_resnums):
                    motif_chain_id = ch.name
                    break
        except Exception:
            motif_chain_id = None

    # Always use AF2 path inside container to avoid external deps; params cached under volume
    af2_settings = {
        "use_af2": True,
        "af2_num_recycles": 1,
        "af2_params_dir": "/mhetase_cache/af2",
    }

    scaffold_kwargs: dict = {
        "tmol_context": {"ligand": {}},
        "supervised_positions": supervised_positions if supervised_positions else None,
        "motif_roles": motif_roles if motif_roles else None,
        "motif_pdb_path": remote_pdb_path,
        "motif_chain_id": motif_chain_id,
        "motif_resnums": motif_resnums,
        "freeze_supervised_positions": bool(freeze_supervised_positions),
        "fix_supervised_identities": fix_supervised_identities,
        "steps": int(total_steps),
    } | af2_settings
    af3_settings: dict = {}
    out = run_mhetase_e2e(
        design_path=design_path,
        binder_len=int(binder_len),
        scaffold_kwargs=scaffold_kwargs,
        af3_settings=af3_settings,
        seed=0,
    )
    print(_json.dumps({"results_dir": design_path, "status": "completed"}))

@app.local_entrypoint()
def main(
    workflow: str = "germinal",
    binder_len: int = 100,
    ser: int | None = None,
    asp: int | None = None,
    his: int | None = None,
    gly: int | None = None,
    glu: int | None = None,
    total_steps: int = 100,
    pdb_path: str = "",
    pdb_residues: str = "",
    freeze_supervised_positions: bool = False,
    fix_supervised_identities: str = "",
    use_jd: bool = False,
):
    if workflow == "germinal":
        run_germinal.remote()
    elif workflow == "bindcraft":
        run_bindcraft.remote()
    elif workflow == "mhetase":
        run_mhetase.remote(
            binder_len=binder_len,
            ser=ser,
            asp=asp,
            his=his,
            gly=gly,
            glu=glu,
            total_steps=total_steps,
            pdb_path=pdb_path,
            pdb_residues=pdb_residues,
            freeze_supervised_positions=freeze_supervised_positions,
            fix_supervised_identities=fix_supervised_identities,
            use_jd=use_jd,
        )
    else:
        raise ValueError("workflow must be one of: germinal, bindcraft, mhetase")


