import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict

import modal


# -------- Modal configuration --------

# Base image with Python + common scientific stack; adjust as needed.
# For GPU runs, set gpu=modal.gpu.A10G() on the function below and add CUDA wheels.
image = (
    modal.Image.debian_slim(python_version="3.12.0")
    .apt_install("git", "aria2")
    .env({
        "BOLTZ_CACHE": "/root/.boltz",
        "JAX_PLATFORMS": "cuda",
        # Prefer tensor core kernels; disable PGLE to avoid TF profiler spam
        "JAX_DEFAULT_MATMUL_PRECISION": "high",
        "JAX_ENABLE_PGLE": "false",
        "TF_CPP_MIN_LOG_LEVEL": "1",
        # Persistent compilation caches
        "JAX_ENABLE_COMPILATION_CACHE": "yes",
        "JAX_COMPILATION_CACHE_DIR": "/root/.cache/jax",
        # XLA flags: disable Triton GEMM to match previous fast config; keep persistent cache
        "XLA_FLAGS": "--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false",
        # NCCL single-host perf knobs
        "NCCL_LL128_BUFFSIZE": "-2",
        "NCCL_LL_BUFFSIZE": "-2",
        "NCCL_PROTO": "SIMPLE,LL,LL128",
    })
    .run_commands(
        "python -m pip install -U pip setuptools wheel && "
        # CUDA PyTorch for GPU (pulls CUDA runtime libs)
        "python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1 && "
        # JAX stack: force-remove mismatched installs and pin compatible versions (NumPy 1.26 for numba/boltz)
        "python -m pip uninstall -y jax jaxlib jax-cuda12-plugin || true && "
        "python -m pip install --no-cache-dir jax==0.6.2 jaxlib==0.6.2 jax-cuda12-plugin==0.6.2 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && "
        "python -m pip install --no-cache-dir numpy==1.26.4 && "
        # PTX toolchain
        "python -m pip install nvidia-cuda-nvcc-cu12==12.8.93 && "
        # Core JAX ecosystem deps used by workflows
        "python -m pip install optax==0.2.4 dm-haiku>=0.0.13 flax>=0.10.2 ml-collections>=1.0.0 httpx>=0.28.1 gemmi>=0.6.0 && "
        # BindCraft/runtime deps
        "python -m pip install loguru ffmpeg-python plotly kaleido pyarrow fastparquet boto3 python-dotenv openmm mdtraj biopython freesasa scipy scikit-learn && "
        # ColabDesign for BindCraft compatibility
        "python -m pip install git+https://github.com/sokrypton/ColabDesign.git && "
        # PDBFixer required by BindCraft openmm_utils
        "python -m pip install git+https://github.com/openmm/pdbfixer.git && "
        # Git-only deps needed at runtime
        "python -m pip install git+https://github.com/escalante-bio/jablang.git && "
        "python -m pip install git+https://github.com/escalante-bio/esmj.git && "
        # Install joltz (JAX translation of Boltz) so joltz.backend is importable
        "python -m pip install git+https://github.com/adaptyvbio/joltz.git && "
        # Omit protenij here to keep numpy compatibility with boltz; but will add later if needed
        # "python -m pip install git+https://github.com/escalante-bio/protenij.git && "
        # Boltz models and tooling (required by mosaic.losses.boltz2)
        "python -m pip install git+https://github.com/jwohlwend/boltz.git && "
        # Additional deps mirrored from pyproject to avoid resolver conflicts
        "python -m pip install esm2quinox==0.1.0 ipymolstar>=0.0.9 matplotlib>=3.10.0 datasets>=2.19.0 && "
        # QP solver deps for UPGrad parity
        "python -m pip install qpsolvers==4.3.3 quadprog==0.1.12 && "
        # IgLM (public repo; non-commercial license)
        "python -m pip install git+https://github.com/Graylab/IgLM.git && "
        # Bake repo source into the image and import via sys.path (avoid pyproject resolution here)
        "git clone --depth 1 https://github.com/adaptyvbio/mosaic_workflows.git /repo && "
        # Re-assert final pins in case any dependency altered them
        "python -m pip install --no-cache-dir --upgrade jax==0.6.2 jaxlib==0.6.2 jax-cuda12-plugin==0.6.2 numpy==1.26.4 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    )
    # Ensure JAX stack is consistent after all pip installs (some deps pin older JAX)
    .run_commands(
        "python -m pip install --upgrade jax==0.6.2 jaxlib==0.6.2 && "
        "python -m pip install --upgrade jax-cuda12-plugin==0.6.2 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    )
    # Final hard pin to keep NumPy compatible with numba/boltz and avoid later upgrades
    .run_commands(
        "python -m pip install --no-cache-dir --force-reinstall numpy==1.26.4 numba==0.61.0 && "
        "python -m pip install --no-cache-dir --upgrade jax==0.6.2 jaxlib==0.6.2 jax-cuda12-plugin==0.6.2 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    )
    # Also include the local src so container runs latest edits
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/src", "/workspace/src")  # type: ignore[attr-defined]
)


app = modal.App("adaptyv-boltzcraft", image=image)


boltz_cache = modal.Volume.from_name("boltz-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("results-boltzcraft", create_if_missing=True)
af2_cache = modal.Volume.from_name("alphafold-cache", create_if_missing=True)
iglm_cache = modal.Volume.from_name("iglm-cache", create_if_missing=True)
af3_models_vol = modal.Volume.from_name("af3-models", create_if_missing=True)
af3_db_vol = modal.Volume.from_name("af3-db", create_if_missing=True)
msa_db_vol = modal.Volume.from_name("msa-db", create_if_missing=True)
dssp_vol = modal.Volume.from_name("dssp-cache", create_if_missing=True)

def _add_paths(workspace: Path):
    sys.path.append(str(workspace / "src"))


def _default_steps(total: int = 20) -> Dict[str, int]:
    """Split total steps into BD1-style phases.

    warmup: hallucination entropy + pLDDT only (stabilize backbone)
    soft:   full loss at temp=1, e_soft=0.8
    anneal: full loss with temperature decay
    """
    w = max(1, total // 5)
    s = max(1, total // 3)
    a = max(1, total - (w + s))
    return {"warmup": w, "soft": s, "anneal": a}


@app.function(volumes={"/results": results_vol})
def inspect_trajectory(results_dir: str, head: int = 10):
    import json
    p = Path(results_dir) / "trajectory.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"trajectory not found at {p}")
    shown = 0
    with open(p, "r") as f:
        for line in f:
            if shown >= int(head):
                break
            row = json.loads(line)
            aux = row.get("aux", {})
            # Some writers nest as {loss, aux:{...}}
            inner = aux.get("aux") if isinstance(aux, dict) and isinstance(aux.get("aux"), dict) else aux
            # try to surface nested boltz2 entries
            b2 = inner.get("boltz2") if isinstance(inner, dict) else None
            if isinstance(inner, dict) and isinstance(b2, list) and len(b2) >= 1 and isinstance(b2[0], dict):
                flat = {}
                for d in b2:
                    if isinstance(d, dict):
                        flat.update(d)
                inner["boltz2_flat"] = {k: float(v) if hasattr(v, "__float__") else v for k, v in flat.items()}
            print({
                "step": row.get("step"),
                "loss": float(aux.get("loss", row.get("loss", 0.0))) if isinstance(aux, dict) else row.get("loss"),
                "aux": aux,
            })
            shown += 1

@app.function(volumes={"/results": results_vol})
def inspect_best(results_dir: str, positions: str = ""):
    p = Path(results_dir) / "best_sequence.txt"
    if not p.exists():
        raise FileNotFoundError(f"best_sequence not found at {p}")
    seq = p.read_text().strip()
    pos_list = [int(x.strip()) for x in positions.split(',') if x.strip()] if positions else []
    aa = {i: (seq[i] if 0 <= i < len(seq) else None) for i in pos_list}
    print({"length": len(seq), "positions": aa, "sequence_head": seq[:min(80, len(seq))]})


@app.function(volumes={"/results": results_vol})
def inspect_motif_positions(results_dir: str, pdb_path: str, pdb_residues: str, motif_chain_id: str = "A"):
    from importlib.machinery import SourceFileLoader
    import json
    import os as _os
    _os.environ["JAX_PLATFORMS"] = "cpu"
    _os.environ["JAX_DISABLE_JAX_PLUGIN_DISCOVERY"] = "1"
    import numpy as np
    import jax
    import jax.numpy as jnp
    # Ensure local src is on path
    local_src = Path("/workspace/src")
    if local_src.exists():
        sys.path.insert(0, str(local_src))
    repo_src = Path("/repo/src")
    if repo_src.exists():
        sys.path.append(str(repo_src))
    from mosaic.losses.boltz2 import load_boltz2, load_features_and_structure_writer, set_binder_sequence, Boltz2Output  # type: ignore[import-not-found]

    # Read best sequence
    res_dir = Path(results_dir)
    seq_path = res_dir / "best_sequence.txt"
    if not seq_path.exists():
        raise FileNotFoundError(f"best_sequence.txt not found in {res_dir}")
    seq = seq_path.read_text().strip()
    L = len(seq)
    vocab = "ARNDCQEGHILKMFPSTWYV"
    onehot = np.zeros((L, 20), dtype=np.float32)
    for i, ch in enumerate(seq):
        if ch in vocab:
            onehot[i, vocab.index(ch)] = 1.0
    # Build boltz2 features for binder length L
    def _yaml(binder_len: int) -> str:
        lines = ["version: 1", "sequences:"]
        lines.append(f"  - protein:\n      id: A\n      sequence: {'X'*binder_len}\n      msa: empty")
        lines.append("  - ligand:\n      id: L\n      smiles: 'OCCOC(=O)c1ccc(cc1)C(=O)O'")
        return "\n".join(lines)

    joltz2 = load_boltz2()
    features, _ = load_features_and_structure_writer(_yaml(L), cache=Path(os.environ.get("BOLTZ_CACHE", "/root/.boltz")).expanduser())
    features = set_binder_sequence(jnp.asarray(onehot), features)
    out = Boltz2Output(joltz2=joltz2, features=features, deterministic=True, key=jax.random.PRNGKey(0))

    # Predicted CA coordinates
    ca = np.array(out.backbone_coordinates[:L, 1, :])

    # Template motif CA from PDB
    from importlib import import_module
    gemmi = import_module("gemmi")
    p = Path(pdb_path)
    if not p.exists():
        raise FileNotFoundError(f"PDB path does not exist: {p}")
    st = gemmi.read_structure(str(p))
    chain = st[0][motif_chain_id]
    resnums = [int(x.strip()) for x in pdb_residues.split(',') if x.strip()]
    tmpl = []
    for rn in resnums:
        res = next(r for r in chain if r.seqid.num == int(rn))
        ca_atom = next((a for a in res if a.name == "CA"), None)
        if ca_atom is None:
            raise ValueError(f"Missing CA for residue {rn}")
        tmpl.append([ca_atom.pos.x, ca_atom.pos.y, ca_atom.pos.z])
    tmpl = np.asarray(tmpl, dtype=np.float32)  # type: ignore[assignment]

    # Helper: pairwise distance matrices
    def pdist(x):
        diff = x[:, None, :] - x[None, :, :]
        return np.sqrt((diff * diff).sum(-1))

    Dp = pdist(ca)  # [L,L]
    Dt = pdist(tmpl)  # [K,K]
    K = Dt.shape[0]

    # Greedy assignment matching pairwise distances to already-mapped residues
    mapped = {}
    used = set()

    # Seed: choose (a,i) minimizing mean min_j |Dp[i,j] - Dt[a,b]|
    seed_scores = np.full((K, L), np.inf, dtype=np.float32)
    for a in range(K):
        row_t = Dt[a]
        for i in range(L):
            diff = np.abs(Dp[i, :] - row_t[:, None]).min(axis=1)  # min over binder positions per motif partner
            seed_scores[a, i] = diff.mean()
    a0, i0 = np.unravel_index(np.argmin(seed_scores), seed_scores.shape)
    mapped[a0] = i0
    used.add(i0)

    while len(mapped) < K:
        best_a: int = -1
        best_i: int = -1
        best_s: float = 1e9
        for a in range(K):
            if a in mapped:
                continue
            for i in range(L):
                if i in used:
                    continue
                # score against already mapped residues
                errs = []
                for c, ic in mapped.items():
                    dij = float(Dp[i, ic])
                    dt = float(Dt[a, c])
                    errs.append(abs(dij - dt))
                s = float(np.mean(errs)) if errs else 0.0
                if s < best_s:
                    best_a, best_i, best_s = a, i, s
        if best_a >= 0 and best_i >= 0:
            mapped[best_a] = best_i
            used.add(best_i)

    # Order positions by template index
    motif_positions = [int(mapped[a]) for a in range(K)]
    print({"motif_positions": motif_positions})

@app.function(
    gpu="H100",
    timeout=3 * 60 * 60,
    volumes={"/root/.boltz": boltz_cache, "/results": results_vol, "/repo/params": af2_cache},
    secrets=[modal.Secret.from_name("github-token")],
)
def run_mhetase(
    *,
    binder_len: int = 20,
    motif_positions: Dict[str, Any] = {"ser": 3, "his": 10, "asp": 15},
    ligand: Dict[str, Any] = {"enzyme_chain": "A", "ligand_chain": "L", "smiles": "OCCOC(=O)c1ccc(cc1)C(=O)O"},
    total_steps: int = 20,
    seed: int = 0,
    pdb_path: str | None = None,
    pdb_bytes: bytes | None = None,
    pdb_residues: str | None = None,
    motif_chain_id: str = "A",
    use_af2: bool = False,
    af2_num_recycles: int = 1,
    af2_model_idx: int | None = None,
    # weights and fixing knobs
    w_contact: float = 1.0,
    w_motif_cce: float = 1.0,
    w_motif_rmsd: float = 0.2,
    w_sc_rmsd: float = 0.1,
    w_plddt: float = 0.0,
    w_pae: float = 0.0,
    w_rg: float = 0.0,
    w_seq_ent: float = 0.2,
    w_cat_dist: float = 0.1,
    w_helix: float = 0.0,
    w_fape: float = 0.0,
    auto_motif: bool = False,
    freeze_supervised_positions: bool = False,
    fix_supervised_identities: str | None = None,
    use_jd: bool = False,
    jd_agg: str = "pcgrad",
    jd_lr_scale: float = 0.5,
):
    """Launch the MHETase scaffolding workflow on Modal with a small budget.

    Parameters
    ----------
    binder_len : int
        Length of the designed enzyme chain.
    motif_positions : dict
        Positions for the catalytic triad (0-indexed): {"ser": i, "his": j, "asp": k}.
    ligand : dict
        Ligand spec for boltz2 predictor: keys include enzyme_chain, ligand_chain, smiles or ccd.
    total_steps : int
        Total optimization steps across warmup/design/refine.
    seed : int
        Random seed.
    """

    # Ensure Boltz cache uses a persisted Modal volume
    os.environ.setdefault("BOLTZ_CACHE", "/root/.boltz")
    Path(os.environ["BOLTZ_CACHE"]).mkdir(parents=True, exist_ok=True)


    # Prefer mounted local source (latest edits), fall back to baked repo
    local_src = Path("/workspace/src")
    if local_src.exists():
        sys.path.insert(0, str(local_src))
    repo_src = Path("/repo/src")
    if repo_src.exists():
        sys.path.append(str(repo_src))
    # Load modules directly to avoid package-level imports
    from importlib.machinery import SourceFileLoader
    dwf = SourceFileLoader("design", "/workspace/src/mosaic_workflows/design.py").load_module()  # type: ignore
    # Choose JD variant if requested
    scaffold_path = "/workspace/src/mosaic_workflows/mhetase_scaffold_jd.py" if bool(use_jd) else "/workspace/src/mosaic_workflows/mhetase_scaffold.py"
    ms = SourceFileLoader("mhetase_scaffold_jd" if bool(use_jd) else "mhetase_scaffold", scaffold_path).load_module()  # type: ignore

    tmol_context = {"ligand": ligand}
    # Determine supervised positions order. If 5 motif residues are provided, order to match
    # [Ser, Asp, His, Gly, Glu] which corresponds to the expected PDB residue order
    # (e.g., 225,492,528,132,226). Otherwise, fall back to triad [Ser, His, Asp].
    order_keys = ["ser", "his", "asp"]
    if pdb_residues is not None:
        num_res = len([x for x in pdb_residues.split(',') if x.strip()])
        if num_res == 5:
            order_keys = ["ser", "asp", "his", "gly", "glu"]
    supervised_positions: tuple[int, ...] = ()
    motif_roles_labels: tuple[str, ...] | None = None
    if not bool(auto_motif):
        supervised_positions_list = [
            int(motif_positions[k]) for k in order_keys if (k in motif_positions and motif_positions[k] is not None)
        ]
        supervised_positions = tuple(supervised_positions_list)
        # preserve mapping of labels
        motif_roles_labels = tuple(order_keys)
    # Optional motif PDB inputs
    # If PDB bytes are provided, write to a temp file in the container
    motif_pdb_path = pdb_path
    if pdb_bytes is not None:
        tmp_pdb = Path("/tmp/motif_input.pdb")
        tmp_pdb.write_bytes(pdb_bytes)
        motif_pdb_path = str(tmp_pdb)
    motif_resnums = tuple(int(x.strip()) for x in pdb_residues.split(',') if x.strip()) if pdb_residues else None

    # If AF2 is requested, require params are available under /repo/params
    if bool(use_af2):
        params_dir = Path("/repo") / "params"
        key_file = params_dir / "params_model_1.npz"
        if not key_file.exists():
            raise FileNotFoundError("AF2 params not found under /repo/params; run download_params.sh beforehand.")

    kwargs = dict(
        binder_len=binder_len,
        tmol_context=tmol_context,
        use_af2=bool(use_af2),
        af2_num_recycles=int(af2_num_recycles),
        af2_params_dir="/repo",
        af2_model_idx=(int(af2_model_idx) if af2_model_idx is not None else None),
        steps=int(total_steps),
        lr=0.5 if bool(use_af2) else 0.1,
        w_contact=float(w_contact),
        w_motif_cce=float(w_motif_cce),
        w_motif_rmsd=float(w_motif_rmsd),
        w_sc_rmsd=float(w_sc_rmsd),
        w_plddt=float(w_plddt),
        w_pae=float(w_pae),
        w_rg=float(w_rg),
        w_seq_ent=float(w_seq_ent),
        w_cat_dist=float(w_cat_dist),
        w_helix=float(w_helix),
        w_fape=float(w_fape),
        freeze_supervised_positions=bool(freeze_supervised_positions),
        # pass raw string so downstream parser can split correctly
        fix_supervised_identities=fix_supervised_identities,
    )
    if supervised_positions:
        kwargs["supervised_positions"] = supervised_positions
        if motif_roles_labels is not None:
            kwargs["motif_roles"] = motif_roles_labels
    if motif_pdb_path:
        kwargs["motif_pdb_path"] = motif_pdb_path
    if motif_chain_id:
        kwargs["motif_chain_id"] = motif_chain_id
    if motif_resnums:
        kwargs["motif_resnums"] = motif_resnums

    # Pass JD knobs if JD is enabled
    if bool(use_jd):
        kwargs["jd_agg"] = jd_agg
        kwargs["jd_lr_scale"] = float(jd_lr_scale)
    wf = ms.make_workflow(**kwargs)

    wf["seed"] = int(seed)
    wf["initial_x"] = (np := __import__("numpy")).random.randn(binder_len, 20).astype(np.float32) * 0.1
    out = dwf.run_workflow(wf)

    # Save outputs to results volume
    import json, time
    run_id = f"mhetase_{int(time.time())}_seed{seed}_len{binder_len}"
    out_dir = Path("/results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save best sequence and best_x
    (out_dir / "best_sequence.txt").write_text(str(out.get("best_sequence", "")))
    np.save(out_dir / "best_x.npy", out.get("best_x"))

    # Save raw trajectory: just step and aux (which includes losses)
    traj = out.get("trajectory") or []
    with open(out_dir / "trajectory.jsonl", "w") as f:
        for rec in traj:
            row = {
                "step": int(rec.get("step", 0)),
                "aux": rec.get("aux", {}),
            }
            f.write(json.dumps(row, default=lambda o: float(o) if hasattr(o, "item") else None) + "\n")

    # Final structure export removed in simplified pipeline

    print({"results_dir": str(out_dir)})


@app.function(
    gpu="H100",
    timeout=3 * 60 * 60,
    volumes={
        "/results": results_vol,
        "/repo/params": af2_cache,
        "/root/.cache": iglm_cache,
        "/vol/af3_models": af3_models_vol,
        "/vol/af3_db": af3_db_vol,
        "/vol/msa_db": msa_db_vol,
        "/vol/dssp": dssp_vol,
    },
)
def run_germinal_pdl1(
    *,
    total_steps_logits: int = 65,
    total_steps_softmax: int = 35,
    total_steps_semigreedy: int = 10,
):
    import os as _os
    from pathlib import Path as _Path
    import subprocess as _sp
    import sys as _sys
    _os.environ.setdefault("JAX_PLATFORMS", "cuda")
    _os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "high")
    _os.environ["JAX_ENABLE_PGLE"] = "false"
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    _os.environ.pop("JAX_COMPILATION_CACHE_EXPECT_PGLE", None)
    _os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/root/.cache/jax")
    _os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_triton_gemm_any=True --xla_gpu_enable_command_buffer=''")
    _os.environ.setdefault("NCCL_LL128_BUFFSIZE", "-2")
    _os.environ.setdefault("NCCL_LL_BUFFSIZE", "-2")
    _os.environ.setdefault("NCCL_PROTO", "SIMPLE,LL,LL128")
    # Ensure HF/torch caches persist under /root/.cache volume
    _os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    _os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface")
    # DSSP cache/bin if needed by external tools
    _os.environ.setdefault("DSSP_CACHE", "/vol/dssp")
    # Ensure local src and baked repo are importable
    local_src = _Path("/workspace/src")
    if local_src.exists():
        _sys.path.insert(0, str(local_src))
    repo_src = _Path("/repo/src")
    if repo_src.exists():
        _sys.path.append(str(repo_src))

    # Require AF2 params under /repo/params
    params_dir = _Path("/repo") / "params"
    key_file = params_dir / "params_model_1.npz"
    if not key_file.exists():
        raise FileNotFoundError("AF2 params not found under /repo/params; run download_params.sh beforehand.")

    # Clone Germinal to fetch PD-L1 PDB and configs
    g_dir = _Path("/tmp/germinal")
    if not g_dir.exists():
        _sp.run(["git", "clone", "--depth", "1", "https://github.com/SantiagoMille/germinal.git", str(g_dir)], check=True)
    # Ensure 'germinal' package is importable
    if str(g_dir) not in _sys.path:
        _sys.path.insert(0, str(g_dir))
    # Ensure 'germinal' package is importable
    if str(g_dir) not in _sys.path:
        _sys.path.insert(0, str(g_dir))

    # Parse PD-L1 config
    import yaml  # type: ignore
    with open(g_dir / "configs/target/pdl1.yaml", "r") as f:
        tgt = yaml.safe_load(f)
    with open(g_dir / "configs/run/vhh.yaml", "r") as f:
        vhh = yaml.safe_load(f)

    target_pdb_path = str(g_dir / tgt["target_pdb_path"])  # pdbs/pdl1.pdb
    target_chain_id = str(tgt.get("target_chain", "A"))
    binder_len = int(tgt.get("length", 133))

    cdr_lengths = list(vhh.get("cdr_lengths", [11, 8, 18]))
    fw_lengths = list(vhh.get("fw_lengths", [25, 17, 38, 14]))

    # Build CDR and framework positions
    # Layout: FW1, CDR1, FW2, CDR2, FW3, CDR3, FW4
    segments = [
        ("fw1", fw_lengths[0]),
        ("cdr1", cdr_lengths[0]),
        ("fw2", fw_lengths[1]),
        ("cdr2", cdr_lengths[1]),
        ("fw3", fw_lengths[2]),
        ("cdr3", cdr_lengths[2]),
        ("fw4", fw_lengths[3]),
    ]
    pos = 0
    cdr_positions: list[int] = []
    framework_positions: list[int] = []
    for name, length in segments:
        idxs = list(range(pos, pos + int(length)))
        if name.startswith("cdr"):
            cdr_positions.extend(idxs)
        else:
            framework_positions.extend(idxs)
        pos += int(length)
    # Safety clip to binder_len
    cdr_positions = [i for i in cdr_positions if 0 <= i < binder_len]
    framework_positions = [i for i in framework_positions if 0 <= i < binder_len]

    # Parse epitope positions from "A37,A39,A41,..." to 0-based indices
    import gemmi  # type: ignore
    st = gemmi.read_structure(target_pdb_path)
    chain = st[0][target_chain_id]
    # map seqid.num -> 0-based index in chain order
    rn_to_idx = {}
    for i, r in enumerate(chain):
        rn_to_idx[int(r.seqid.num)] = i
    raw_hotspots = str(tgt.get("target_hotspots", "")).split(",")
    epitope_idx = []
    for tok in raw_hotspots:
        tok = tok.strip()
        if not tok:
            continue
        # Tokens like "A37" or "37"; extract number
        num_str = "".join(ch for ch in tok if ch.isdigit())
        if num_str:
            rn = int(num_str)
            if rn in rn_to_idx:
                epitope_idx.append(int(rn_to_idx[rn]))

    # Framework sequence for bias: use provided nb scaffold if present
    fw_seq = "G" * binder_len
    nb_pdb = g_dir / "pdbs/nb.pdb"
    if nb_pdb.exists():
        st_nb = gemmi.read_structure(str(nb_pdb))
        ch_nb = st_nb[0]["A"] if "A" in [c.name for c in st_nb[0]] else st_nb[0][0]
        fw_seq = gemmi.one_letter_code([r.name for r in ch_nb])
        if len(fw_seq) < binder_len:
            fw_seq = fw_seq + ("G" * (binder_len - len(fw_seq)))
        fw_seq = fw_seq[:binder_len]

    # Import Mosaic workflow and run
    import importlib as _importlib
    if str(local_src) not in _sys.path:
        _sys.path.insert(0, str(local_src))
    design_mod = _importlib.import_module("mosaic_workflows.design")
    germinal_mod = _importlib.import_module("mosaic_workflows.germinal")

    wf = germinal_mod.make_workflow(
        binder_len=binder_len,
        target_pdb_path=target_pdb_path,
        target_chain_id=target_chain_id,
        target_hotspots=tuple(epitope_idx),
        cdr_positions=tuple(cdr_positions),
        framework_positions=tuple(framework_positions),
        framework_sequence=str(fw_seq),
        af2_params_dir="/repo",
        af2_num_recycles=int(vhh.get("num_recycles_design", 3)),
        w_plddt=float(vhh.get("weights_plddt", 1.0)),
        w_iptm=float(vhh.get("weights_iptm", 0.7)),
        w_pae_bt=float(vhh.get("weights_pae_inter", 0.5)),
        w_intra_con=float(vhh.get("weights_con_intra", 0.1)),
        w_rg=float(vhh.get("weights_rg", 0.1)),
        w_dgram_cce=float(vhh.get("dgram_cce", 0.01)),
        w_fw_penalty=0.5,
        w_cdr_helix_suppress=float(vhh.get("weights_helix", 0.1)),
        w_cdr_beta_suppress=float(vhh.get("weights_beta", 0.1)),
        steps_logits=int(total_steps_logits),
        steps_softmax=int(total_steps_softmax),
        steps_semigreedy=int(total_steps_semigreedy),
        framework_bias=float(vhh.get("bias_redesign", 10)),
        framework_contact_offset=float(vhh.get("framework_contact_offset", 1.0)),
        lr=float(vhh.get("learning_rate", 0.1)),
        plddt_thr=float(vhh.get("plddt_threshold", 0.82)),
        iptm_thr=float(vhh.get("i_ptm_threshold", 0.68)),
        ipae_thr=float(vhh.get("i_pae_threshold", 0.27)),
        seq_entropy_thr=float(vhh.get("seq_entropy_threshold", 0.10)),
        grad_merge_method=str(vhh.get("grad_merge_method", "pcgrad")),
        omit_aas=str(vhh.get("omit_AAs", "C")),
        seq_init_mode=(vhh.get("seq_init_mode", ["gumbel"]) or ["gumbel"])[0],
    )

    wf["seed"] = 0
    wf["initial_x"] = (np := __import__("numpy")).random.randn(binder_len, 20).astype(np.float32) * 0.1
    # Prepare results dir and stream path before running
    import json as _json, time as _time
    run_id = f"germinal_pdl1_{int(_time.time())}_len{binder_len}"
    out_dir = _Path("/results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stream trajectory to results
    wf["trajectory_path"] = str(out_dir / "trajectory.jsonl")
    out = design_mod.run_workflow(wf)
    (out_dir / "best_sequence.txt").write_text(str(out.get("best_sequence", "")))
    np.save(out_dir / "best_x.npy", out.get("best_x"))
    traj = out.get("trajectory") or []
    with open(out_dir / "trajectory.jsonl", "w") as f:
        for rec in traj:
            row = {"step": int(rec.get("step", 0)), "aux": rec.get("aux", {})}
            f.write(_json.dumps(row, default=lambda o: float(o) if hasattr(o, "item") else None) + "\n")
    print({"results_dir": str(out_dir)})
@app.function(
    gpu="H100",
    timeout=6 * 60 * 60,
    volumes={
        "/results": results_vol,
        "/repo/params": af2_cache,
        "/root/.cache": iglm_cache,
        "/vol/af3_models": af3_models_vol,
        "/vol/af3_db": af3_db_vol,
        "/vol/msa_db": msa_db_vol,
        "/vol/dssp": dssp_vol,
    },
    mounts=[
        modal.Mount.from_local_dir(
            "/Users/tudorcotet/Documents/Adaptyv/adaptyv_bindcraft/src/BindCraft",
            remote_path="/root/BindCraft",
        ),
        modal.Mount.from_local_dir(
            "/Users/tudorcotet/Documents/Adaptyv/adaptyv_bindcraft/utilities",
            remote_path="/root/utilities",
        ),
    ],
    secrets=[modal.Secret.from_name("github-token")],
)
def run_germinal_full():
    import os as _os
    from pathlib import Path as _Path
    import subprocess as _sp
    import sys as _sys
    _os.environ.setdefault("JAX_PLATFORMS", "cuda")
    _os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "high")
    _os.environ["JAX_ENABLE_PGLE"] = "false"
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    _os.environ.pop("JAX_COMPILATION_CACHE_EXPECT_PGLE", None)
    _os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/root/.cache/jax")
    _os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_persistent_cache_dir=/root/.cache/xla")
    _os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_persistent_cache_dir=/root/.cache/xla")
    _os.environ.setdefault("NCCL_LL128_BUFFSIZE", "-2")
    _os.environ.setdefault("NCCL_LL_BUFFSIZE", "-2")
    _os.environ.setdefault("NCCL_PROTO", "SIMPLE,LL,LL128")
    _os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    _os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface")

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

    # Clone Germinal (external)
    g_dir = _Path("/tmp/germinal")
    if not g_dir.exists():
        _sp.run(["git", "clone", "--depth", "1", "https://github.com/SantiagoMille/germinal.git", str(g_dir)], check=True)
    # Ensure 'germinal' package is importable
    if str(g_dir) not in _sys.path:
        _sys.path.insert(0, str(g_dir))
    # Seed AF3 models volume if params exist in repo
    params_dir = g_dir / "params"
    if params_dir.exists():
        # Copy once if volume appears empty
        if not any((_Path("/vol/af3_models")).iterdir()):
            _sp.run(["bash", "-lc", f"cp -r {params_dir}/* /vol/af3_models/ || true"], check=False)

    # Build configs
    import yaml as _yaml
    vhh = {}
    with open(g_dir / "configs/run/vhh.yaml", "r") as f:
        vhh = _yaml.safe_load(f) or {}
    with open(g_dir / "configs/target/pdl1.yaml", "r") as f:
        tgt = _yaml.safe_load(f) or {}

    target_pdb_path = str(g_dir / tgt["target_pdb_path"])  # pdbs/pdl1.pdb
    # Germinal-compatible wrapper
    from mosaic_workflows.bindcraft_compat import run_germinal_compat

    design_path = "/results/germinal_full"
    _Path(design_path).mkdir(parents=True, exist_ok=True)

    target_settings = {
        "starting_pdb": target_pdb_path,
        "chains": tgt.get("target_chain", "A"),
        # For AF3 validation (needed when running AF3). If absent, AF3 step will compute target seq from pdb internally where possible.
        "target_seq": "",
    }
    af3_settings = {
        # Paths mounted as volumes
        "af_params_dir": "/repo",
        "af3_repo_path": "/vol/af3_repo",  # optional; if you bind a local AF3 repo, ensure it's present on this volume
        "af3_sif_path": "/vol/af3_models/alphafold3.sif",
        "af3_model_dir": "/vol/af3_models",
        "af3_db_dir": "/vol/af3_db",
        "msa_db_dir": "/vol/msa_db",
        "use_metagenomic_db": False,
    }
    # If AF3 SIF missing, we’ll skip AF3; run_germinal_compat will still design + redesign
    if not _Path(af3_settings["af3_sif_path"]).exists():
        # Log and proceed without AF3
        print({"warning": "AF3 SIF not found; AF3 validation will be skipped", "sif_path": af3_settings["af3_sif_path"]})

    # Stream trajectory to results dir used by bindcraft_compat emitter
    out = run_germinal_compat(
        design_path=design_path,
        target_settings=target_settings,
        vhh_config=vhh,
        af3_settings=af3_settings,
        max_trajectories=10,
        runtime_seed=0,
    )
    print({"results_dir": design_path, "status": "completed"})

@app.local_entrypoint()
def main(
    workflow: str = "mhetase",
    binder_len: int = 20,
    ser: int = 3,
    his: int = 10,
    asp: int = 15,
    gly: int | None = None,
    glu: int | None = None,
    oxyanion: str | None = None,
    ligand_smiles: str = "OCCOC(=O)c1ccc(cc1)C(=O)O",
    total_steps: int = 20,
    seed: int = 0,
    pdb_path: str | None = None,
    pdb_residues: str | None = None,
    pdb_oxyanion_residues: str | None = None,
    random_len_min: int = 0,
    random_len_max: int = 0,
    use_af2: bool = False,
    af2_num_recycles: int = 1,
    af2_model_idx: int | None = None,
    results_dir: str | None = None,
    head: int = 10,
    positions: str = "",
    use_jd: bool = False,
    jd_agg: str = "pcgrad",
    jd_lr_scale: float = 0.5,
    # weights and fixing knobs for convenience via local entrypoint
    w_contact: float = 1.0,
    w_motif_cce: float = 1.0,
    w_motif_rmsd: float = 1.0,
    w_sc_rmsd: float = 0.1,
    w_plddt: float = 0.0,
    w_pae: float = 0.0,
    w_rg: float = 0.0,
    w_seq_ent: float = 0.1,
    w_cat_dist: float = 0.1,
    w_helix: float = 0.0,
    w_fape: float = 1.0,
    auto_motif: bool = False,
    freeze_supervised_positions: bool = False,
    fix_supervised_identities: str | None = None,
    # utility
    inspect_results_dir: str | None = None,
):
    """Local entrypoint to kick off a workflow on Modal.

    Examples (local):
      modal run scripts.modal_app --workflow mhetase --binder-len 20 --ser 3 --his 10 --asp 15 --total-steps 20
    """
    if workflow == "mhetase":
        ligand = {"enzyme_chain": "A", "ligand_chain": "L", "smiles": ligand_smiles}
        motif_pos: dict[str, int] = {"ser": ser, "his": his, "asp": asp}
        if gly is not None:
            motif_pos["gly"] = int(gly)
        if glu is not None:
            motif_pos["glu"] = int(glu)
        pdb_bytes = None
        if pdb_path and Path(pdb_path).exists():
            pdb_bytes = Path(pdb_path).read_bytes()
        run_mhetase.remote(
            binder_len=binder_len,
            motif_positions=motif_pos,
            ligand=ligand,
            total_steps=total_steps,
            seed=seed,
            pdb_path=pdb_path,
            pdb_bytes=pdb_bytes,
            pdb_residues=pdb_residues,
            motif_chain_id="A",
            use_af2=use_af2,
            af2_num_recycles=af2_num_recycles,
            af2_model_idx=af2_model_idx,
            use_jd=use_jd,
            jd_agg=jd_agg,
            jd_lr_scale=jd_lr_scale,
            w_contact=w_contact,
            w_motif_cce=w_motif_cce,
            w_motif_rmsd=w_motif_rmsd,
            w_sc_rmsd=w_sc_rmsd,
            w_plddt=w_plddt,
            w_pae=w_pae,
            w_rg=w_rg,
            w_seq_ent=w_seq_ent,
            w_cat_dist=w_cat_dist,
            w_helix=w_helix,
            w_fape=w_fape,
            auto_motif=auto_motif,
            freeze_supervised_positions=freeze_supervised_positions,
            fix_supervised_identities=fix_supervised_identities,
        )
    elif workflow == "germinal_pdl1":
        run_germinal_pdl1.remote()
    elif workflow == "germinal_full":
        run_germinal_full.remote()
    elif workflow == "inspect":
        if results_dir is None:
            raise ValueError("--results-dir is required for inspect workflow")
        inspect_trajectory.remote(results_dir=results_dir, head=int(head))
    elif workflow == "inspect_best":
        if results_dir is None:
            raise ValueError("--results-dir is required for inspect_best workflow")
        inspect_best.remote(results_dir=results_dir, positions=positions)
    elif workflow == "motif_positions":
        if inspect_results_dir is None or pdb_path is None or pdb_residues is None:
            raise ValueError("--inspect-results-dir, --pdb-path and --pdb-residues are required")
        inspect_motif_positions.remote(results_dir=inspect_results_dir, pdb_path=pdb_path, pdb_residues=pdb_residues, motif_chain_id="A")
    else:
        raise ValueError(f"Unknown workflow: {workflow}")


