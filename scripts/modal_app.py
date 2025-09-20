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
        "JAX_PLATFORMS": "cuda"
    })
    .run_commands(
        "python -m pip install -U pip setuptools wheel && "
        # CUDA PyTorch for GPU (pulls CUDA runtime libs)
        "python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1 && "
        # JAX core and CUDA plugin (pin compatible versions)
        "python -m pip install --upgrade jax==0.7.1 && "
        "python -m pip install --upgrade jax-cuda12-plugin==0.7.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && "
        # PTX toolchain
        "python -m pip install nvidia-cuda-nvcc-cu12==12.8.93 && "
        # Core JAX ecosystem deps used by workflows
        "python -m pip install optax==0.2.4 dm-haiku>=0.0.13 flax>=0.10.2 ml-collections>=1.0.0 httpx>=0.28.1 gemmi>=0.6.0 && "
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
        # Bake repo source into the image and import via sys.path (avoid pyproject resolution here)
        "git clone --depth 1 https://github.com/adaptyvbio/mosaic_workflows.git /repo"
    )
    # Also include the local src so container runs latest edits
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/src", "/workspace/src")
)


app = modal.App("adaptyv-boltzcraft", image=image)


boltz_cache = modal.Volume.from_name("boltz-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("results-boltzcraft", create_if_missing=True)
af2_cache = modal.Volume.from_name("alphafold-cache", create_if_missing=True)

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
        print({"error": f"trajectory not found at {p}"})
        return
    shown = 0
    with open(p, "r") as f:
        for line in f:
            if shown >= int(head):
                break
            try:
                row = json.loads(line)
            except Exception:
                continue
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
        print({"error": f"best_sequence not found at {p}"})
        return
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
    from mosaic.losses.boltz2 import load_boltz2, load_features_and_structure_writer, set_binder_sequence, Boltz2Output

    # Read best sequence
    res_dir = Path(results_dir)
    seq_path = res_dir / "best_sequence.txt"
    if not seq_path.exists():
        print({"error": f"best_sequence.txt not found in {res_dir}"})
        return
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
        # Try to fetch from RCSB if given like 6QZ4.pdb or a path ending with such
        try:
            code = p.name.split(".")[0]
            import httpx
            resp = httpx.get(f"https://files.rcsb.org/download/{code}.pdb", timeout=30.0)
            resp.raise_for_status()
            tmp_p = Path("/tmp") / f"{code}.pdb"
            tmp_p.write_bytes(resp.content)
            p = tmp_p
        except Exception as e:
            print({"error": f"Failed to fetch PDB: {e}"})
            return
    st = gemmi.read_structure(str(p))
    chain = st[0][motif_chain_id]
    resnums = [int(x.strip()) for x in pdb_residues.split(',') if x.strip()]
    tmpl = []
    for rn in resnums:
        res = next(r for r in chain if r.seqid.num == int(rn))
        ca_atom = next((a for a in res if a.name == "CA"), None)
        if ca_atom is None:
            print({"error": f"Missing CA for residue {rn}"})
            return
        tmpl.append([ca_atom.pos.x, ca_atom.pos.y, ca_atom.pos.z])
    tmpl = np.asarray(tmpl, dtype=np.float32)

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
        best = (None, None, 1e9)
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
                if s < best[2]:
                    best = (a, i, s)
        a_sel, i_sel, _ = best
        mapped[a_sel] = i_sel
        used.add(i_sel)

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
    # weights and fixing knobs
    w_contact: float = 1.0,
    w_motif_cce: float = 1.0,
    w_motif_rmsd: float = 0.0,
    w_plddt: float = 0.0,
    w_pae: float = 0.0,
    w_rg: float = 0.0,
    w_seq_ent: float = 0.1,
    w_cat_dist: float = 0.0,
    auto_motif: bool = False,
    freeze_supervised_positions: bool = False,
    fix_supervised_identities: str | None = None,
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
    ms = SourceFileLoader("mhetase_scaffold", "/workspace/src/mosaic_workflows/mhetase_scaffold.py").load_module()  # type: ignore

    tmol_context = {"ligand": ligand}
    # Determine supervised positions order. If 5 motif residues are provided, order to match
    # [Ser, Asp, His, Gly, Glu] which corresponds to the expected PDB residue order
    # (e.g., 225,492,528,132,226). Otherwise, fall back to triad [Ser, His, Asp].
    order_keys = ["ser", "his", "asp"]
    if pdb_residues is not None:
        try:
            num_res = len([x for x in pdb_residues.split(',') if x.strip()])
            if num_res == 5:
                order_keys = ["ser", "asp", "his", "gly", "glu"]
        except Exception:
            pass
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

    # If AF2 is requested, ensure params are available under /repo/params
    if bool(use_af2):
        params_dir = Path("/repo") / "params"
        key_file = params_dir / "params_model_1.npz"
        if not key_file.exists():
            script = Path("/repo") / "download_params.sh"
            if script.exists():
                subprocess.run(["bash", str(script), "/repo"], check=True)

    kwargs = dict(
        binder_len=binder_len,
        tmol_context=tmol_context,
        use_af2=bool(use_af2),
        af2_num_recycles=int(af2_num_recycles),
        af2_params_dir="/repo",
        steps=int(total_steps),
        lr=0.5 if bool(use_af2) else 0.1,
        w_contact=float(w_contact),
        w_motif_cce=float(w_motif_cce),
        w_motif_rmsd=float(w_motif_rmsd),
        w_plddt=float(w_plddt),
        w_pae=float(w_pae),
        w_rg=float(w_rg),
        w_seq_ent=float(w_seq_ent),
        w_cat_dist=float(w_cat_dist),
        freeze_supervised_positions=bool(freeze_supervised_positions),
        fix_supervised_identities=tuple(x.strip() for x in fix_supervised_identities.split(',')) if fix_supervised_identities else None,
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
    results_dir: str | None = None,
    head: int = 10,
    # weights and fixing knobs for convenience via local entrypoint
    w_contact: float = 1.0,
    w_motif_cce: float = 1.0,
    w_motif_rmsd: float = 0.0,
    w_plddt: float = 0.0,
    w_pae: float = 0.0,
    w_rg: float = 0.0,
    w_seq_ent: float = 0.1,
    w_cat_dist: float = 0.0,
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
        # Optional random binder length sampling
        if int(random_len_max) > int(random_len_min) and int(random_len_min) > 0:
            import random
            binder_len = random.randint(int(random_len_min), int(random_len_max))
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
            w_contact=w_contact,
            w_motif_cce=w_motif_cce,
            w_motif_rmsd=w_motif_rmsd,
            w_plddt=w_plddt,
            w_pae=w_pae,
            w_rg=w_rg,
            w_seq_ent=w_seq_ent,
            w_cat_dist=w_cat_dist,
            auto_motif=auto_motif,
            freeze_supervised_positions=freeze_supervised_positions,
            fix_supervised_identities=fix_supervised_identities,
        )
    elif workflow == "inspect":
        if results_dir is None:
            raise ValueError("--results-dir is required for inspect workflow")
        inspect_trajectory.remote(results_dir=results_dir, head=int(head))
    elif workflow == "motif_positions":
        if inspect_results_dir is None or pdb_path is None or pdb_residues is None:
            raise ValueError("--inspect-results-dir, --pdb-path and --pdb-residues are required")
        inspect_motif_positions.remote(results_dir=inspect_results_dir, pdb_path=pdb_path, pdb_residues=pdb_residues, motif_chain_id="A")
    else:
        raise ValueError(f"Unknown workflow: {workflow}")


