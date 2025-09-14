import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict

import modal


# -------- Modal configuration (reuse current image setup) --------
image = (
    modal.Image.debian_slim(python_version="3.12.0")
    .apt_install("git", "aria2")
    .env({
        "BOLTZ_CACHE": "/root/.boltz",
        "JAX_PLATFORMS": "cuda"
    })
    .run_commands(
        "python -m pip install -U pip setuptools wheel && "
        "python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1 && "
        "python -m pip install --upgrade jax==0.7.1 && "
        "python -m pip install --upgrade jax-cuda12-plugin==0.7.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && "
        "python -m pip install nvidia-cuda-nvcc-cu12==12.8.93 && "
        "python -m pip install optax==0.2.4 dm-haiku>=0.0.13 flax>=0.10.2 ml-collections>=1.0.0 httpx>=0.28.1 gemmi>=0.6.0 && "
        "python -m pip install git+https://github.com/escalante-bio/jablang.git && "
        "python -m pip install git+https://github.com/escalante-bio/esmj.git && "
        "python -m pip install git+https://github.com/adaptyvbio/joltz.git && "
        "python -m pip install git+https://github.com/jwohlwend/boltz.git && "
        "python -m pip install esm2quinox==0.1.0 ipymolstar>=0.0.9 matplotlib>=3.10.0 datasets>=2.19.0 && "
        "git clone --depth 1 https://github.com/adaptyvbio/mosaic_workflows.git /repo"
    )
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/src", "/workspace/src")
)


app = modal.App("adaptyv-boltzcraft-old", image=image)


boltz_cache = modal.Volume.from_name("boltz-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("results-boltzcraft", create_if_missing=True)


@app.function(volumes={"/results": results_vol})
def inspect_trajectory_old(results_dir: str, head: int = 10):
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
            print({
                "step": row.get("step"),
                "loss": row.get("loss"),
                "aux": row.get("aux"),
            })
            shown += 1


@app.function(
    gpu="H100",
    timeout=3 * 60 * 60,
    volumes={"/root/.boltz": boltz_cache, "/results": results_vol},
    secrets=[modal.Secret.from_name("github-token")],
)
def run_mhetase_old(
    *,
    binder_len: int = 20,
    motif_positions: Dict[str, Any] = {"ser": 3, "his": 10, "asp": 15},
    ligand: Dict[str, Any] = {"enzyme_chain": "A", "ligand_chain": "L", "smiles": "OCCOC(=O)c1ccc(cc1)C(=O)O"},
    total_steps: int = 20,
    seed: int = 0,
    pdb_path: str | None = None,
    pdb_bytes: bytes | None = None,
    pdb_residues: str | None = None,
):
    os.environ.setdefault("BOLTZ_CACHE", "/root/.boltz")
    Path(os.environ["BOLTZ_CACHE"]).mkdir(parents=True, exist_ok=True)

    # Prefer mounted local source (latest edits), fall back to baked repo
    local_src = Path("/workspace/src")
    if local_src.exists():
        sys.path.insert(0, str(local_src))
    repo_src = Path("/repo/src")
    if repo_src.exists():
        sys.path.append(str(repo_src))

    # Avoid package __init__ side-effects; load modules directly
    from importlib.machinery import SourceFileLoader
    ms = SourceFileLoader("mhetase_scaffold_old", "/workspace/src/mosaic_workflows/mhetase_scaffold_old.py").load_module()
    dwf = SourceFileLoader("design", "/workspace/src/mosaic_workflows/design.py").load_module()

    tmol_context = {"ligand": ligand}
    mp = dict(motif_positions)

    # If bytes provided, write to temp
    if pdb_bytes is not None:
        tmp_pdb = Path("/tmp/motif_input.pdb")
        tmp_pdb.write_bytes(pdb_bytes)
        pdb_path = str(tmp_pdb)

    # Build optional motif template from PDB CA or backbone (N,CA,C)
    motif_template_backbone = None
    motif_template_ca = None
    if pdb_path and pdb_residues:
        import gemmi
        st = gemmi.read_structure(str(pdb_path))
        chain = st[0]["A"]  # default to A
        resnums = [int(x.strip()) for x in pdb_residues.split(',') if x.strip()]
        bb = []
        ca_list = []
        for rn in resnums:
            # find residue by seqid
            res = next(r for r in chain if r.seqid.num == int(rn))
            # collect N, CA, C
            atoms = {a.name: a for a in res}
            n = atoms.get("N"); ca = atoms.get("CA"); c = atoms.get("C")
            if n is None or ca is None or c is None:
                raise RuntimeError(f"Missing backbone atoms for residue {rn}")
            bb.append([[n.pos.x, n.pos.y, n.pos.z], [ca.pos.x, ca.pos.y, ca.pos.z], [c.pos.x, c.pos.y, c.pos.z]])
            ca_list.append([ca.pos.x, ca.pos.y, ca.pos.z])
        import numpy as np
        motif_template_backbone = np.asarray(bb, dtype=np.float32)
        motif_template_ca = np.asarray(ca_list, dtype=np.float32)

    # Build predictor and workflow
    predict_fn = ms.build_boltz2_predict_fn_mhetase(
        binder_len=int(binder_len),
        enzyme_chain=ligand.get("enzyme_chain", "A"),
        ligand_chain=ligand.get("ligand_chain", "L"),
        ligand_ccd=ligand.get("ccd"),
        ligand_smiles=ligand.get("smiles"),
    )

    wf = ms.make_workflow(
        binder_len=int(binder_len),
        motif_positions=mp,
        tmol_context=tmol_context,
        predict_fn=predict_fn,
        motif_template_backbone=motif_template_backbone,
        motif_template_ca=motif_template_ca,
    )
    # init logits with motif fixed (like snippet)
    import numpy as np
    vocab = "ARNDCQEGHILKMFPSTWYV"
    x0 = np.random.randn(binder_len, 20).astype(np.float32) * 0.1
    def set_pos(pos, aa):
        if 0 <= pos < binder_len:
            x0[pos, :] = -10.0
            x0[pos, vocab.index(aa)] = 10.0
    if "ser" in mp: set_pos(int(mp["ser"]), "S")
    if "his" in mp: set_pos(int(mp["his"]), "H")
    if "asp" in mp: set_pos(int(mp["asp"]), "D")
    wf["initial_x"] = x0
    wf["seed"] = int(seed)
    out = dwf.run_workflow(wf)

    # Save outputs to results volume
    import json, time, numpy as np
    run_id = f"mhetase_old_{int(time.time())}_seed{seed}_len{binder_len}"
    out_dir = Path("/results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "best_sequence.txt").write_text(str(out.get("best_sequence", "")))
    np.save(out_dir / "best_x.npy", out.get("best_x"))

    traj = out.get("trajectory") or []
    with open(out_dir / "trajectory.jsonl", "w") as f:
        for rec in traj:
            row = {
                "step": int(rec.get("step", 0)),
                "aux": rec.get("aux", {}),
            }
            f.write(json.dumps(row, default=lambda o: float(o) if hasattr(o, "item") else None) + "\n")

    print({"results_dir": str(out_dir)})


@app.local_entrypoint()
def main(
    workflow: str = "mhetase_old",
    binder_len: int = 20,
    ser: int = 3,
    his: int = 10,
    asp: int = 15,
    oxyanion: str | None = None,
    ligand_smiles: str = "OCCOC(=O)c1ccc(cc1)C(=O)O",
    total_steps: int = 20,
    seed: int = 0,
    pdb_path: str | None = None,
    pdb_residues: str | None = None,
    head: int = 10,
):
    if workflow == "mhetase_old":
        ligand = {"enzyme_chain": "A", "ligand_chain": "L", "smiles": ligand_smiles}
        motif_pos: dict[str, int] = {"ser": ser, "his": his, "asp": asp}
        pdb_bytes = None
        if pdb_path and Path(pdb_path).exists():
            pdb_bytes = Path(pdb_path).read_bytes()
        run_mhetase_old.remote(
            binder_len=binder_len,
            motif_positions=motif_pos,
            ligand=ligand,
            total_steps=total_steps,
            seed=seed,
            pdb_path=pdb_path,
            pdb_bytes=pdb_bytes,
            pdb_residues=pdb_residues,
        )
    elif workflow == "inspect_old":
        if pdb_path is None:
            raise ValueError("--pdb-path used as results_dir for inspect_old")
        inspect_trajectory_old.remote(results_dir=pdb_path, head=int(head))
    else:
        raise ValueError(f"Unknown workflow: {workflow}")


