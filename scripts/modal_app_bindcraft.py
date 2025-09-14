import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

import modal


# Base image modeled after adaptyv_bindcraft modal app with extras for Mosaic
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "wget",
        "git",
        "aria2",
        "ffmpeg",
        "build-essential",
        "libxml2-dev",
        "libxslt1-dev",
        "cmake",
    )
    .pip_install(
        # BindCraft stack
        "pdb-tools==2.4.8",
        "ffmpeg-python==0.2.0",
        "plotly==5.18.0",
        "kaleido==0.2.1",
        "pyarrow",
        "fastparquet",
        "boto3",
        "python-dotenv",
        "loguru",
        "openmm>=7.7.0",
        "mdtraj",
        "biopython",
        "freesasa",
        "scipy",
        "scikit-learn",
    )
    .pip_install("git+https://github.com/openmm/pdbfixer.git")
    .pip_install("git+https://github.com/sokrypton/ColabDesign.git")
    .run_commands(
        "mkdir -p /root/BindCraft/functions && mkdir -p /params"
    )
    .run_commands(
        "ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign"
    )
    # Removed AF params download; rely on mounted/cached volumes for params
    # JAX
    .pip_install("jax[cuda]")
    .pip_install("jaxlib")
    # Misc deps
    .pip_install("gemmi")
    # Mosaic dependencies: Joltz and Boltz wrappers
    .pip_install("git+https://github.com/adaptyvbio/joltz.git")
    .pip_install("git+https://github.com/jwohlwend/boltz.git")
    # Add only local src/ to the image to minimize snapshot/mount size
    .add_local_dir(str(Path(__file__).resolve().parents[1] / "src"), "/repo/src", copy=True)
    .run_commands(
        "cp /repo/src/../download_params.sh /root/download_params.sh || true && chmod +x /root/download_params.sh || true"
    )
)


app = modal.App("mosaic-bindcraft-compat", image=image)


out_vol = modal.Volume.from_name("mosaic-bindcraft-out", create_if_missing=True)
boltz_vol = modal.Volume.from_name("boltz-cache", create_if_missing=False)
af_params_vol = modal.Volume.from_name("alphafold-cache", create_if_missing=False)


def _add_paths():
    # Add mosaic_workflows and BindCraft into sys.path
    for p in ("/repo/src", "/root/adaptyv_bindcraft", "/root/BindCraft", "/root/utilities"):
        if p not in sys.path and Path(p).exists():
            sys.path.insert(0, p)


@app.function(
    gpu="A10",
    timeout=8 * 60 * 60,
    volumes={
        "/output": out_vol,
        "/root/.boltz": boltz_vol,
        "/root/BindCraft/params": af_params_vol,
    },
    mounts=[
        # Mount only the BindCraft package directory needed by bindcraft_compat
        modal.Mount.from_local_dir(
            "/Users/tudorcotet/Documents/Adaptyv/adaptyv_bindcraft/src/BindCraft",
            remote_path="/root/BindCraft",
        ),
        # Mount utilities required by BindCraft (e.g., adaptyv_scoring_improved)
        modal.Mount.from_local_dir(
            "/Users/tudorcotet/Documents/Adaptyv/adaptyv_bindcraft/utilities",
            remote_path="/root/utilities",
        ),
    ],
)
def run_bindcraft_outer(
    *,
    task_name: str = "TEST",
    binder_chain: str = "B",
    target_sequence: str = "",
    chains: str = "A",
    lengths: List[int] = [80],
    number_of_final_designs: int = 50,
    max_trajectories: int = 50,
    runtime_seed: int | None = 1234,
):
    _add_paths()
    import os
    import subprocess

    from mosaic_workflows.bindcraft_compat import run_bindcraft_compat
    import gemmi

    design_path = f"/output/{task_name}"
    Path(design_path).mkdir(parents=True, exist_ok=True)

    # Load PDL1 template and sequence from repo
    pdl1_pdb = "/repo/src/PDL1_stable_region.pdb"
    try:
        st = gemmi.read_structure(pdl1_pdb)
        st.setup_entities()
        target_chain_seq = gemmi.one_letter_code([r.name for r in st[0][0]])
    except Exception:
        target_chain_seq = target_sequence or ""

    # Ensure AF2 params exist (fallback)
    try:
        os.chdir("/root")
    except Exception:
        pass
    try:
        if not (Path("/root/BindCraft") / "params").exists():
            subprocess.run(["bash", "/root/download_params.sh", "/root/BindCraft"], check=True)
    except Exception:
        pass

    # Minimal settings
    target_settings: Dict[str, Any] = {
        "binder_name": task_name,
        "task_name": task_name,
        "target_sequence": target_chain_seq,
        "starting_pdb": pdl1_pdb,
        "chains": chains,
        "lengths": lengths,
        "number_of_final_designs": number_of_final_designs,
        "target_hotspot_residues": "",
    }
    # Optional external tools/paths
    filters_json_path = Path("/root/BindCraft") / "settings_filters" / "openmm_filters.json"
    dssp_path = Path("/root/BindCraft") / "functions" / "dssp"
    dalphaball_path = Path("/root/BindCraft") / "functions" / "DAlphaBall.gcc"

    advanced_settings: Dict[str, Any] = {
        "binder_chain": binder_chain,
        "use_multimer_design": True,
        "num_recycles_design": 3,
        "num_recycles_validation": 3,
        "af_params_dir": "/root/BindCraft",
        "filters_json": str(filters_json_path) if filters_json_path.exists() else "",
        # MPNN defaults
        "backbone_noise": 0.0,
        "model_path": "v_48_020",
        "mpnn_weights": "soluble",
        "mpnn_fix_interface": False,
        "omit_AAs": None,
        "sampling_temp": 0.1,
        "num_seqs": 5,
        # Predict flags
        "rm_template_seq_predict": False,
        "rm_template_sc_predict": False,
        # Design algo label for CSV parity
        "design_algorithm": "3stage",
        # External tools
        "dssp_path": str(dssp_path) if dssp_path.exists() else "",
        "dalphaball_path": str(dalphaball_path) if dalphaball_path.exists() else "",
        # No predict_fn here; child predictions handled in emit
    }
    filters: Dict[str, Any] = {}

    out = run_bindcraft_compat(
        repo_root="/root",  # BindCraft content under /root/BindCraft and /root/adaptyv_bindcraft
        design_path=design_path,
        build_parent=None,
        spawn_children=None,
        emit_row=None,
        max_trajectories=max_trajectories,
        target_settings=target_settings,
        advanced_settings=advanced_settings,
        filters=filters,
        runtime_seed=runtime_seed,
        runtime_length=(lengths[0] if lengths else 20),
        stop=None,
    )

    return {"design_dir": design_path, "num_rows": len(out.get("rows", []))}


@app.local_entrypoint()
def main(
    task_name: str = "TEST",
    binder_chain: str = "B",
    target_sequence: str = "MFEARLVQGSI",
    chains: str = "A",
    length: int = 20,
    number_of_final_designs: int = 1,
    max_trajectories: int = 1,
    runtime_seed: int | None = 1234,
):
    res = run_bindcraft_outer.remote(
        task_name=task_name,
        binder_chain=binder_chain,
        target_sequence=target_sequence,
        chains=chains,
        lengths=[int(length)],
        number_of_final_designs=number_of_final_designs,
        max_trajectories=max_trajectories,
        runtime_seed=runtime_seed,
    )
    print(json.dumps(res, indent=2))


