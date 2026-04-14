"""
run_prediction.py — run a single structure prediction with Boltz1 and save
the result as a PDB file.  Good first real test of the installation.

Usage:
    uv run python hpc/run_prediction.py [--out-dir results/]

The script uses PDL1.pdb (included in the repo) as an example target and
predicts its structure.  On a GH200 the first run takes ~4 min for JAX
JIT compilation, then < 1 min per prediction afterwards.

Typical SLURM usage:
    sbatch hpc/job.slurm prediction   (see job.slurm for the full command)
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import gemmi
import numpy as np


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Print device info
    # ------------------------------------------------------------------
    devices = jax.devices()
    print(f"[predict] JAX devices: {devices}")

    # ------------------------------------------------------------------
    # Load target structure from the bundled PDB file
    # The target sequence is extracted from PDL1.pdb (PD-L1 checkpoint).
    # We could also just hard-code a sequence string if preferred.
    # ------------------------------------------------------------------
    repo_root = Path(__file__).parent.parent
    pdb_path = repo_root / "PDL1.pdb"
    print(f"[predict] Loading target from {pdb_path}")

    st = gemmi.read_structure(str(pdb_path))
    st.remove_ligands_and_waters()
    target_sequence = gemmi.one_letter_code([r.name for r in st[0][0]])
    print(f"[predict] Target sequence ({len(target_sequence)} aa): {target_sequence[:40]}...")

    # ------------------------------------------------------------------
    # Initialise Boltz1 (downloads weights on first run, cached after that)
    # ------------------------------------------------------------------
    print("[predict] Loading Boltz1 model (may download weights on first run)...")
    from mosaic.models.boltz1 import Boltz1
    from mosaic.structure_prediction import TargetChain

    t0 = time.time()
    boltz1 = Boltz1()
    print(f"[predict] Model loaded in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Build features and run prediction
    # ------------------------------------------------------------------
    print("[predict] Building features...")
    features, writer = boltz1.target_only_features(
        chains=[TargetChain(sequence=target_sequence)]
    )

    print("[predict] Running structure prediction (first call JIT-compiles — expect ~4 min)...")
    t1 = time.time()
    prediction = boltz1.predict(
        features=features,
        writer=writer,
        key=jax.random.key(42),
        recycling_steps=3,
    )
    elapsed = time.time() - t1
    print(f"[predict] Prediction complete in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Print quality metrics and save PDB
    # ------------------------------------------------------------------
    plddt_mean = float(np.mean(prediction.plddt))
    print(f"[predict] Mean pLDDT : {plddt_mean:.3f}")

    out_pdb = out_dir / "PDL1_predicted.pdb"
    out_pdb.write_text(prediction.st.make_pdb_string())
    print(f"[predict] Saved PDB  : {out_pdb}")

    # Save per-residue pLDDT as a simple CSV
    out_csv = out_dir / "PDL1_plddt.csv"
    with out_csv.open("w") as f:
        f.write("residue,plddt\n")
        for i, v in enumerate(prediction.plddt):
            f.write(f"{i},{float(v):.4f}\n")
    print(f"[predict] Saved pLDDT: {out_csv}")
    print("[predict] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a protein structure with Boltz1")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/prediction"),
        help="Directory to write PDB and CSV output files (default: results/prediction/)",
    )
    args = parser.parse_args()
    main(args.out_dir)
