"""
run_binder_design.py — design a protein binder against a target sequence
using Boltz1 + ProteinMPNN + ESM2 + Trigram + Stability losses.

This is a non-interactive, batch-friendly version of the example notebooks.
All output is written to files; nothing requires a display or browser.

Usage:
    uv run python hpc/run_binder_design.py --target PDL1.pdb --binder-length 80

    # Or supply the target as a plain amino-acid string:
    uv run python hpc/run_binder_design.py \
        --target-seq "SFPASVQLHTAVEMHHWCIPFSVDGQPAPSLRWLFNGSVLNETSFIFTEFLEPAANETVRHGCLRLNQPTHVNNGNYTLLAANPFGQASASIMAAF" \
        --binder-length 60 \
        --n-steps 150 \
        --out-dir results/my_run

Outputs (in --out-dir):
    design_<seed>.pdb       predicted complex structure
    design_<seed>.fasta     designed sequence
    design_<seed>_plddt.csv per-residue pLDDT
    design_<seed>_pae.npy   PAE matrix (numpy)
    summary.csv             one row per run with aggregate metrics

Notes:
    • JAX JIT compilation happens on the first call — expect ~4 min on GH200.
    • Subsequent optimisation steps are fast (~30 s each for 150 steps total).
    • Run multiple seeds in separate SLURM array jobs for diversity.
"""

import argparse
import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import gemmi
import matplotlib
matplotlib.use("Agg")          # no display needed on HPC
import matplotlib.pyplot as plt

from mosaic.optimizers import simplex_APGM
from mosaic.structure_prediction import TargetChain
from mosaic.common import TOKENS
import mosaic.losses.structure_prediction as sp
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.proteinmpnn.mpnn import ProteinMPNN
from mosaic.losses.stability import StabilityModel
from mosaic.losses.trigram import TrigramLL
from mosaic.losses.transformations import ClippedLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pssm_to_sequence(pssm: np.ndarray) -> str:
    """Convert a soft PSSM (N, 20) → one-letter sequence via argmax."""
    return "".join(TOKENS[i] for i in pssm.argmax(-1))


def save_pssm_plot(pssm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(pssm.T, aspect="auto", cmap="viridis")
    ax.set_xlabel("Sequence position")
    ax.set_ylabel("Amino acid")
    ax.set_yticks(range(20))
    ax.set_yticklabels(TOKENS, fontsize=7)
    ax.set_title("Designed PSSM")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main design routine
# ---------------------------------------------------------------------------

def design(
    target_sequence: str,
    binder_length: int,
    n_steps: int,
    seed: int,
    out_dir: Path,
) -> dict:
    """Run one binder design trajectory and save results."""

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    key = jax.random.key(rng.randint(0, 100_000))

    print(f"\n[design] seed={seed}  binder_length={binder_length}  n_steps={n_steps}")
    print(f"[design] Target ({len(target_sequence)} aa): {target_sequence[:50]}...")
    print(f"[design] JAX devices: {jax.devices()}")

    # ------------------------------------------------------------------
    # Load models (weights downloaded/cached automatically)
    # ------------------------------------------------------------------
    print("[design] Loading Boltz1...")
    from mosaic.models.boltz1 import Boltz1
    boltz1 = Boltz1()

    print("[design] Loading ProteinMPNN...")
    mpnn = ProteinMPNN.from_pretrained()

    print("[design] Loading ESM2...")
    import esm
    import esm2quinox
    from mosaic.losses.esm import ESM2PseudoLikelihood
    torch_esm2, _ = esm.pretrained.esm2_t33_650M_UR50D()
    esm2 = ESM2PseudoLikelihood(esm2quinox.from_torch(torch_esm2))

    print("[design] Loading ESM-C (required for stability model)...")
    from esmj import from_torch as esmc_from_torch
    from esm.models.esmc import ESMC as TorchESMC
    esmc_model = esmc_from_torch(TorchESMC.from_pretrained("esmc_300m").to("cpu"))

    print("[design] Loading Trigram...")
    trigram = TrigramLL.from_pkl()

    print("[design] Loading Stability model...")
    repo_root = Path(__file__).parent.parent
    stability = StabilityModel.from_pretrained(esmc_model, path=repo_root / "stability.eqx")

    # ------------------------------------------------------------------
    # Build features for the binder-target complex and the binder alone
    # ------------------------------------------------------------------
    print("[design] Building features...")
    complex_features, complex_writer = boltz1.binder_features(
        binder_length=binder_length,
        chains=[TargetChain(sequence=target_sequence)],
    )
    mono_features, mono_writer = boltz1.binder_features(
        binder_length=binder_length,
        chains=[],
    )

    # ------------------------------------------------------------------
    # Build the combined loss function
    # Based on the README example: Boltz1 (complex) + Boltz1 (monomer) +
    # ESM2 + Trigram + Stability + ProteinMPNN inverse folding
    # ------------------------------------------------------------------
    combined_loss = (
        boltz1.build_loss(
            loss=(
                4.0 * sp.BinderTargetContact()
                + sp.RadiusOfGyration(target_radius=15.0)
                + sp.WithinBinderContact()
                + 0.3 * sp.HelixLoss()
                + 5.0 * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(0.01))
            ),
            features=complex_features,
            recycling_steps=1,
        )
        + 0.5 * ClippedLoss(esm2, 2.0, 100.0)
        + trigram
        + 0.1 * stability
        + 0.5 * boltz1.build_loss(
            loss=(
                0.2 * sp.PLDDTLoss()
                + sp.RadiusOfGyration(target_radius=15.0)
                + 0.3 * sp.HelixLoss()
            ),
            features=mono_features,
            recycling_steps=1,
        )
    )

    # ------------------------------------------------------------------
    # Phase 1: continuous optimisation on the probability simplex
    # ------------------------------------------------------------------
    print(f"[design] Phase 1: simplex_APGM for {n_steps} steps "
          f"(first call JIT-compiles — expect several minutes)...")
    t0 = time.time()
    x0 = jax.nn.softmax(
        0.5 * jax.random.gumbel(key=key, shape=(binder_length, 20))
    )
    _, pssm_soft = simplex_APGM(
        loss_function=combined_loss,
        x=x0,
        n_steps=n_steps,
        stepsize=0.1 * np.sqrt(binder_length),
        momentum=0.9,
    )
    print(f"[design] Phase 1 done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Phase 2: sharpen the PSSM to approach a single sequence
    # ------------------------------------------------------------------
    print("[design] Phase 2: sharpening PSSM...")
    pssm_sharp, _ = simplex_APGM(
        loss_function=combined_loss,
        x=pssm_soft,
        n_steps=25,
        stepsize=0.2,
        scale=1.1,
    )
    pssm_sharp, _ = simplex_APGM(
        loss_function=combined_loss,
        x=pssm_sharp,
        n_steps=25,
        stepsize=0.2,
        scale=1.5,
    )

    # ------------------------------------------------------------------
    # Evaluate: predict the complex structure for the final design
    # ------------------------------------------------------------------
    print("[design] Predicting final complex structure...")
    t1 = time.time()
    prediction = boltz1.predict(
        PSSM=pssm_sharp,
        features=complex_features,
        writer=complex_writer,
        key=jax.random.key(seed),
    )
    print(f"[design] Final prediction done in {time.time()-t1:.1f}s")

    # ------------------------------------------------------------------
    # Collect metrics
    # ------------------------------------------------------------------
    sequence = pssm_to_sequence(np.array(pssm_sharp))
    plddt_binder = float(np.mean(prediction.plddt[:binder_length]))
    plddt_target = float(np.mean(prediction.plddt[binder_length:]))
    plddt_mean   = float(np.mean(prediction.plddt))
    pae_bt       = float(np.mean(
        prediction.pae[:binder_length, binder_length:]
    ))

    metrics = {
        "seed": seed,
        "binder_length": binder_length,
        "sequence": sequence,
        "plddt_mean": plddt_mean,
        "plddt_binder": plddt_binder,
        "plddt_target": plddt_target,
        "pae_binder_target": pae_bt,
    }
    print(f"[design] pLDDT(binder)={plddt_binder:.3f}  "
          f"pLDDT(target)={plddt_target:.3f}  "
          f"PAE(binder→target)={pae_bt:.2f}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    prefix = out_dir / f"design_{seed}"

    # PDB
    (prefix.parent / f"design_{seed}.pdb").write_text(prediction.st.make_pdb_string())
    print(f"[design] Saved PDB   : {prefix}.pdb")

    # FASTA
    (prefix.parent / f"design_{seed}.fasta").write_text(
        f">design_seed{seed}\n{sequence}\n"
    )
    print(f"[design] Saved FASTA : {prefix}.fasta")

    # Per-residue pLDDT
    with open(f"{prefix}_plddt.csv", "w") as f:
        f.write("residue,chain,plddt\n")
        for i, v in enumerate(prediction.plddt):
            chain = "binder" if i < binder_length else "target"
            f.write(f"{i},{chain},{float(v):.4f}\n")

    # PAE matrix
    np.save(f"{prefix}_pae.npy", np.array(prediction.pae))

    # PSSM plot
    save_pssm_plot(np.array(pssm_sharp), prefix.parent / f"design_{seed}_pssm.png")
    print(f"[design] Saved PSSM  : {prefix}_pssm.png")

    # Append to summary CSV
    summary_path = out_dir / "summary.csv"
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"[design] Summary row : {summary_path}")

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Design a protein binder with mosaic (batch / SLURM compatible)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--target",
        metavar="PDB_OR_CIF",
        help="Path to a PDB or CIF file containing the target structure. "
             "The sequence is extracted from the first chain.",
    )
    group.add_argument(
        "--target-seq",
        metavar="AA_STRING",
        help="Target amino-acid sequence as a plain string (e.g. MGSSHHHHH...).",
    )
    parser.add_argument(
        "--binder-length",
        type=int,
        default=80,
        help="Length of the binder to design (default: 80)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=150,
        help="Number of simplex_APGM optimisation steps (default: 150)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (use SLURM_ARRAY_TASK_ID for array jobs, default: 0)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/binder_design"),
        help="Directory to write outputs (default: results/binder_design/)",
    )
    args = parser.parse_args()

    # Resolve target sequence
    if args.target_seq:
        target_seq = args.target_seq
    else:
        st = gemmi.read_structure(args.target)
        st.remove_ligands_and_waters()
        target_seq = gemmi.one_letter_code([r.name for r in st[0][0]])

    design(
        target_sequence=target_seq,
        binder_length=args.binder_length,
        n_steps=args.n_steps,
        seed=args.seed,
        out_dir=args.out_dir,
    )
