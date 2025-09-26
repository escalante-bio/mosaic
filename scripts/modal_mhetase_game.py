import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import modal


# -------- Modal image --------
image = (
    modal.Image.debian_slim(python_version="3.12.0")
    .apt_install("git", "aria2")
    .env({
        "BOLTZ_CACHE": "/root/.boltz",
        "JAX_PLATFORMS": "cuda",
    })
    .run_commands(
        "python -m pip install -U pip setuptools wheel && "
        # CUDA PyTorch for CUDA runtime libs
        "python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1 && "
        # JAX core and CUDA plugin
        "python -m pip install --upgrade jax==0.7.1 && "
        "python -m pip install --upgrade jax-cuda12-plugin==0.7.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && "
        # PTX toolchain
        "python -m pip install nvidia-cuda-nvcc-cu12==12.8.93 && "
        # Core JAX ecosystem deps
        "python -m pip install optax==0.2.4 dm-haiku>=0.0.13 flax>=0.10.2 ml-collections>=1.0.0 httpx>=0.28.1 gemmi>=0.6.0 matplotlib>=3.10.0 && "
        # Git-only deps used at runtime
        "python -m pip install git+https://github.com/escalante-bio/jablang.git && "
        "python -m pip install git+https://github.com/escalante-bio/esmj.git && "
        # joltz (JAX translation of Boltz)
        "python -m pip install git+https://github.com/adaptyvbio/joltz.git && "
        # boltz utils
        "python -m pip install git+https://github.com/jwohlwend/boltz.git && "
        # Bake repo (fallback)
        "git clone --depth 1 https://github.com/adaptyvbio/mosaic_workflows.git /repo"
    )
    # Include local src so latest edits are available
    .add_local_dir("/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/src", "/workspace/src")
)


app = modal.App("adaptyv-mhetase-game", image=image)

boltz_cache = modal.Volume.from_name("boltz-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("results-boltzcraft", create_if_missing=True)
af2_cache = modal.Volume.from_name("alphafold-cache", create_if_missing=True)


@app.function(
    gpu="H100",
    timeout=3 * 60 * 60,
    volumes={"/root/.boltz": boltz_cache, "/results": results_vol, "/repo/params": af2_cache},
    secrets=[modal.Secret.from_name("github-token")],
)
def run_mhetase_game(
    *,
    binder_len: int = 100,
    total_steps: int = 100,
    seed: int = 0,
    # motif config
    ser: int = 10,
    asp: int = 30,
    his: int = 50,
    gly: Optional[int] = None,
    glu: Optional[int] = None,
    pdb_path: Optional[str] = None,
    pdb_bytes: Optional[bytes] = None,
    pdb_residues: Optional[str] = None,
    motif_chain_id: str = "A",
    # backend
    use_af2: bool = True,
    af2_num_recycles: int = 1,
    # weights
    w_contact: float = 1.0,
    w_motif_cce: float = 1.0,
    w_motif_rmsd: float = 0.2,
    w_sc_rmsd: float = 0.1,
    w_plddt: float = 0.1,
    w_pae: float = 0.4,
    w_rg: float = 0.0,
    w_seq_ent: float = 0.1,
    w_cat_dist: float = 0.1,
    # fixing
    freeze_supervised_positions: bool = True,
    fix_supervised_identities: Optional[str] = "S,D,H,G,E",
    # game
    game_type: str = "none",  # one of: none|minimax|stackelberg
):
    """Run MHETase optimization with optional 2-player game (minimax/stackelberg)."""

    # Ensure Boltz cache dir
    os.environ.setdefault("BOLTZ_CACHE", "/root/.boltz")
    Path(os.environ["BOLTZ_CACHE"]).mkdir(parents=True, exist_ok=True)

    # Prefer mounted local source, then baked repo
    local_src = Path("/workspace/src")
    if local_src.exists():
        sys.path.insert(0, str(local_src))
    repo_src = Path("/repo/src")
    if repo_src.exists():
        sys.path.append(str(repo_src))

    # Load modules directly
    from importlib.machinery import SourceFileLoader
    dwf = SourceFileLoader("design", "/workspace/src/mosaic_workflows/design.py").load_module()  # type: ignore
    ms = SourceFileLoader("mhetase_scaffold", "/workspace/src/mosaic_workflows/mhetase_scaffold.py").load_module()  # type: ignore

    # Build supervised mapping and roles
    order_keys: list[str] = ["ser", "his", "asp"]
    if pdb_residues is not None:
        try:
            num_res = len([x for x in pdb_residues.split(',') if x.strip()])
            if num_res == 5:
                order_keys = ["ser", "asp", "his", "gly", "glu"]
        except Exception:
            pass
    motif_pos: Dict[str, int | None] = {"ser": ser, "his": his, "asp": asp, "gly": gly, "glu": glu}
    supervised_positions_list = [
        int(motif_pos[k]) for k in order_keys if (k in motif_pos and motif_pos[k] is not None)
    ]
    supervised_positions = tuple(supervised_positions_list)
    motif_roles = tuple(order_keys[: len(supervised_positions_list)])

    # Handle PDB template
    motif_pdb_path = pdb_path
    if pdb_bytes is not None:
        tmp_pdb = Path("/tmp/motif_input.pdb"); tmp_pdb.write_bytes(pdb_bytes)
        motif_pdb_path = str(tmp_pdb)
    motif_resnums = tuple(int(x.strip()) for x in pdb_residues.split(',') if x.strip()) if pdb_residues else None

    # Build a single-player MHETase loss (AlphaFoldLoss only; AF2 required for game)
    if not bool(use_af2):
        raise ValueError("Game modes currently require --use-af2")

    from mosaic.models.af2 import AlphaFold2
    from mosaic.structure_prediction import TargetChain
    import jax
    import jax.numpy as jnp
    import numpy as np

    # Ensure AF2 params are present under /repo/params
    params_dir = Path("/repo") / "params"
    key_file = params_dir / "params_model_1.npz"
    if not key_file.exists():
        script = Path("/repo") / "download_params.sh"
        if script.exists():
            import subprocess
            subprocess.run(["bash", str(script), "/repo"], check=True)

    af2_model = AlphaFold2(data_dir="/repo")
    # Use partial template for AF2 binder when motif info is provided (ColabDesign-style)
    use_partial_template = bool(motif_pdb_path) and bool(motif_resnums) and bool(supervised_positions)
    if use_partial_template:
        # Build partial binder template chain: CA (and sidechains if available) at motif positions
        from importlib import import_module
        gemmi = import_module("gemmi")
        chain = gemmi.Chain("A")
        for i in range(int(binder_len)):
            res = gemmi.Residue(); res.name = "GLY"; res.seqid = gemmi.SeqId(int(i + 1), " "); chain.add_residue(res)
        # derive motif CA and optional sidechains using mhetase_scaffold helpers already loaded as ms
        mt = ms._build_motif_from_pdb(pdb_path=str(motif_pdb_path), chain_id=str(motif_chain_id), residue_numbers=tuple(int(x) for x in motif_resnums))
        sc_list = ms._build_motif_sidechains_from_pdb(pdb_path=str(motif_pdb_path), chain_id=str(motif_chain_id), residue_numbers=tuple(int(x) for x in motif_resnums))
        mp = tuple(int(x) for x in supervised_positions)
        for idx_local, pos in enumerate(mp):
            if 0 <= int(pos) < int(binder_len):
                res = chain[int(pos)]
                # set residue name from PDB regardless of sidechain atoms
                if idx_local < len(sc_list):
                    try:
                        res.name = str(sc_list[idx_local][0])
                    except Exception:
                        pass
                atom = gemmi.Atom(); atom.name = "CA"; xyz = mt[idx_local]
                atom.pos.x, atom.pos.y, atom.pos.z = float(xyz[0]), float(xyz[1]), float(xyz[2]); res.add_atom(atom)
                if idx_local < len(sc_list):
                    _, atom_names, coords = sc_list[idx_local]
                    for nm, q in zip(atom_names, coords):
                        if nm in ("N","CA","C","O"): continue
                        a2 = gemmi.Atom(); a2.name = str(nm); a2.pos.x, a2.pos.y, a2.pos.z = float(q[0]), float(q[1]), float(q[2]); res.add_atom(a2)
        feats, _ = af2_model.target_only_features([TargetChain(sequence="G"*int(binder_len), use_msa=False, template_chain=chain)])
    else:
        feats, _ = af2_model.binder_features(int(binder_len), chains=[])

    # Compose MHETase losses using the same terms as mhetase_scaffold
    excl = tuple(supervised_positions) if supervised_positions else ()
    aux = float(w_contact) * ms.ContactLoss(cutoff=14.0, binary=True, num=2, num_pos=1, seqsep=9, exclude_positions=excl)
    conf = ms._sum_losses([
        (float(w_plddt) * ms.PLDDTLoss(exclude_positions=excl)) if float(w_plddt) != 0.0 else None,
        (float(w_pae) * ms.PAELoss(seqsep=9, exclude_positions=excl)) if float(w_pae) != 0.0 else None,
    ])

    # Motif geometry
    def motif_geo():
        terms: list = []
        if motif_pdb_path and motif_resnums:
            # Build CA template
            mt = ms._build_motif_from_pdb(pdb_path=str(motif_pdb_path), chain_id=str(motif_chain_id), residue_numbers=tuple(int(x) for x in motif_resnums))
            if supervised_positions:
                mp = tuple(int(x) for x in supervised_positions)
                if float(w_motif_cce) != 0.0:
                    terms.append(float(w_motif_cce) * ms.MotifDistogramCCE(motif_positions=mp, motif_template_ca=mt))
                if float(w_motif_rmsd) != 0.0:
                    terms.append(float(w_motif_rmsd) * ms.ClippedLoss(loss=ms.MotifRMSDCA(motif_positions=mp, motif_template_ca=mt), l=0.0, u=10.0, name="motif_rmsd_clip"))
                if float(w_sc_rmsd) != 0.0:
                    terms.append(float(w_sc_rmsd) * ms.AF2SidechainRMSD_Outer(positions=mp))
                if float(w_cat_dist) != 0.0:
                    s_i, h_i, a_i = None, None, None
                    roles_l = [str(r).lower() for r in motif_roles]
                    s_i = mp[roles_l.index("ser")]
                    h_i = mp[roles_l.index("his")]
                    a_i = mp[roles_l.index("asp")]
                    terms.append(float(w_cat_dist) * ms.CatalyticProximityCA(ser_idx=s_i, his_idx=h_i, asp_idx=a_i))
            else:
                if float(w_motif_cce) != 0.0:
                    terms.append(float(w_motif_cce) * ms.MotifAutoDistogramCCE(motif_template_ca=mt, beta=10.0))
        return ms._sum_losses(terms)

    # Priors shared
    from mosaic.proteinmpnn.mpnn import ProteinMPNN
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    mpnn = ProteinMPNN.from_pretrained()
    mpnn_prior = InverseFoldingSequenceRecovery(mpnn=mpnn, temp=jnp.asarray(0.05), num_samples=8, jacobi_iterations=8)
    seq_prior = ms.ClippedLoss(loss=5.0 * mpnn_prior, l=-np.inf, u=100.0, name="mpnn_clipped")
    no_cys = 0.1 * ms.NoCysteine()
    rg_term = float(w_rg) * ms.MaskedDistogramRadiusOfGyration(exclude_positions=excl)

    # Mask catalytic residues in sequence entropy if supervised
    cat_positions: tuple[int, ...] = ()
    if supervised_positions and motif_roles:
        roles_l = [str(r).lower() for r in motif_roles]
        if all(k in roles_l for k in ("ser","his","asp")):
            mp = tuple(int(x) for x in supervised_positions)
            cat_positions = (mp[roles_l.index("ser")], mp[roles_l.index("his")], mp[roles_l.index("asp")])
    seq_ent = float(w_seq_ent) * ms.SeqEntropyLoss(exclude_positions=cat_positions)

    struct_full = ms.ClippedGradient(loss=ms._sum_losses([
        aux, motif_geo(), seq_ent, conf, rg_term, seq_prior, no_cys
    ]), max_norm=1.0)

    loss_single = af2_model.build_loss(
        loss=struct_full,
        features=feats,
        recycling_steps=int(af2_num_recycles),
        name="af2_mhetase",
    )

    # Build game phases
    from binder_games.builders import build_minmax_phase, build_stackelberg_phase
    from binder_games.analyzers import (
        saddle_gap_estimate,
        decode_sequences_xy,
        value_components,
        probs_entropy_xy,
        kl_divergence_xy,
        sequence_hamming_xy,
        grad_norms_xy,
    )
    from mosaic_workflows.transforms import temperature_on_logits, e_soft_on_logits, gradient_normalizer

    def build_loss_pair():
        # Two independent views over the same AF2 features/params
        lx = loss_single
        ly = loss_single
        margin = 0.1
        lambda_hinge = 0.5
        def loss_fn(x_probs, y_probs, key=None):
            vx, auxx = lx(x_probs, key=key)
            vy, auxy = ly(y_probs, key=key)
            h = jnp.maximum(0.0, margin + vx - vy)
            v = (vx - vy) + lambda_hinge * h
            aux = {"x": auxx, "y": auxy, "value_x": vx, "value_y": vy, "ranking_hinge": h}
            return v, aux
        return loss_fn

    import numpy as _np
    x0 = _np.random.randn(binder_len, 20).astype(_np.float32) * 0.1

    game = str(game_type).lower()
    if game == "none":
        # Fall back to standard single-player MHETase workflow
        kwargs = dict(
            binder_len=binder_len,
            tmol_context={"ligand": {"enzyme_chain": "A", "ligand_chain": "L", "smiles": "OCCOC(=O)c1ccc(cc1)C(=O)O"}},
            use_af2=True,
            af2_num_recycles=af2_num_recycles,
            af2_params_dir="/repo",
            steps=total_steps,
            lr=0.1,
            w_contact=w_contact,
            w_motif_cce=w_motif_cce,
            w_motif_rmsd=w_motif_rmsd,
            w_sc_rmsd=w_sc_rmsd,
            w_plddt=w_plddt,
            w_pae=w_pae,
            w_seq_ent=w_seq_ent,
            w_cat_dist=w_cat_dist,
        )
        if supervised_positions:
            kwargs["supervised_positions"] = supervised_positions
            kwargs["motif_roles"] = motif_roles
        if motif_pdb_path:
            kwargs["motif_pdb_path"] = motif_pdb_path
        if motif_chain_id:
            kwargs["motif_chain_id"] = motif_chain_id
        if motif_resnums:
            kwargs["motif_resnums"] = motif_resnums
        if fix_supervised_identities:
            kwargs["fix_supervised_identities"] = fix_supervised_identities
        wf = ms.make_workflow(**kwargs)
        wf["seed"] = int(seed)
        wf["initial_x"] = x0
        out = dwf.run_workflow(wf)
        import json, time
        run_id = f"mhetase_{int(time.time())}_seed{seed}_len{binder_len}"
        out_dir = Path("/results") / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        best_seq = str(out.get("best_sequence", ""))
        (out_dir / "best_sequence.txt").write_text(best_seq)
        _np.save(out_dir / "best_x.npy", out.get("best_x"))
        traj = out.get("trajectory") or []
        with open(out_dir / "trajectory.jsonl", "w") as f:
            for rec in traj:
                f.write(
                    json.dumps(
                        {"step": int(rec.get("step", 0)), "aux": rec.get("aux", {})},
                        default=lambda o: (o.tolist() if hasattr(o, "tolist") else (o.item() if hasattr(o, "item") else (float(o) if hasattr(o, "__float__") else None))),
                    )
                    + "\n"
                )
        # Print quick verification: residues at supervised positions
        cat_res = {}
        if supervised_positions and best_seq:
            try:
                cat_res = {int(i): best_seq[int(i)] for i in supervised_positions if 0 <= int(i) < len(best_seq)}
            except Exception:
                pass
        print({"results_dir": str(out_dir), "best_sequence": best_seq, "supervised_positions": supervised_positions, "residues_at_supervised": cat_res})
        return

    # Two-player setups
    if game == "minimax":
        phase = build_minmax_phase(
            name="mhetase_minmax",
            build_loss=build_loss_pair,
            steps=int(total_steps),
            schedule=lambda g, p: {"lr_x": 0.05, "lr_y": 0.05, "temperature": 1.0, "y_init": "random"},
            transforms={
                "x": {"pre_logits": [temperature_on_logits(), e_soft_on_logits()], "grad": [gradient_normalizer(mode="per_chain", log_norm=True)]},
                "y": {"pre_logits": [temperature_on_logits(), e_soft_on_logits()], "grad": [gradient_normalizer(mode="per_chain", log_norm=True)]},
            },
            analyzers=[saddle_gap_estimate(), value_components(), decode_sequences_xy(), probs_entropy_xy(), kl_divergence_xy(), sequence_hamming_xy(), grad_norms_xy()],
            analyze_every=1,
        )
    elif game == "stackelberg":
        phase = build_stackelberg_phase(
            name="mhetase_stackelberg",
            build_loss=build_loss_pair,
            steps=int(total_steps),
            schedule=lambda g, p: {"lr_x": 0.05, "lr_y": 0.05, "br_steps": 5, "reinit_y_each_step": False, "temperature": 1.0, "y_init": "random"},
            transforms={
                "x": {"pre_logits": [temperature_on_logits(), e_soft_on_logits()], "grad": [gradient_normalizer(mode="per_chain", log_norm=True)]},
                "y": {"pre_logits": [temperature_on_logits(), e_soft_on_logits()], "grad": [gradient_normalizer(mode="per_chain", log_norm=True)]},
            },
            analyzers=[saddle_gap_estimate(), value_components(), decode_sequences_xy(), probs_entropy_xy(), kl_divergence_xy(), sequence_hamming_xy(), grad_norms_xy()],
            analyze_every=1,
        )
    else:
        raise ValueError("--game-type must be one of: none|minimax|stackelberg")

    wf = {"phases": [phase], "binder_len": int(binder_len), "seed": int(seed), "initial_x": x0}
    out = dwf.run_workflow(wf)

    import json, time
    run_id = f"mhetase_game_{int(time.time())}_seed{seed}_len{binder_len}_{game}"
    out_dir = Path("/results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "best_sequence.txt").write_text(str(out.get("best_sequence", "")))
    if out.get("best_x") is not None:
        _np.save(out_dir / "best_x.npy", out.get("best_x"))

    traj = out.get("trajectory", []) or []
    with open(out_dir / "trajectory.jsonl", "w") as f:
        for rec in traj:
            step_i = int(rec.get("step", 0))
            aux_i = rec.get("aux", {}) or {}
            f.write(
                json.dumps(
                    {"step": step_i, "aux": aux_i},
                    default=lambda o: (o.tolist() if hasattr(o, "tolist") else (o.item() if hasattr(o, "item") else (float(o) if hasattr(o, "__float__") else None))),
                )
                + "\n"
            )

    print({"results_dir": str(out_dir)})


@app.local_entrypoint()
def main(
    binder_len: int = 100,
    total_steps: int = 100,
    seed: int = 0,
    ser: int = 10,
    asp: int = 30,
    his: int = 50,
    gly: Optional[int] = 65,
    glu: Optional[int] = 11,
    pdb_path: Optional[str] = "/Users/tudorcotet/Downloads/6QZ4.pdb",
    pdb_residues: Optional[str] = "225,492,528,132,226",
    motif_chain_id: str = "A",
    use_af2: bool = True,
    af2_num_recycles: int = 1,
    w_contact: float = 1.0,
    w_motif_cce: float = 1.0,
    w_motif_rmsd: float = 0.2,
    w_sc_rmsd: float = 0.1,
    w_plddt: float = 0.1,
    w_pae: float = 0.4,
    w_rg: float = 0.0,
    w_seq_ent: float = 0.1,
    w_cat_dist: float = 0.1,
    freeze_supervised_positions: bool = True,
    fix_supervised_identities: Optional[str] = "S,D,H,G,E",
    game_type: str = "none",
):
    pdb_bytes = None
    if pdb_path and Path(pdb_path).exists():
        pdb_bytes = Path(pdb_path).read_bytes()
    run_mhetase_game.remote(
        binder_len=binder_len,
        total_steps=total_steps,
        seed=seed,
        ser=ser,
        asp=asp,
        his=his,
        gly=gly,
        glu=glu,
        pdb_path=pdb_path,
        pdb_bytes=pdb_bytes,
        pdb_residues=pdb_residues,
        motif_chain_id=motif_chain_id,
        use_af2=use_af2,
        af2_num_recycles=af2_num_recycles,
        w_contact=w_contact,
        w_motif_cce=w_motif_cce,
        w_motif_rmsd=w_motif_rmsd,
        w_sc_rmsd=w_sc_rmsd,
        w_plddt=w_plddt,
        w_pae=w_pae,
        w_rg=w_rg,
        w_seq_ent=w_seq_ent,
        w_cat_dist=w_cat_dist,
        freeze_supervised_positions=freeze_supervised_positions,
        fix_supervised_identities=fix_supervised_identities,
        game_type=game_type,
    )


