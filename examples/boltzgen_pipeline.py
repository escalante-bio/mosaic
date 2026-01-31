import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from mosaic.models.boltzgen import load_boltzgen, load_features_and_structure_writer, Sampler
    from mosaic.common import TOKENS
    from mosaic.models.boltz2 import Boltz2, pad_atom_features
    from mosaic.models.boltzgen import BoltzGenOutput, CoordsToToken
    from mosaic.losses.structure_prediction import BinderPTMLoss, BinderTargetPAE, IPTMLoss, BinderTargetIPTM
    from mosaic.util import calculate_rmsd, bond_info
    from mosaic.structure_prediction import TargetChain
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.losses.protein_mpnn import jacobi_inverse_fold

    import time
    import json
    import secrets
    from typing import Optional, List
    from jaxtyping import Array
    import gemmi
    import polars as pl 
    import torch 
    import numpy as np
    import jax.numpy as jnp
    import jax 
    from pathlib import Path
    import equinox as eqx 
    from dataclasses import dataclass
    from os import devnull
    from contextlib import redirect_stdout, redirect_stderr

    start = time.time()
    return (
        BinderPTMLoss,
        BinderTargetIPTM,
        BinderTargetPAE,
        Boltz2,
        BoltzGenOutput,
        CoordsToToken,
        List,
        Optional,
        Path,
        Sampler,
        TOKENS,
        TargetChain,
        bond_info,
        calculate_rmsd,
        dataclass,
        devnull,
        eqx,
        gemmi,
        jacobi_inverse_fold,
        jax,
        jnp,
        json,
        load_boltzgen,
        load_features_and_structure_writer,
        load_mpnn_sol,
        np,
        pad_atom_features,
        pl,
        redirect_stderr,
        redirect_stdout,
        secrets,
        start,
        time,
        torch,
    )


@app.cell
def _():
    L_BINDER=70
    N_SAMPLES=24
    return L_BINDER, N_SAMPLES


@app.cell
def _(
    L_BINDER,
    N_SAMPLES,
    TARGET_SEQUENCE,
    TargetChain,
    boltz2,
    boltzgen,
    jax,
    run_boltzgen_pipeline,
    start,
    target_path,
    target_structure,
    time,
):
    samples = run_boltzgen_pipeline(
        N_SAMPLES,
        L_BINDER,
        target_path,
        TargetChain(
            TARGET_SEQUENCE, use_msa=False, template_chain=target_structure[0][0]
        ),
        jax.random.key(0),
        boltzgen=boltzgen,
        boltz2=boltz2,
    )
    print(f"Generating {N_SAMPLES} samples took {time.time() - start:.2f} seconds")
    return


@app.cell
def _(Path, gemmi):
    target_path = Path("/home/sam/targets/cif/IL7RA.cif")
    target_structure = gemmi.read_structure(str(target_path))
    target_structure.remove_ligands_and_waters()
    return target_path, target_structure


@app.cell
def _(gemmi, target_structure):
    TARGET_SEQUENCE = "".join(gemmi.one_letter_code([_r.name for _r in target_structure[0][0]]))
    return (TARGET_SEQUENCE,)


@app.cell
def _(load_features_and_structure_writer):
    def load_diffusion_features(binder_len, target_path):

        yaml_binder = r"""
        entities:
          - protein:
              id: A
              sequence: {N}

          - file:
              path: TARG.CIF

              include: 
                - chain:
                    id: B
        """.format(N=binder_len)
        features, _ = load_features_and_structure_writer(
            yaml_string=yaml_binder,
            files={"TARG.CIF": target_path},
        )
        return features
    return (load_diffusion_features,)


@app.cell
def _(Optional, dataclass, gemmi, np):
    @dataclass
    class BinderSample:
        diffusion_seq: str 
        seq: str 
        struct: gemmi.Structure
        bb_rmsd: float
        bb_rmsd_binder: float
        bb_rmsd_binder_alone: float
        binder_ptm: float
        binder_iptm: float
        neg_min_binder_target_pae: float
        n_hbonds: float
        n_saltbridges: float
        delta_sasa: float
        rank: Optional[int] = np.nan
        n_filters_passed: Optional[int] = np.nan 
    return (BinderSample,)


@app.cell
def _(Boltz2, load_boltzgen):
    boltzgen = load_boltzgen()
    boltz2 = Boltz2()
    return boltz2, boltzgen


@app.cell
def _(L_BINDER, TOKENS, jnp, load_mpnn_sol):
    MPNN = load_mpnn_sol()
    MPNN_BIAS = jnp.zeros((L_BINDER, 20)).at[:, TOKENS.index('C')].set(-1e6)
    MPNN_TEMP = 0.1
    return MPNN, MPNN_BIAS, MPNN_TEMP


@app.cell
def _(
    BoltzGenOutput,
    MPNN,
    MPNN_BIAS,
    MPNN_TEMP,
    eqx,
    jacobi_inverse_fold,
    jax,
    jnp,
):
    @eqx.filter_jit
    def sample_and_inverse_fold(
        key,
        binder_len,
        features,
        coords2token,
        sampler,
        structure_module,
        num_sampling_steps=300,
        mpnn=MPNN,
        mpnn_bias=MPNN_BIAS,
        mpnn_temp=MPNN_TEMP,
    ):
        key, k1 = jax.random.split(key)
        sample = sampler(
            structure_module=structure_module,
            num_sampling_steps=num_sampling_steps,
            step_scale=jnp.array(2.0),
            noise_scale=jnp.array(0.88),
            key=k1,
        )
        model_output = BoltzGenOutput(sample, features, coords2token)
        key, k2 = jax.random.split(key)
        mpnn_seq = jacobi_inverse_fold(
            mpnn,
            binder_len,
            model_output,
            mpnn_temp,  # mpnn temperature
            k2,
            bias=mpnn_bias,
        )
        return (
            jnp.argmax(model_output.full_sequence, -1)[:binder_len],
            model_output.backbone_coordinates,
            mpnn_seq,
        )
    return (sample_and_inverse_fold,)


@app.cell
def _(TOKENS):
    def tokens_to_str(tokens):
        return "".join([TOKENS[i] for i in tokens])
    return (tokens_to_str,)


@app.cell
def _(calculate_rmsd, eqx, jax, jnp):
    @eqx.filter_jit
    def batched_backbone_rmsd(x, y):
        ## calculate_rmsd expects (N,3) but backbone atoms are (M,4,3)
        return jax.vmap(calculate_rmsd)(jnp.vstack(x), jnp.vstack(y))
    return (batched_backbone_rmsd,)


@app.cell
def _(
    BinderPTMLoss,
    BinderSample,
    BinderTargetIPTM,
    BinderTargetPAE,
    CoordsToToken,
    Sampler,
    batched_backbone_rmsd,
    bond_info,
    jax,
    jnp,
    load_diffusion_features,
    load_padded_refold_features,
    rank,
    refold,
    sample_and_inverse_fold,
    tokens_to_str,
):
    def run_boltzgen_pipeline(
        num_samples,
        binder_len,
        target_path,
        target_chain,
        key,
        boltzgen,
        boltz2,
    ):
        diffusion_features = load_diffusion_features(binder_len, target_path)
        coords2token = CoordsToToken(diffusion_features)

        key, k1 = jax.random.split(key)
        sampler = Sampler.from_features(
            model=boltzgen,
            features=diffusion_features,
            key=k1,
            deterministic=True,
            recycling_steps=3,
        )

        key, k2 = jax.random.split(key)
        diffusion_seqs, diffusion_bb, mpnn_seqs = jax.vmap(
            lambda k: sample_and_inverse_fold(
                k,
                binder_len,
                diffusion_features,
                coords2token,
                sampler,
                boltzgen.structure_module,
            )
        )(jax.random.split(k2, num_samples))

        refold_complex_features, refold_writers = load_padded_refold_features(
            mpnn_seqs, boltz2, [target_chain]
        )

        metrics = {
            "ptm": BinderPTMLoss(),
            "neg_min_pae": BinderTargetPAE(reduce=jnp.min),
            "iptm": BinderTargetIPTM(),
        }

        key, k3 = jax.random.split(key)
        refold_coordinates, refold_bb, refold_metrics = jax.vmap(
            lambda k, feat: refold(k, feat, model=boltz2, metrics=metrics)
        )(
            jax.random.split(k3, num_samples),
            jax.tree.map(lambda *feat: jnp.stack(feat), *refold_complex_features),
        )

        refold_alone_features, _ = load_padded_refold_features(
            mpnn_seqs, boltz2, []
        )

        key, k4 = jax.random.split(key)
        _, refold_alone_bb, __ = jax.vmap(
            lambda k, feat: refold(k, feat, model=boltz2, metrics={})
        )(
            jax.random.split(k4, num_samples),
            jax.tree.map(lambda *feat: jnp.stack(feat), *refold_alone_features),
        )
        backbone_rmsd = batched_backbone_rmsd(diffusion_bb, refold_bb)
        backbone_rmsd_binder = batched_backbone_rmsd(
            diffusion_bb[:, :binder_len], refold_bb[:, :binder_len]
        )
        backbone_rmsd_binder_alone = batched_backbone_rmsd(
            diffusion_bb[:, :binder_len], refold_alone_bb
        )

        binder_samples = []
        for i in range(num_samples):
            refold_struct = refold_writers[i](refold_coordinates[i])
            n_hbonds, n_saltbridges, delta_sasa = bond_info(refold_struct)
            binder_samples.append(
                BinderSample(
                    diffusion_seq=tokens_to_str(diffusion_seqs[i]),
                    seq=tokens_to_str(mpnn_seqs[i]),
                    struct=refold_struct,
                    bb_rmsd=backbone_rmsd[i].item(),
                    bb_rmsd_binder=backbone_rmsd_binder[i].item(),
                    bb_rmsd_binder_alone=backbone_rmsd_binder_alone[i].item(),
                    binder_ptm=refold_metrics["ptm"][i].item(), 
                    binder_iptm=refold_metrics["iptm"][i].item(), 
                    neg_min_binder_target_pae=refold_metrics["neg_min_pae"][i].item(), 
                    n_hbonds=n_hbonds.item(),
                    n_saltbridges=n_saltbridges.item(),
                    delta_sasa=delta_sasa.item(),

                )
            )

        return rank(binder_samples)
    return (run_boltzgen_pipeline,)


@app.cell
def _(L_BINDER, eqx, jnp):
    @eqx.filter_jit
    def refold(key, features, model, metrics={}):
        output = model.model_output(features=features, key=key)
        pssm = jnp.zeros((L_BINDER, 20))
        metric_evals = {m: -fun(pssm, output, key=key)[0] for m,fun in metrics.items()}
        return output.structure_coordinates, output.backbone_coordinates, metric_evals
    return (refold,)


@app.cell
def _(
    TargetChain,
    devnull,
    np,
    pad_atom_features,
    redirect_stderr,
    redirect_stdout,
    tokens_to_str,
    torch,
):
    def load_padded_refold_features(sequences, folding_model, target_chains=[]):
        with redirect_stdout(open(devnull, "w")), redirect_stderr(open(devnull, "w")):
            unpadded_features_writers = [
                folding_model.target_only_features(
                    [
                        TargetChain(tokens_to_str(seq), use_msa=False),
                        *target_chains,
                    ]
                )
                for seq in sequences
            ]
        pad_length = max(
            fw[0]["atom_pad_mask"].shape[1] for fw in unpadded_features_writers
        )
        assert pad_length % 32 == 0

        padded_features, writers = [], []
        for f, w in unpadded_features_writers:
            padded_f = pad_atom_features(f, pad_length)
            w.atom_pad_mask = torch.Tensor(
                np.array(padded_f["atom_pad_mask"])[None]
            )
            padded_features.append(padded_f)
            writers.append(w)

        return padded_features, writers
    return (load_padded_refold_features,)


@app.cell
def _(BinderSample, L_BINDER, List, pl):
    # even though the boltzgen papers talks about ranking with binder_iiptm (what they call design_iiptm) they actually rank with binder_iptm in their code

    # bb_rmsd seems to fail often and heavily. Very correlated with iptm -- bad binders get reoriented vis-a-vis the target. Why does this not seem to happen in boltzgen?

    ranking_weights = {
        "binder_iptm": 1,
        "binder_ptm": 1,
        "neg_min_binder_target_pae": 1,
        "n_hbonds": 2,
        "n_saltbridges": 2,
        "delta_sasa": 2,
    }

    def filter(binder_sample: BinderSample):
        n_pass = binder_sample.bb_rmsd < 2.5
        n_pass += binder_sample.bb_rmsd_binder < 2.5
        n_pass += binder_sample.bb_rmsd_binder_alone < 2.5
        n_pass += sum(
            binder_sample.seq.count(aa) < 0.3 * L_BINDER for aa in "AGELV"
        )
        n_pass += binder_sample.seq.count("X") == 0
        return n_pass


    def rank(binder_samples: List[BinderSample], weights=ranking_weights):
        df = (
            pl.from_dicts(
                [
                    {k: _binder_sample.__dict__[k] for k in weights}
                    | {
                        "seq": _binder_sample.seq,
                        "num_filters_passed": filter(_binder_sample),
                    }
                    for _binder_sample in binder_samples
                ]
            )
            .with_columns(
                [
                    (
                        pl.struct(["num_filters_passed", col]).rank(
                            method="min", descending=True
                        )
                        / weights[col]
                    ).alias(f"rank_{col}")
                    for col in weights
                ]
            )
            .with_columns(
                pl.max_horizontal([f"rank_{col}" for col in weights]).alias(
                    "max_rank"
                )
            )
            .with_columns(
                pl.struct(["max_rank", -1.0 * pl.col("binder_iptm")])
                .rank(method="dense")
                .alias("final_rank")
            )
        )
        ranking = {_["seq"]: (_["final_rank"], _['num_filters_passed']) for _ in df.to_dicts()}
        for binder_sample in binder_samples:
            binder_sample.rank, binder_sample.n_filters_passed = ranking[binder_sample.seq]
        return binder_samples
    return (rank,)


@app.cell
def _(BinderSample, List, Path, json, secrets):
    def write_samples(samples: List[BinderSample], id, path: Path = Path(".")):
        stats_path = path / f"{id}.json"
        if (stats_path).exists():
            raise FileExistsError("output file {stats_path} exists, aborting write")
        def _write_pdb(sample: BinderSample, id, path: Path):
            struct_id = f"{id}_{secrets.token_hex(6)}.pdb"
            sample.struct.write_pdb(str(path / struct_id))
            sample.struct = struct_id
        out = []
        for sample in samples:
            _write_pdb(sample)
            out.append(sample.__dict__)    
        with open(stats_path, "w") as _jf:
                json.dump(out, _jf)
    return


if __name__ == "__main__":
    app.run()
