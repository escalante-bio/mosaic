import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    from mosaic.models.boltzgen import (
        load_boltzgen,
        load_features_and_structure_writer,
        Sampler,
        BoltzGenOutput,
        CoordsToToken,
    )
    from mosaic.models.boltz2 import Boltz2, pad_atom_features
    from mosaic.losses.boltz2 import Boltz2Output, Boltz2FromTrunkOutput
    from mosaic.losses.structure_prediction import (
        BinderTargetIPSAE,
        TargetBinderIPSAE,
        IPTMLoss,
    )
    from mosaic.losses.protein_mpnn import jacobi_inverse_fold
    from mosaic.util import calculate_rmsd, fold_in
    from mosaic.structure_prediction import TargetChain
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.common import TOKENS

    import time
    import json
    import secrets
    import urllib
    import gemmi
    import polars as pl
    import torch
    import numpy as np
    import jax.numpy as jnp
    import jax
    import equinox as eqx
    from dataclasses import dataclass
    from pathlib import Path
    from os import devnull
    from contextlib import redirect_stdout, redirect_stderr
    from tempfile import NamedTemporaryFile
    from typing import Optional, List
    from jaxtyping import Array

    return (
        BinderTargetIPSAE,
        Boltz2,
        Boltz2FromTrunkOutput,
        Boltz2Output,
        BoltzGenOutput,
        CoordsToToken,
        IPTMLoss,
        List,
        NamedTemporaryFile,
        Path,
        Sampler,
        TOKENS,
        TargetBinderIPSAE,
        TargetChain,
        calculate_rmsd,
        dataclass,
        devnull,
        eqx,
        fold_in,
        gemmi,
        jacobi_inverse_fold,
        jax,
        jnp,
        load_boltzgen,
        load_features_and_structure_writer,
        load_mpnn_sol,
        np,
        pad_atom_features,
        pl,
        redirect_stderr,
        redirect_stdout,
        secrets,
        time,
        torch,
        urllib,
    )


@app.cell
def _():
    FILTER_RMSD: float = 2.5
    return (FILTER_RMSD,)


@app.cell
def _(Path, download_target):
    ### User settings

    BINDER_LEN = 80

    target = download_target("3di3")  # Interleukin-7 Complex
    TARGET_CHAIN = target[0]["B"]  # Chain B is IL7Ra

    RUN_ID = "3di3_0"
    OUT_PATH = Path(".") / "3di3"

    N_SAMPLES = 60
    BATCH_SIZE = 12

    # N_SAMPLES will be rounded up to the nearest multiple of BATCH_SIZE to prevent recompilation
    return BATCH_SIZE, BINDER_LEN, N_SAMPLES, OUT_PATH, RUN_ID, TARGET_CHAIN


@app.cell
def _(
    BATCH_SIZE,
    BINDER_LEN,
    N_SAMPLES,
    TARGET_CHAIN,
    jax,
    np,
    ranking_score,
    run_boltzgen_pipeline,
    time,
):
    samples = []
    start = time.time()
    start_batch = start
    for _i in range(0, N_SAMPLES, BATCH_SIZE):
        batch = run_boltzgen_pipeline(
            BATCH_SIZE,
            BINDER_LEN,
            TARGET_CHAIN,
            key=jax.random.key(np.random.randint(1000000)),
        )
        samples.extend(batch)
        end_batch = time.time()
        print(
            f"Batch {_i // BATCH_SIZE}: generated {BATCH_SIZE} samples in {end_batch - start_batch:.2f} seconds"
        )
        start_batch = end_batch

    print(f"Generated {len(samples)} samples in {time.time() - start:.2f} seconds")
    samples = sorted(samples, key=ranking_score)
    return (samples,)


@app.cell
def _(OUT_PATH, RUN_ID, samples, write_structures):
    # write results + structures to disk
    results_dataframe = write_structures(samples, run_id=RUN_ID, path=OUT_PATH)
    results_dataframe.write_json(OUT_PATH / f"{RUN_ID}.json")
    results_dataframe
    return


@app.cell
def _(Boltz2, load_boltzgen, load_mpnn_sol):
    BOLTZGEN = load_boltzgen()
    BOLTZ2 = Boltz2()
    MPNN = load_mpnn_sol()

    return BOLTZ2, BOLTZGEN, MPNN


@app.cell
def _(BINDER_LEN, TOKENS, jnp):
    MPNN_BIAS = jnp.zeros((BINDER_LEN, 20)).at[:, TOKENS.index("C")].set(-1e6)
    MPNN_TEMP = 0.1
    return MPNN_BIAS, MPNN_TEMP


@app.cell
def _(gemmi, urllib):
    def download_target(pdb_id: str) -> gemmi.Structure:
        with urllib.request.urlopen(
            f"https://files.rcsb.org/download/{pdb_id}.cif"
        ) as response:
            st = gemmi.make_structure_from_block(
                gemmi.cif.read_string(response.read().decode("utf-8"))[0]
            )
        st.remove_ligands_and_waters()
        st.remove_empty_chains()
        return st

    return (download_target,)


@app.cell
def _(
    BOLTZ2,
    BOLTZGEN,
    BinderSample,
    BinderTargetIPSAE,
    CoordsToToken,
    IPTMLoss,
    MPNN,
    MPNN_BIAS,
    MPNN_TEMP,
    Sampler,
    TargetBinderIPSAE,
    TargetChain,
    batched_backbone_rmsd,
    fold_in,
    gemmi,
    jax,
    jnp,
    load_diffusion_features,
    load_padded_refold_features,
    multifold,
    sample_and_inverse_fold,
    tokens_to_str,
):
    def run_boltzgen_pipeline(
        num_samples: int,
        binder_len: int,
        target_chain: gemmi.Chain,
        key,
        boltzgen=BOLTZGEN,
        boltz2=BOLTZ2,
        mpnn=MPNN,
        mpnn_temp=MPNN_TEMP,
        mpnn_bias=MPNN_BIAS,
    ):
        diffusion_features = load_diffusion_features(binder_len, target_chain)
        coords2token = CoordsToToken(diffusion_features)

        sampler = Sampler.from_features(
            model=boltzgen,
            features=diffusion_features,
            key=fold_in(key, "sampler"),
            deterministic=True,
            recycling_steps=3,
        )

        diffusion_seqs, diffusion_bb, mpnn_seqs = jax.vmap(
            lambda k: sample_and_inverse_fold(
                k,
                binder_len,
                diffusion_features,
                coords2token,
                sampler,
                boltzgen.structure_module,
            )
        )(jax.random.split(fold_in(key, "diffusion"), num_samples))

        target_sequence = "".join(
            gemmi.one_letter_code([_r.name for _r in target_chain])
        )

        refold_complex_features, refold_writers = load_padded_refold_features(
            mpnn_seqs,
            boltz2,
            [
                TargetChain(
                    target_sequence, use_msa=False, template_chain=target_chain
                )
            ],
        )

        ranking_loss = (
            1.0 * IPTMLoss() + 0.5 * TargetBinderIPSAE() + 0.5 * BinderTargetIPSAE()
        )

        refold_outputs = jax.vmap(
            lambda k, feat: multifold(
                k, feat, model=boltz2, loss=ranking_loss, num_samples=5
            )
        )(
            jax.random.split(fold_in(key, "refold"), num_samples),
            jax.tree.map(lambda *feat: jnp.stack(feat), *refold_complex_features),
        )

        refold_alone_features, _ = load_padded_refold_features(
            mpnn_seqs,
            boltz2,
            [],
        )

        refold_alone_outputs = jax.vmap(
            lambda k, feat: multifold(
                k, feat, model=boltz2, loss=lambda sequence, output, key: (0.0, {'zero': 0.0}), num_samples=1
            )
        )(
            jax.random.split(fold_in(key, "monomer"), num_samples),
            jax.tree.map(lambda *feat: jnp.stack(feat), *refold_alone_features),
        )
        backbone_rmsd = batched_backbone_rmsd(
            diffusion_bb, refold_outputs.backbone_coordinates
        )
        backbone_rmsd_binder = batched_backbone_rmsd(
            diffusion_bb[:, :binder_len],
            refold_outputs.backbone_coordinates[:, :binder_len],
        )
        backbone_rmsd_binder_alone = batched_backbone_rmsd(
            diffusion_bb[:, :binder_len], refold_alone_outputs.backbone_coordinates
        )

        binder_samples = []
        for i in range(num_samples):
            refold_struct = refold_writers[i](
                refold_outputs.structure_coordinates[i]
            )
            binder_samples.append(
                BinderSample(
                    diffusion_seq=tokens_to_str(diffusion_seqs[i]),
                    seq=tokens_to_str(mpnn_seqs[i]),
                    struct=refold_struct,
                    bb_rmsd=backbone_rmsd[i].item(),
                    bb_rmsd_binder=backbone_rmsd_binder[i].item(),
                    bb_rmsd_binder_alone=backbone_rmsd_binder_alone[i].item(),
                    ranking_loss=refold_outputs.loss[i].item(),
                )
            )

        return binder_samples

    return (run_boltzgen_pipeline,)


@app.cell
def _(NamedTemporaryFile, gemmi, load_features_and_structure_writer):
    def load_diffusion_features(binder_len: int, target_chain: gemmi.Chain):

        struct = gemmi.Structure()
        model = gemmi.Model("0")
        model.add_chain(target_chain)
        struct.add_model(model)
        struct[0][0].name = "A"  # reset chain name

        with NamedTemporaryFile(suffix=".pdb", mode="w") as tf:
            struct.write_pdb(tf.name)
            yaml_binder = r"""
            entities:
              - protein:
                  id: B
                  sequence: {N}

              - file:
                  path: {target_file}

                  include: 
                    - chain:
                        id: A
            """.format(N=binder_len, target_file=tf.name)
            features, _ = load_features_and_structure_writer(
                yaml_string=yaml_binder
            )
        return features

    return (load_diffusion_features,)


@app.cell
def _(dataclass, gemmi):
    @dataclass(frozen=True)
    class BinderSample:
        diffusion_seq: str
        seq: str
        struct: gemmi.Structure
        bb_rmsd: float
        bb_rmsd_binder: float
        bb_rmsd_binder_alone: float
        ranking_loss: float

    return (BinderSample,)


@app.cell
def _(BinderSample, FILTER_RMSD: float):
    def ranking_score(sample: BinderSample, filter_rmsd: float = FILTER_RMSD):
        passes_filters = (
            sample.bb_rmsd < filter_rmsd
            and sample.bb_rmsd_binder < filter_rmsd
            and sample.bb_rmsd_binder_alone < filter_rmsd
        )
        return sample.ranking_loss + float("inf") if not passes_filters else 0.0

    return (ranking_score,)


@app.cell
def _(
    BoltzGenOutput,
    MPNN,
    MPNN_BIAS,
    MPNN_TEMP,
    eqx,
    fold_in,
    jacobi_inverse_fold,
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
        num_sampling_steps=500,
        mpnn=MPNN,
        mpnn_bias=MPNN_BIAS,
        mpnn_temp=MPNN_TEMP,
    ):

        sample = sampler(
            structure_module=structure_module,
            num_sampling_steps=num_sampling_steps,
            step_scale=jnp.array(2.0),
            noise_scale=jnp.array(0.88),
            key=fold_in(key, "sampler"),
        )
        model_output = BoltzGenOutput(sample, features, coords2token)

        mpnn_seq = jacobi_inverse_fold(
            mpnn,
            binder_len,
            model_output,
            mpnn_temp,  # mpnn temperature
            fold_in(key, "inverse fold"),
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
def _(calculate_rmsd, eqx, jax):
    @eqx.filter_jit
    def batched_backbone_rmsd(x, y):
        ## calculate_rmsd expects (N,3) but backbone atoms are (M,4,3)
        return jax.vmap(
            lambda i, j: calculate_rmsd(i.reshape(-1, 3), j.reshape(-1, 3))
        )(x, y)

    return (batched_backbone_rmsd,)


@app.cell
def _(BINDER_LEN, Boltz2FromTrunkOutput, Boltz2Output, eqx, fold_in, jax, jnp):
    class FoldOutput(eqx.Module):
        loss: float
        structure_coordinates: jax.Array
        backbone_coordinates: jax.Array


    @eqx.filter_jit
    def multifold(key, features, model, loss, num_samples):
        """
        Refold multiple times (one trunk run, multiple diffusion samples)
        and pick the best (lowest) according to a mosaic loss functional
        """
        output = Boltz2Output(
            joltz2=model.model,
            features=features,
            deterministic=True,
            key=fold_in(key, "trunk"),
            recycling_steps=3,
        )

        def apply_loss_to_single_sample(key):
            from_trunk_output = Boltz2FromTrunkOutput(
                joltz2=model.model,
                features=features,
                deterministic=True,
                key=key,
                initial_embedding=output.initial_embedding,
                trunk_state=output.trunk_state,
                recycling_steps=3,
            )
            v, aux = loss(
                sequence=jnp.zeros((BINDER_LEN, 20)),
                output=from_trunk_output,
                key=fold_in(key, "loss"),
            )
            return FoldOutput(
                v,
                from_trunk_output.structure_coordinates,
                from_trunk_output.backbone_coordinates,
            )

        output = jax.vmap(apply_loss_to_single_sample)(
            jax.random.split(fold_in(key, "samples"), num_samples)
        )
        indmin = jnp.argmin(output.loss)
        return jax.tree.map(lambda v: v[indmin], output)

    return (multifold,)


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
        """
        load folding features for a number of sequences, and pad them all to the same number of atoms
        """
        with (
            redirect_stdout(open(devnull, "w")),
            redirect_stderr(open(devnull, "w")),
        ):
            if target_chains:
                target_feat, _ = folding_model.target_only_features(target_chains)
                target_atom_size = target_feat["atom_pad_mask"].shape[-1]
            else:
                target_atom_size = 0

            unpadded_features_writers = [
                folding_model.target_only_features(
                    [
                        TargetChain(tokens_to_str(seq), use_msa=False),
                        *target_chains,
                    ]
                )
                for seq in sequences
            ]
        max_atom_size = max(
            fw[0]["atom_pad_mask"].shape[-1] for fw in unpadded_features_writers
        )

        pad_length = (
            sequences[0].size * 14 + target_atom_size
        )  # max 14 heavy atoms per residue
        pad_length = ((pad_length + 31) // 32) * 32  # boltz needs a multiple of 32

        assert pad_length >= max_atom_size
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
def _(BinderSample, List, Path, pl, secrets):
    def write_structures(
        samples: List[BinderSample], run_id, path: Path = Path(".")
    ) -> pl.DataFrame:
        """
        Write a list of samples. Stats will be written to <path>/<run_id>_stats.json and
        refolded structures to <path>/structs/
        """
        path.mkdir(exist_ok=True, parents=True)
        (path / "structs").mkdir(exist_ok=True)

        def _write_pdb(sample: BinderSample, id, path: Path):
            struct_id = f"{id}_{secrets.token_hex(6)}.pdb"
            struct_path = path / "structs" / struct_id
            sample.struct.write_pdb(str(struct_path))
            return struct_path

        def sample_to_row(sample):
            return {
                k: getattr(sample, k)
                for k in [
                    "diffusion_seq",
                    "seq",
                    "bb_rmsd",
                    "bb_rmsd_binder",
                    "bb_rmsd_binder_alone",
                    "ranking_loss",
                ]
            } | {"struct": str(_write_pdb(sample, run_id, path))}

        return pl.DataFrame([sample_to_row(sample) for sample in samples])

    return (write_structures,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
