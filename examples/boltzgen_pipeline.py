import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from mosaic.models.boltzgen import load_boltzgen, load_features_and_structure_writer, Sampler
    from mosaic.common import TOKENS
    from mosaic.models.boltz2 import Boltz2, pad_atom_features
    from mosaic.losses.boltz2 import Boltz2Output, Boltz2FromTrunkOutput
    from mosaic.losses.structure_prediction import BinderTargetIPSAE, TargetBinderIPSAE      
    from mosaic.models.boltzgen import BoltzGenOutput, CoordsToToken
    from mosaic.losses.structure_prediction import BinderPTMLoss, BinderTargetPAE, IPTMLoss, BinderTargetIPTM
    from mosaic.util import calculate_rmsd, bond_info
    from mosaic.structure_prediction import TargetChain
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.losses.protein_mpnn import jacobi_inverse_fold

    import time
    import copy
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
    return (
        BinderTargetIPSAE,
        Boltz2,
        Boltz2FromTrunkOutput,
        Boltz2Output,
        BoltzGenOutput,
        CoordsToToken,
        IPTMLoss,
        List,
        Optional,
        Path,
        Sampler,
        TOKENS,
        TargetBinderIPSAE,
        TargetChain,
        calculate_rmsd,
        copy,
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
        redirect_stderr,
        redirect_stdout,
        secrets,
        time,
        torch,
    )


@app.cell
def _(Path, gemmi):
    target_path = Path("/home/sam/targets/PDL1_TED_18_131.pdb")
    target_structure = gemmi.read_structure(str(target_path))
    target_structure.remove_ligands_and_waters()
    return target_path, target_structure


@app.cell
def _():
    # N_SAMPLES will be rounded up to the nearest multiple of BATCH_SIZE to prevent recompilation
    L_BINDER=80
    N_SAMPLES=60
    BATCH_SIZE=12
    return BATCH_SIZE, L_BINDER, N_SAMPLES


@app.cell
def _(
    BATCH_SIZE,
    L_BINDER,
    N_SAMPLES,
    TARGET_SEQUENCE,
    TargetChain,
    boltz2,
    boltzgen,
    jax,
    np,
    rank,
    run_boltzgen_pipeline,
    target_path,
    target_structure,
    time,
):
    samples = []
    start = time.time()
    for _i in range(0, N_SAMPLES, BATCH_SIZE):
        batch = run_boltzgen_pipeline(
                BATCH_SIZE,
                L_BINDER,
                target_path,
                TargetChain(
                    TARGET_SEQUENCE, use_msa=False, template_chain=target_structure[0][0]
                ),
                key=jax.random.key(np.random.randint(1000000)),
                boltzgen=boltzgen,
                boltz2=boltz2,
            )    
        samples.extend(batch)
        end = time.time()
        print(f"Generating {BATCH_SIZE} samples took {end - start:.2f} seconds")
        start = end
    
    ranked_samples = rank(samples, filter_rmsd=2.5)
    return


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
              id: B
              sequence: {N}

          - file:
              path: TARG{SUFFIX}

              include: 
                - chain:
                    id: A
        """.format(N=binder_len, SUFFIX=target_path.suffix)
        features, _ = load_features_and_structure_writer(
            yaml_string=yaml_binder,
            files={f"TARG{target_path.suffix}": target_path},
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
        ranking_loss: float
        rank: Optional[int] = np.nan
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
        return jax.vmap(lambda i, j: calculate_rmsd(jnp.vstack(i), jnp.vstack(j)))(
            x, y
        )
    return (batched_backbone_rmsd,)


@app.cell
def _(
    BinderSample,
    BinderTargetIPSAE,
    CoordsToToken,
    IPTMLoss,
    Sampler,
    TargetBinderIPSAE,
    batched_backbone_rmsd,
    jax,
    jnp,
    load_diffusion_features,
    load_padded_refold_features,
    multifold,
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
        complex_pad=0,
        alone_pad=0,
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
            mpnn_seqs, boltz2, [target_chain],
        )

        ranking_loss = 1.0 * IPTMLoss() + 0.5 * TargetBinderIPSAE() + 0.5 * BinderTargetIPSAE()
    
        key, k3 = jax.random.split(key)
        refold_outputs = jax.vmap(
            lambda k, feat: multifold(k, feat, model=boltz2, loss=ranking_loss, num_samples=5)
        )(
            jax.random.split(k3, num_samples),
            jax.tree.map(lambda *feat: jnp.stack(feat), *refold_complex_features),
        )
    
        refold_alone_features, _ = load_padded_refold_features(
            mpnn_seqs, boltz2, [],
        )

        key, k4 = jax.random.split(key)
        refold_alone_outputs = jax.vmap(
            lambda k, feat: refold(k, feat, model=boltz2)
        )(
            jax.random.split(k4, num_samples),
            jax.tree.map(lambda *feat: jnp.stack(feat), *refold_alone_features),
        )
        backbone_rmsd = batched_backbone_rmsd(diffusion_bb, refold_outputs.backbone_coordinates)
        backbone_rmsd_binder = batched_backbone_rmsd(
            diffusion_bb[:, :binder_len], refold_outputs.backbone_coordinates[:, :binder_len]
        )
        backbone_rmsd_binder_alone = batched_backbone_rmsd(
            diffusion_bb[:, :binder_len], refold_alone_outputs.backbone_coordinates
        )

        binder_samples = []
        for i in range(num_samples):
            refold_struct = refold_writers[i](refold_outputs.structure_coordinates[i])
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
def _(Boltz2FromTrunkOutput, Boltz2Output, L_BINDER, eqx, jax, jnp):
    class FoldOutput(eqx.Module):
        loss: float
        structure_coordinates: jax.Array
        backbone_coordinates: jax.Array
        def best(self):
            if self.loss.ndim == 0:
                return self
            i = jnp.argmin(self.loss)
            return jax.tree.map(lambda v: v[i], self)

    @eqx.filter_jit
    def multifold(key, features, model, loss, num_samples):
        key, subkey = jax.random.split(key, 2)
        output=Boltz2Output(
                joltz2=model.model,
                features=features,
                deterministic=True,
                key=subkey,
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
                    sequence=jnp.zeros((L_BINDER, 20)),
                    output=from_trunk_output,
                    key=key,
                )
            return FoldOutput(v, from_trunk_output.structure_coordinates, from_trunk_output.backbone_coordinates)

        output = jax.vmap(apply_loss_to_single_sample)(
                jax.random.split(key, num_samples)
            )
        return output.best()

    return FoldOutput, multifold


@app.cell
def _(FoldOutput, eqx):
    @eqx.filter_jit
    def refold(key, features, model):
        output = model.model_output(features=features, key=key)
        return FoldOutput(0.0, output.structure_coordinates, output.backbone_coordinates)
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
    def load_padded_refold_features(
        sequences, folding_model, target_chains=[]
    ):        
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

        pad_length = sequences[0].size * 14 + target_atom_size #max 14 heavy atoms per residue
        pad_length = ((pad_length + 31) // 32) * 32 #boltz needs a multiple of 32
    
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
def _(BinderSample, List):
    def rank(binder_samples: List[BinderSample], filter_rmsd=2.5):
        binder_samples = sorted(binder_samples, key=lambda x: x.ranking_loss)
        for rank, sample in enumerate(binder_samples):
            sample.rank = rank
            if (
                sample.bb_rmsd > filter_rmsd
                or sample.bb_rmsd_binder > filter_rmsd
                or sample.bb_rmsd_binder_alone > filter_rmsd
            ):
                sample.rank = len(binder_samples)
        return binder_samples
    return (rank,)


@app.cell
def _(BinderSample, List, Path, copy, json, secrets):
    def write_samples(samples: List[BinderSample], run_id, path: Path = Path(".")):
        path.mkdir(exist_ok=True, parents=True)
        (path / "structs").mkdir(exist_ok=True)
        stats_path = path / f"{run_id}.json"
        if (stats_path).exists():
            raise FileExistsError("output file {stats_path} exists, aborting write")
        def _write_pdb(sample: BinderSample, id, path: Path):
            sample = copy.copy(sample)
            struct_id = f"{id}_{secrets.token_hex(6)}.pdb"
            sample.struct.write_pdb(str(path / "structs" / struct_id))
            sample.struct = "structs/" + struct_id
            return sample
        out = []
        for sample in samples:
            sample = _write_pdb(sample, run_id, path)
            out.append(sample.__dict__)    
        with open(stats_path, "w") as _jf:
                json.dump(out, _jf)
    return


if __name__ == "__main__":
    app.run()
