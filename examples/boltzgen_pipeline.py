import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from mosaic.models.boltzgen import load_boltzgen, load_features_and_structure_writer, Sampler
    from mosaic.notebook_utils import gemmi_structure_from_models
    from mosaic.common import TOKENS
    from mosaic.models.boltz2 import Boltz2, pad_atom_features
    from mosaic.models.boltzgen import BoltzGenOutput
    from mosaic.notebook_utils import pdb_viewer
    from mosaic.losses.structure_prediction import BinderPTMLoss, BinderTargetPAE, bond_info
    from mosaic.losses.boltz2 import calculate_iiptm
    from mosaic.util import calculate_rmsd
    from mosaic.structure_prediction import TargetChain
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.losses.protein_mpnn import jacobi_inverse_fold

    import time
    from typing import Optional
    from jaxtyping import Array
    import gemmi
    import torch 
    from tempfile import NamedTemporaryFile
    import numpy as np
    import jax.numpy as jnp
    import jax 
    from pathlib import Path
    import equinox as eqx 
    from dataclasses import dataclass
    from os import devnull
    from contextlib import redirect_stdout, redirect_stderr
    return (
        Array,
        BinderPTMLoss,
        BinderTargetPAE,
        Boltz2,
        BoltzGenOutput,
        Optional,
        Path,
        Sampler,
        TOKENS,
        TargetChain,
        bond_info,
        calculate_iiptm,
        calculate_rmsd,
        dataclass,
        devnull,
        eqx,
        gemmi,
        jacobi_inverse_fold,
        jax,
        jnp,
        load_boltzgen,
        load_features_and_structure_writer,
        load_mpnn_sol,
        np,
        pad_atom_features,
        redirect_stderr,
        redirect_stdout,
        torch,
    )


@app.cell
def _():
    L_BINDER=70
    N_SAMPLES=16
    return L_BINDER, N_SAMPLES


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
def _(L_BINDER, load_features_and_structure_writer, target_path):
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
    """.format(N=L_BINDER)
    features_boltzgen, writer_boltzgen = load_features_and_structure_writer(
        yaml_string=yaml_binder,
        files={"TARG.CIF": target_path},
    )
    return features_boltzgen, writer_boltzgen


@app.cell
def _(Array, Optional, dataclass, gemmi):
    @dataclass
    class BinderSample:
        diffusion_seq: Optional[str] = None
        diffusion_backbone: Optional[Array] = None
        seq: Optional[str] = None
        refold_backbone: Optional[Array] = None
        struct: Optional[gemmi.Structure] = None
        bb_rmsd: Optional[float] = None
        bb_rmsd_binder: Optional[float] = None
        bb_rmsd_binder_alone: Optional[float] = None
        binder_ptm: Optional[float] = None
        binder_iiptm: Optional[float] = None
        neg_min_binder_target_pae: Optional[float] = None
        n_hbonds: Optional[float] = None
        n_saltbridges: Optional[float] = None
        delta_sasa: Optional[float] = None
    return (BinderSample,)


@app.cell
def _(load_boltzgen):
    boltzgen = load_boltzgen()
    return (boltzgen,)


@app.cell
def _(Sampler, boltzgen, eqx, features_boltzgen, jax, np):
    sampler = eqx.filter_jit(
            Sampler.from_features(
                model=boltzgen,
                features=features_boltzgen,
                key=jax.random.key(np.random.randint(10000)),
                deterministic=True,
                recycling_steps=3,
            )
        )
    return (sampler,)


@app.cell
def _(eqx, jax, jnp):
    @eqx.filter_jit
    def batch_sample(sampler, structure_module, num_samples, key):
        print("JIT!")
        return jax.vmap(
            lambda k: sampler(
                structure_module=structure_module,
                num_sampling_steps=300,
                step_scale=jnp.array(2.0),
                noise_scale=jnp.array(0.88),
                key=k,
            )
        )(jax.random.split(key, num_samples))
    return (batch_sample,)


@app.cell
def _(L_BINDER, TOKENS, jacobi_inverse_fold, jax, jnp, load_mpnn_sol, np):
    MPNN = load_mpnn_sol()
    MPNN_BIAS = jnp.zeros((L_BINDER, 20)).at[:, TOKENS.index('C')].set(-1e6)
    MPNN_TEMP = 0.1

    def inverse_fold(model_output):
        return jacobi_inverse_fold(
            MPNN,
            L_BINDER,
            model_output,
            MPNN_TEMP,
            jax.random.key(np.random.randint(1000000)),
            bias=MPNN_BIAS,
        )
    return (inverse_fold,)


@app.cell
def _(
    BinderSample,
    BoltzGenOutput,
    N_SAMPLES,
    TOKENS,
    batch_sample,
    boltzgen,
    features_boltzgen,
    gemmi,
    inverse_fold,
    jax,
    sampler,
    writer_boltzgen,
):
    ## Sample from boltzgen

    boltzgen_coords = batch_sample(sampler, boltzgen.structure_module, N_SAMPLES, jax.random.key(0))

    def process_boltzgen_sample(sample):

        _boltzgen_output = BoltzGenOutput(sample=sample, features=features_boltzgen)
        _boltzgen_struct = writer_boltzgen(sample)
        _boltzgen_seq = "".join(gemmi.one_letter_code([r.name for r in _boltzgen_struct[0][0]]))
        return BinderSample(
            diffusion_seq = _boltzgen_seq,
            diffusion_backbone = _boltzgen_output.backbone_coordinates, 
            seq = "".join([TOKENS[_idx] for _idx in inverse_fold(_boltzgen_output)]),
        )

    binder_samples = [process_boltzgen_sample(_coord) for _coord in boltzgen_coords]
    return (binder_samples,)


@app.cell
def _(Boltz2):
    refolding_model = Boltz2()
    return (refolding_model,)


@app.cell
def _(
    BinderPTMLoss,
    BinderTargetPAE,
    L_BINDER,
    calculate_iiptm,
    eqx,
    jax,
    jnp,
    refolding_model,
):
    ptm_fun = BinderPTMLoss()
    pae_fun = BinderTargetPAE(reduce=jnp.min)

    @eqx.filter_jit
    def refold_and_conf(features, key, model=refolding_model):
        key, k_ptm, k_pae, k_iiptm = jax.random.split(key, 4)
        output = model.model_output(features=features, key=key)
        pssm = jnp.zeros((L_BINDER, 20))
        ptm = -ptm_fun(pssm, output, key=k_ptm)[0]
        pae = -pae_fun(pssm, output, key=k_pae)[0]
        return output.structure_coordinates, ptm, pae, calculate_iiptm(output)
    return (refold_and_conf,)


@app.cell
def _(eqx, refolding_model):
    @eqx.filter_jit
    def refold(features, key, model=refolding_model):
        return model.model_output(features=features, key=key).structure_coordinates
    return (refold,)


@app.cell
def _(
    TARGET_SEQUENCE,
    TargetChain,
    binder_samples,
    devnull,
    redirect_stderr,
    redirect_stdout,
    refolding_model,
    target_structure,
):
    with redirect_stdout(open(devnull, "w")), redirect_stderr(open(devnull, "w")):
        refold_features_writers = [
            refolding_model.target_only_features(
                [
                    TargetChain(binder_sample.seq, use_msa=False),
                    TargetChain(
                        TARGET_SEQUENCE,
                        use_msa=False,
                        template_chain=target_structure[0][0],
                    ),
                ]
            )
            for binder_sample in binder_samples
        ]

    pad_length = max(
        _feature_writer[0]["atom_pad_mask"].shape[1]
        for _feature_writer in refold_features_writers
    )
    assert pad_length % 32 == 0
    return pad_length, refold_features_writers


@app.cell
def _(
    L_BINDER,
    binder_samples,
    bond_info,
    calculate_rmsd,
    jax,
    jnp,
    np,
    pad_atom_features,
    pad_length,
    refold_and_conf,
    refold_features_writers,
    torch,
):
    for _sample, (_features, _writer) in zip(
        binder_samples, refold_features_writers
    ):
        _features = pad_atom_features(_features, pad_length)
        (
            _coords,
            _sample.binder_ptm,
            _sample.neg_min_binder_target_pae,
            _sample.binder_iiptm,
        ) = refold_and_conf(_features, jax.random.key(0))

        _sample.refold_backbone = _coords[
            _features["atom_backbone_mask"].astype(bool)
        ].reshape((-1, 4, 3))
        _sample.bb_rmsd = calculate_rmsd(
            jnp.vstack(_sample.diffusion_backbone),
            jnp.vstack(_sample.refold_backbone),
        )
        _sample.bb_rmsd_binder = calculate_rmsd(
            jnp.vstack(_sample.diffusion_backbone[:L_BINDER]),
            jnp.vstack(_sample.refold_backbone[:L_BINDER]),
        )

        _writer.atom_pad_mask = torch.Tensor(
            np.array(_features["atom_pad_mask"])[None]
        )
        _sample.struct = _writer(_coords)
        _sample.n_hbonds, _sample.n_saltbridges, _sample.delta_sasa = bond_info(
            _sample.struct
        )
    return


@app.cell
def _(
    TargetChain,
    binder_samples,
    devnull,
    redirect_stderr,
    redirect_stdout,
    refolding_model,
):
    with redirect_stdout(open(devnull, "w")), redirect_stderr(open(devnull, "w")):

        binder_alone_features = [
            refolding_model.target_only_features(
                [TargetChain(binder_sample.seq, use_msa=False)]
            )[0] #dont need writers
            for binder_sample in binder_samples
        ]

    pad_length_binder = max(
        _feature["atom_pad_mask"].shape[1]
        for _feature in binder_alone_features
    )
    assert pad_length_binder % 32 == 0
    return binder_alone_features, pad_length_binder


@app.cell
def _(
    L_BINDER,
    binder_alone_features,
    binder_samples,
    calculate_rmsd,
    jax,
    jnp,
    pad_atom_features,
    pad_length_binder,
    refold,
):
    for _sample, _binder_features in zip(binder_samples, binder_alone_features):
        _binder_features = pad_atom_features(_binder_features, pad_length_binder)
        _binder_coords = refold(_binder_features, jax.random.key(0))
        _binder_backbone = _binder_coords[
            _binder_features["atom_backbone_mask"].astype(bool)
        ]
        _sample.bb_rmsd_binder_alone = calculate_rmsd(
            jnp.vstack(_sample.diffusion_backbone[:L_BINDER]), _binder_backbone
        )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
