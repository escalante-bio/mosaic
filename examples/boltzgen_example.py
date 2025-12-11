import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    from mosaic.models.boltzgen import load_boltzgen, load_features_and_structure_writer, Sampler
    from mosaic.notebook_utils import gemmi_structure_from_models
    from mosaic.common import TOKENS
    from mosaic.models.boltz2 import Boltz2
    from mosaic.notebook_utils import pdb_viewer
    import gemmi
    import torch 
    import numpy as np
    import jax.numpy as jnp
    from mosaic.structure_prediction import TargetChain
    import jax
    import marimo as mo 
    from pathlib import Path
    import equinox as eqx
    return (
        Boltz2,
        Path,
        Sampler,
        TOKENS,
        TargetChain,
        eqx,
        gemmi,
        gemmi_structure_from_models,
        jax,
        jnp,
        load_boltzgen,
        load_features_and_structure_writer,
        np,
        pdb_viewer,
    )


@app.cell
def _(Boltz2):
    folding_model = Boltz2() # load a model for refolding -- because this is mosaic we're not limited to boltz2
    return (folding_model,)


@app.cell
def _(Path, gemmi):
    target_path= Path("IL7RA.cif")
    target_structure = gemmi.read_structure(str(target_path))
    target_structure.setup_entities()
    return target_path, target_structure


@app.cell
def _(gemmi, target_structure):
    TARGET_SEQUENCE = gemmi.one_letter_code([r.name for r in target_structure[0][0]])
    return (TARGET_SEQUENCE,)


@app.cell
def _():
    L_BINDER = 76
    return (L_BINDER,)


@app.cell
def _(load_boltzgen):
    boltzgen = load_boltzgen()
    return (boltzgen,)


@app.cell
def _():
    _helix = 23 * "H"
    secondary_structure_string = _helix + "L" * 4 + _helix + "L" *3 + _helix
    return (secondary_structure_string,)


@app.cell
def _(pdb_viewer, target_structure):
    pdb_viewer(target_structure)
    return


@app.cell
def _(
    L_BINDER,
    load_features_and_structure_writer,
    secondary_structure_string,
    target_path,
):
    yaml_binder = r"""
    entities:
      - protein:
          id: B
          sequence: {N}
          secondary_structure: {s}

      - file:
          path: TARG.CIF

          include: 
            - chain:
                id: A

    structure_groups:
      - group:
          id: A
          visibility: 2
    """.format(N=L_BINDER, s=secondary_structure_string)
    features_binder, writer_binder = load_features_and_structure_writer(
        yaml_string=yaml_binder,
        files={"TARG.CIF": target_path},
    )
    return features_binder, writer_binder


@app.cell
def _():
    # precompute trunk embedding and diffusion conditioning to save time when sampling
    return


@app.cell
def _(Sampler, boltzgen, eqx, features_binder, jax, np):
    # run trunk to precompute conditioning
    sampler = eqx.filter_jit(
        Sampler.from_features(
            model=boltzgen,
            features=features_binder,
            key=jax.random.key(np.random.randint(10000)),
            deterministic=True,
            recycling_steps=3,
        )
    )
    return (sampler,)


@app.cell
def _(boltzgen, jax, jnp, np, sampler):
    # sampler is a callable object that ... samples...
    coords = sampler(
            structure_module=boltzgen.structure_module,
            num_sampling_steps=300,
            step_scale=jnp.array(2.0),
            noise_scale=jnp.array(0.88),
            key=jax.random.key(np.random.randint(10000000)),
        )
    return (coords,)


@app.cell
def _(coords, writer_binder):
    # we can then transform these coordinates into a gemmi structure using the writer
    complex_structure = writer_binder(coords)
    return (complex_structure,)


@app.cell
def _(
    TOKENS,
    complex_structure,
    folding_model,
    gemmi,
    jax,
    np,
    p_features,
    p_writer,
):
    # and repredict with boltz2 if we'd like
    binder_seq = gemmi.one_letter_code(
        [r.name for r in complex_structure[0][0]]
    )
    prediction = folding_model.predict(
        PSSM=jax.nn.one_hot([TOKENS.index(c) for c in binder_seq], 20),
        features=p_features,
        writer=p_writer,
        key=jax.random.key(np.random.randint(1000000)),
        recycling_steps=1,
    )
    return (prediction,)


@app.cell
def _(pdb_viewer, prediction):
    pdb_viewer(prediction.st)
    return


@app.cell
def _():
    # we can wrap this all up in an efficient function... - should take about 3 seconds on an H100 after JIT
    return


@app.cell
def _():
    # sampler is compatible with vmap and other nice JAX things. This is *very* fast.
    return


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
def _(batch_sample, boltzgen, jax, sampler):
    _ = batch_sample(sampler, boltzgen.structure_module, 16, jax.random.key(0))
    return


@app.cell
def _(batch_sample, boltzgen, jax, np, sampler):
    batched_samples = np.array(batch_sample(sampler, boltzgen.structure_module, 16, jax.random.key(0)))
    return (batched_samples,)


@app.cell
def _(L_BINDER, TARGET_SEQUENCE, TargetChain, folding_model, target_structure):
    p_features, p_writer = folding_model.binder_features(
        L_BINDER,
        chains=[
            TargetChain(
                sequence=TARGET_SEQUENCE,
                use_msa=False,
                template_chain=target_structure[0][0],
            ),
        ],
    )
    return p_features, p_writer


@app.cell
def _(batched_samples, gemmi_structure_from_models, pdb_viewer, writer_binder):
    pdb_viewer(
        gemmi_structure_from_models(
            "", models=[writer_binder(c)[0] for c in batched_samples]
        )
    )
    return


if __name__ == "__main__":
    app.run()
