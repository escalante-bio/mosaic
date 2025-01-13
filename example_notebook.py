import marimo

__generated_with = "0.10.12"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from ipymolstar import PDBeMolstar
    from pathlib import Path
    from boltz.model.model import Boltz1
    from boltz.main import BoltzDiffusionParams
    from dataclasses import asdict
    import joltz
    import jax
    import equinox as eqx
    import numpy as np
    import optax
    import boltz
    import matplotlib.pyplot as plt
    from boltz_binder_design import (
        make_binder_monomer_features,
        make_binder_features,
        design_bregman_optax,
        make_monomer_features,
        target_fasta_seq,
        load_features_and_structure_writer,
    )

    from boltz_binder_design.losses import (
        set_binder_sequence,
        RadiusOfGyration,
        WithinBinderContact,
        BinderTargetContact,
        HelixLoss,
        StructurePrediction,
    )
    return (
        BinderTargetContact,
        Boltz1,
        BoltzDiffusionParams,
        HelixLoss,
        PDBeMolstar,
        Path,
        RadiusOfGyration,
        StructurePrediction,
        WithinBinderContact,
        asdict,
        boltz,
        design_bregman_optax,
        eqx,
        jax,
        joltz,
        load_features_and_structure_writer,
        make_binder_features,
        make_binder_monomer_features,
        make_monomer_features,
        mo,
        np,
        optax,
        plt,
        set_binder_sequence,
        target_fasta_seq,
    )


@app.cell
def _(PDBeMolstar, Path):
    def pdb_viewer(file: Path):
        custom_data = {
            "data": file.read_text(),
            "format": "pdb",
            "binary": False,
        }
        return PDBeMolstar(
            custom_data=custom_data,
            hide_settings=True,
            hide_controls_icon=False,
            hide_expand_icon=True,
            hide_settings_icon=True,
            hide_selection_icon=True,
            hide_animation_icon=False,
            hide_water=True,
            hide_carbs=True,
            alphafold_view=True,
            theme="dark",
        )
    return (pdb_viewer,)


@app.cell
def _(Boltz1, BoltzDiffusionParams, Path, asdict, eqx, jax, joltz):
    # load boltz model and convert to JAX
    predict_args = {
        "recycling_steps": 1,
        "sampling_steps": 25,
        "diffusion_samples": 1,
    }

    _torch_model = Boltz1.load_from_checkpoint(
        Path("~/.boltz/boltz1_conf.ckpt").expanduser(),
        strict=True,
        map_location="cpu",
        predict_args=predict_args,
        diffusion_process_args=asdict(BoltzDiffusionParams()),
        ema=False,
    )
    model = joltz.from_torch(_torch_model)
    _model_params, _model_static = eqx.partition(model, eqx.is_inexact_array)
    model = eqx.combine(jax.device_put(_model_params), _model_static)
    return model, predict_args


@app.cell
def _(eqx, model):
    j_model = eqx.filter_jit(model)
    return (j_model,)


@app.cell
def _(eqx, j_model, jax, pdb_viewer, set_binder_sequence):
    def predict(sequence, features, writer):
        o = j_model(
            eqx.filter_jit(set_binder_sequence)(sequence, features),
            key=jax.random.key(5),
            sample_structure=True,
            confidence_prediction=True,
        )
        out_path = writer(o["sample_atom_coords"])
        viewer = pdb_viewer(out_path)
        print(out_path)
        print(o["plddt"].mean())
        return o, viewer
    return (predict,)


@app.cell
def _():
    binder_length = 80
    return (binder_length,)


@app.cell
def _(binder_length, make_binder_features):
    boltz_features, boltz_writer = make_binder_features(
        binder_length,
        "ETRECIYYNANWELERTNQSGLERCEGEQDKRLHCYASWRNSSGTIELVKKGCWLDDFNCYDRQECVATEENPQVYFCCCEGNFCNERFTHLP",
    )
    return boltz_features, boltz_writer


@app.cell
def _(
    BinderTargetContact,
    HelixLoss,
    RadiusOfGyration,
    StructurePrediction,
    WithinBinderContact,
    boltz_features,
    model,
):
    loss = StructurePrediction(
        model=model,
        name="boltz",
        loss=4 * BinderTargetContact()
        + 1.0
        * (
            1.0 * RadiusOfGyration(target_radius=15.0)
            + WithinBinderContact()
            + 0.3 * HelixLoss()
        ),
        features=boltz_features,
        recycling_steps=0,
    )
    return (loss,)


@app.cell
def _(binder_length, design_bregman_optax, loss, np, optax):
    logits = design_bregman_optax(
        loss_function=loss,
        n_steps=150,
        x=np.random.randn(binder_length, 20) * 0.1,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0), optax.sgd(np.sqrt(binder_length))
        ),
    )
    return (logits,)


@app.cell
def _(binder_length, design_bregman_optax, logits, loss, np, optax):
    # we can sharpen these logits using weight decay (which is equivalent to adding entropic regularization)
    logits_sharper = design_bregman_optax(
        loss_function=loss,
        n_steps=50,
        x=logits,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.025),
            optax.sgd(np.sqrt(binder_length)),
        ),
    )
    logits_sharper = design_bregman_optax(
        loss_function=loss,
        n_steps=50,
        x=logits_sharper,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.05),
            optax.sgd(np.sqrt(binder_length)),
        ),
    )
    return (logits_sharper,)


@app.cell
def _(boltz_features, boltz_writer, jax, logits_sharper, predict):
    model_output, viewer = predict(
        jax.nn.softmax(logits_sharper), boltz_features, boltz_writer
    )
    viewer
    return model_output, viewer


@app.cell
def _(model_output, plt):
    _f = plt.imshow(model_output["pae"][0])
    plt.colorbar()
    _f
    return


@app.cell
def _(jax, logits_sharper, plt):
    plt.imshow(jax.nn.softmax(logits_sharper))
    return


@app.cell
def _(jax, logits, plt):
    plt.imshow(jax.nn.softmax(logits))
    return


@app.cell
def _(
    Path,
    boltz,
    load_features_and_structure_writer,
    logits_sharper,
    mo,
    predict,
    target_fasta_seq,
):
    # Let's repredict our designed sequence with the correct sidechains, hopefully Boltz still likes it


    def _repredict():
        binder_seq = "".join(
            boltz.data.const.prot_token_to_letter[boltz.data.const.tokens[i]]
            for i in logits_sharper.argmax(-1) + 2
        )
        out_dir = Path(f"/tmp/proteins/{binder_seq}")
        out_dir.mkdir(exist_ok=True, parents=True)
        fasta_path = out_dir / "protein.fasta"
        target_sequence = "ETRECIYYNANWELERTNQSGLERCEGEQDKRLHCYASWRNSSGTIELVKKGCWLDDFNCYDRQECVATEENPQVYFCCCEGNFCNERFTHLP"
        fasta_path.write_text(
            target_fasta_seq(binder_seq, chain="A", use_msa=False)
            + target_fasta_seq(target_sequence)
        )
        return load_features_and_structure_writer(fasta_path, out_dir)


    f_r, _w = _repredict()

    repredicted_output, repredicted_viewer = predict(
        f_r["res_type"][0][:, 2:22], f_r, _w
    )

    with open(next(_w.out_dir.glob("*/*.pdb")), "r") as _f:
        download_structure = mo.download(_f.read(), filename="next.pdb")

    repredicted_viewer
    return download_structure, f_r, repredicted_output, repredicted_viewer


@app.cell
def _(plt, repredicted_output):
    _f = plt.imshow(repredicted_output["pae"][0])
    plt.colorbar()
    _f
    return


@app.cell
def _(download_structure):
    download_structure
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
