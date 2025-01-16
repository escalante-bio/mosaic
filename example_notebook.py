import marimo

__generated_with = "0.10.13"
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
        PLDDTLoss,
    )
    return (
        BinderTargetContact,
        Boltz1,
        BoltzDiffusionParams,
        HelixLoss,
        PDBeMolstar,
        PLDDTLoss,
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
            hide_settings=False,
            hide_controls_icon=False,
            hide_expand_icon=False,
            hide_settings_icon=False,
            hide_selection_icon=False,
            hide_animation_icon=False,
            hide_water=True,
            hide_carbs=True,
            alphafold_view=True,
            theme="dark",
        )
    return (pdb_viewer,)


@app.cell
def _():
    target_sequence = "ETRECIYYNANWELERTNQSGLERCEGEQDKRLHCYASWRNSSGTIELVKKGCWLDDFNCYDRQECVATEENPQVYFCCCEGNFCNERFTHLP"
    return (target_sequence,)


@app.cell
def _(Boltz1, BoltzDiffusionParams, Path, asdict, eqx, jax, joltz):
    # load boltz model and convert to JAX
    predict_args = {
        "recycling_steps": 0,
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
        print(o["plddt"][: sequence.shape[0]].mean())
        return o, viewer
    return (predict,)


@app.cell
def _():
    binder_length = 80
    return (binder_length,)


@app.cell
def _(binder_length, make_binder_features, target_sequence):
    boltz_features, boltz_writer = make_binder_features(
        binder_length,
        target_sequence,
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
        name="ART2B",
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
    target_fasta_seq,
    target_sequence,
):
    # Let's repredict our designed sequence with the correct sidechains, hopefully Boltz still likes it
    def repredict(logits_sharper, target_sequence=target_sequence):
        binder_seq = "".join(
            boltz.data.const.prot_token_to_letter[boltz.data.const.tokens[i]]
            for i in logits_sharper.argmax(-1) + 2
        )
        print(binder_seq)
        out_dir = Path(f"/tmp/proteins/{binder_seq}")
        out_dir.mkdir(exist_ok=True, parents=True)
        fasta_path = out_dir / "protein.fasta"
        fasta_path.write_text(
            target_fasta_seq(binder_seq, chain="A", use_msa=True)
            + target_fasta_seq(target_sequence)
        )
        return load_features_and_structure_writer(fasta_path, out_dir)
    return (repredict,)


@app.cell
def _(logits_sharper, mo, predict, repredict):
    f_r, _w = repredict(logits_sharper)

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
    # Okay, but what if we wanted to optimize multiple objectives?
    return


@app.cell
def _():
    from boltz_binder_design.esm.pretrained import load_pretrained_esm
    from boltz_binder_design.loss.esm import ESM2PseudoLikelihood
    from boltz_binder_design.loss.trigram import TrigramLL
    from boltz_binder_design.loss.stability import StabilityModel
    return (
        ESM2PseudoLikelihood,
        StabilityModel,
        TrigramLL,
        load_pretrained_esm,
    )


@app.cell
def _():
    from boltz_binder_design.util import At
    return (At,)


@app.cell
def _(ESM2PseudoLikelihood, load_pretrained_esm):
    esm, _ = load_pretrained_esm()
    esm_loss = ESM2PseudoLikelihood(esm)
    return esm, esm_loss


@app.cell
def _(TrigramLL):
    trigram_ll = TrigramLL.from_pkl()
    return (trigram_ll,)


@app.cell
def _(binder_length, make_binder_monomer_features):
    monomer_features, monomer_writer = make_binder_monomer_features(
        binder_length,
    )
    return monomer_features, monomer_writer


@app.cell
def _(
    BinderTargetContact,
    HelixLoss,
    PLDDTLoss,
    RadiusOfGyration,
    StabilityModel,
    StructurePrediction,
    WithinBinderContact,
    boltz_features,
    esm,
    esm_loss,
    model,
    monomer_features,
    trigram_ll,
):
    # modify structural loss to add stability term, add ESM + trigram
    combined_loss = (
        StructurePrediction(
            model=model,
            name="ART2B",
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
        + 0.5 * esm_loss
        + trigram_ll
        + StructurePrediction(
            model=model,
            name="mono",
            loss=1 * PLDDTLoss() + 0.1 * StabilityModel.from_pretrained(esm),
            features=monomer_features,
            recycling_steps=0,
        )
    )
    return (combined_loss,)


@app.cell
def _(binder_length, combined_loss, design_bregman_optax, np, optax):
    logits_combined_objective = design_bregman_optax(
        loss_function=combined_loss,
        n_steps=150,
        x=np.random.randn(binder_length, 20) * 0.1,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0), optax.sgd(np.sqrt(binder_length))
        ),
    )
    return (logits_combined_objective,)


@app.cell
def _(
    binder_length,
    combined_loss,
    design_bregman_optax,
    logits_combined_objective,
    np,
    optax,
):
    # we can sharpen these logits using weight decay (which is equivalent to adding entropic regularization)
    logits_combined_sharper = design_bregman_optax(
        loss_function=combined_loss,
        n_steps=50,
        x=logits_combined_objective,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.01),
            optax.sgd(1.0 * np.sqrt(binder_length)),
        ),
    )
    logits_combined_sharper = design_bregman_optax(
        loss_function=combined_loss,
        n_steps=50,
        x=logits_combined_sharper,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.05),
            optax.sgd(0.25 * np.sqrt(binder_length)),
        ),
    )
    return (logits_combined_sharper,)


@app.cell
def _(jax, logits_combined_sharper, plt):
    plt.imshow(jax.nn.softmax(logits_combined_sharper))
    return


@app.cell
def _(jax, logits_combined_objective, plt):
    plt.imshow(jax.nn.softmax(logits_combined_objective))
    return


@app.cell
def _(boltz_features, boltz_writer, jax, logits_combined_sharper, predict):
    _model_output, _viewer = predict(
        jax.nn.softmax(logits_combined_sharper),
        boltz_features,
        boltz_writer,
    )

    _viewer
    return


@app.cell
def _(download_structure_stab):
    download_structure_stab
    return


@app.cell
def _(
    binder_length,
    j_model,
    jax,
    logits_combined_sharper,
    mo,
    pdb_viewer,
    repredict,
):
    _f_r, _w = repredict(logits_combined_sharper)

    o_combined = j_model(
        _f_r,
        key=jax.random.key(5),
        sample_structure=True,
        confidence_prediction=True,
    )

    _out_path = _w(o_combined["sample_atom_coords"])
    _repredicted_viewer = pdb_viewer(_out_path)

    print(o_combined["plddt"][:binder_length].mean())
    print(_w.out_dir)
    with open(next(_w.out_dir.glob("*/*.pdb")), "r") as _f:
        download_structure_stab = mo.download(_f.read(), filename="next.pdb")

    _repredicted_viewer
    return download_structure_stab, o_combined


@app.cell
def _(download_structure_stab):
    download_structure_stab
    return


@app.cell
def _(o_combined, plt):
    _f = plt.imshow(o_combined["pae"][0])
    plt.colorbar()
    _f
    return


if __name__ == "__main__":
    app.run()
