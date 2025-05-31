import marimo

__generated_with = "0.13.10"
app = marimo.App(width="full")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import jax
    from boltz_binder_design.af2.alphafold2 import AF2
    from boltz_binder_design.losses.af2 import AlphaFold
    from boltz_binder_design.losses.boltz import make_monomer_features, load_boltz
    import boltz_binder_design.losses.af2 as aflosses
    from boltz_binder_design.common import TOKENS
    from ipymolstar import PDBeMolstar
    from boltz_binder_design.optimizers import design_bregman_optax
    import numpy as np
    import equinox as eqx
    import gemmi
    import optax
    return (
        AF2,
        AlphaFold,
        PDBeMolstar,
        TOKENS,
        aflosses,
        design_bregman_optax,
        eqx,
        gemmi,
        jax,
        load_boltz,
        make_monomer_features,
        np,
        optax,
        plt,
    )


@app.cell
def _(PDBeMolstar, gemmi):
    def pdb_viewer(st: gemmi.Structure):
        """Display a PDB file using Molstar"""
        custom_data = {
            "data": st.make_pdb_string(),
            "format": "pdb",
            "binary": False,
        }
        return PDBeMolstar(custom_data=custom_data, theme="dark")
    return (pdb_viewer,)


@app.cell
def _(AF2):
    af2 = AF2()
    return (af2,)


@app.cell
def _():
    target_sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    return (target_sequence,)


@app.cell
def _(eqx, gemmi, jax, load_boltz, make_monomer_features, target_sequence):
    # predict the target alone with boltz-1 to build a template for AF2
    model = load_boltz()
    j_model = eqx.filter_jit(lambda model, *args, **kwargs: model(*args, **kwargs))
    target_features, target_writer = make_monomer_features(target_sequence)


    o_target = j_model(
        model,
        target_features,
        key=jax.random.key(5),
        sample_structure=True,
        confidence_prediction=True,
    )

    out_path_target = target_writer(o_target["sample_atom_coords"])
    target_st = gemmi.read_pdb(str(out_path_target))
    return (target_st,)


@app.cell
def _():
    binder_length = 90
    return (binder_length,)


@app.cell
def _(af2, binder_length, target_sequence, target_st):
    af_features, initial_guess = af2.build_features(
        chains=["G" * binder_length, target_sequence],
        template_chains={1: target_st[0][0]},
    )
    return (af_features,)


@app.cell
def _(AlphaFold, af2, af_features, aflosses, jax):
    af_loss = AlphaFold(
        name="af",
        forward=af2.alphafold_apply,
        stacked_params=jax.device_put(af2.stacked_model_params),
        features=af_features,
        losses=0.005 * aflosses.PLDDTLoss()
        + 1 * aflosses.BinderTargetContact()
        + 0.1 * aflosses.TargetBinderPAE()
        + 0.1 * aflosses.BinderTargetPAE()
        + 1.0 * aflosses.WithinBinderContact()
        + 0.0 * aflosses.IPTM()
        + 0.05 * aflosses.BinderPAE()
        + 0.05 * aflosses.pTMEnergy(),
    )
    return (af_loss,)


@app.cell
def _(af_loss, binder_length, design_bregman_optax, np, optax):
    _, logits_af = design_bregman_optax(
        loss_function=af_loss,
        n_steps=150,
        x=np.random.randn(binder_length, 20) * 0.1,
        optim=optax.chain(
            # optax.normalize_by_update_norm(1.0),
            optax.clip_by_global_norm(1.0),
            optax.sgd(0.1 * np.sqrt(binder_length), momentum=0.0),
        ),
    )
    return (logits_af,)


@app.cell
def _(af_loss, binder_length, design_bregman_optax, logits_af, np, optax):
    mult = 0.25
    logits_af_sharper, _ = design_bregman_optax(
        loss_function=af_loss,
        n_steps=25,
        x=logits_af,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.01),
            optax.sgd(mult * np.sqrt(binder_length)),
        ),
    )
    logits_af_sharper, _ = design_bregman_optax(
        loss_function=af_loss,
        n_steps=25,
        x=logits_af_sharper,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.05),
            optax.sgd(mult * np.sqrt(binder_length)),
        ),
    )
    logits_af_sharper, _ = design_bregman_optax(
        loss_function=af_loss,
        n_steps=25,
        x=logits_af_sharper,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.1),
            optax.sgd(mult * np.sqrt(binder_length)),
        ),
    )
    return (logits_af_sharper,)


@app.cell
def _(TOKENS, af2, jax, logits_af_sharper, target_sequence, target_st):
    # make final prediction
    o_af_pred, st_af_pred = af2.predict(
        [
            "".join(TOKENS[i] for i in logits_af_sharper.argmax(-1)),
            target_sequence,
        ],
        template_chains={1: target_st[0][0]},
        key=jax.random.key(0),
        model_idx=1,
    )
    return o_af_pred, st_af_pred


@app.cell
def _(o_af_pred, plt):
    _f = plt.imshow(o_af_pred.predicted_aligned_error)
    plt.colorbar()
    plt.title("PAE")
    _f
    return


@app.cell
def _(pdb_viewer, st_af_pred):
    pdb_viewer(st_af_pred)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
