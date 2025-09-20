import marimo

__generated_with = "0.15.5"
app = marimo.App()

with app.setup:
    import marimo as mo
    from mosaic.optimizers import simplex_APGM
    import mosaic.losses.structure_prediction as sp
    import matplotlib.pyplot as plt
    import jax
    import numpy as np
    from mosaic.notebook_utils import pdb_viewer
    from mosaic.losses.protein_mpnn import (
        InverseFoldingSequenceRecovery,
    )
    from mosaic.proteinmpnn.mpnn import ProteinMPNN
    import importlib
    import mosaic
    import equinox as eqx

    import jax.numpy as jnp
    from protenix.protenij import TrunkEmbedding
    from mosaic.structure_prediction import TargetChain
    from mosaic.models.protenix import ProtenixMini


@app.cell
def _():
    from mosaic.models.af2 import AlphaFold2
    return (AlphaFold2,)


@app.cell
def _(AlphaFold2):
    af2 = AlphaFold2()
    return (af2,)


@app.cell
def _():
    protenix = ProtenixMini()
    return (protenix,)


@app.cell
def _():
    binder_length = 80
    return (binder_length,)


@app.cell
def _():
    target_sequence = "DYSFSCYSQLEVNGSQHSLTCAFEDPDVNTTNLEFEICGALVEVKCLNFRKLQEIYFIETKKFLLIGKSNICVKVGEKSLTCKKIDLTTIVKPEAPFDLSVVYREGANDFVVTFNTSHLQKKYVKVLMHDVAYRQEKDENKWTHVNLSSTKLTLLQRKLQPAAMYEIKVRSIPDHYFKGFWSEWSPSYYFRT"
    return (target_sequence,)


@app.cell
def _(protenix, target_sequence):
    target_only_features, target_only_structure = protenix.target_only_features(
        [TargetChain(target_sequence)]
    )
    return target_only_features, target_only_structure


@app.cell
def _(binder_length, protenix, target_sequence):
    design_features, design_structure = protenix.binder_features(
        binder_length = binder_length, chains = [TargetChain(target_sequence)]
    )
    return design_features, design_structure


@app.cell
def _():
    mpnn = ProteinMPNN.from_pretrained(
            importlib.resources.files(mosaic)
            / "proteinmpnn/weights/soluble_v_48_020.pt"
        )
    return (mpnn,)


@app.cell
def _(protenix, target_only_features, target_only_structure):
    pred_target = protenix.predict(
        features=target_only_features,
        writer=target_only_structure,
        key=jax.random.key(0),
        recycling_steps=10,
    )
    st_target_only = pred_target.st
    return pred_target, st_target_only


@app.cell
def _(pred_target):
    plt.plot(pred_target.plddt)
    return


@app.cell
def _(st_target_only):
    pdb_viewer(st_target_only)
    return


@app.cell
def _(binder_length, mpnn):
    structure_loss = (
        sp.BinderTargetContact()
        + sp.WithinBinderContact()
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.05 * sp.IPTMLoss()
        + 0.4 * sp.WithinBinderPAE()
        + 0.00 * sp.pTMEnergy()
        + 0.1 * sp.PLDDTLoss()
        + 5.0 * InverseFoldingSequenceRecovery(mpnn, temp=jax.numpy.array(0.001))
        + 0.025*sp.ActualRadiusOfGyration(target_radius = 2.38 * binder_length**0.365)
        - 0.0*sp.HelixLoss()
        + 0.0*sp.BinderTargetIPSAE()
        + 0.0*sp.TargetBinderIPSAE()
    )
    return (structure_loss,)


@app.cell(hide_code=True)
def _(PSSM_sharper):
    plt.imshow(PSSM_sharper)
    return


@app.cell
def _(PSSM_sharper, design_features, design_structure, protenix):
    # repredict design with recycling
    protenix_pred = protenix.predict(
        PSSM = PSSM_sharper, 
        features = design_features,
        recycling_steps = 5, 
        key = jax.random.key(0),
        writer = design_structure
    )
    return (protenix_pred,)


@app.cell
def _(protenix_pred):
    plt.plot(protenix_pred.plddt)
    return


@app.cell
def _(protenix_pred):
    plt.imshow(protenix_pred.pae)
    return


@app.cell
def _(protenix_pred):
    pdb_viewer(protenix_pred.st)
    return


@app.cell
def _(PSSM_sharper, af2, af_features):
    # repredict with AF2 models
    o_pred_af = max(
        [
            af2.predict(
                PSSM=jax.nn.one_hot(PSSM_sharper.argmax(-1), 20),
                features=af_features,
                model_idx=idx,
                key=jax.random.key(1),
            )
            for idx in range(5)
        ],
        key=lambda T: T.iptm,
    )
    return (o_pred_af,)


@app.cell
def _(o_pred_af):
    _f = plt.imshow(o_pred_af.pae)
    plt.title(f"AF2 IPTM {o_pred_af.iptm : 0.2f}")
    _f
    return


@app.cell
def _(o_pred_af):
    pdb_viewer(o_pred_af.st)
    return


@app.cell
def _(af2, binder_length, st_target_only, target_sequence):
    af_features, _ = af2.binder_features(
        binder_length=binder_length,
        chains=[
            TargetChain(
                target_sequence, template_chain=st_target_only[0][0], use_msa=False
            )
        ],
    )
    return (af_features,)


@app.cell(hide_code=True)
def _():
    mo.md(
        """
    Pre-cycle the target. Because the publically released Protenix mini doesn't support templates we first run 10 cycles of recycling on the target alone and use the final representation to initialize the recycling representation for the loss.

    Without this step design will fail for challenging targets because the model won't be able to properly fold the target.
    """
    )
    return


@app.cell
def _(protenix, target_only_features):
    te_target = protenix.model_output(features = target_only_features, key = jax.random.key(0), recycling_steps=10).trunk_state
    return (te_target,)


@app.cell
def _(binder_length, target_sequence, te_target):
    N = len(target_sequence) + binder_length

    _te = TrunkEmbedding(s=jnp.zeros((N, 384)), z=jnp.zeros((N, N, 128)))
    te = eqx.tree_at(
        lambda s: (s.s, s.z),
        _te,
        (
            _te.s.at[binder_length:].set(te_target.s),
            _te.z.at[binder_length:, binder_length:].set(te_target.z),
        ),
    )
    return (te,)


@app.cell
def _(design_features, protenix, structure_loss, te):
    loss = protenix.build_loss(
        loss=structure_loss, features=design_features, initial_recycling_state=te
    )
    return (loss,)


@app.cell
def _(binder_length, loss):
    # JIT compile value + gradient
    x = jax.nn.softmax(
        0.50
        * jax.random.gumbel(
            key=jax.random.key(np.random.randint(100000)),
            shape=(binder_length, 20),
        )
    )

    (_, aux), _ = mosaic.optimizers._eval_loss_and_grad(
        x=x, loss_function=loss, key=jax.random.key(0)
    )
    return


@app.cell
def _(binder_length, loss):
    PSSM = jax.nn.softmax(
                    0.5
                    * jax.random.gumbel(
                        key=jax.random.key(np.random.randint(100000)),
                        shape=(binder_length, 20),
                    )
                )

    for _outer in range(20):
        print(_outer)
        PSSM,_ = simplex_APGM(
                loss_function=loss,
                x=PSSM,
                n_steps=2,
                stepsize=0.15,
                momentum=0.9,
                scale=1.0,
                update_loss_state=True
            )
    return (PSSM,)


@app.cell
def _(PSSM, loss):
    PSSM_sharper = PSSM
    for _ in range(5*2):
        _,PSSM_sharper = simplex_APGM(
                loss_function=loss,
                x=PSSM_sharper,
                n_steps=2,
                stepsize=0.1,
                momentum=0.0,
                scale = 1.5,
                update_loss_state=True,
                logspace=False
            )
    return (PSSM_sharper,)


@app.cell
def _(PSSM_sharper):
    plt.imshow(PSSM_sharper)
    return


@app.cell
def _(PSSM):
    plt.imshow(PSSM)
    return


@app.cell
def _(protenix_pred):
    mo.download(data = protenix_pred.st.make_minimal_pdb(), filename = "tnf.pdb")
    return


@app.cell
def _(o_pred_af):
    mo.download(data = o_pred_af.st.make_minimal_pdb(), filename = "af_tnf.pdb")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
