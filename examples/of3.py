import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import gemmi
    import jopenfold3
    from mosaic.models.of3 import OF3
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.structure_prediction import TargetChain
    import mosaic.losses.structure_prediction as sp
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    import jax
    import mosaic
    from mosaic.optimizers import simplex_APGM
    import jax.numpy as jnp
    import numpy as np

    return (
        InverseFoldingSequenceRecovery,
        OF3,
        TargetChain,
        gemmi,
        jax,
        jnp,
        load_mpnn_sol,
        mo,
        np,
        simplex_APGM,
        sp,
    )


@app.cell
def _():
    from mosaic.notebook_utils import pdb_viewer
    import matplotlib.pyplot as plt

    return pdb_viewer, plt


@app.cell
def _(OF3):
    of3 = OF3()
    return (of3,)


@app.cell
def _():
    binder_length = 120
    return (binder_length,)


@app.cell
def _(TargetChain, binder_length, of3, target_sequence, target_structure):
    features, writer = of3.binder_features(
        binder_length,
        chains=[
            TargetChain(
                sequence=target_sequence,
                use_msa=True,
                template_chain=target_structure[0][0],
            )
        ],
    )
    return features, writer


@app.cell
def _(gemmi):
    target_structure = gemmi.read_structure("IL7RA.cif")
    target_structure.remove_ligands_and_waters()
    target_sequence = gemmi.one_letter_code(
        [r.name for r in target_structure[0][0]]
    )
    return target_sequence, target_structure


@app.cell
def _(load_mpnn_sol):
    mpnn = load_mpnn_sol(0.05)
    return (mpnn,)


@app.cell
def _(InverseFoldingSequenceRecovery, binder_length, jax, mpnn, sp):
    structure_loss = (
        sp.BinderTargetContact()
        + 2 * sp.WithinBinderContact()
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.025 * sp.IPTMLoss()
        + 0.5 * sp.WithinBinderPAE()
        + 0.00 * sp.pTMEnergy()
        + 0.5 * sp.PLDDTLoss()
        + 10.0 * InverseFoldingSequenceRecovery(mpnn, temp=jax.numpy.array(0.001))
        + 1.00
        * sp.ActualRadiusOfGyration(target_radius=2.38 * binder_length**0.365 + 1.0)
        - 0.0 * sp.HelixLoss()
        + 0.0 * sp.BinderTargetIPSAE()
        + 0.0 * sp.TargetBinderIPSAE()
    )
    return (structure_loss,)


@app.cell
def _(features, of3, structure_loss):
    loss = of3.build_multisample_loss(
        loss=structure_loss,
        features=features,
        recycling_steps=1,
        sampling_steps=20,
        num_samples=2,
    )
    return (loss,)


@app.cell
def _(mo):
    mo.callout(
        value="OF3 implementation is currently a bit finicky -- you may have to run many samples before getting a decent binder."
    )
    return


@app.cell
def _(binder_length, jax, loss, np, simplex_APGM):
    PSSM = jax.nn.softmax(
        0.1
        * jax.random.gumbel(
            key=jax.random.key(np.random.randint(1000000)),
            shape=(binder_length, 20),
        )
    )

    _, PSSM = simplex_APGM(
        loss_function=loss,
        x=PSSM,
        n_steps=100,
        stepsize=0.15 * np.sqrt(binder_length),
        momentum=0.1,
        scale=1.0,
        update_loss_state=False,
        max_gradient_norm=1.0,
    )
    return (PSSM,)


@app.cell
def _(PSSM, binder_length, jnp, loss, np, simplex_APGM):
    PSSM_sharper, _ = simplex_APGM(
        loss_function=loss,
        x=jnp.log(PSSM + 1e-5),
        n_steps=20,
        stepsize=0.1 * np.sqrt(binder_length),
        momentum=0.0,
        scale=1.3,
        update_loss_state=False,
        logspace=False,
        max_gradient_norm=1.0,
    )
    return (PSSM_sharper,)


@app.cell
def _(PSSM_sharper, features, jax, of3, writer):
    pred = of3.predict(
        PSSM=PSSM_sharper,
        features=features,
        recycling_steps=10,
        key=jax.random.key(0),
        writer=writer,
    )
    return (pred,)


@app.cell
def _(plt, pred):
    plt.imshow(pred.pae)
    return


@app.cell
def _(binder_length, plt, pred):
    plt.plot(pred.plddt)
    plt.vlines(
        [binder_length],
        pred.plddt.min(),
        pred.plddt.max(),
        linestyle="dashed",
        color="red",
    )
    return


@app.cell
def _(pred):
    pred.iptm
    return


@app.cell
def _(pdb_viewer, pred):
    pdb_viewer(pred.st)
    return


@app.cell
def _(mo, pred):
    mo.download(data=pred.st.make_pdb_string(), filename="binder.pdb")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
