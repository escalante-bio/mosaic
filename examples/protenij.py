import marimo

__generated_with = "0.19.8"
app = marimo.App()

with app.setup:
    import marimo as mo
    from mosaic.optimizers import simplex_APGM
    import mosaic.losses.structure_prediction as sp
    import matplotlib.pyplot as plt
    import jax
    import numpy as np
    import gemmi
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
    from mosaic.models.protenix import ProtenixMini, Protenix2025, ProtenixBase


@app.cell(hide_code=True)
def _():
    mo.callout("Demo design using Protenix v1.0", kind="success")
    return


@app.cell
def _():
    protenix = Protenix2025()
    return (protenix,)


@app.cell
def _():
    binder_length = 120
    return (binder_length,)


@app.cell
def _():
    target_structure = gemmi.read_structure("IL7RA.cif")
    target_structure.remove_ligands_and_waters()
    target_sequence = gemmi.one_letter_code(
        [r.name for r in target_structure[0][0]]
    )
    return target_sequence, target_structure


@app.cell
def _(binder_length, protenix, target_sequence, target_structure):
    design_features, design_structure = protenix.binder_features(
        binder_length=binder_length,
        chains=[
            TargetChain(
                target_sequence,
                use_msa=True,
                template_chain=target_structure[0][0],
            )
        ],
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
def _(binder_length, mpnn):
    structure_loss = (
        sp.BinderTargetContact()
        + 1 * sp.WithinBinderContact()
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.025 * sp.IPTMLoss()
        + 0.4 * sp.WithinBinderPAE()
        + 0.025 * sp.pTMEnergy()
        + 0.1 * sp.PLDDTLoss()
        + 10.0 * InverseFoldingSequenceRecovery(mpnn, temp=jax.numpy.array(0.001))
        + 0.00
        * sp.ActualRadiusOfGyration(target_radius=2.38 * binder_length**0.365)
        - 0.0 * sp.HelixLoss()
        + 0.0 * sp.BinderTargetIPSAE()
        + 0.0 * sp.TargetBinderIPSAE()
    )
    return (structure_loss,)


@app.cell
def _():
    mo.md("""
    Bit finicky to optimize through this model. Becomes much easier with n. recycling steps > 1.
    """)
    return


@app.cell
def _(design_features, protenix, structure_loss):
    loss = protenix.build_multisample_loss(
        loss=structure_loss,
        features=design_features,
        recycling_steps=1,
        sampling_steps=20,
    )
    return (loss,)


@app.cell(hide_code=True)
def _():
    mo.callout(
        "It can take up to 4 MINUTES to JIT Protenix -- this should only happen the first time you run the following cell. You may need to rerun it multiple times to get a good sample!",
        kind="warn",
    )
    return


@app.cell
def _(binder_length, loss):
    PSSM = jax.random.dirichlet(
        key=jax.random.key(np.random.randint(100000)),
        shape=(binder_length,),
        alpha=1.9 * np.ones(20),
    )

    _, PSSM = simplex_APGM(
        loss_function=loss,
        x=PSSM,
        n_steps=100,
        stepsize=0.15 * np.sqrt(binder_length),
        momentum=0.3,
        scale=1.0,
        update_loss_state=False,
        max_gradient_norm=1.0,
    )
    return (PSSM,)


@app.cell
def _(PSSM, binder_length, loss):
    PSSM_sharper, _ = simplex_APGM(
        loss_function=loss,
        x=jnp.log(PSSM + 1e-5),
        n_steps=20,
        stepsize=0.5 * np.sqrt(binder_length),
        momentum=0.0,
        scale=1.3,
        update_loss_state=False,
        logspace=False,
        max_gradient_norm=1.0,
    )
    return (PSSM_sharper,)


@app.cell
def _(PSSM_sharper):
    plt.imshow(PSSM_sharper)
    return


@app.cell
def _():
    mo.callout("Let's evaluate our design", "info")
    return


@app.cell
def _(protenix_pred):
    pdb_viewer(protenix_pred.st)
    return


@app.cell
def _(PSSM_sharper, design_features, design_structure, protenix):
    # repredict design with recycling
    protenix_pred = protenix.predict(
        PSSM=PSSM_sharper,
        features=design_features,
        recycling_steps=4,
        key=jax.random.key(0),
        writer=design_structure,
    )
    return (protenix_pred,)


@app.cell
def _(protenix_pred):
    plt.imshow(protenix_pred.pae)
    return


@app.cell
def _(binder_length, protenix_pred):
    plt.plot(protenix_pred.plddt)
    plt.vlines(
        [binder_length],
        protenix_pred.plddt.min(),
        protenix_pred.plddt.max(),
        linestyle="dashed",
        color="red",
    )
    return


@app.cell
def _(protenix_pred):
    protenix_pred.iptm
    return


@app.cell
def _(protenix_pred):
    mo.download(
        data=protenix_pred.st.make_minimal_pdb(), filename="protenix_complex.pdb"
    )
    return


if __name__ == "__main__":
    app.run()
