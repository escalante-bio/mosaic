import marimo

__generated_with = "0.19.9"
app = marimo.App()

with app.setup:
    import gemmi
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from jfold.blocks.pairformer import with_triangle_chunks

    import mosaic.losses.structure_prediction as sp
    from mosaic.common import TOKENS
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    from mosaic.models.protenix import Protenix
    from mosaic.notebook_utils import pdb_viewer
    from mosaic.optimizers import (
        batch_greedy_descent,
        batched_eval,
        batched_simplex_APGM,
        simplex_APGM,
    )
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.structure_prediction import TargetChain


@app.cell(hide_code=True)
def _():
    mo.callout("Demo design using Protenix v1.0", kind="success")
    return


@app.cell
def _():
    protenix = Protenix.base_20250630()
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
        key=jax.random.key(0),
    )
    return design_features, design_structure


@app.cell
def _():
    mpnn = load_mpnn_sol()
    return (mpnn,)


@app.cell
def _(mpnn):
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
    loss = protenix.build_loss(
        loss=structure_loss,
        features=design_features,
        recycling_steps=1,
        sampler=protenix.sampler(20),
        num_samples=4,
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
    PSSM = jax.nn.softmax(
        0.5
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
        momentum=0.3,
        scale=1.0,
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


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Several designs at once

    The same loss, optimized from several random starts in parallel: the batched
    optimizer runs them together, the lowest-loss design is kept, and a greedy search
    over single mutations finishes it as a discrete sequence. Each start needs its own trunk
    and gradient, so the batched loss draws two diffusion samples instead of four and runs
    triangle attention 32 rows at a time: about 22 GiB for two starts, 43 GiB for four. The
    greedy search tries `batch_size` mutations at a time.
    """)
    return


@app.cell
def _(design_features, protenix, structure_loss):
    batch_size = 2
    batch_loss = with_triangle_chunks(protenix, 32).build_loss(
        loss=structure_loss,
        features=design_features,
        recycling_steps=1,
        sampler=protenix.sampler(20),
        num_samples=2,
    )
    return batch_loss, batch_size


@app.cell
def _(batch_loss, batch_size, binder_length):
    starts = jnp.stack(
        [
            jax.nn.softmax(
                0.5 * jax.random.gumbel(key=jax.random.key(seed), shape=(binder_length, 20))
            )
            for seed in np.random.randint(0, 1000000, size=batch_size)
        ]
    )
    _, batch_pssms = batched_simplex_APGM(
        loss_function=batch_loss,
        x=starts,
        n_steps=80,
        stepsize=0.15 * np.sqrt(binder_length),
        momentum=0.2,
        scale=1.0,
        max_gradient_norm=1.0,
    )
    sharp_pssms, _ = batched_simplex_APGM(
        loss_function=batch_loss,
        x=jnp.log(batch_pssms + 1e-5),
        n_steps=30,
        stepsize=0.5 * np.sqrt(binder_length),
        momentum=0.0,
        scale=1.3,
        logspace=True,
        max_gradient_norm=1.0,
    )
    return (sharp_pssms,)


@app.cell
def _(batch_loss, batch_size, sharp_pssms):
    # The lowest-loss design, scored with one key for all, then greedy single mutations.
    _key = jax.random.key(0)
    _values, _, _ = batched_eval(
        batch_loss, sharp_pssms, jnp.broadcast_to(_key, (sharp_pssms.shape[0], *_key.shape))
    )
    _losses = np.array(_values)
    _best = int(np.argmin(_losses))
    print(f"Best of the batch: {_best}, loss {_losses[_best]:.4f}")

    best_sequence, _best_loss = batch_greedy_descent(
        batch_loss,
        jnp.argmax(sharp_pssms[_best], axis=-1),
        batch_size=batch_size,
        steps=50,
        key=jax.random.key(0),
    )
    print(f"After greedy descent: loss {_best_loss:.4f}")
    print("".join(TOKENS[int(i)] for i in best_sequence))
    return (best_sequence,)


@app.cell
def _(best_sequence, design_features, design_structure, protenix):
    batch_pred = protenix.predict(
        PSSM=jax.nn.one_hot(best_sequence, 20),
        features=design_features,
        recycling_steps=4,
        key=jax.random.key(0),
        writer=design_structure,
    )
    mo.vstack([mo.md(f"ipTM {float(batch_pred.iptm):.3f}"), pdb_viewer(batch_pred.st)])
    return (batch_pred,)


@app.cell
def _(batch_pred):
    mo.download(data=batch_pred.st.make_minimal_pdb(), filename="protenix_batched.pdb")
    return


if __name__ == "__main__":
    app.run()
