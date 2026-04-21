import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import gemmi
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.models.protenix import Protenix2025
    from mosaic.structure_prediction import TargetChain
    import mosaic.losses.structure_prediction as sp
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.common import TOKENS
    from mosaic.optimizers import (
        batched_simplex_APGM,
        batch_greedy_descent,
        batched_eval,
    )

    return (
        InverseFoldingSequenceRecovery,
        Protenix2025,
        TOKENS,
        TargetChain,
        batch_greedy_descent,
        batched_eval,
        batched_simplex_APGM,
        gemmi,
        jax,
        jnp,
        load_mpnn_sol,
        mo,
        np,
        sp,
    )


@app.cell
def _():
    from mosaic.notebook_utils import pdb_viewer
    import matplotlib.pyplot as plt

    return pdb_viewer, plt


@app.cell
def _(Protenix2025):
    protenix = Protenix2025()
    return (protenix,)


@app.cell
def _():
    binder_length = 80
    batch_size = 4
    return batch_size, binder_length


@app.cell
def _(gemmi):
    target_structure = gemmi.read_structure("IL7RA.cif")
    target_structure.remove_ligands_and_waters()
    target_sequence = gemmi.one_letter_code(
        [r.name for r in target_structure[0][0]]
    )
    return target_sequence, target_structure


@app.cell
def _(TargetChain, binder_length, protenix, target_sequence, target_structure):
    features, writer = protenix.binder_features(
        binder_length=binder_length,
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
def _(load_mpnn_sol):
    mpnn = load_mpnn_sol(0.05)
    return (mpnn,)


@app.cell
def _(InverseFoldingSequenceRecovery, jax, mpnn, sp):
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
    )
    return (structure_loss,)


@app.cell
def _(features, protenix, structure_loss):
    loss = protenix.build_multisample_loss(
        loss=structure_loss,
        features=features,
        recycling_steps=1,
        sampling_steps=20,
        num_samples=2,
    )
    return (loss,)


@app.cell
def _(batch_size, batched_simplex_APGM, binder_length, jax, jnp, loss, np):
    init_pssms = jnp.stack(
        [
            jax.nn.softmax(
                0.5
                * jax.random.gumbel(
                    key=jax.random.key(seed),
                    shape=(binder_length, 20),
                )
            )
            for seed in np.random.randint(0, 1000000, size=batch_size)
        ]
    )

    _, best_pssms = batched_simplex_APGM(
        loss_function=loss,
        x=init_pssms,
        n_steps=80,
        stepsize=0.15 * np.sqrt(binder_length),
        momentum=0.2,
        scale=1.0,
        max_gradient_norm=1.0,
    )
    return (best_pssms,)


@app.cell
def _(batched_simplex_APGM, best_pssms, binder_length, jnp, loss, np):
    sharp_pssms, _ = batched_simplex_APGM(
        loss_function=loss,
        x=jnp.log(best_pssms + 1e-5),
        n_steps=30,
        stepsize=0.5 * np.sqrt(binder_length),
        momentum=0.0,
        scale=1.3,
        logspace=True,
        max_gradient_norm=1.0,
    )
    return (sharp_pssms,)


@app.cell
def _(
    TOKENS,
    batch_greedy_descent,
    batched_eval,
    jax,
    jnp,
    loss,
    np,
    sharp_pssms,
):
    # pick the best PSSM and discretize
    key = jax.random.key(0)
    shared_keys = jnp.broadcast_to(key, (sharp_pssms.shape[0], *key.shape))
    values, _, _ = batched_eval(loss, sharp_pssms, shared_keys)
    losses = np.array(values)

    best_idx = int(np.argmin(losses))
    init_seq = jnp.argmax(sharp_pssms[best_idx], axis=-1)
    print(f"Best PSSM: idx={best_idx}, loss={losses[best_idx]:.4f}")
    print("".join(TOKENS[int(i)] for i in init_seq))

    best_seq, best_val = batch_greedy_descent(
        loss,
        init_seq,
        batch_size=4,
        steps=50,
        key=jax.random.key(0),
    )
    print(f"Greedy result: {best_val:.4f}")
    print("".join(TOKENS[int(i)] for i in best_seq))
    return (best_seq,)


@app.cell
def _(best_seq, features, jax, protenix, writer):
    pred = protenix.predict(
        PSSM=jax.nn.one_hot(best_seq, 20),
        features=features,
        recycling_steps=4,
        key=jax.random.key(0),
        writer=writer,
    )
    return (pred,)


@app.cell
def _(pred):
    pred.iptm
    return


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
def _(pdb_viewer, pred):
    pdb_viewer(pred.st)
    return


@app.cell
def _(mo, pred):
    mo.download(data=pred.st.make_minimal_pdb(), filename="protenix_batched.pdb")
    return


if __name__ == "__main__":
    app.run()
