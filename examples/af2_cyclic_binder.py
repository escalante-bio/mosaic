import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")

with app.setup:
    import jax
    import marimo as mo
    import numpy as np

    import mosaic.losses.structure_prediction as sp
    from mosaic.models.af2 import AlphaFold2
    from mosaic.notebook_utils import pdb_viewer
    from mosaic.optimizers import simplex_APGM
    from mosaic.structure_prediction import TargetChain


@app.cell(hide_code=True)
def _():
    mo.md("""
    ---
    **Warning**

    1. You'll almost certainly need a GPU or TPU
    2. Because JAX uses JIT compilation the first execution of a cell may take quite a while
    3. You might have to run these optimization methods multiple times before you get a reasonable binder
    ---

    This example designs a head-to-tail **cyclic** binder against a target using AF2. Two
    things make a binder "cyclic" here:

    1. `AlphaFold2.binder_features(..., cyclic=True)`, which biases the structure module's
       residue-index offsets (`mosaic.geometry.cyclic_offset_matrix`) so the trunk treats the
       binder as a closed loop rather than a linear chain.
    2. `sp.CyclicClosureLoss()`, an explicit contact loss between the binder's first and last
       residues. The offset bias alone isn't consistently strong enough to force closure once
       it's combined with other contact losses during full sequence design, so we add this term
       directly to the loss.
    """)
    return


@app.cell
def _():
    target_sequence = "SFPASVQLHTAVEMHHWCIPFSVDGQPAPSLRWLFNGSVLNETSFIFTEFLEPAANETVRHGCLRLNQPTHVNNGNYTLLAANPFGQASASIMAAF"
    return (target_sequence,)


@app.cell
def _():
    binder_length = 20
    return (binder_length,)


@app.cell
def _():
    af2 = AlphaFold2()
    return (af2,)


@app.cell
def _(af2, binder_length, target_sequence):
    af_features, af_writer = af2.binder_features(
        binder_length=binder_length,
        chains=[TargetChain(sequence=target_sequence, use_msa=False)],
        cyclic=True,
    )
    return af_features, af_writer


@app.cell(hide_code=True)
def _():
    mo.md("""
    We combine the usual binder-design losses with `CyclicClosureLoss` to enforce
    head-to-tail closure, and pass `cyclic=True` to `WithinBinderContact` so its
    sequence-separation mask wraps around the cyclic binder instead of treating it as linear.
    """)
    return


@app.cell
def _(af2, af_features):
    loss = af2.build_loss(
        loss=sp.BinderTargetContact()
        + sp.WithinBinderContact(cyclic=True)
        + sp.CyclicClosureLoss(),
        features=af_features,
        recycling_steps=2,
    )
    return (loss,)


@app.cell
def _(af2, af_features, af_writer):
    def binder_closure_distance(st):
        """CA-CA distance between the binder's first and last residues (chain 0)."""
        binder_chain = st[0][0]
        first_ca = binder_chain[0].get_ca().pos
        last_ca = binder_chain[len(binder_chain) - 1].get_ca().pos
        return first_ca.dist(last_ca)

    def predict(sequence):
        pred = af2.predict(
            PSSM=sequence, features=af_features, writer=af_writer, key=jax.random.key(0)
        )
        closure_distance = binder_closure_distance(pred.st)
        return pred, pdb_viewer(pred.st), closure_distance

    return (predict,)


@app.cell
def _(binder_length, loss):
    _, PSSM = simplex_APGM(
        loss_function=loss,
        x=jax.nn.softmax(
            0.5
            * jax.random.gumbel(
                key=jax.random.key(np.random.randint(100000)),
                shape=(binder_length, 20),
            )
        ),
        n_steps=100,
        stepsize=0.1,
        momentum=0.9,
    )
    return (PSSM,)


@app.cell
def _(PSSM, predict):
    output, viewer, closure_distance = predict(PSSM)
    print(f"binder first-to-last CA distance: {closure_distance:.2f} A")
    viewer
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    Let's sharpen the PSSM to a single sequence and repredict.
    """)
    return


@app.cell
def _(PSSM, loss):
    pssm_sharper, _ = simplex_APGM(
        loss_function=loss,
        n_steps=25,
        x=PSSM,
        stepsize=0.2,
        scale=1.5,
    )
    return (pssm_sharper,)


@app.cell
def _(predict, pssm_sharper):
    sharp_output, sharp_viewer, sharp_closure_distance = predict(pssm_sharper)
    print(f"binder first-to-last CA distance: {sharp_closure_distance:.2f} A")
    sharp_viewer
    return


if __name__ == "__main__":
    app.run()
