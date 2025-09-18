import marimo

__generated_with = "0.15.5"
app = marimo.App(width="full")

with app.setup:
    import jax
    # jax.config.update("jax_debug_nans", True)

    import marimo as mo

    from ipymolstar import PDBeMolstar
    from pathlib import Path
    from boltz.model.models.boltz1 import Boltz1
    from boltz.main import BoltzDiffusionParams
    from dataclasses import asdict
    import joltz

    import equinox as eqx
    import numpy as np
    import optax
    import boltz
    import matplotlib.pyplot as plt
    from mosaic.optimizers import (
        simplex_APGM,
        gradient_MCMC,
    )
    import mosaic.losses.structure_prediction as sp
    from mosaic.models.boltz1 import Boltz1

    from mosaic.common import TOKENS, tokenize
    from mosaic.losses.transformations import SoftClip
    from mosaic.notebook_utils import pdb_viewer
    from jaxtyping import Float, Array
    from mosaic.common import LossTerm
    import jax.numpy as jnp
    from mosaic.structure_prediction import TargetChain
    from mosaic.models.af2 import AlphaFold2
    from mosaic.proteinmpnn.mpnn import ProteinMPNN
    import gemmi


@app.cell(hide_code=True)
def _():
    mo.md(
        """
    ---
    **Warning**

    1. You'll almost certainly need a GPU or TPU
    2. Because JAX uses JIT compilation the first execution of a cell may take quite a while
    3. You might have to run these optimization methods multiple times before you get a reasonable binder
    4. If you wanted to, you could certainly find better hyperparameters for these examples (for faster or better optimization)
    ---
    """
    )
    return


@app.cell
def _():
    boltz1 = Boltz1()
    return (boltz1,)


@app.cell
def _():
    target_sequence = "SFPASVQLHTAVEMHHWCIPFSVDGQPAPSLRWLFNGSVLNETSFIFTEFLEPAANETVRHGCLRLNQPTHVNNGNYTLLAANPFGQASASIMAAF"
    return (target_sequence,)


@app.cell
def _(boltz1):
    def predict(sequence, features, writer):
        pred = boltz1.predict(PSSM = sequence, features=features, writer=writer, key = jax.random.key(11))
        return pred, pdb_viewer(pred.st)
    return (predict,)


@app.cell
def _(scaffold_sequence):
    binder_length = len(scaffold_sequence)
    return (binder_length,)


@app.cell
def _(binder_length, boltz1, target_sequence):
    boltz_features, boltz_writer = boltz1.binder_features(
        binder_length=binder_length,
        chains=[TargetChain(sequence=target_sequence)],
    )
    return boltz_features, boltz_writer


@app.cell(hide_code=True)
def _():
    mo.md("""First let's define a simple loss function to optimize.""")
    return


@app.cell
def _(boltz1, boltz_features):
    loss = boltz1.build_loss(
        loss=2 * sp.BinderTargetContact() + sp.WithinBinderContact(),
        features=boltz_features,
        recycling_steps=1,
    )
    return (loss,)


@app.cell(hide_code=True)
def _():
    mo.md("""Now we run an optimizer -- in this case an accelerated proximal gradient method -- to get an initial soluton""")
    return


@app.cell
def _(PSSM, boltz_features, boltz_writer, predict):
    _o, _viewer = predict(PSSM, boltz_features, boltz_writer)
    _viewer
    return


@app.cell
def _():
    # _, PSSM = simplex_APGM(
    #     loss_function=loss,
    #     x=jax.nn.softmax(
    #         0.5
    #         * jax.random.gumbel(
    #             key=jax.random.key(np.random.randint(100000)),
    #             shape=(binder_length, 20),
    #         )
    #     ),
    #     n_steps=100,
    #     stepsize=0.1 * np.sqrt(binder_length),
    #     momentum=0.9,
    # )
    return


@app.cell
def _(PSSM, boltz_features, boltz_writer, predict):
    soft_output, _viewer = predict(
        PSSM, boltz_features, boltz_writer
    )
    _viewer
    return (soft_output,)


@app.cell
def _(PSSM, soft_output):
    visualize_output(soft_output, PSSM)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""This looks pretty good (usually), but it isn't a single sequence (check out the PSSM above)! We could inverse fold the structure but instead let's try to 'sharpen' the PSSM to get to an extreme point of the probability simplex.""")
    return


@app.cell
def _(PSSM, binder_length, loss):
    # we can sharpen these logits using weight decay (which is equivalent to adding entropic regularization)
    pssm_sharper, _ = simplex_APGM(
        loss_function=loss,
        n_steps=25,
        x=PSSM,
        stepsize = 0.2 * np.sqrt(binder_length),
        scale = 1.1
    )
    pssm_sharper, _ = simplex_APGM(
        loss_function=loss,
        n_steps=25,
        x=pssm_sharper,
        stepsize = 0.2 * np.sqrt(binder_length),
        scale = 1.5
    )
    return (pssm_sharper,)


@app.cell
def _(boltz_features, boltz_writer, predict, pssm_sharper):
    sharp_outputs, _viewer = predict(
        pssm_sharper, boltz_features, boltz_writer
    )
    _viewer
    return (sharp_outputs,)


@app.cell
def _(pssm_sharper, sharp_outputs):
    visualize_output(sharp_outputs, pssm_sharper)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""Hopefully this still looks pretty good and is now a single sequence!""")
    return


@app.cell
def _(download_structure):
    download_structure
    return


@app.cell
def _(af2, pssm_sharper, target_sequence, target_st):
    mo.md("""Finally, let's repredict with AF2-multimer""")
    _o_af_scaffold, _st_af_scaffold = af2.predict(
        ["".join([TOKENS[i] for i in pssm_sharper.argmax(-1)]), target_sequence],
        template_chains={1: target_st[0][0]},
        key=jax.random.key(0),
        model_idx=0,
    )
    print(_o_af_scaffold.iptm)
    pdb_viewer(_st_af_scaffold)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        """
    Okay, that was fun but let's do a something a little more complicated: we'll use AlphaFold2 (instead of Boltz) to design a binder that adheres to a specified fold. [7S5B](https://www.rcsb.org/structure/7S5B) is a denovo triple-helix bundle originally designed to bind IL-7r; let's see if we can find a sequence _with the same fold_ that AF thinks will bind to our target instead.

    To do so we'll add two terms to our loss function:

    1. The log-likelihood of our sequence according to ProteinMPNN applied to the scaffold structure
    2. Cross-entropy between the predicted distogram of our sequence and the original 7S5B sequence

    We'll also show how easy it is to modify loss terms by clipping these two functionals.
    """
    )
    return


@app.cell
def _():
    from mosaic.af2.alphafold2 import AF2
    from mosaic.losses.af2 import AlphaFoldLoss
    import mosaic.losses.af2 as aflosses
    from mosaic.losses.protein_mpnn import (
        FixedStructureInverseFoldingLL,
    )
    return (FixedStructureInverseFoldingLL,)


@app.cell
def _():
    scaffold_sequence = "SVIEKLRKLEKQARKQGDEVLVMLARMVLEYLEKGWVSEEDADESADRIEEVLKK"
    return (scaffold_sequence,)


@app.cell
def _():
    af2 = AlphaFold2()
    return (af2,)


@app.cell(hide_code=True)
def _():
    mo.md("""Let's add a loss term that penalizes cysteines.""")
    return


@app.class_definition
class NoCysteine(LossTerm):
    def __call__(self, seq: Float[Array, "N 20"], *, key):
        p_cys = seq[:, TOKENS.index("C")].sum()
        return p_cys, {"p_cys": p_cys}


@app.cell(hide_code=True)
def _():
    mo.md(
        """
    Next, we'll predict the scaffold alone using AF2 (we could use the crystal structure instead but this works fine). We'll use the predicted structure in two loss terms:

    1. Cross entropy between the distograms for the scaffold ground truth sequence and our designed binder
    2. Inverse folding log probability of our designed binder as predicted by proteinMPNN applied to the scaffold structure
    """
    )
    return


@app.cell
def _(af2, scaffold_sequence):
    _scaffold_features, _= af2.target_only_features(chains = [TargetChain(sequence=scaffold_sequence, use_msa = False)])


    o_af_scaffold = af2.predict(
        features = _scaffold_features,
        recycling_steps = 3,
        key=jax.random.key(0),
        writer = None
    )

    af_scaffold_logits = af2.model_output(
         features = _scaffold_features,
        recycling_steps = 1,
        key=jax.random.key(0),
    ).distogram_logits

    pdb_viewer(o_af_scaffold.st)
    return af_scaffold_logits, o_af_scaffold


@app.cell
def _(FixedStructureInverseFoldingLL, o_af_scaffold):
    # Create inverse folding LL term
    scaffold_inverse_folding_LL = FixedStructureInverseFoldingLL.from_structure(
        o_af_scaffold.st,
        ProteinMPNN.from_pretrained(),
    )
    return (scaffold_inverse_folding_LL,)


@app.cell
def _():
    # class SoftClip(LossTerm):
    #     loss: LossTerm
    #     l: eqx.field(converter=jnp.array)
    #     alpha: eqx.field(converter=jnp.array)
    #     name: str = "elu"

    #     def __call__(self, *args, key, **kwargs):
    #         v, aux = self.loss(*args, key=key, **kwargs)
    #         z = jax.nn.elu((v - self.l)*self.alpha)
    #         return z, aux | {self.name: z}
    return


@app.cell
def _(af2, binder_length, target_sequence, target_st):
    # ### Generate input features for alphafold
    # # We use a template for the target chain!
    af_features, _ = af2.binder_features(
        binder_length=binder_length,
        chains=[
            TargetChain(
                target_sequence, use_msa=False, template_chain=target_st[0][0]
            )
        ],
    )
    return (af_features,)


@app.cell
def _():
    # af_features, initial_guess = af2_old.build_features(
    #     chains=["G" * binder_length, target_sequence],
    #     template_chains={1: target_st[0][0]},
    # )
    return


@app.cell
def _(af_scaffold_logits):
    plt.imshow(jax.nn.softmax(af_scaffold_logits)[..., 35])
    return


@app.cell
def _():
    # af_loss = (
    #         AlphaFoldLoss(
    #             name="af",
    #             forward=af2_old.alphafold_apply,
    #             stacked_params=jax.device_put(af2_old.stacked_model_params),
    #             features=af_features,
    #             loss=0.1 * sp.PLDDTLoss()
    #             + 1 * sp.BinderTargetContact()
    #             + 1 * sp.WithinBinderContact()
    #             + 0.1 * sp.TargetBinderPAE()
    #             + 0.1 * sp.BinderTargetPAE(),
    #             + 2.5
    #             * ClippedLoss(
    #                 sp.DistogramCE(
    #                     jax.nn.softmax(o_af_scaffold.distogram.logits),
    #                     name="scaffoldCE",
    #                 ),
    #                 2,
    #                 100,
    #             ),
    #         )
    #         + ClippedLoss(scaffold_inverse_folding_LL, 2, 100)
    #         + NoCysteine()
    #     )
    return


@app.cell
def _(af2, af_features, af_scaffold_logits, scaffold_inverse_folding_LL):
    af_loss = (
        af2.build_loss(
            loss=1.0 * sp.PLDDTLoss()
            + 1 * sp.BinderTargetContact()
            + 0.05 * sp.TargetBinderPAE()
            + 0.05 * sp.BinderTargetPAE()
            + 0.025 * sp.IPTMLoss()
            + 0.0 * sp.PLDDTLoss()
            + 0.4 * sp.WithinBinderPAE()
            + 1.0 * sp.WithinBinderContact()
            + 2.5*SoftClip(
                sp.DistogramCE(
                    jax.nn.softmax(af_scaffold_logits),
                    name="scaffoldCE",
                ),
                2.5,
                3
            ),
            # + 5.5
            # * ClippedLoss(
            #     sp.DistogramCE(
            #         jax.nn.softmax(af_scaffold_logits),
            #         name="scaffoldCE",
            #     ),
            #     2.75,
            #     100,
            # ),
            features=af_features,
        )
        + 2.0*SoftClip(scaffold_inverse_folding_LL, 2.5, 3.0)
        + NoCysteine()
    )
    return (af_loss,)


@app.cell
def _(af_loss, binder_length):
    _, pssm_af = simplex_APGM(
        loss_function=af_loss,
        x=jax.nn.softmax(
            0.5
            * jax.random.gumbel(
                key=jax.random.key(np.random.randint(100000)),
                shape=(binder_length, 20),
            )
        ),
        n_steps=100,
        stepsize=0.1,
        momentum=0.0,
        serial_evaluation=True,
    )
    return (pssm_af,)


@app.cell
def _(af_loss, pssm_af):
    pssm_sharper_af, _ = simplex_APGM(
        loss_function=af_loss,
        n_steps=25,
        x=pssm_af,
        stepsize = 0.2,
        scale = 1.2,
        serial_evaluation=True
    )
    pssm_sharper_af, _ = simplex_APGM(
        loss_function=af_loss,
        n_steps=25,
        x=pssm_sharper_af,
        stepsize = 0.2,
        scale = 1.25,
        serial_evaluation=True
    )
    return (pssm_sharper_af,)


@app.cell
def _():
    mo.md("""Let's test this out by predicting the complex structure with Boltz and AF2""")
    return


@app.cell
def _(boltz_features, boltz_writer, predict, pssm_sharper_af):
    boltz_output, _viewer = predict(pssm_sharper_af, boltz_features, boltz_writer)
    _viewer
    return (boltz_output,)


@app.cell
def _(boltz_output, pssm_sharper_af):
    visualize_output(boltz_output, pssm_sharper_af)
    return


@app.cell
def _(af2, pssm_sharper_af, target_sequence, target_st):
    o_pred, st_pred = af2.predict(
        [
            "".join([TOKENS[i] for i in pssm_sharper_af.argmax(-1)]),
            target_sequence,
        ],
        template_chains={1: target_st[0][0]},
        key=jax.random.key(0),
        model_idx=0,
    )
    return (o_pred,)


@app.cell
def _(o_pred):
    _f = plt.imshow(o_pred.predicted_aligned_error)
    plt.title(f"AF2 PAE, iptm: {o_pred.iptm: 0.3f}")
    plt.colorbar()
    _f
    return


@app.cell
def _():
    mo.md("""For fun (and to show how easy it is to use different optimization algorithms) let's try polishing this design using gradient-assisted MCMC""")
    return


@app.cell
def _(af_loss, pssm_sharper_af):
    seq_mcmc = gradient_MCMC(
        af_loss,
        jax.device_put(pssm_sharper_af.argmax(-1)),
        temp=0.001,
        proposal_temp=0.01,
        steps=100,
        fix_loss_key=False,
        serial_evaluation=True
    )
    return (seq_mcmc,)


@app.cell
def _():
    return


@app.cell
def _(boltz_features, boltz_writer, predict, seq_mcmc):
    predict(jax.nn.one_hot(seq_mcmc, 20), boltz_features, boltz_writer)
    return


@app.cell
def _(af2, seq_mcmc, target_sequence, target_st):
    _o_pred, mcmc_st = af2.predict(
        [
            "".join([TOKENS[i] for i in seq_mcmc]),
            target_sequence,
        ],
        template_chains={1: target_st[0][0]},
        key=jax.random.key(3),
        model_idx=0,
    )
    print(_o_pred.iptm)
    plt.imshow(_o_pred.predicted_aligned_error)
    return (mcmc_st,)


@app.cell
def _(mcmc_st):
    pdb_viewer(mcmc_st)
    return


@app.cell
def _(mcmc_st):
    mo.download(
        mcmc_st.make_pdb_string(),
        filename="mcmc.pdb",
        label="AF2 predicted complex",
    )
    return


@app.cell
def _(seq_mcmc):
    plt.imshow(jax.nn.one_hot(seq_mcmc, 20))
    return


@app.cell
def _(boltz_features, boltz_writer, predict, pssm_af):
    predict(pssm_af, boltz_features, boltz_writer)
    return


@app.cell
def _(pssm_af):
    plt.imshow(pssm_af)
    return


@app.cell
def _(boltz1, target_sequence):
    # predict target - we'll use this as a template for alphafold





    target_features, target_writer = boltz1.target_only_features(chains = [TargetChain(sequence = target_sequence)])

    o_target = boltz1.predict(features = target_features, writer = target_writer, key = jax.random.key(0))



    target_st = o_target.st
    viewer_target = pdb_viewer(target_st)
    viewer_target
    return (target_st,)


@app.cell
def _(seq_mcmc):
    "".join([TOKENS[i] for i in seq_mcmc])
    return


@app.cell
def _(pssm_sharper_af):
    "".join([TOKENS[i] for i in pssm_sharper_af.argmax(-1)])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""As a final example let's try optimizing the *sum* of these loss terms; so we're calling both AF2 and Boltz1 at every iteration. In `mosaic` this is trivial.""")
    return


@app.cell
def _(af_loss, binder_length, loss):
    _, pssm_both = simplex_APGM(
        loss_function=af_loss + loss,
        x=jax.nn.softmax(
            0.5
            * jax.random.gumbel(
                key=jax.random.key(np.random.randint(100000)),
                shape=(binder_length, 20),
            )
        ),
        n_steps=100,
        stepsize=0.15, #* np.sqrt(binder_length),
        momentum=0.0,
        serial_evaluation=True
    )
    return (pssm_both,)


@app.cell
def _(boltz_features, boltz_writer, predict, pssm_both):
    predict(pssm_both, boltz_features, boltz_writer)
    return


@app.function(hide_code=True)
def visualize_output(outputs, pssm):
    _f = plt.imshow(outputs.pae)
    plt.title(f"Boltz PAE")
    plt.colorbar()
    _f

    _g = plt.figure(dpi=125)
    plt.plot(outputs.plddt)
    plt.title("pLDDT")
    plt.vlines([pssm.shape[0]], 0, 1, color="red", linestyles="--")

    _h = plt.figure(dpi=125)
    plt.imshow(pssm)
    plt.xlabel("Amino acid")
    plt.ylabel("Sequence position")

    return mo.ui.tabs({"PAE": _f, "pLDDT": _g, "PSSM": _h})


if __name__ == "__main__":
    app.run()
