import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full")

with app.setup:
    # Initialization code that runs before all other cells
    import mosaic.losses.boltz
    boltz1 = mosaic.losses.boltz.load_boltz()


@app.cell(hide_code=True)
def _(mo):
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
    from mosaic.proteinmpnn.mpnn import ProteinMPNN
    import gemmi
    return ProteinMPNN, gemmi


@app.cell
def _():
    from mosaic.common import LossTerm
    import jax.numpy as jnp
    return (LossTerm,)


@app.cell
def _():
    from mosaic.losses.transformations import ClippedLoss
    return (ClippedLoss,)


@app.cell
def _():
    from mosaic.common import TOKENS
    return (TOKENS,)


@app.cell
def _():
    import marimo as mo
    from ipymolstar import PDBeMolstar
    from pathlib import Path
    from boltz.model.models.boltz1 import Boltz1
    from boltz.main import BoltzDiffusionParams
    from dataclasses import asdict
    import joltz
    import jax
    import equinox as eqx
    import numpy as np
    import optax
    import boltz
    import matplotlib.pyplot as plt
    from mosaic.optimizers import (
        simplex_APGM,
        gradient_MCMC,
    )
    import mosaic.losses.boltz as bl
    import mosaic.losses.structure_prediction as sp

    return Path, bl, eqx, gradient_MCMC, jax, mo, np, plt, simplex_APGM, sp


@app.cell
def _():
    from mosaic.notebook_utils import pdb_viewer
    return (pdb_viewer,)


@app.cell
def _():
    from jaxtyping import Float, Array
    return Array, Float


@app.cell
def _():
    target_sequence = "SFPASVQLHTAVEMHHWCIPFSVDGQPAPSLRWLFNGSVLNETSFIFTEFLEPAANETVRHGCLRLNQPTHVNNGNYTLLAANPFGQASASIMAAF"
    return (target_sequence,)


@app.cell
def _(eqx):
    j_model = eqx.filter_jit(lambda model, *args, **kwargs: model(*args, **kwargs))
    return (j_model,)


@app.cell
def _(bl, gemmi, j_model, jax, pdb_viewer):
    def predict(sequence, features, writer):
        o = j_model(
            boltz1,
            bl.set_binder_sequence(sequence, features),
            key=jax.random.key(5),
            sample_structure=True,
            confidence_prediction=True,
            deterministic=True,
        )
        out_path = writer(o["sample_atom_coords"])
        viewer = pdb_viewer(gemmi.read_structure(str(out_path)))
        print("plddt", o["plddt"][: sequence.shape[0]].mean())
        print("ipae", o["complex_ipae"].item())
        return o, viewer
    return (predict,)


@app.cell
def _(scaffold_sequence):
    binder_length = len(scaffold_sequence)
    return (binder_length,)


@app.cell
def _(binder_length, bl, target_sequence):
    boltz_features, boltz_writer = bl.make_binder_features(
        binder_length,
        target_sequence,
    )
    return boltz_features, boltz_writer


@app.cell(hide_code=True)
def _(mo):
    mo.md("""First let's define a simple loss function to optimize.""")
    return


@app.cell
def _(bl, boltz_features, sp):
    loss = bl.Boltz1Loss(
        joltz1=boltz1,
        loss=2 * sp.BinderTargetContact() + sp.WithinBinderContact(),
        features=boltz_features,
        recycling_steps=0,
        deterministic=False,
    )
    return (loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Now we run an optimizer -- in this case an accelerated proximal gradient method -- to get an initial soluton""")
    return


@app.cell
def _(PSSM, boltz_features, boltz_writer, predict):
    _o, _viewer = predict(PSSM, boltz_features, boltz_writer)
    _viewer
    return


@app.cell
def _(binder_length, jax, loss, np, simplex_APGM):
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
        stepsize=0.1 * np.sqrt(binder_length),
        momentum=0.9,
    )

    return (PSSM,)


@app.cell
def _(PSSM, boltz_features, boltz_writer, predict):
    soft_output, _viewer = predict(
        PSSM, boltz_features, boltz_writer
    )
    _viewer
    return (soft_output,)


@app.cell
def _(PSSM, soft_output, visualize_output):
    visualize_output(soft_output, PSSM)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""This looks pretty good (usually), but it isn't a single sequence (check out the PSSM above)! We could inverse fold the structure but instead let's try to 'sharpen' the PSSM to get to an extreme point of the probability simplex.""")
    return


@app.cell
def _(PSSM, binder_length, loss, np, simplex_APGM):
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
def _(pssm_sharper, sharp_outputs, visualize_output):
    visualize_output(sharp_outputs, pssm_sharper)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    Hopefully this still looks pretty good and is now a single sequence!

    One final check: when we run Boltz properly (i.e with all side-chain atoms) does it still like this sequence?
    """
    )
    return


@app.cell
def _(Path, TOKENS, bl, target_sequence):
    # Let's repredict our designed sequence with the correct sidechains, hopefully Boltz still likes it
    def repredict(pssm, target_sequence=target_sequence):
        binder_seq = "".join(TOKENS[i] for i in pssm.argmax(-1))
        print(binder_seq)
        out_dir = Path(f"/tmp/proteins/{binder_seq[:10]}_{target_sequence[:10]}")
        out_dir.mkdir(exist_ok=True, parents=True)

        return bl.load_features_and_structure_writer(
            bl.get_input_yaml(
                binder_sequence=binder_seq, targets_sequence=target_sequence
            )
        )
    return (repredict,)


@app.cell
def _(mo, predict, pssm_sharper, repredict, target_sequence):
    f_r, _w = repredict(pssm_sharper, target_sequence=target_sequence)

    repredicted_output, repredicted_viewer = predict(
        f_r["res_type"][0][:, 2:22], f_r, _w
    )

    with open(next(_w.out_dir.glob("*/*.cif")), "r") as _f:
        download_structure = mo.download(_f.read(), filename="next.cif")

    repredicted_viewer
    return (download_structure,)


@app.cell
def _(download_structure):
    download_structure
    return


@app.cell
def _(
    TOKENS,
    af2,
    jax,
    mo,
    pdb_viewer,
    pssm_sharper,
    target_sequence,
    target_st,
):
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
def _(mo):
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
    return AF2, AlphaFoldLoss, FixedStructureInverseFoldingLL


@app.cell
def _():
    scaffold_sequence = "SVIEKLRKLEKQARKQGDEVLVMLARMVLEYLEKGWVSEEDADESADRIEEVLKK"
    return (scaffold_sequence,)


@app.cell
def _(AF2):
    af2 = AF2()
    return (af2,)


@app.cell
def _(mo):
    mo.md("""Now let's define a loss term that wraps and clips another loss functional.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Let's add a loss term that penalizes cysteines.""")
    return


@app.cell
def _(Array, Float, LossTerm, TOKENS):
    class NoCysteine(LossTerm):
        def __call__(self, seq: Float[Array, "N 20"], *, key):
            p_cys = seq[:, TOKENS.index("C")].sum()
            return p_cys, {"p_cys": p_cys}
    return (NoCysteine,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    Next, we'll predict the scaffold alone using AF2 (we could use the crystal structure instead but this works fine). We'll use the predicted structure in two loss terms:

    1. Cross entropy between the distograms for the scaffold ground truth sequence and our designed binder
    2. Inverse folding log probability of our designed binder as predicted by proteinMPNN applied to the scaffold structure
    """
    )
    return


@app.cell
def _(af2, jax, pdb_viewer, scaffold_sequence):
    o_af_scaffold, st_af_scaffold = af2.predict(
        [scaffold_sequence],
        template_chains={},
        key=jax.random.key(0),
        model_idx=0,
    )

    pdb_viewer(st_af_scaffold)
    return o_af_scaffold, st_af_scaffold


@app.cell
def _(FixedStructureInverseFoldingLL, ProteinMPNN, st_af_scaffold):
    # Create inverse folding LL term
    scaffold_inverse_folding_LL = FixedStructureInverseFoldingLL.from_structure(
        st_af_scaffold,
        ProteinMPNN.from_pretrained(),
    )
    return (scaffold_inverse_folding_LL,)


@app.cell
def _(af2, binder_length, target_sequence, target_st):
    ### Generate input features for alphafold
    # We use a template for the target chain!
    af_features, initial_guess = af2.build_features(
        chains=["G" * binder_length, target_sequence],
        template_chains={1: target_st[0][0]},
    )
    return (af_features,)


@app.cell
def _(
    AlphaFoldLoss,
    ClippedLoss,
    NoCysteine,
    af2,
    af_features,
    jax,
    o_af_scaffold,
    scaffold_inverse_folding_LL,
    sp,
):
    af_loss = (
        AlphaFoldLoss(
            name="af",
            forward=af2.alphafold_apply,
            stacked_params=jax.device_put(af2.stacked_model_params),
            features=af_features,
            losses=1.0 * sp.PLDDTLoss()
            + 1 * sp.BinderTargetContact()
            + 0.1 * sp.TargetBinderPAE()
            + 0.1 * sp.BinderTargetPAE()
            + 0.5
            * ClippedLoss(
                sp.DistogramCE(
                    jax.nn.softmax(o_af_scaffold.distogram.logits),
                    name="scaffoldCE",
                ),
                2,
                100,
            ),
        )
        + ClippedLoss(scaffold_inverse_folding_LL, 2, 100)
        + NoCysteine()
    )
    return (af_loss,)


@app.cell
def _(af_loss, binder_length, jax, np, simplex_APGM):
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
        stepsize=0.1 * np.sqrt(binder_length),
        momentum=0.5,
    )
    return (pssm_af,)


@app.cell
def _(binder_length, loss, np, pssm_af, simplex_APGM):
    pssm_sharper_af, _ = simplex_APGM(
        loss_function=loss,
        n_steps=25,
        x=pssm_af,
        stepsize = 0.2 * np.sqrt(binder_length),
        scale = 1.1
    )
    pssm_sharper_af, _ = simplex_APGM(
        loss_function=loss,
        n_steps=25,
        x=pssm_sharper_af,
        stepsize = 0.2 * np.sqrt(binder_length),
        scale = 1.5
    )
    return (pssm_sharper_af,)


@app.cell
def _(mo):
    mo.md("""Let's test this out by predicting the complex structure with Boltz and AF2""")
    return


@app.cell
def _(boltz_features, boltz_writer, predict, pssm_sharper_af):
    boltz_output, _viewer = predict(pssm_sharper_af, boltz_features, boltz_writer)
    _viewer
    return (boltz_output,)


@app.cell
def _(boltz_output, pssm_sharper_af, visualize_output):
    visualize_output(boltz_output, pssm_sharper_af)
    return


@app.cell
def _(TOKENS, af2, jax, pssm_sharper_af, target_sequence, target_st):
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
def _(o_pred, plt):
    _f = plt.imshow(o_pred.predicted_aligned_error)
    plt.title(f"AF2 PAE, iptm: {o_pred.iptm: 0.3f}")
    plt.colorbar()
    _f
    return


@app.cell
def _(mo):
    mo.md("""For fun (and to show how easy it is to use different optimization algorithms) let's try polishing this design using gradient-assisted MCMC""")
    return


@app.cell
def _(af_loss, gradient_MCMC, jax, pssm_sharper_af):
    seq_mcmc = gradient_MCMC(
        af_loss,
        jax.device_put(pssm_sharper_af.argmax(-1)),
        temp=0.001,
        proposal_temp=0.01,
        steps=100,
    )
    return (seq_mcmc,)


@app.cell
def _(boltz_features, boltz_writer, jax, predict, seq_mcmc):
    predict(jax.nn.one_hot(seq_mcmc, 20), boltz_features, boltz_writer)
    return


@app.cell
def _(TOKENS, af2, jax, plt, seq_mcmc, target_sequence, target_st):
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
def _(mcmc_st, pdb_viewer):
    pdb_viewer(mcmc_st)
    return


@app.cell
def _(mcmc_st, mo):
    mo.download(
        mcmc_st.make_pdb_string(),
        filename="mcmc.pdb",
        label="AF2 predicted complex",
    )
    return


@app.cell
def _(jax, plt, seq_mcmc):
    plt.imshow(jax.nn.one_hot(seq_mcmc, 20))
    return


@app.cell
def _(mo):
    mo.md("""As a final example we'll try minimizing the same loss function using projected gradient descent on the simplex -- which also seems to work just fine.""")
    return


@app.cell
def _(boltz_features, boltz_writer, predict, pssm_af):
    predict(pssm_af, boltz_features, boltz_writer)
    return


@app.cell
def _(plt, pssm_af):
    plt.imshow(pssm_af)
    return


@app.cell
def _(bl, gemmi, j_model, jax, pdb_viewer, target_sequence):
    # predict target - we'll use this as a template for alphafold

    target_features, target_writer = bl.make_monomer_features(target_sequence)


    o_target = j_model(
        boltz1,
        target_features,
        key=jax.random.key(5),
        sample_structure=True,
        confidence_prediction=True,
    )

    out_path_target = target_writer(o_target["sample_atom_coords"])
    target_st = gemmi.read_structure(str(out_path_target))
    viewer_target = pdb_viewer(target_st)
    viewer_target
    return (target_st,)


@app.cell
def _(TOKENS, seq_mcmc):
    "".join([TOKENS[i] for i in seq_mcmc])
    return


@app.cell
def _(TOKENS, pssm_sharper_af):
    "".join([TOKENS[i] for i in pssm_sharper_af.argmax(-1)])
    return


@app.cell(hide_code=True)
def _(mo, plt):
    def visualize_output(outputs, pssm):
        _f = plt.imshow(outputs["i_pae"][0])
        plt.title(f"Boltz PAE")
        plt.colorbar()
        _f

        _g = plt.figure(dpi=125)
        plt.plot(outputs["plddt"][0])
        plt.title("pLDDT")
        plt.vlines([pssm.shape[0]], 0, 1, color="red", linestyles="--")

        _h = plt.figure(dpi=125)
        plt.imshow(pssm)
        plt.xlabel("Amino acid")
        plt.ylabel("Sequence position")

        return mo.ui.tabs({"PAE": _f, "pLDDT": _g, "PSSM": _h})
    return (visualize_output,)


if __name__ == "__main__":
    app.run()
