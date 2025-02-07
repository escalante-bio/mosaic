import marimo

__generated_with = "0.10.19"
app = marimo.App(width="full")


@app.cell
def _():
    from boltz_binder_design.proteinmpnn.mpnn import ProteinMPNN
    import gemmi
    return ProteinMPNN, gemmi


@app.cell
def _():
    from boltz_binder_design.common import LossTerm
    import jax.numpy as jnp
    return LossTerm, jnp


@app.cell
def _():
    from boltz_binder_design.common import TOKENS
    return (TOKENS,)


@app.cell
def _():
    from boltz.data.const import prot_token_to_letter, tokens
    return prot_token_to_letter, tokens


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
    from boltz_binder_design import design_bregman_optax
    from boltz_binder_design.losses.boltz import (
        make_binder_monomer_features,
        make_binder_features,
        make_monomer_features,
        target_fasta_seq,
        load_features_and_structure_writer,
        set_binder_sequence,
        RadiusOfGyration,
        WithinBinderContact,
        BinderTargetContact,
        HelixLoss,
        StructurePrediction,
        PLDDTLoss,
        ActualRadiusOfGyration,
        load_boltz_model,
    )
    return (
        ActualRadiusOfGyration,
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
        load_boltz_model,
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
        """Display a PDB file using Molstar"""
        custom_data = {
            "data": file.read_text(),
            "format": "pdb",
            "binary": False,
        }
        return PDBeMolstar(custom_data=custom_data, theme="dark")
    return (pdb_viewer,)


@app.cell
def _():
    target_sequence = "SFPASVQLHTAVEMHHWCIPFSVDGQPAPSLRWLFNGSVLNETSFIFTEFLEPAANETVRHGCLRLNQPTHVNNGNYTLLAANPFGQASASIMAAF"
    return (target_sequence,)


@app.cell
def _(load_boltz_model):
    model = load_boltz_model()
    return (model,)


@app.cell
def _(eqx):
    j_model = eqx.filter_jit(lambda model, *args, **kwargs: model(*args, **kwargs))
    return (j_model,)


@app.cell
def _(j_model, jax, model, pdb_viewer, set_binder_sequence):
    def predict(sequence, features, writer):
        o = j_model(
            model,
            set_binder_sequence(sequence, features),
            key=jax.random.key(5),
            sample_structure=True,
            confidence_prediction=True,
            deterministic=True
        )
        out_path = writer(o["sample_atom_coords"])
        viewer = pdb_viewer(out_path)
        print("plddt", o["plddt"][: sequence.shape[0]].mean())
        print("ipae", o["complex_ipae"].item())
        return o, viewer
    return (predict,)


@app.cell
def _(scaffold_sequence):
    binder_length = len(scaffold_sequence)
    return (binder_length,)


@app.cell
def _(binder_length, make_binder_features, target_sequence):
    boltz_features, boltz_writer = make_binder_features(
        binder_length,
        target_sequence,
    )
    return boltz_features, boltz_writer


@app.cell
def _(mo):
    mo.md("""First let's define a simple loss function to optimize.""")
    return


@app.cell
def _(
    BinderTargetContact,
    StructurePrediction,
    WithinBinderContact,
    boltz_features,
    model,
):
    loss = StructurePrediction(
        model=model,
        name="target",
        loss= 2*BinderTargetContact() + WithinBinderContact(),
        features=boltz_features,
        recycling_steps=0,
        deterministic=False
    )
    return (loss,)


@app.cell
def _(mo):
    mo.md("""Now we run an optimizer to get an initial soluton""")
    return


@app.cell
def _(binder_length, design_bregman_optax, loss, np, optax):
    _, logits = design_bregman_optax(
        loss_function=loss,
        n_steps=100,
        x=np.random.randn(binder_length, 20) * 0.1,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(0.5*np.sqrt(binder_length), momentum=0.5),
        ),
    )
    return (logits,)


@app.cell
def _(boltz_features, boltz_writer, jax, logits, predict):
    soft_output, _viewer = predict(
        jax.nn.softmax(logits), boltz_features, boltz_writer
    )
    _viewer
    return (soft_output,)


@app.cell
def _(logits, soft_output, visualize_output):
    visualize_output(soft_output, logits)
    return


@app.cell
def _(mo):
    mo.md("""This looks pretty good (usually), but this isn't a single sequence! We could inverse fold the structure but instead let's try to 'sharpen' the PSSM to get an extreme point of the probability simplex.""")
    return


@app.cell
def _(binder_length, design_bregman_optax, logits, loss, np, optax):
    # we can sharpen these logits using weight decay (which is equivalent to adding entropic regularization)
    logits_sharper, _ = design_bregman_optax(
        loss_function=loss,
        n_steps=25,
        x=logits,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.01),
            optax.sgd(np.sqrt(binder_length)),
        ),
    )
    logits_sharper, _ = design_bregman_optax(
        loss_function=loss,
        n_steps=25,
        x=logits_sharper,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.05),
            optax.sgd(np.sqrt(binder_length)),
        ),
    )
    logits_sharper, _ = design_bregman_optax(
        loss_function=loss,
        n_steps=25,
        x=logits_sharper,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.1),
            optax.sgd(np.sqrt(binder_length)),
        ),
    )
    return (logits_sharper,)


@app.cell
def _(boltz_features, boltz_writer, jax, logits_sharper, predict):
    sharp_outputs, _viewer = predict(
        jax.nn.softmax(logits_sharper), boltz_features, boltz_writer
    )
    _viewer
    return (sharp_outputs,)


@app.cell
def _(logits_sharper, sharp_outputs, visualize_output):
    visualize_output(sharp_outputs, logits_sharper)
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


@app.cell(hide_code=True)
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
        out_dir = Path(f"/tmp/proteins/{binder_seq[:10]}_{target_sequence[:10]}")
        out_dir.mkdir(exist_ok=True, parents=True)
        fasta_path = out_dir / "protein.fasta"
        fasta_path.write_text(
            target_fasta_seq(binder_seq, chain="A", use_msa=True)
            + target_fasta_seq(target_sequence)
        )
        return load_features_and_structure_writer(fasta_path, out_dir)
    return (repredict,)


@app.cell
def _(logits_sharper, mo, predict, repredict, target_sequence):
    f_r, _w = repredict(logits_sharper, target_sequence=target_sequence)

    repredicted_output, repredicted_viewer = predict(
        f_r["res_type"][0][:, 2:22], f_r, _w
    )

    with open(next(_w.out_dir.glob("*/*.pdb")), "r") as _f:
        download_structure = mo.download(_f.read(), filename="next.pdb")

    repredicted_viewer
    return download_structure, f_r, repredicted_output, repredicted_viewer


@app.cell
def _(download_structure):
    download_structure
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
    from boltz_binder_design.af2.alphafold2 import AF2
    from boltz_binder_design.losses.af2 import AlphaFold
    import boltz_binder_design.losses.af2 as aflosses
    from boltz_binder_design.losses.protein_mpnn import (
        FixedChainInverseFoldingLL,
    )
    return AF2, AlphaFold, FixedChainInverseFoldingLL, aflosses


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


@app.cell
def _(LossTerm):
    class ClippedLoss(LossTerm):
        loss: LossTerm
        l: float
        u: float

        def __call__(self, *args, key, **kwargs):
            v, aux = self.loss(*args, key=key, **kwargs)
            return v.clip(self.l, self.u), aux
    return (ClippedLoss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""While we're at it let's also add a term that penalizes cysteines.""")
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
def _(Path, af2, jax, pdb_viewer, scaffold_sequence):
    o_af_scaffold, st_af_scaffold = af2.predict(
        [scaffold_sequence],
        template_chains={},
        key=jax.random.key(0),
        model_idx=0,
    )

    st_af_scaffold.write_minimal_pdb("af_scaffold.pdb")
    pdb_viewer(Path("af_scaffold.pdb"))
    return o_af_scaffold, st_af_scaffold


@app.cell
def _(
    FixedChainInverseFoldingLL,
    Path,
    ProteinMPNN,
    gemmi,
    st_af_scaffold,
):
    # Create inverse folding LL term
    st_af_scaffold.write_minimal_pdb("af_scaffold.pdb")
    scaffold_inverse_folding_LL = FixedChainInverseFoldingLL.from_structure(
        gemmi.read_structure("af_scaffold.pdb"),
        ProteinMPNN.from_pretrained(
            Path("protein_mpnn_weights/vanilla/v_48_020.pt")
        ),
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
    return af_features, initial_guess


@app.cell
def _(
    AlphaFold,
    ClippedLoss,
    NoCysteine,
    af2,
    af_features,
    aflosses,
    jax,
    o_af_scaffold,
    scaffold_inverse_folding_LL,
):
    af_loss = (
        AlphaFold(
            name="af",
            forward=af2.alphafold_apply,
            stacked_params=jax.device_put(af2.stacked_model_params),
            features=af_features,
            losses=0.01 * aflosses.PLDDTLoss()
            + 1 * aflosses.BinderTargetContact()
            + 0.1 * aflosses.TargetBinderPAE()
            + 0.1 * aflosses.BinderTargetPAE()
            + 0.5
            * ClippedLoss(
                aflosses.DistogramCE(
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
def _(af_loss, binder_length, design_bregman_optax, np, optax):
    _, logits_af = design_bregman_optax(
        loss_function=af_loss,
        n_steps=150,
        x=np.random.randn(binder_length, 20) * 0.1,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(1.0 * np.sqrt(binder_length), momentum=0.0),
        ),
    )
    return (logits_af,)


@app.cell
def _(af_loss, binder_length, design_bregman_optax, logits_af, np, optax):
    logits_af_sharper, _ = design_bregman_optax(
        loss_function=af_loss,
        n_steps=25,
        x=logits_af,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.01),
            optax.sgd(0.5 * np.sqrt(binder_length)),
        ),
    )
    logits_af_sharper, _ = design_bregman_optax(
        loss_function=af_loss,
        n_steps=25,
        x=logits_af_sharper,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.05),
            optax.sgd(0.5 * np.sqrt(binder_length)),
        ),
    )
    logits_af_sharper, _ = design_bregman_optax(
        loss_function=af_loss,
        n_steps=25,
        x=logits_af_sharper,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.1),
            optax.sgd(np.sqrt(binder_length)),
        ),
    )
    return (logits_af_sharper,)


@app.cell
def _(mo):
    mo.md("""Let's test this out by predicting the complex structure with Boltz and AF2""")
    return


@app.cell
def _(boltz_features, boltz_writer, jax, logits_af_sharper, predict):
    boltz_output, _viewer = predict(
        jax.nn.softmax(10000 * logits_af_sharper), boltz_features, boltz_writer
    )
    _viewer
    return (boltz_output,)


@app.cell
def _(boltz_output, logits_af_sharper, visualize_output):
    visualize_output(boltz_output, 10000 * logits_af_sharper)
    return


@app.cell
def _(TOKENS, af2, jax, logits_af_sharper, target_sequence, target_st):
    o_pred, st_pred = af2.predict(
        [
            "".join([TOKENS[i] for i in logits_af_sharper.argmax(-1)]),
            target_sequence,
        ],
        template_chains={1: target_st[0][0]},
        key=jax.random.key(0),
        model_idx=0,
    )
    return o_pred, st_pred


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
def _(af_loss, gradient_MCMC, logits_af_sharper):
    seq_mcmc = gradient_MCMC(
        af_loss,
        logits_af_sharper.argmax(-1),
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
def _(Path, mcmc_st, pdb_viewer):
    mcmc_st.write_minimal_pdb("test.pdb")
    pdb_viewer(Path("test.pdb"))
    return


@app.cell
def _(mcmc_st, mo):
    mcmc_st.write_minimal_pdb("mcmc.pdb")
    with open("mcmc.pdb") as _f:
        _download = mo.download(
            _f.read(), filename="mcmc.pdb", label="AF2 predicted complex"
        )
    _download
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
def _(
    af_loss,
    binder_length,
    np,
    optax,
    simplex_projected_gradient_descent,
):
    _, exp_logits_af = simplex_projected_gradient_descent(
        loss_function=af_loss,
        x=np.random.randn(binder_length, 20) * 0.1,
        n_steps=150,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(0.1 * np.sqrt(binder_length)),
        ),
    )
    return (exp_logits_af,)


@app.cell
def _(
    gemmi,
    j_model,
    jax,
    make_monomer_features,
    model,
    pdb_viewer,
    target_sequence,
):
    # predict target - we'll use this as a template for alphafold

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
    viewer_target = pdb_viewer(out_path_target)
    viewer_target
    return (
        o_target,
        out_path_target,
        target_features,
        target_st,
        target_writer,
        viewer_target,
    )


@app.cell
def _(TOKENS, seq_mcmc):
    "".join([TOKENS[i] for i in seq_mcmc])
    return


@app.cell
def _(TOKENS, logits_af_sharper):
    "".join([TOKENS[i] for i in logits_af_sharper.argmax(-1)])
    return


@app.cell
def _():
    from boltz_binder_design.optimizers import (
        gradient_MCMC,
        simplex_projected_gradient_descent,
    )
    return gradient_MCMC, simplex_projected_gradient_descent


@app.cell(hide_code=True)
def _(jax, mo, plt):
    def visualize_output(outputs, logits):
        _f = plt.imshow(outputs["i_pae"][0])
        plt.title(f"Boltz PAE")
        plt.colorbar()
        _f

        _g = plt.figure(dpi=125)
        plt.plot(outputs["plddt"][0])
        plt.title("pLDDT")
        plt.vlines([logits.shape[0]], 0, 1, color="red", linestyles="--")

        _h = plt.figure(dpi=125)
        plt.imshow(jax.nn.softmax(logits))
        plt.xlabel("Amino acid")
        plt.ylabel("Sequence position")

        return mo.ui.tabs({"PAE": _f, "pLDDT": _g, "PSSM": _h})
    return (visualize_output,)


if __name__ == "__main__":
    app.run()
