import marimo

__generated_with = "0.10.15"
app = marimo.App(width="full")


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
    target_sequence = "ETRECIYYNANWELERTNQSGLERCEGEQDKRLHCYASWRNSSGTIELVKKGCWLDDFNCYDRQECVATEENPQVYFCCCEGNFCNERFTHLP"
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
        )
        out_path = writer(o["sample_atom_coords"])
        viewer = pdb_viewer(out_path)
        print("plddt", o["plddt"][: sequence.shape[0]].mean())
        print("ipae", o["complex_ipae"].item())
        return o, viewer
    return (predict,)


@app.cell
def _():
    binder_length = 100
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
        loss=14 * BinderTargetContact() + 3.0 * WithinBinderContact(),
        features=boltz_features,
        recycling_steps=0,
    )
    return (loss,)


@app.cell
def _(mo):
    mo.md("""Now we run an optimizer to get an initial soluton""")
    return


@app.cell
def _(binder_length, design_bregman_optax, loss, np, optax):
    logits = design_bregman_optax(
        loss_function=loss,
        n_steps=150,
        x=np.random.randn(binder_length, 20) * 0.1,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(np.sqrt(binder_length), momentum=0.5),
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


@app.cell(hide_code=True)
def _(jax, mo, plt):
    def visualize_output(outputs, logits):
        _f = plt.imshow(outputs["i_pae"][0])
        plt.title("PAE")
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
    logits_sharper = design_bregman_optax(
        loss_function=loss,
        n_steps=25,
        x=logits,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.01),
            optax.sgd(np.sqrt(binder_length)),
        ),
    )
    logits_sharper = design_bregman_optax(
        loss_function=loss,
        n_steps=25,
        x=logits_sharper,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.05),
            optax.sgd(np.sqrt(binder_length)),
        ),
    )
    logits_sharper = design_bregman_optax(
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


@app.cell
def _(mo):
    mo.md(
        """
        Hopefully this still looks pretty good and is now a single sequence!

            One final check: when we run Boltz properly (i.e with all side-chain atoms) does it still like this sequence?
        """
    )
    return


@app.cell
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


@app.cell
def _():
    # What if we wanted to optimize a combination of multiple objectives?
    # Let's load a bunch of different models to predict different properties of our binder
    return


@app.cell
def _():
    from boltz_binder_design.esm.pretrained import load_pretrained_esm
    from boltz_binder_design.losses.esm import ESM2PseudoLikelihood
    from boltz_binder_design.losses.trigram import TrigramLL
    from boltz_binder_design.losses.stability import StabilityModel
    from boltz_binder_design.losses.protein_mpnn import (
        BoltzProteinMPNNLoss,
        FixedChainInverseFoldingLL,
    )
    from boltz_binder_design.proteinmpnn.mpnn import ProteinMPNN
    import gemmi
    return (
        BoltzProteinMPNNLoss,
        ESM2PseudoLikelihood,
        FixedChainInverseFoldingLL,
        ProteinMPNN,
        StabilityModel,
        TrigramLL,
        gemmi,
        load_pretrained_esm,
    )


@app.cell
def _():
    from boltz_binder_design.losses.boltz import DistogramCE
    return (DistogramCE,)


@app.cell
def _(ProteinMPNN):
    mpnn = ProteinMPNN.from_pretrained()
    return (mpnn,)


@app.cell
def _(BoltzProteinMPNNLoss, mpnn):
    complex_inverse_folding_LL = BoltzProteinMPNNLoss(mpnn, 16, stop_grad=False)
    return (complex_inverse_folding_LL,)


@app.cell
def _(FixedChainInverseFoldingLL, Path, ProteinMPNN, gemmi):
    scaffold_inverse_folding_LL = FixedChainInverseFoldingLL.from_structure(
        gemmi.read_structure("7s5b.pdb"),
        ProteinMPNN.from_pretrained(
            Path("protein_mpnn_weights/vanilla/v_48_020.pt")
        ),
    )
    return (scaffold_inverse_folding_LL,)


@app.cell
def _(ESM2PseudoLikelihood, load_pretrained_esm):
    esm, _ = load_pretrained_esm()
    esm_pseudo_LL = ESM2PseudoLikelihood(esm, stop_grad=False)
    return esm, esm_pseudo_LL


@app.cell
def _(TrigramLL):
    trigram_ll = TrigramLL.from_pkl()
    return (trigram_ll,)


@app.cell
def _(make_binder_features, target_sequence):
    scaffolded_binder_length = 55
    scaffolded_binder_features, scaffolded_binder_viewer = make_binder_features(
        scaffolded_binder_length, target_sequence
    )
    return (
        scaffolded_binder_features,
        scaffolded_binder_length,
        scaffolded_binder_viewer,
    )


@app.cell
def _(make_binder_monomer_features, scaffolded_binder_length):
    monomer_features, monomer_writer = make_binder_monomer_features(
        scaffolded_binder_length,
    )
    return monomer_features, monomer_writer


@app.cell
def _():
    # predict distogram for 7S5B
    return


@app.cell
def _(make_monomer_features, predict):
    scaffold_features, scaffold_writer = make_monomer_features(
        "SVIEKLRKLEKQARKQGDEVLVMLARMVLEYLEKGWVSEEDADESADRIEEVLKK"
    )

    o_scaffold, v_scaffold = predict(
        scaffold_features["res_type"][0][:, 2:22],
        scaffold_features,
        scaffold_writer,
    )
    v_scaffold
    return o_scaffold, scaffold_features, scaffold_writer, v_scaffold


@app.cell
def _(
    BinderTargetContact,
    DistogramCE,
    PLDDTLoss,
    StabilityModel,
    StructurePrediction,
    WithinBinderContact,
    complex_inverse_folding_LL,
    esm,
    esm_pseudo_LL,
    jax,
    model,
    monomer_features,
    o_scaffold,
    scaffold_inverse_folding_LL,
    scaffolded_binder_features,
    trigram_ll,
):
    # modify structural loss to add stability term, add ESM + trigram
    combined_loss = (
        StructurePrediction(
            model=model,
            name="ART2B",
            loss=8 * BinderTargetContact()
            + WithinBinderContact()
            + complex_inverse_folding_LL,
            features=scaffolded_binder_features,
            recycling_steps=0,
        )
        + 0.5 * esm_pseudo_LL
        + trigram_ll
        + 10 * scaffold_inverse_folding_LL
        + 0.5
        * StructurePrediction(
            model=model,
            name="mono",
            loss=0.2 * PLDDTLoss()
            + 0.1 * StabilityModel.from_pretrained(esm)
            + (-3)
            * DistogramCE(
                jax.nn.softmax(o_scaffold["pdistogram"])[0], name="scaffold"
            ),
            features=monomer_features,
            recycling_steps=0,
        )
    )
    return (combined_loss,)


@app.cell
def _(mo):
    mo.md("""Here we use a combination of ~5 models: Boltz, ESM, ProteinMPNN, a trigram model, and a small stability predictor. This specific example is contrived, but if we want to design a binder with a vaguely similar fold to 7S5B that is thermostable, has high PLL under ESM2, etc...""")
    return


@app.cell
def _(
    combined_loss,
    design_bregman_optax,
    np,
    optax,
    scaffolded_binder_length,
):
    logits_combined = design_bregman_optax(
        loss_function=combined_loss,
        n_steps=100,
        x=np.random.randn(scaffolded_binder_length, 20) * 0.1,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(np.sqrt(scaffolded_binder_length)),
        ),
    )
    return (logits_combined,)


@app.cell
def _(
    jax,
    logits_combined,
    predict,
    scaffolded_binder_features,
    scaffolded_binder_viewer,
):
    combined_outputs, _viewer = predict(
        jax.nn.softmax(logits_combined),
        scaffolded_binder_features,
        scaffolded_binder_viewer,
    )
    _viewer
    return (combined_outputs,)


@app.cell
def _(combined_outputs, logits_combined, visualize_output):
    visualize_output(combined_outputs, logits_combined)
    return


@app.cell
def _(
    binder_length,
    combined_loss,
    design_bregman_optax,
    logits_combined,
    np,
    optax,
):
    # we can sharpen these logits using weight decay (which is equivalent to adding entropic regularization)
    logits_combined_sharper = design_bregman_optax(
        loss_function=combined_loss,
        n_steps=50,
        x=logits_combined,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.01),
            optax.sgd(1.0 * np.sqrt(binder_length)),
        ),
    )
    logits_combined_sharper = design_bregman_optax(
        loss_function=combined_loss,
        n_steps=50,
        x=logits_combined_sharper,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.05),
            optax.sgd(0.25 * np.sqrt(binder_length)),
        ),
    )
    return (logits_combined_sharper,)


@app.cell
def _(
    jax,
    logits_combined_sharper,
    predict,
    scaffolded_binder_features,
    scaffolded_binder_viewer,
):
    output_combined_sharper, _viewer = predict(
        jax.nn.softmax(1000 * logits_combined_sharper),
        scaffolded_binder_features,
        scaffolded_binder_viewer,
    )

    _viewer
    return (output_combined_sharper,)


@app.cell
def _(logits_combined_sharper, output_combined_sharper, visualize_output):
    visualize_output(output_combined_sharper, logits_combined_sharper)
    return


@app.cell
def _(
    binder_length,
    j_model,
    jax,
    logits_combined_sharper,
    mo,
    model,
    pdb_viewer,
    repredict,
):
    _f_r, _w = repredict(logits_combined_sharper)

    o_combined = j_model(
        model,
        _f_r,
        key=jax.random.key(5),
        sample_structure=True,
        confidence_prediction=True,
    )

    _out_path = _w(o_combined["sample_atom_coords"])
    _repredicted_viewer = pdb_viewer(_out_path)

    print(o_combined["plddt"][:binder_length].mean())
    print(_w.out_dir)
    with open(next(_w.out_dir.glob("*/*.pdb")), "r") as _f:
        download_structure_stab = mo.download(_f.read(), filename="next.pdb")

    _repredicted_viewer
    return download_structure_stab, o_combined


@app.cell
def _(download_structure_stab):
    download_structure_stab
    return


if __name__ == "__main__":
    app.run()
