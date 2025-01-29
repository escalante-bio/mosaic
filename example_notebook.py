import marimo

__generated_with = "0.10.16"
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
    binder_length = 55
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
    _, logits = design_bregman_optax(
        loss_function=loss,
        n_steps=50,
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
        FixedChainInverseFoldingLL,
    )
    from boltz_binder_design.proteinmpnn.mpnn import ProteinMPNN
    import gemmi
    return (
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
    from boltz_binder_design.losses.boltz import DistogramCE, BoltzProteinMPNNLoss
    return BoltzProteinMPNNLoss, DistogramCE


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
def _():
    scaffold_sequence = "SVIEKLRKLEKQARKQGDEVLVMLARMVLEYLEKGWVSEEDADESADRIEEVLKK"
    return (scaffold_sequence,)


@app.cell
def _(make_monomer_features, predict, scaffold_sequence):
    scaffold_features, scaffold_writer = make_monomer_features(scaffold_sequence)

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
            + 3
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
def _():
    # logits_combined = design_bregman_optax(
    #     loss_function=combined_loss,
    #     n_steps=100,
    #     x=np.random.randn(scaffolded_binder_length, 20) * 0.1,
    #     optim=optax.chain(
    #         optax.clip_by_global_norm(1.0),
    #         optax.sgd(np.sqrt(scaffolded_binder_length)),
    #     ),
    # )
    return


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


@app.cell
def _(jax, logits_sharper, plt):
    _f = plt.imshow(jax.nn.softmax(logits_sharper * 10000))
    plt.colorbar()
    _f
    return


@app.cell
def _():
    from boltz_binder_design import _eval_loss_and_grad
    return


@app.cell
def _(jax, logits_sharper, loss):
    input = jax.nn.softmax(logits_sharper * 10000)
    og_output, g = _eval_loss_and_grad(loss, input, key=jax.random.key(0))
    return g, input, og_output


@app.cell
def _(g, plt):
    _f = plt.imshow(g)
    plt.colorbar()
    _f
    return


@app.cell
def _(g, input, plt):
    _f = plt.imshow(g * input)
    plt.colorbar()
    _f
    return


@app.cell
def _(np):
    def naive_proposal(input, g):
        r = np.zeros_like(g)
        inner_g_input = (input * g).sum()
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                x_prime = np.copy(input)
                x_prime[i] = 0.0
                x_prime[i, j] = 1.0
                r[i, j] = (x_prime * g).sum() - inner_g_input

        return r
    return (naive_proposal,)


@app.cell
def _():
    def proposal(input, g):
        g_i_x_i = (g * input).sum(-1, keepdims=True)
        return (input * g).sum() - g_i_x_i + g

        # return r
    return (proposal,)


@app.cell
def _(g, input, naive_proposal, np, plt):
    p = np.exp(-0.5 * naive_proposal(input, g))
    p = p / sum(p)
    _f = plt.imshow(p)
    plt.colorbar()
    _f
    return (p,)


@app.cell
def _(g, input, np, plt, proposal):
    _p = np.exp(proposal(input, g))
    _p = _p / sum(_p)
    _f = plt.imshow(_p)
    plt.colorbar()
    _f
    return


@app.cell
def _(g, input, proposal):
    proposal(input, g)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(mut_idx):
    mut_idx
    return


@app.cell
def _(mut_idx, np, p):
    I, J = np.unravel_index(mut_idx, p.shape)
    return I, J


@app.cell
def _(I, J):
    I, J
    return


@app.cell
def _():
    def apply_mut(one_hot, i, j):
        return one_hot.at[i].set(0.0).at[i, j].set(1.0)
    return (apply_mut,)


@app.cell
def _(I, J, apply_mut, input, jax):
    muts = jax.vmap(lambda i, j: apply_mut(input, i, j))(I, J)
    return (muts,)


@app.cell
def _(jax, np, p):
    mut_idx = jax.random.choice(
        key=jax.random.key(np.random.randint(10000)),
        a=len(np.ravel(p)),
        p=np.ravel(p),
        shape=(32,),
    )
    return (mut_idx,)


@app.cell
def _(eqx, jax):
    @eqx.filter_jit
    def test_muts(loss, muts, key):
        print("JIT")
        n_muts = muts.shape[0]
        return jax.vmap(lambda mut, key: loss(mut, key=key))(
            muts, jax.random.split(key, n_muts)
        )
    return (test_muts,)


@app.cell
def _(jax, loss, muts, test_muts):
    sorted(test_muts(loss, muts, jax.random.key(0))[0])
    return


@app.cell
def _(og_output):
    og_output
    return


@app.cell
def _(apply_mut, eqx, jax):
    @eqx.filter_jit
    def make_muts(seq, I, J):
        return jax.vmap(lambda i, j: apply_mut(seq, i, j))(I, J)
    return (make_muts,)


@app.cell
def _(apply_mut, jax, make_muts, np, proposal, test_muts):
    def greedy_optimize_path(loss, input, proposal_temp=0.1, pathlength=3):
        key = jax.random.key(0)
        for iter in range(50):
            og_output, g = _eval_loss_and_grad(loss, input, key=key)
            p = jax.nn.softmax(-proposal(input, g) / proposal_temp)
            first_mut_idx = jax.random.choice(
                key=jax.random.key(np.random.randint(10000)),
                a=len(np.ravel(p)),
                p=np.ravel(p),
                shape=(32,),
                replace=False,
            )
            I, J = np.unravel_index(first_mut_idx, p.shape)
            muts = make_muts(
                input, I, J
            )  # jax.vmap(lambda i, j: apply_mut(input, i, j))(I, J)
            for _ in range(pathlength - 1):
                next_mut_idx = jax.random.choice(
                    key=jax.random.key(np.random.randint(10000)),
                    a=len(np.ravel(p)),
                    p=np.ravel(p),
                    shape=(32,),
                    replace=False,
                )
                I, J = np.unravel_index(next_mut_idx, p.shape)
                muts = jax.vmap(apply_mut)(muts, I, J)

            v_muts, _ = test_muts(loss, muts, key)
            best_mut = np.argmin(v_muts)

            print(iter, og_output[0], v_muts[best_mut])
            if v_muts[best_mut] < og_output[0]:
                input = muts[best_mut]

        return input
    return (greedy_optimize_path,)


@app.cell
def _(jax, make_muts, np, proposal, test_muts):
    def greedy_optimize(loss, input, proposal_temp=0.1):
        key = jax.random.key(0)
        for iter in range(50):
            og_output, g = _eval_loss_and_grad(loss, input, key=key)
            p = jax.nn.softmax(-proposal(input, g) / proposal_temp)
            mut_idx = jax.random.choice(
                key=jax.random.key(np.random.randint(10000)),
                a=len(np.ravel(p)),
                p=np.ravel(p),
                shape=(32,),
            )
            I, J = np.unravel_index(mut_idx, p.shape)
            muts = make_muts(input, I, J)

            v_muts, _ = test_muts(loss, muts, key)
            best_mut = np.argmin(v_muts)

            print(iter, og_output[0], v_muts[best_mut])
            if v_muts[best_mut] < og_output[0]:
                input = muts[best_mut]

        return input
    return (greedy_optimize,)


@app.cell
def _(greedy_optimize_path, jax, logits_sharper, loss):
    discreteo = greedy_optimize_path(
        loss,
        jax.nn.softmax(10000 * logits_sharper),
        pathlength=1,  # jax.nn.one_hot(np.random.randn(100, 20).argmax(-1), 20)
    )  # input)
    return (discreteo,)


@app.cell
def _(discreteo, plt):
    plt.imshow(discreteo)
    return


@app.cell
def _(jax, logits_sharper, plt):
    plt.imshow(jax.nn.softmax(10000 * logits_sharper))
    return


@app.cell
def _(boltz_features, boltz_writer, discreteo, predict):
    predict(discreteo, boltz_features, boltz_writer)
    return


@app.cell
def _():
    from boltz_binder_design.alphafold.common import protein, residue_constants
    return protein, residue_constants


@app.cell
def _():
    from boltz_binder_design.alphafold.model import config, data, modules_multimer
    return config, data, modules_multimer


@app.cell
def _():
    from boltz_binder_design.af2.alphafold2 import AF2
    from boltz_binder_design.losses.af2 import AlphaFold
    import boltz_binder_design.losses.af2 as aflosses
    return AF2, AlphaFold, aflosses


@app.cell
def _(AF2):
    af2 = AF2()
    return (af2,)


@app.cell
def _(af2, binder_length, target_sequence, target_st):
    af_features, initial_guess = af2.build_features(
        # ["SVIEKLRKLEKQARKQGDEVLVMLARMVLEYLEKGWVSEEDADESADRIEEVLKK"],
        ["G" * binder_length, target_sequence],
        template_chains={1: target_st[0][0]},
    )
    return af_features, initial_guess


@app.cell
def _(
    j_model,
    jax,
    make_monomer_features,
    model,
    pdb_viewer,
    target_sequence,
):
    # predict target

    target_features, target_writer = make_monomer_features(target_sequence)


    o_target = j_model(
        model,
        target_features,
        key=jax.random.key(5),
        sample_structure=True,
        confidence_prediction=True,
    )

    out_path_target = target_writer(o_target["sample_atom_coords"])
    viewer_target = pdb_viewer(out_path_target)
    viewer_target
    return (
        o_target,
        out_path_target,
        target_features,
        target_writer,
        viewer_target,
    )


@app.cell
def _(gemmi, out_path_target):
    target_st = gemmi.read_pdb(str(out_path_target))
    return (target_st,)


@app.cell
def _(
    AlphaFold,
    ClippedLoss,
    af2,
    af_features,
    aflosses,
    jax,
    mpnn,
    o_af_scaffold,
    scaffold_inverse_folding_LL,
):
    af_loss = AlphaFold(
        name="af",
        forward=af2.alphafold_apply,
        stacked_params=jax.device_put(af2.stacked_model_params),
        features=af_features,
        losses=0.01 * aflosses.PLDDTLoss()
        + 0.0 * aflosses.BinderPAE()
        # + 0.1 * aflosses.RadiusOfGyration(12.0)
        + 0.0 * aflosses.RadiusOfGyration(12.0)
        + 0.0 * aflosses.WithinBinderContact(num_contacts_per_residue=5)
        + 1 * aflosses.BinderTargetContact()
        + 0.1 * aflosses.TargetBinderPAE()
        + 0.1 * aflosses.BinderTargetPAE()
        + 1
        * ClippedLoss(
            aflosses.DistogramCE(
                jax.nn.softmax(o_af_scaffold.distogram.logits), name="scaffoldCE"
            ),
            2,
            100,
        )
        + 0.01 * aflosses.AFProteinMPNNLoss(mpnn, num_samples=1),
    ) + 1 * ClippedLoss(scaffold_inverse_folding_LL, 2, 100)
    # # + StructurePrediction(
    # #     model=model,
    # #     name="ART2B",
    # #     loss=8 * BinderTargetContact()
    # #     + WithinBinderContact()
    # #     + complex_inverse_folding_LL,
    # #     features=scaffolded_binder_features,
    # #     recycling_steps=0,
    # # )
    return (af_loss,)


@app.cell
def _(af_loss):
    lossy = af_loss
    return (lossy,)


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


@app.cell
def _(af_loss, binder_length, design_bregman_optax, np, optax):
    _, logits_af = design_bregman_optax(
        loss_function=af_loss,
        n_steps=100,
        x=np.random.randn(binder_length, 20) * 0.1,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(1.0 * np.sqrt(binder_length), momentum=0.0),
        ),
    )
    return (logits_af,)


@app.cell
def _(binder_length, design_bregman_optax, logits_af, lossy, np, optax):
    logits_af_sharper, _ = design_bregman_optax(
        loss_function=lossy,
        n_steps=100,
        x=logits_af,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.01),
            optax.sgd(0.1 * np.sqrt(binder_length), momentum=0.0),
        ),
    )

    logits_af_sharper, _ = design_bregman_optax(
        loss_function=lossy,
        n_steps=100,
        x=logits_af_sharper,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.15),
            optax.sgd(0.1 * np.sqrt(binder_length), momentum=0.0),
        ),
    )
    return (logits_af_sharper,)


@app.cell
def _(
    binder_length,
    design_bregman_optax,
    logits_af_sharper,
    lossy,
    np,
    optax,
):
    logits_af_sharpest, _ = design_bregman_optax(
        loss_function=lossy,
        n_steps=50,
        x=logits_af_sharper,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.5),
            optax.sgd(0.1 * np.sqrt(binder_length), momentum=0.0),
        ),
    )
    return (logits_af_sharpest,)


@app.cell
def _():
    from boltz_binder_design.common import LossTerm
    import jax.numpy as jnp
    return LossTerm, jnp


@app.cell
def _():
    from boltz_binder_design import _eval_loss_and_grad, _print_iter
    return


@app.cell
def _(jax, optax):
    def colab_grad(optim, opt_state, params, t, loss_function, key):
        def colab_x(param):
            return t * jax.nn.softmax(param) + (1 - t) * param

        x, vjp = jax.vjp(colab_x, params)
        (v, aux), g = _eval_loss_and_grad(
            loss_function=loss_function, x=x, key=key
        )

        (g,) = vjp(g)
        updates, opt_state = optim.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, v, aux, opt_state
    return (colab_grad,)


@app.cell
def _():
    # seq_colab = design_colab(
    #     loss_function=lossy,
    #     n_steps=75,
    #     x=np.random.randn(binder_length, 20) * 0.01,
    #     optim=optax.chain(
    #         optax.clip_by_global_norm(1.0),
    #         optax.sgd(0.25 * np.sqrt(binder_length), momentum=0.0),
    #     ),
    # )
    return


@app.cell
def _(colab_grad, jax, np, optax):
    def design_colab(
        *,
        loss_function,
        x,
        n_steps: int,
        optim=optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(1e-1)),
    ):
        opt_state = optim.init(x)

        for _iter in range(n_steps):
            t = _iter / n_steps
            x, v, aux, opt_state = colab_grad(
                params=x,
                t=t,
                loss_function=loss_function,
                key=jax.random.key(np.random.randint(0, 10000)),
                optim=optim,
                opt_state=opt_state,
            )

            entropy = -(jax.nn.log_softmax(x) * jax.nn.softmax(x)).sum(-1).mean()
            _print_iter(_iter, aux, entropy, v)

        return x
    return (design_colab,)


@app.cell
def _(LossTerm, jax, jnp):
    class MinLoss(LossTerm):
        loss: LossTerm
        n_samples: int

        def __call__(self, *args, key, **kwargs):
            keys = jax.random.split(key, self.n_samples)

            vs, auxs = jax.vmap(lambda k: self.loss(*args, key=k, **kwargs))(keys)
            i_min = jnp.argmin(vs)

            return jax.tree.map(lambda a: a[i_min], (vs, auxs))
    return (MinLoss,)


@app.cell
def _(jax, logits_af_sharpest, plt):
    plt.imshow(jax.nn.softmax(logits_af_sharpest))
    return


@app.cell
def _(jax, plt, seq_colab):
    plt.imshow(jax.nn.softmax(seq_colab))
    return


@app.cell
def _(boltz_features, boltz_writer, jax, logits_af_sharpest, predict):
    af_output, _viewer = predict(
        jax.nn.softmax(10000 * logits_af_sharpest), boltz_features, boltz_writer
    )
    _viewer
    return (af_output,)


@app.cell
def _(boltz_writer, mo):
    with open(next(boltz_writer.out_dir.glob("*/*.pdb")), "r") as _f:
        _download_structure = mo.download(_f.read(), filename="next.pdb")

    _download_structure
    return


@app.cell
def _(TOKENS, logits_af_sharper):
    "".join([TOKENS[i] for i in logits_af_sharper.argmax(-1)])
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
        model_idx=1,
    )
    return o_pred, st_pred


@app.cell
def _():
    from boltz_binder_design.common import TOKENS
    return (TOKENS,)


@app.cell
def _(Path, pdb_viewer, st_pred):
    st_pred.write_minimal_pdb("test.pdb")
    pdb_viewer(Path("test.pdb"))
    return


@app.cell
def _(mo, st_pred):
    st_pred.write_minimal_pdb("test.pdb")
    with open("test.pdb", "r") as _f:
        _download_structure = mo.download(_f.read(), filename="next.pdb")

    _download_structure
    return


@app.cell
def _(o_pred, plt):
    _f = plt.imshow(o_pred.predicted_aligned_error)
    plt.colorbar()
    _f
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


if __name__ == "__main__":
    app.run()
