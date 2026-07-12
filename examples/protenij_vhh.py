import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from mosaic.optimizers import simplex_APGM, batched_simplex_APGM
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
    from mosaic.models.protenix import ProtenixV2
    from mosaic.proteinmpnn.mpnn import load_abmpnn
    from mosaic.losses.ablang import AbLangPseudoLikelihood, load_ablang
    from mosaic.losses.esmc import ESMCPseudoLikelihood, load_esmc
    from mosaic.losses.transformations import SetPositions


@app.cell
def _():
    from mosaic.common import TOKENS

    return (TOKENS,)


@app.cell
def _():
    mo.callout("Demo VHH CDR design using Protenix v1.0", kind="success")
    return


@app.cell
def _():
    # we'll use a sequence + MSA for the VHH. We could also add a template
    masked_framework_sequence = "QVQLVESGGGLVQPGGSLRLSCAASXXXXXXXXXXXLGWFRQAPGQGLEAVAAXXXXXXXXYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCXXXXXXXXXXXXXXXXXXWGQGTLVTVS"
    return (masked_framework_sequence,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    We'll use PDL1 as a target; the world will never have enough de novo binders to PDL1. We'll use both an MSA and a template for the target structure.
    """)
    return


@app.cell
def _():
    target_structure = gemmi.read_structure("PDL1.pdb")
    target_sequence = gemmi.one_letter_code(
        [r.name for r in target_structure[0][0]]
    )
    return target_sequence, target_structure


@app.cell
def _(target_structure):
    pdb_viewer(target_structure)
    return


@app.cell
def _():
    # for fun we'll incorporate ESMC and AbLang PLLs into our loss function. In theory this should make our sequences more "natural."
    ablang, ablang_tokenizer = load_ablang("heavy")
    ablang_pll = AbLangPseudoLikelihood(
        model=ablang,
        tokenizer=ablang_tokenizer,
        stop_grad=True,
    )
    # and ESMC PLL
    ESMCPLL = ESMCPseudoLikelihood(load_esmc("esmc_300m"), stop_grad=True)
    return ESMCPLL, ablang_pll


@app.cell
def _():
    protenix = ProtenixV2()
    return (protenix,)


@app.cell
def _(masked_framework_sequence, protenix, target_sequence, target_structure):
    design_features, design_structure = protenix.target_only_features(
        # instead of binder_features, we'll use target_only_features so we properly handle sidechains on the framework. this shouldn't make a huge difference but feels right.
        chains=[
            TargetChain(
                masked_framework_sequence,
                use_msa=True,
            ),
            TargetChain(
                target_sequence,
                use_msa=True,
                template_chain=target_structure[0][0],
            ),
        ],
    )
    return design_features, design_structure


@app.cell
def _(
    ESMCPLL,
    ablang_pll,
    design_features,
    masked_framework_sequence,
    protenix,
):
    structure_loss = (
        sp.BinderTargetContact(
            paratope_idx=np.array(
                [
                    i
                    for (i, c) in enumerate(masked_framework_sequence)
                    if c == "X"
                ]  # encourage binding with the CDRs rather than the framework.
            )
            # if you have a particular hotspot you're going for you could use `epitope_idx` here.
        )
        - 0.0
        * sp.BinderTargetContact(
            paratope_idx=np.array(
                [
                    i
                    for (i, c) in enumerate(masked_framework_sequence)
                    if c != "X"
                ]  # discourage binding with the framewor
            )
        )
        # if we really don't like "sidebinders" and want contact ONLY with the CDRs we could add a *negative* BinderTargetContact term here that only applies to framework residues
        # I set this to zero because it seems silly to me: plenty of natural VHHs have framework-target contacts
        # to really discourage these kinds of poses you'd also need to downweight some of the terms below that prefer secondary structure within the binder (WithinBinderPAE, pLDDT, etc)
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.025 * sp.IPTMLoss()
        + 0.4 * sp.WithinBinderPAE()
        + 0.025 * sp.pTMEnergy()
        + 0.1 * sp.PLDDTLoss()
    )

    model_loss = protenix.build_multisample_loss(
        loss=structure_loss,
        features=design_features,
        recycling_steps=2,
        sampling_steps=20,
        # single diffusion sample per gradient eval; we get robustness/diversity from
        # running a batch of independent designs in parallel (see the batched optimizer below)
        num_samples=1,
    )

    # we use SetPositions to fix the framework AAs
    loss = SetPositions.from_sequence(
        wildtype=masked_framework_sequence,
        loss=0.1 * ESMCPLL + 2 * model_loss + 0.1 * ablang_pll,
    )
    return (loss,)


@app.cell
def _():
    mo.callout(
        "JIT will take a very long time the first time we run the following cell. Rerun for more samples!"
    )
    return


@app.cell
def _(TOKENS, loss, masked_framework_sequence):
    batch_size = 2
    num_designed_residues = len(
        [c for c in masked_framework_sequence if c == "X"]
    )

    # initial soft sequences: [B, N, 20]
    _pssm = 0.5 * jax.random.gumbel(
        key=jax.random.key(np.random.randint(1000000)),
        shape=(batch_size, num_designed_residues, 20),
    )

    # NOTE: stepsize scales with sqrt(N) where N = num_designed_residues (axis 1),
    # NOT the batch dimension.
    _, partial_pssm = batched_simplex_APGM(
        loss_function=loss,
        x=_pssm,
        n_steps=50,
        stepsize=1.5 * np.sqrt(num_designed_residues),
        momentum=0.2,
        scale=1.00,
        logspace=True,
        max_gradient_norm=1.0,
    )
    _, partial_pssm = batched_simplex_APGM(
        loss_function=loss,
        x=partial_pssm,
        n_steps=30,
        stepsize=0.5 * np.sqrt(num_designed_residues),
        momentum=0.0,
        scale=1.1,
        logspace=False,
        max_gradient_norm=1.0,
    )
    for _b in range(batch_size):
        print(
            f"[{_b}] " + "".join(TOKENS[i] for i in partial_pssm[_b].argmax(-1))
        )

    _, partial_pssm = batched_simplex_APGM(
        loss_function=loss,
        x=jnp.log(partial_pssm + 1e-5),
        n_steps=30,
        stepsize=0.25 * np.sqrt(num_designed_residues),
        momentum=0.0,
        scale=1.1,
        logspace=True,
        max_gradient_norm=1.0,
    )
    return batch_size, partial_pssm


@app.cell
def _():
    mo.callout(
        "Let's try to improve our best design using batched greedy descent"
    )
    return


@app.cell
def _():
    from mosaic.optimizers import batch_greedy_descent

    return (batch_greedy_descent,)


@app.cell
def _(loss):
    j_loss = eqx.filter_jit(loss)
    return (j_loss,)


@app.cell
def _(batch_greedy_descent, batch_size, j_loss, loss, partial_pssm):
    _score_key = jax.random.key(0)
    _seqs = partial_pssm.argmax(-1)  # [B, N] discrete
    _design_vals = []
    for _b in range(batch_size):
        _v, _ = j_loss(jax.nn.one_hot(_seqs[_b], 20), key=_score_key)
        _design_vals.append(float(jnp.nan_to_num(_v, nan=1e6)))
    _best = int(np.argmin(_design_vals))
    print(
        "batched design losses:",
        [round(v, 3) for v in _design_vals],
        "-> seeding greedy from",
        _best,
    )

    # Batched greedy descent: each step ranks all single-point mutations by first-order
    # predicted delta and evaluates the top `batch_size` unseen candidates in parallel,
    # greedily accepting the best improvement.
    s_greedy, s_greedy_val = batch_greedy_descent(
        loss,
        sequence=jax.device_put(_seqs[_best]),
        batch_size=2,
        steps=30,
    )
    return (s_greedy,)


@app.cell
def _():
    mo.callout(
        "It's very important we add the framework residues back into our sequence before e.g. repredicting!",
        kind="danger",
    )
    return


@app.cell
def _(loss, s_greedy):
    final_pssm = loss.sequence(jax.nn.one_hot(s_greedy, 20))
    return (final_pssm,)


@app.cell
def _(TOKENS, design_features, design_structure, final_pssm, protenix):
    # repredict with more recycling steps
    prediction_inpaint = protenix.predict(
        PSSM=final_pssm,
        writer=design_structure,
        features=design_features,
        recycling_steps=10,
        key=jax.random.key(np.random.randint(10000)),
    )

    design_str = "".join(TOKENS[i] for i in final_pssm.argmax(-1))
    return design_str, prediction_inpaint


@app.cell
def _(prediction_inpaint):
    plt.imshow(prediction_inpaint.pae)
    plt.title(f"Predicted Aligned Error, IPTM {prediction_inpaint.iptm: 0.3f}")
    return


@app.cell
def _(masked_framework_sequence, prediction_inpaint):
    _f = plt.figure()
    plt.plot(prediction_inpaint.plddt)
    plt.title("pLDDT")
    plt.vlines(
        [len(masked_framework_sequence)],
        ymin=prediction_inpaint.plddt.min(),
        ymax=prediction_inpaint.plddt.max(),
        linestyles="dashed",
        color="red",
    )
    _f
    return


@app.cell
def _(prediction_inpaint):
    pdb_viewer(prediction_inpaint.st)
    return


@app.cell
def _(design_str):
    design_str
    return


@app.cell
def _(prediction_inpaint):
    mo.download(data=prediction_inpaint.st.make_pdb_string(), filename="vhh.pdb")
    return


if __name__ == "__main__":
    app.run()
