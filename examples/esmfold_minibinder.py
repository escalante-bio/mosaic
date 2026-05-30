import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Designing a de novo minibinder against PD-L1

    A recreation of the binder-design algorithm from ESMFold2. Here we design a
    small *de novo* **minibinder** to bind PD-L1: there is no fixed framework —
    the entire binder is `X`, so the optimizer chooses every residue. (Contrast
    the VHH notebook, which fixes an Ig framework and designs only the CDR loops.)

    We use two models:
    - **ESMFold2-Experimental-Fast-2025** — folds the binder + target complex
    - **ESMC-6B** — a protein language model used in a *pseudo-perplexity loss term*

    1. Load both models.
    2. Pick a binder length; every position is designable (`X`).
    3. Featurize.
    4. Build a differentiable loss over all binder positions.
    5. Optimize in two stages (soft → sharpen) by gradient descent on the simplex.
    6. Re-predict each design with full sidechains and ESMC embeddings as
       two-chain complexes and read off iPTM.
    """)
    return


@app.cell
def _():
    import time

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np

    import mosaic.losses.structure_prediction as sp
    from mosaic.common import TOKENS
    from mosaic.losses.esmc import ESMCPseudoPerplexity, load_esmc
    from mosaic.losses.transformations import NormedGradient, SetPositions
    from mosaic.models.esmfold2 import (
        ESMFold2ExperimentalFast,
        ESMFold2ExperimentalFast2025,
        ESMFold2Fast,
    )
    from mosaic.optimizers import batched_simplex_APGM
    from mosaic.structure_prediction import TargetChain

    return (
        ESMCPseudoPerplexity,
        ESMFold2ExperimentalFast2025,
        NormedGradient,
        TOKENS,
        TargetChain,
        batched_simplex_APGM,
        jax,
        jnp,
        load_esmc,
        mo,
        np,
        sp,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Load the models
    """)
    return


@app.cell
def _(ESMFold2ExperimentalFast2025, load_esmc):
    model = ESMFold2ExperimentalFast2025()
    ppl_esmc = load_esmc("biohub/ESMC-6B")
    return model, ppl_esmc


@app.cell
def _():
    PDL1_SEQUENCE = (
        "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQ"
        "HSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA"
    )

    BINDER_LENGTH = 80
    return BINDER_LENGTH, PDL1_SEQUENCE


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Featurize
    """)
    return


@app.cell
def _(BINDER_LENGTH, PDL1_SEQUENCE, TargetChain, model, np):
    target_chains = [TargetChain(PDL1_SEQUENCE, use_msa=False)]

    pack, _ = model.binder_features(
        BINDER_LENGTH, target_chains, placeholder_char="G"
    )
    sqrtM = float(np.sqrt(BINDER_LENGTH))
    return pack, sqrtM, target_chains


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Construct a structure loss

    A small linear combination of structure-prediction terms, all read off the
    folded complex:

    - `WithinBinderContact` — the binder should be a compact, self-contacting
      domain (≥2 contacts/residue).
    - `BinderTargetContact` — reward proximity between the binder and PD-L1. With
      no epitope specified the binder is free to dock anywhere on the target.
    - `DistogramRadiusOfGyration` — discourage the binder from unfolding/spreading.
    """)
    return


@app.cell
def _(sp):
    structure_loss = (
        0.5 * sp.WithinBinderContact(num_contacts_per_residue=2)
        + 0.5
        * sp.BinderTargetContact(
            contact_distance=22.0,
        )
        + 0.2 * sp.DistogramRadiusOfGyration()
    )
    return (structure_loss,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Assemble the full design loss


    - **`NormedGradient(inner, scale)`** centers the gradient across the vocab,
      normalizes it to unit norm, then multiplies by `scale`. Doing this *per
      term* makes the structure : prior ratio exactly `1.0 : ppl_weight` on the
      positions we actually move — independent of each term's raw magnitude.

    `ESMCPseudoPerplexity(..., design_idx=variable_positions)` scores the binder
    tokens, so the prior pulls the designed residues toward natural-looking
    sequence.
    """)
    return


@app.cell
def _(
    ESMCPseudoPerplexity,
    NormedGradient,
    model,
    pack,
    ppl_esmc,
    structure_loss,
):
    ppl_weight = 0.15

    structure_term = NormedGradient(
        model.build_loss(
            loss=structure_loss,
            features=pack,
            recycling_steps=2,
            msa_max_depth=1024,
        ),
        1.0,
    )
    ppl_term = NormedGradient(
        ESMCPseudoPerplexity(esm=ppl_esmc),
        ppl_weight,
    )
    loss = structure_term + ppl_term
    return (loss,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Optimization config

    `B` designs are optimized in parallel from independent random inits. `B=2` works on an H100.
    """)
    return


@app.cell
def _():
    B = 2  # batch: parallel designs from different gumbel inits
    SEED = 1
    N_SOFT = 64  # stage-1 soft steps
    N_SHARP = 30  # stage-2 sharpening steps
    GUMBEL = 0.75  # init concentration
    S1_MULT = 0.2  # stage-1 stepsize multiplier (× sqrt(M))
    S2_MULT = 0.10  # stage-2 stepsize multiplier
    S2_SCALE = 1.3  # stage-2 anneal rate
    S1_MOM = 0.1  # stage-1 momentum
    return B, GUMBEL, N_SHARP, N_SOFT, S1_MOM, S1_MULT, S2_MULT, S2_SCALE, SEED


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Stage 1 — soft optimization
    """)
    return


@app.cell
def _(
    B,
    BINDER_LENGTH,
    GUMBEL,
    N_SOFT,
    S1_MOM,
    S1_MULT,
    SEED,
    batched_simplex_APGM,
    jax,
    loss,
    sqrtM,
    time,
):
    x0 = jax.nn.softmax(
        GUMBEL
        * jax.random.gumbel(jax.random.key(SEED), shape=(B, BINDER_LENGTH, 20))
    )
    t0_soft = time.time()
    _, pssm = batched_simplex_APGM(
        loss_function=loss,
        x=x0,
        n_steps=N_SOFT,
        stepsize=S1_MULT * sqrtM,
        momentum=S1_MOM,
        scale=1.0,
        max_gradient_norm=1.0,
        key=jax.random.key(SEED + 1),
    )
    print(f"stage 1 done in {time.time() - t0_soft:.1f}s  ")
    return (pssm,)


@app.cell
def _(plt, pssm):
    plt.imshow(pssm[1])
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Stage 2 — sharpen
    """)
    return


@app.cell
def _(
    BINDER_LENGTH,
    N_SHARP,
    S2_MULT,
    S2_SCALE,
    SEED,
    batched_simplex_APGM,
    jax,
    jnp,
    loss,
    np,
    pssm,
    time,
):
    t0_sharp = time.time()
    pssm_sharp, _ = batched_simplex_APGM(
        loss_function=loss,
        x=jnp.log(pssm + 1e-5),
        n_steps=N_SHARP,
        stepsize=S2_MULT * float(np.sqrt(BINDER_LENGTH)),
        momentum=0.0,
        scale=S2_SCALE,
        logspace=True,
        max_gradient_norm=1.0,
        key=jax.random.key(SEED + 2),
    )
    print(f"stage 2 done in {time.time() - t0_sharp:.1f}s  ")
    return (pssm_sharp,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · Repredict with full sidechains
    """)
    return


@app.cell
def _(
    B,
    BINDER_LENGTH,
    SEED,
    TOKENS,
    TargetChain,
    jax,
    model,
    np,
    pssm_sharp,
    target_chains,
):
    predictions = []
    for i in range(B):
        tokens = np.argmax(np.asarray(pssm_sharp[i]), axis=-1)
        seq = "".join(TOKENS[i] for i in tokens)

        full_features, writer = model.target_only_features(
            chains=[TargetChain(seq, use_msa=False), *target_chains],
        )

        pred = model.predict(
            PSSM=jax.nn.one_hot(tokens, 20),
            features=full_features,
            writer=writer,
            recycling_steps=20,
            sampling_steps=100,
            key=jax.random.key(SEED + 100 + i),
        )
        predictions.append(pred)
        print(
            f"design {i}: iPTM={float(pred.iptm):.3f}  pLDDT(b)={float(pred.plddt[:BINDER_LENGTH].mean()):.3f}  seq={seq}"
        )
    return (predictions,)


@app.cell
def _():
    from mosaic.notebook_utils import pdb_viewer


    return (pdb_viewer,)


@app.cell
def _(pdb_viewer, predictions):
    pdb_viewer(predictions[0].st)
    return


@app.cell
def _(pdb_viewer, predictions):
    pdb_viewer(predictions[1].st)
    return


@app.cell
def _(mo, predictions):
    mo.download(data = predictions[0].st.make_pdb_string(), filename = "a.pdb")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
