import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.callout(
        """Demo de novo minibinder design against PD-L1, using a as-faithful-as-possible recreation of the (rather good) ESMFold2 
        binder design algorithm""",
        kind="success",
    )
    return


@app.cell
def _():
    import os

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

    import time

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np

    import mosaic.losses.structure_prediction as sp
    from mosaic.common import TOKENS
    from mosaic.losses.esmfold2 import StackedEsmFold2PLL
    from mosaic.losses.esmc import ESMCPseudoPerplexity
    from mosaic.losses.transformations import NormedGradient, SetPositions
    from mosaic.models.esmfold2 import (
        ESMFold2ExperimentalFast,
        ESMFold2ExperimentalFast2025,
        ESMFold2Fast,
    )
    from mosaic.common import LossTerm
    from mosaic.losses.esmc import load_esmc
    from mosaic.structure_prediction import TargetChain
    import equinox as eqx

    return (
        ESMCPseudoPerplexity,
        ESMFold2ExperimentalFast,
        ESMFold2ExperimentalFast2025,
        LossTerm,
        StackedEsmFold2PLL,
        TOKENS,
        TargetChain,
        eqx,
        jax,
        load_esmc,
        mo,
        np,
        sp,
    )


@app.cell
def _():
    from mosaic.optimizers import biohub_optimizer

    return (biohub_optimizer,)


@app.cell
def _(ESMFold2ExperimentalFast, ESMFold2ExperimentalFast2025, eqx, load_esmc):
    model_0 = ESMFold2ExperimentalFast()
    model_1 = ESMFold2ExperimentalFast2025()
    # remove esmc models
    model_0 = eqx.tree_at(lambda m: m.esmc, model_0, None)
    model_1 = eqx.tree_at(lambda m: m.esmc, model_1, None)

    esmc = load_esmc(model_name="esmc_6b")
    return esmc, model_0, model_1


@app.cell
def _(ESMFold2ExperimentalFast2025):
    validation_model = ESMFold2ExperimentalFast2025()
    return (validation_model,)


@app.cell
def _():
    TARGET_SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

    BINDER_LENGTH = 120
    return BINDER_LENGTH, TARGET_SEQUENCE


@app.cell
def _(BINDER_LENGTH, TARGET_SEQUENCE, TargetChain, eqx, esmc, model_0, np):
    target_chains = [TargetChain(TARGET_SEQUENCE, use_msa=False)]

    features, _ = eqx.tree_at(
        lambda m: m.esmc, model_0, esmc, is_leaf=lambda l: l is None
    ).binder_features(BINDER_LENGTH, target_chains)
    sqrtM = float(np.sqrt(BINDER_LENGTH))
    return features, target_chains


@app.cell
def _(sp):
    structure_loss = (
        0.5 * sp.WithinBinderContact(num_contacts_per_residue=2)
        + 0.5
        * sp.ESMFoldInterContact(
            contact_distance=22.0,
        )
        + 0.2 * sp.ESMFoldGlobularity()
    )
    return (structure_loss,)


@app.cell
def _(
    ESMCPseudoPerplexity,
    StackedEsmFold2PLL,
    esmc,
    features,
    model_0,
    model_1,
    structure_loss,
):
    ppl_weight = 0.15

    loss = StackedEsmFold2PLL(
        [
            model.build_loss(
                loss=structure_loss,
                features=features,
                recycling_steps=1,
                msa_max_depth=1024,
                # geom + LM refresh are tied to the (design) features from binder_features.
                lm_dropout=0.5,
            )
            for model in (model_0, model_1)
        ],
        esmc,
        ESMCPseudoPerplexity(None),
    )
    return (loss,)


@app.cell
def _():
    B = 4  # batch: parallel designs from different inits
    return (B,)


@app.cell
def _(mo):
    mo.md(r"""
    We can also give the biohub optimizer a "tail loss" to select the final iterate. This gets a bit complicated: we want to avoid computing this loss on every iteration, and we don't want to differentiate through the structure and confidence modules.
    """)
    return


@app.cell
def _(LossTerm, sp):
    class IPTMMonitor(LossTerm):
        """Adapter to report iptm as a ranking_loss for biohub_optimizer -- returns 0.0 to avoid backprop through structure + confidence modules."""

        def __call__(self, seq, output, key):
            neg_iptm, _ = sp.IPTMLoss()(seq, output, key)
            return 0.0, {"ranking_loss": neg_iptm, "iptm": -neg_iptm}

    return (IPTMMonitor,)


@app.cell
def _(
    ESMCPseudoPerplexity,
    IPTMMonitor,
    StackedEsmFold2PLL,
    esmc,
    features,
    model_0,
    model_1,
    structure_loss,
):
    structure_term_tail = StackedEsmFold2PLL(
        [
            model.build_loss(
                loss=structure_loss + IPTMMonitor(),
                features=features,
                recycling_steps=1,
                msa_max_depth=1024,
                # geom + LM refresh are tied to the (design) features from binder_features.
                lm_dropout=0.5,
            )
            for model in (model_0, model_1)
        ],
        esmc,
        ESMCPseudoPerplexity(None),
    )
    return (structure_term_tail,)


@app.cell
def _(TOKENS, np):
    def no_cysteine_mask(length: int):
        """[N, 20] mask that is 0 at cysteine (biohub masks Cys out), else 1."""
        mask = np.ones((length, len(TOKENS)), dtype=np.float32)
        mask[:, TOKENS.index("C")] = 0.0
        return mask

    return (no_cysteine_mask,)


@app.cell
def _(
    B,
    BINDER_LENGTH,
    biohub_optimizer,
    jax,
    loss,
    no_cysteine_mask,
    structure_term_tail,
):
    SEED = 1
    x0 = 1e-4 * jax.random.normal(
        jax.random.key(SEED), shape=(B, BINDER_LENGTH, 20)
    )
    pssm, neg_iptm = biohub_optimizer(
        loss_function=loss,
        logits=x0,
        verbose=True,
        mask=no_cysteine_mask(BINDER_LENGTH),
        tail_loss_function=structure_term_tail,
        key=jax.random.key(SEED + 1),
        beta = 0.5
    )
    return neg_iptm, pssm


@app.cell
def _(neg_iptm):
    neg_iptm
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(mo):
    mo.md("""
    Let's repredict the best design
    """)
    return


@app.cell
def _(BINDER_LENGTH, target_chains, validation_model):
    validation_features, val_writer = validation_model.binder_features(
        binder_length=BINDER_LENGTH, chains=target_chains
    )
    return val_writer, validation_features


@app.cell
def _(plt, pssm):
    _f = plt.figure()
    plt.imshow(pssm[0])
    plt.colorbar()
    _f
    return


@app.cell
def _(jax, pssm, val_writer, validation_features, validation_model):
    prediction = validation_model.predict(
        PSSM=jax.nn.one_hot(pssm[0].argmax(-1), 20),
        features=validation_features,
        writer=val_writer,
        key=jax.random.key(0),
        recycling_steps=20,
        sampling_steps=100,
    )
    return (prediction,)


@app.cell
def _(prediction):
    print(prediction.iptm)
    return


@app.cell
def _(plt, prediction):
    plt.imshow(prediction.pae)
    return


@app.cell
def _(pdb_viewer, prediction):
    pdb_viewer(prediction.st)
    return


@app.cell
def _():
    from mosaic.notebook_utils import pdb_viewer

    return (pdb_viewer,)


if __name__ == "__main__":
    app.run()
