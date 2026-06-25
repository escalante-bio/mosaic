import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Minibinder design (mosaic × jpromera)

        """
    )
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import jax

    jax.config.update("jax_compilation_cache_dir", "/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    import jax.numpy as jnp
    import numpy as np

    import mosaic.losses.structure_prediction as sp
    from mosaic.common import TOKENS, LossTerm
    from mosaic.losses.protein_mpnn import (
        InverseFoldingSequenceRecovery,
        inverse_fold,
    )
    from mosaic.optimizers import batched_simplex_APGM
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.structure_prediction import TargetChain

    from mosaic.models.promera import JPromeraModel

    return (
        InverseFoldingSequenceRecovery,
        JPromeraModel,
        Path,
        TOKENS,
        TargetChain,
        batched_simplex_APGM,
        inverse_fold,
        jax,
        jnp,
        load_mpnn_sol,
        np,
        sp,
    )


@app.cell
def _():
    import equinox as eqx

    return (eqx,)


@app.cell
def _():
    from jaxtyping import Array, Int
    from mosaic.structure_prediction import StructureModelOutput

    return Array, Int, StructureModelOutput


@app.cell
def _():
    from mosaic.losses.trigram import UnigramExcess, BigramExcess

    return BigramExcess, UnigramExcess


@app.cell
def _():
    from mosaic.losses.transformations import NormedGradient

    return (NormedGradient,)


@app.cell
def _():
    from mosaic.losses.esmc import ESMCPseudoPerplexity, load_esmc

    return ESMCPseudoPerplexity, load_esmc


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _():
    import mosaic.notebook_utils as nb

    return (nb,)


@app.cell
def _(Path):
    # PD-L1 IgV domain (target) + design knobs.
    TARGET = (
        "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRAR"
        "LLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA"
    )
    BINDER_LEN = 80
    MASK_FRACTION = 0.1  # fraction of binder residues UNK-masked each loss call
    OUT = Path("design_out")
    OUT.mkdir(exist_ok=True)
    return BINDER_LEN, OUT, TARGET


@app.cell
def _(JPromeraModel, load_mpnn_sol):
    # Load Promera and soluble ProteinMPNN (torch loads the MPNN weights only).
    model = JPromeraModel(subsample=1024)
    mpnn = load_mpnn_sol()
    return model, mpnn


@app.cell
def _(load_esmc):
    esmc = load_esmc("biohub/ESMC-6B")
    return (esmc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · Design a backbone with Promera

    To use Promera to design a binder backbone we simply use an all "X" TargetChain and call predict or model_output
    """)
    return


@app.cell
def _(BINDER_LEN, TARGET, TargetChain, model):
    des_features, des_writer = model.target_only_features(
        [
            TargetChain("X" * BINDER_LEN, use_msa=False),
            TargetChain(TARGET, use_msa=True),
        ]
    )
    return des_features, des_writer


@app.cell
def _(des_features, jax, model):
    des = model.model_output(
        features=des_features,
        recycling_steps=4,
        sampling_steps=200,
        key=jax.random.key(1),
        step_scale=1.0,
    )
    return (des,)


@app.cell
def _(des, nb):
    nb.pdb_viewer(des.to_structure())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · Inverse-fold the backbone

    Run soluble ProteinMPNN on the `des` backbone to get a real binder sequence for that fold.
    """)
    return


@app.cell
def _(BINDER_LEN, TOKENS, des, inverse_fold, jax, mpnn):
    # Inverse-fold the `des` backbone (all-X binder docked to target) with soluble
    # ProteinMPNN -> a real designed sequence for that predicted fold.
    des_seq_idx = inverse_fold(
        mpnn, BINDER_LEN, des, temp=0.1, key=jax.random.key(0)
    )
    des_seq = "".join(TOKENS[int(i)] for i in des_seq_idx)
    print(f"inverse-folded binder ({len(des_seq)} aa, Cys {des_seq.count('C')}):")
    print(des_seq)
    des_seq
    return (des_seq_idx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Re-predict & score the designed sequence

    Fold the inverse-folded sequence as a real binder chain and read off ipTM.
    """)
    return


@app.cell
def _(BINDER_LEN, TARGET, TargetChain, model):
    # to make a full structure prediction we use binder_features (fairly technical detail but des_features will pass incorrect side chain atomic information into the model and omit sidechains on the output)
    features, writer = model.binder_features(
        BINDER_LEN, [TargetChain(TARGET, use_msa=True)]
    )
    return features, writer


@app.cell
def _(des_seq_idx, des_writer, features, jax, model):
    des_repredict = model.predict(
        PSSM=jax.nn.one_hot(des_seq_idx, 20),
        features=features,
        recycling_steps=4,
        sampling_steps=200,
        key=jax.random.key(0),
        writer=des_writer,
    )
    return (des_repredict,)


@app.cell
def _(des_repredict, nb):
    nb.pdb_viewer(des_repredict.model_output.to_structure())
    return


@app.cell
def _(des_repredict):
    des_repredict.iptm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · Batched: sample → inverse-fold → re-predict

    Let's JIT and vmap that process (sample → inverse-fold → re-predict) -- we need to take many samples to get a good sequence.
    """)
    return


@app.cell
def _(Array, Int, StructureModelOutput, eqx):
    class PromeraSample(eqx.Module):
        backbone: StructureModelOutput
        sequence: Int[Array, "B"]
        refolded: StructureModelOutput

    return (PromeraSample,)


@app.cell
def _(BINDER_LEN, PromeraSample, eqx, inverse_fold, jax, np):
    @eqx.filter_jit
    def sample_promera(
        model, mpnn, key, *, des_features, binder_features, temp=0.1
    ) -> PromeraSample:
        des = model.model_output(
            features=des_features,
            recycling_steps=4,
            sampling_steps=200,
            key=key,
            step_scale=1.0,
        )
        seq = inverse_fold(
            mpnn, BINDER_LEN, des, temp=temp, key=jax.random.fold_in(key, 1)
        )
        repredicted = model.model_output(
            features=binder_features,
            recycling_steps=4,
            sampling_steps=200,
            key=jax.random.fold_in(key, 2),
            PSSM=jax.nn.one_hot(seq, 20),
        )
        return PromeraSample(des, seq, repredicted)


    @eqx.filter_jit
    def _batch_sample_promera(
        model, mpnn, key, *, des_features, binder_features, temp=0.1, n_samples: int
    ) -> PromeraSample:
        return jax.vmap(
            lambda k: sample_promera(
                model,
                mpnn,
                key=k,
                des_features=des_features,
                binder_features=binder_features,
                temp=temp,
            )
        )(jax.random.split(key, n_samples))


    def batch_sample_promera(
        model, mpnn, key, *, des_features, binder_features, temp=0.1, n_samples: int
    ) -> list[PromeraSample]:
        batched = _batch_sample_promera(
            model,
            mpnn,
            key,
            des_features=des_features,
            binder_features=binder_features,
            temp=temp,
            n_samples=n_samples,
        )
        batched = jax.tree.map(np.array, batched)
        return [jax.tree.map(lambda v: v[i], batched) for i in range(n_samples)]

    return (batch_sample_promera,)


@app.cell
def _(batch_sample_promera, des_features, features, jax, model, mpnn):
    batch_sample = [
        a
        for a in batch_sample_promera(
            model,
            mpnn,
            jax.random.key(0),
            des_features=des_features,
            binder_features=features,
            n_samples=16,
        )
        for _ in range(2)
    ]
    return (batch_sample,)


@app.cell
def _(batch_sample, nb):
    nb.pdb_viewer(batch_sample[0].backbone.to_structure())
    return


@app.cell
def _(batch_sample, nb):
    nb.pdb_viewer(batch_sample[0].refolded.to_structure())
    return


@app.cell
def _(batch_sample):
    batch_sample[0].refolded.chain_pair_iptm()
    return


@app.cell
def _(batch_sample, plt):
    plt.plot([s.refolded.chain_pair_iptm()[(0, 1)] for s in batch_sample])
    plt.xlabel("Sample idx")
    plt.ylabel("IPTM")
    return


@app.cell
def _(batch_sample, np):
    best_sample_idx = np.argmax(
        [s.refolded.chain_pair_iptm()[(0, 1)] for s in batch_sample]
    )
    return (best_sample_idx,)


@app.cell
def _(batch_sample, best_sample_idx, nb):
    nb.pdb_viewer(batch_sample[best_sample_idx].backbone.to_structure())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Differentiable PSSM design

    Optimize a soft binder PSSM directly against a composite structure loss with APGM.
    """)
    return


@app.cell
def _(
    BigramExcess,
    ESMCPseudoPerplexity,
    InverseFoldingSequenceRecovery,
    NormedGradient,
    UnigramExcess,
    esmc,
    features,
    jnp,
    model,
    mpnn,
    sp,
):
    # Composite objective: structure terms + MPNN sequence recovery + ESMC + n-gram priors.
    structure_loss = (
        1.0 * sp.BinderTargetContact()
        + 1.0 * sp.WithinBinderContact()
        + 10.0
        * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(0.001), num_samples=8)
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.025 * sp.IPTMLoss()
        + 0.4 * sp.WithinBinderPAE()
        + 0.025 * sp.pTMEnergy()
        + 0.1 * sp.PLDDTLoss()
    )
    loss = (
        NormedGradient(
            model.build_loss(
                loss=structure_loss,
                features=features,
                recycling_steps=2,
                sampling_steps=20,
                mask_fraction=0.1,  # mask 10% of residues at each call
                num_samples=2,
            ),
            1.0,
        )
        + NormedGradient(ESMCPseudoPerplexity(esm=esmc), 0.15)
        + 10 * UnigramExcess()
        + 10 * BigramExcess()
    )
    return (loss,)


@app.cell
def _(BINDER_LEN, jax, np):
    # Differentiable PSSM design: random init -> soft APGM -> sharpening anneal.
    # (Re-run this cell with a different SEED for another independent design.)
    SEED = 1
    sqrtN = np.sqrt(BINDER_LEN)
    x0 = 0.5 * jax.random.normal(jax.random.key(SEED), (1, BINDER_LEN, 20))
    return SEED, sqrtN, x0


@app.cell
def _(SEED, batched_simplex_APGM, jax, loss, sqrtN, x0):
    # Soft pass: relax the simplex from the random init.
    _, pssm_soft = batched_simplex_APGM(
        loss_function=loss,
        x=x0,
        n_steps=64,
        stepsize=0.5 * sqrtN,
        momentum=0.5,
        scale=1.0,
        logspace=True,
        max_gradient_norm=1.0,
        key=jax.random.key(SEED),
    )
    return (pssm_soft,)


@app.cell
def _(SEED, batched_simplex_APGM, jax, jnp, loss, pssm_soft, sqrtN):
    # Sharpening anneal toward a near-one-hot PSSM.
    pssm, _ = batched_simplex_APGM(
        loss_function=loss,
        x=jnp.log(pssm_soft + 1e-5),
        n_steps=20,
        stepsize=0.75 * sqrtN,
        momentum=0.0,
        scale=1.15,
        logspace=True,
        max_gradient_norm=1.0,
        key=jax.random.key(SEED + 2),
    )
    return (pssm,)


@app.cell
def _(TOKENS, np, pssm):
    seq = "".join(TOKENS[i] for i in np.asarray(pssm[0]).argmax(-1))
    seq
    return (seq,)


@app.cell
def _(BINDER_LEN, TOKENS, features, jax, jnp, model, seq, writer):
    # Fold the hard argmax sequence in the design context (masked-framework feats).
    hard = jax.nn.one_hot(
        jnp.asarray([TOKENS.index(c) for c in seq]),
        20,
    )
    pred = model.predict(
        PSSM=hard,
        features=features,
        writer=writer,
        recycling_steps=3,
        sampling_steps=100,
        key=jax.random.key(0),
    )
    iptm = float(pred.iptm)
    bplddt = float(pred.plddt[:BINDER_LEN].mean())
    print(
        f"in-design fold: iptm {iptm:.3f}  binder pLDDT {bplddt:.3f}  Cys {seq.count('C')}"
    )
    print(seq)
    return (pred,)


@app.cell
def _(nb, pred):
    nb.pdb_viewer(pred.model_output.to_structure())
    return


@app.cell
def _(plt, pred):
    plt.plot(pred.plddt)
    return


@app.cell
def _(BINDER_LEN, OUT, TARGET, TargetChain, jax, model, seq):
    # refold the designed binder as a real chain (its own
    # sequence, full sidechains) + target via target_only_features.
    ref_feats, ref_writer = model.target_only_features(
        [
            TargetChain(
                seq,
                use_msa=False,
            ),  # designed binder
            TargetChain(TARGET, use_msa=True),  # target (cached MSA)
        ]
    )
    ref = model.predict(
        features=ref_feats,
        writer=ref_writer,
        recycling_steps=4,
        sampling_steps=200,
        key=jax.random.key(1),
    )
    ref_iptm = list(ref.model_output.chain_pair_iptm().values())[0]
    ref_bplddt = float(ref.plddt[:BINDER_LEN].mean())
    print(f"refold: iptm {ref_iptm:.3f}  binder pLDDT {ref_bplddt:.3f}")
    ref.st.write_pdb(str(OUT / "mosaic_binder.pdb"))
    print("saved", OUT / "mosaic_binder.pdb")
    return (ref,)


@app.cell
def _(nb, ref):
    nb.pdb_viewer(ref.model_output.to_structure())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6 · Using `promera` + `mpnn` to initialize `mosaic` to finetune a binder

    We can take the best promera backbone and refine it using `simplex_APGM`: to turn the backbone into a soft(ish) PSSM we first inverse fold 64 times with protein mpnn.
    """)
    return


@app.cell
def _(BINDER_LEN, StructureModelOutput, eqx, inverse_fold, jax):
    @eqx.filter_jit
    def batched_inverse_fold(
        mpnn, structure_output: StructureModelOutput, key, *, n_samples, temp=0.1
    ):
        return jax.nn.one_hot(
            jax.vmap(
                lambda k: inverse_fold(mpnn, BINDER_LEN, structure_output, temp, k)
            )(jax.random.split(key, n_samples)),
            20,
        ).mean(0)

    return (batched_inverse_fold,)


@app.cell
def _(batch_sample, batched_inverse_fold, best_sample_idx, jax, mpnn):
    mpnn_pssm = batched_inverse_fold(
        mpnn,
        batch_sample[best_sample_idx].backbone,
        jax.random.key(0),
        n_samples=64,
    )
    return (mpnn_pssm,)


@app.cell
def _(mpnn_pssm, plt):
    plt.imshow(mpnn_pssm)
    return


@app.cell
def _(SEED, batched_simplex_APGM, jax, loss, mpnn_pssm, sqrtN):
    _, pssm_mpnn_soft = batched_simplex_APGM(
        loss_function=loss,
        x=mpnn_pssm[None],
        n_steps=64,
        stepsize=0.25 * sqrtN,
        momentum=0.5,
        scale=1.0,
        logspace=True,
        max_gradient_norm=1.0,
        key=jax.random.key(SEED),
    )
    return (pssm_mpnn_soft,)


@app.cell
def _(SEED, batched_simplex_APGM, jax, jnp, loss, pssm_mpnn_soft, sqrtN):
    pssm_mpnn, _ = batched_simplex_APGM(
        loss_function=loss,
        x=jnp.log(pssm_mpnn_soft + 1e-5),
        n_steps=20,
        stepsize=0.75 * sqrtN,
        momentum=0.0,
        scale=1.15,
        logspace=True,
        max_gradient_norm=1.0,
        key=jax.random.key(SEED + 2),
    )
    return (pssm_mpnn,)


@app.cell
def _(features, jax, model, pssm_mpnn, writer):
    pred_mpnn = model.predict(
        PSSM=pssm_mpnn[0],
        features=features,
        writer=writer,
        recycling_steps=3,
        sampling_steps=100,
        key=jax.random.key(2),
    )
    pred_mpnn.model_output.chain_pair_iptm()
    return (pred_mpnn,)


@app.cell
def _(nb, pred_mpnn):
    nb.pdb_viewer(pred_mpnn.model_output.to_structure())
    return


if __name__ == "__main__":
    app.run()
