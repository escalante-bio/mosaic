import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.callout(
        "Demo VHH CDR design against PD-L1, recreating the ESMFold2 binder "
        "design algorithm. We fold the binder + target complex with "
        "ESMFold2-Experimental-Fast and add an ESMC-6B pseudo-perplexity term.",
        kind="success",
    )
    return


@app.cell
def _():
    import os

    # Must precede `import jax`: JAX reads XLA_PYTHON_CLIENT_MEM_FRACTION at
    # init. The binder+target complex with ESMC-6B needs a high fraction (the
    # 0.75 default OOMs). See esmfold2-6b-design memory.
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

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
        SetPositions,
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
def _(ESMFold2ExperimentalFast2025, load_esmc):
    model = ESMFold2ExperimentalFast2025()
    ppl_esmc = load_esmc("biohub/ESMC-6B")
    return model, ppl_esmc


@app.cell
def _(mo):
    mo.md(r"""
    The target is PD-L1. The VHH is a single domain: four framework regions
    (FR1–FR4) with three CDR loops between them. We keep the framework fixed at
    its real residues and mark every designable CDR position with `X`.
    """)
    return


@app.cell
def _():
    PDL1_SEQUENCE = (
        "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQ"
        "HSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA"
    )

    FR1 = "QVQLVESGGGLVQPGGSLRLSCAAS"
    CDR1 = "X" * 11
    FR2 = "LGWFRQAPGQGLEAVAA"
    CDR2 = "X" * 8
    FR3 = "YYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC"
    CDR3 = "X" * 18
    FR4 = "WGQGTLVTVS"
    VHH_TEMPLATE = FR1 + CDR1 + FR2 + CDR2 + FR3 + CDR3 + FR4

    CDR_RANGES = {}
    _offset = 0
    for _name, _seg in [
        ("FR1", FR1),
        ("CDR1", CDR1),
        ("FR2", FR2),
        ("CDR2", CDR2),
        ("FR3", FR3),
        ("CDR3", CDR3),
        ("FR4", FR4),
    ]:
        if _name.startswith("CDR"):
            CDR_RANGES[_name] = (_offset, _offset + len(_seg) - 1)
        _offset += len(_seg)

    print(f"VHH template ({len(VHH_TEMPLATE)} aa, {VHH_TEMPLATE.count('X')} X):")
    print(VHH_TEMPLATE)
    print("CDR loops (0-indexed inclusive):", CDR_RANGES)
    return CDR_RANGES, PDL1_SEQUENCE, VHH_TEMPLATE


@app.cell
def _(PDL1_SEQUENCE, TargetChain, VHH_TEMPLATE, model, np):
    target_chains = [TargetChain(PDL1_SEQUENCE, use_msa=False)]

    pack, _ = model.target_only_features(
        [TargetChain(VHH_TEMPLATE, use_msa=False), *target_chains],
        design_char="X",
        design_geometry="G",
    )
    # design_positions are the token indices of the `X`s. The binder is chain 0
    # at token offset 0, so they're exactly the `X` positions in the template.
    design_positions = np.array([i for i, c in enumerate(VHH_TEMPLATE) if c == "X"])
    M = len(design_positions)
    return M, design_positions, pack, target_chains


@app.cell
def _(mo):
    mo.md(r"""
    A small linear combination of structure-prediction terms: a within-binder
    contact term, a paratope–target contact term, and a radius-of-gyration term.
    """)
    return


@app.cell
def _(design_positions, sp):
    structure_loss = (
        0.5 * sp.WithinBinderContact(num_contacts_per_residue=2)
        + 0.5
        * sp.BinderTargetContact(
            contact_distance=22.0,
            paratope_idx=design_positions,
        )
        + 0.2 * sp.DistogramRadiusOfGyration()
    )
    return (structure_loss,)


@app.cell
def _(
    ESMCPseudoPerplexity,
    NormedGradient,
    SetPositions,
    VHH_TEMPLATE,
    design_positions,
    model,
    pack,
    ppl_esmc,
    structure_loss,
):
    ppl_weight = 0.15

    structure_term = NormedGradient(
        SetPositions.from_sequence(
            VHH_TEMPLATE,
            model.build_loss(
                loss=structure_loss,
                features=pack,
                recycling_steps=2,
                msa_max_depth=1024,
            ),
        ),
        1.0,
    )
    ppl_term = NormedGradient(
        SetPositions.from_sequence(
            VHH_TEMPLATE,
            ESMCPseudoPerplexity(esm=ppl_esmc, design_idx=design_positions),
        ),
        ppl_weight,
    )
    loss = structure_term + ppl_term
    return (loss,)


@app.cell
def _():
    B = 4  # batch: parallel designs from different gumbel inits
    SEED = 0
    N_SOFT = 64  # stage-1 soft steps
    N_SHARP = 30  # stage-2 sharpening steps
    GUMBEL = 0.75  # init concentration
    S1_MULT = 0.15  # stage-1 stepsize multiplier (× sqrt(M))
    S2_MULT = 0.10  # stage-2 stepsize multiplier
    S2_SCALE = 1.3  # stage-2 anneal rate
    S1_MOM = 0.2  # stage-1 momentum
    return B, GUMBEL, N_SHARP, N_SOFT, S1_MOM, S1_MULT, S2_MULT, S2_SCALE, SEED


@app.cell
def _(
    B,
    GUMBEL,
    M,
    N_SOFT,
    S1_MOM,
    S1_MULT,
    SEED,
    batched_simplex_APGM,
    jax,
    loss,
    np,
    time,
):
    x0 = jax.nn.softmax(
        GUMBEL * jax.random.gumbel(jax.random.key(SEED), shape=(B, M, 20))
    )
    t0_soft = time.time()
    _, pssm = batched_simplex_APGM(
        loss_function=loss,
        x=x0,
        n_steps=N_SOFT,
        stepsize=S1_MULT * np.sqrt(M),
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
def _(
    M,
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
        stepsize=S2_MULT * float(np.sqrt(M)),
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
    During design the CDRs are fed to ESMC as `UNK` and to ESMFold2 with the
    atomic geometry of glycine. Here we repredict with the correct ESMC features
    and atomic geometry, then read off iPTM.
    """)
    return


@app.cell
def _(TOKENS):
    def full_sequence(template, positions, tokens):
        """Splice argmax'd design residues into the fixed template at `positions`."""
        chars = list(template)
        for p, t in zip(positions, tokens):
            chars[int(p)] = TOKENS[int(t)]
        return "".join(chars)

    return (full_sequence,)


@app.cell
def _(
    B,
    SEED,
    TOKENS,
    TargetChain,
    VHH_TEMPLATE,
    full_sequence,
    jax,
    jnp,
    model,
    np,
    pssm_sharp,
    target_chains,
    design_positions,
):
    L = len(VHH_TEMPLATE)
    designs = []
    predictions = []
    for i in range(B):
        tokens = np.argmax(np.asarray(pssm_sharp[i]), axis=-1)
        seq = full_sequence(VHH_TEMPLATE, design_positions, tokens)
        cdr = "".join(TOKENS[int(t)] for t in tokens)

        full_features, writer = model.target_only_features(
            chains=[TargetChain(seq, use_msa=False), *target_chains],
        )
        binder_ids = jnp.array([TOKENS.index(c) for c in seq])
        pred = model.predict(
            PSSM=jax.nn.one_hot(binder_ids, 20),
            features=full_features,
            writer=writer,
            recycling_steps=20,
            sampling_steps=100,
            key=jax.random.key(SEED + 100 + i),
        )
        predictions.append(pred)
        iptm = float(pred.iptm)
        plddt_b = float(pred.plddt[:L].mean())
        designs.append(
            dict(i=i, iptm=iptm, plddt_binder=plddt_b, cdr_seq=cdr, seq=seq)
        )
        print(
            f"design {i}: iPTM={iptm:.3f}  pLDDT(b)={plddt_b:.3f}  CDRs={cdr} seq={seq}"
        )
    designs
    return (predictions,)


@app.cell
def _(CDR_RANGES):
    from ipymolstar import PDBeMolstar
    from mosaic.notebook_utils import pdb_viewer

    _CDR_COLORS = {
        "CDR1": {"r": 230, "g": 70, "b": 70},
        "CDR2": {"r": 60, "g": 170, "b": 90},
        "CDR3": {"r": 70, "g": 110, "b": 235},
    }


    def cdr_viewer(st, binder_chain="A", target_chain="B"):
        """Cartoon viewer: binder CDR loops colored (CDR1 red, CDR2 green, CDR3
        blue), everything else light grey.

        CDR_RANGES holds 0-indexed inclusive binder spans (the written structure
        numbers residues from 1, so +1).
        """
        data = [
            {
                "struct_asym_id": binder_chain,
                "start_residue_number": start + 1,
                "end_residue_number": end + 1,
                "color": _CDR_COLORS[name],
            }
            for name, (start, end) in CDR_RANGES.items()
        ]
        color_data = {
            "data": data,
            "nonSelectedColor": {"r": 214, "g": 214, "b": 214},
        }
        custom_data = {
            "data": st.make_pdb_string(),
            "format": "pdb",
            "binary": False,
        }
        return PDBeMolstar(
            custom_data=custom_data,
            color_data=color_data,
            visual_style="cartoon",
        )

    return (cdr_viewer,)


@app.cell
def _(cdr_viewer, predictions):
    cdr_viewer(predictions[0].st)
    return


@app.cell
def _(cdr_viewer, predictions):
    cdr_viewer(predictions[2].st)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
