import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Designing a VHH against PD-L1
     This is a recreation of the binder design algorithm from [ESMFold2](https://bhp-papers-prod.s3.us-west-2.amazonaws.com/esm_protein.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=ASIAU6GD3FYNPLX52H7J%2F20260530%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260530T211639Z&X-Amz-Expires=3600&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEB0aCXVzLXdlc3QtMiJGMEQCICo4VQqHmDUk362dR5YyXxSkGEUaUMH5s6M1PLN0S6MVAiBUQfBgpuBioGJ2fHtUuoC9nFAbf%2FETrM6HBzFKntfirCqyBQjm%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDMzOTcxMzE0MjI5OCIMmfdJe4%2BPuyPURDS9KoYF%2B41DBMN4730OL3YsxuLSRIk%2BLAviRp%2BCYdzTlUp3oGzMFx4bzu3r9TtKmJjBc%2FjYv2G2qVlGhlj6fSKC9IsJSc0yG5sBAkhqIe%2BjVja8H9X5fBzPQeI92rdWpT8nP21Yjwa9j%2FphDR4keLzEdz9OvSt%2FNnEvS4U3HEt6uVhaBgcN5DjmIvJOUlNpKGFBiJXoGHk0Low%2BL8NARvxaD4nyXg4Yajz2FB%2B2ptyBrMU9wn2RFxE6yzI9C9l%2F0QqomnoUl1VsYyTyS40%2FnvChN5lrm0VMZycDowzx3pp1PcEjN4RiLhaY4VMR6nAp6ImuZqIL6NWzJB5Fcszbjgz%2B8o%2Bm5ic5QH%2BZEQQH7HdpK%2BQc3qhu8MFVxO8QX9OindiuFCCpuqfpqBKCLAM5uNeNwO%2B6HrkkOsEcjMngczJjCwyvg%2FIZCXfayZ6wZXtHbPCx2dalcXQlXMF9KIkAxLITw8d01htmpL1coCZSEYnug7QzTbHXoM0h5LglspixoaEgvGvoILTHiFc%2F09%2F%2FORj6pJuko2sj%2B5Nem3%2BrUku74PR9vOOuYuB%2B6qr%2BGd%2BRWzYjr684T0SiWxcvcUILP00WxpfoO17xBt06m8j%2FrwsS0%2Fl11nGiljtUrIBI14ZGBbLi0kQWrpZTXGsZ2vF%2B7prKC4FPEz80EATwh034GPsv2Wcrc1cHO4pQCWNibRndm6t9CAxk8PdNjXo4fQEukkkeqz%2B4SbHrckbnKkiA%2Fh3ZzD32xq4Ug6RRz8MzFcYAd%2FspMTNXTfw1HZuxBRXQH361b2eBRWsrTfRPglH9Duwoqfk9QL0xRri7wKLmqiN5sMe2Plfzt5EFEwJpfntsZd%2FF%2B6xqlfqIqnEr9TD3oO3QBjqaAVbLBR9l0iuuX9PmE3jsELuzwCqVSCW%2Bpwx7C1IJrKuiJz28O4jZ3UyFZtZECQsmlq48A75osxgrIakJ3pRJzlTwre3yjBQ6wuk4nJ%2FmbPold2UkMZ%2BvOY6pl6qLq0IbW7AfVG4zeVhGmIEP17KrjDWTaHgx%2Frqshzrt%2FV358ekRHIXVrFYXSxUAbdITJrBsPkoS2VdxtKf78mM%3D&X-Amz-Signature=233b4d6cb14353ec098619afdbca486ff8d67857acf7f22cb8f80565afdf0b5e&X-Amz-SignedHeaders=host&x-amz-checksum-mode=ENABLED&x-id=GetObject)

    Here we design the CDR loops of a VHH to hopefully bind PD-L1.

    We use two models:
    - **ESMFold2-Experimental-Fast-2025** — folds the binder + target complex
    - **ESMC-6B** — a protein language model used in a *pseudo-perplexity loss term*

    1. Load both models.
    2. Write the VHH template: framework residues + `X` at designable CDR
       positions.
    3. Featurize.
    4. Build a differentiable loss over the CDR positions only.
    5. Optimize the CDRs in two stages (soft → sharpen) by gradient descent on
       the simplex.
    6. Re-predict each design with full sidechains and ESMC embeddings as two-chain complexes and read off iPTM.
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
def _(mo):
    mo.md(r"""
    ## 2 · The target and the VHH template

    `PDL1_SEQUENCE` is the antigen (115 aa). The VHH is a single domain:
    four framework regions (FR1–FR4) with three CDR loops between them. The
    framework here is the **humanized universal scaffold h-NbBcII10FGLA** (PDB
    3EAK; human IGHV3-23/DP-47 acceptor, a validated CDR-grafting scaffold) — for
    better developability/immunogenicity than the camelid framework. We keep the
    framework at its real residues and mark every designable CDR position with
    `X`. A handful of structural anchors stay fixed on purpose (the leading `G` of
    CDR1, the `I…T` clamp of CDR2).
    """)
    return


@app.cell
def _():
    PDL1_SEQUENCE = (
        "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQ"
        "HSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA"
    )

    # VHH framework (fixed) with 'X' at the designable CDR positions.
    # Framework = the humanized universal scaffold h-NbBcII10FGLA (PDB 3EAK,
    # Vincke et al. 2009): camelid CDRs grafted onto human germline IGHV3-23/DP-47,
    # validated as a universal CDR-grafting acceptor. Same FR lengths as the
    # original camelid framework, so the CDR anchors and ranges are unchanged; the
    # humanization lives mostly in FR2 (the "FGLA" hallmark-tetrad residues).
    FR1 = "QVQLVESGGGLVQPGGSLRLSCAAS"
    CDR1 = "GXXXXXXX"
    FR2 = "LGWFRQAPGQGLEAVAA"
    CDR2 = "IXXXXXXT"
    FR3 = "YYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC"
    CDR3 = "X" * 12
    FR4 = "WGQGTLVTVSS"
    VHH_TEMPLATE = FR1 + CDR1 + FR2 + CDR2 + FR3 + CDR3 + FR4

    # 0-indexed (start, end) inclusive span of each CDR loop within the binder
    # chain — derived from the segment lengths so it stays correct if you edit
    # the template. Used to highlight the loops in the structure viewer below.
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
def _(mo):
    mo.md(r"""
    ## 3 · Featurize
    """)
    return


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
    variable_positions = design_positions
    M = len(variable_positions)
    sqrtM = float(np.sqrt(M))

    print(f"M = {M} designable positions (token indices):")
    print(variable_positions.tolist())
    return M, design_positions, pack, sqrtM, target_chains, variable_positions


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Construct a structure loss

    A small linear combination of structure-prediction terms, all read off the
    folded complex:

    - `WithinBinderContact` — the binder should be a compact, self-contacting
      domain (≥2 contacts/residue).
    - `BinderTargetContact` — reward proximity (≤22 Å) between the **paratope**
      (the designed CDR positions) and PD-L1. With no epitope specified the loops
      are free to dock anywhere on the antigen.
    - `DistogramRadiusOfGyration` — discourage the binder from unfolding/spreading.
    """)
    return


@app.cell
def _(design_positions, sp):
    structure_loss = (
        0.5 * sp.WithinBinderContact(num_contacts_per_residue=2)
        + 0.5
        * sp.BinderTargetContact(
            contact_distance=14.0,#22.0,
            paratope_idx=design_positions,
        )
        + 0.2 * sp.DistogramRadiusOfGyration()
    )
    return (structure_loss,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Assemble the full design loss

    Two wrappers do the work:

    - **`SetPositions.from_sequence(template, inner)`** lifts a loss that lives on
      the `[M, 20]` *design* positions into the full `[N, 20]` sequence: the `X`
      positions get the optimizer's soft PSSM, every other position is pinned to
      its wildtype one-hot. So the optimization variable is just `[B, M, 20]`.
    - **`NormedGradient(inner, scale)`** centers the gradient across the vocab,
      normalizes it to unit norm, then multiplies by `scale`. Doing this *per
      term* makes the structure : prior ratio exactly `1.0 : ppl_weight` on the
      positions we actually move — independent of each term's raw magnitude.

    `ESMCPseudoPerplexity(..., design_idx=variable_positions)` masks/scores only
    the CDR tokens, so the prior pulls just the designed residues toward
    natural-looking sequence.
    """)
    return


@app.cell
def _(
    ESMCPseudoPerplexity,
    NormedGradient,
    SetPositions,
    VHH_TEMPLATE,
    model,
    pack,
    ppl_esmc,
    structure_loss,
    variable_positions,
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
            ESMCPseudoPerplexity(esm=ppl_esmc, design_idx=variable_positions),
        ),
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
    N_SOFT = 128#64  # stage-1 soft steps
    N_SHARP = 30  # stage-2 sharpening steps
    GUMBEL = 0.75  # init concentration
    S1_MULT = 0.1#0.20  # stage-1 stepsize multiplier (× sqrt(M))
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
    GUMBEL,
    M,
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
        GUMBEL * jax.random.gumbel(jax.random.key(SEED), shape=(B, M, 20))
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

    Continue in log-space with an annealing `scale > 1` that progressively
    concentrates each distribution toward a single residue, so the soft PSSM
    collapses to a discrete sequence we can actually fold.
    """)
    return


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
    ## 9 · Decode and re-predict with full sidechains

    During design the CDRs are fed to ESMC as `UNK` and with the atomic geometry of glycine to ESMFold2. Here we repredict with the correct ESMC features and atomic geometry.
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
    variable_positions,
):
    L = len(VHH_TEMPLATE)
    designs = []
    predictions = []
    for i in range(B):
        tokens = np.argmax(np.asarray(pssm_sharp[i]), axis=-1)
        seq = full_sequence(VHH_TEMPLATE, variable_positions, tokens)
        cdr = "".join(TOKENS[int(t)] for t in tokens)

        pack_real, writer = model.target_only_features(
            chains=[TargetChain(seq, use_msa=False), *target_chains],
        )
        # `pack_real` already holds the binder as a real chain (real geometry,
        # real ESMC features), so the structure is an honest reprediction. We
        # also hand `predict` the binder as a hard one-hot PSSM: on this pack
        # it re-writes res_type/MSA-query to the same residues (a no-op on the
        # structure), but it tells the built-in iPTM the binder length so its
        # binder↔target split is correct.
        binder_ids = jnp.array([TOKENS.index(c) for c in seq])
        pred = model.predict(
            PSSM=jax.nn.one_hot(binder_ids, 20),
            features=pack_real,
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
def _(mo):
    mo.md(r"""
    ## 10 · View the complex with CDRs highlighted

    Cartoon of each predicted binder–PD-L1 complex. The binder's three CDR loops
    are colored **CDR1 red · CDR2 green · CDR3 blue**, and everything else is
    light grey. The designed loops should be reaching across and contacting PD-L1.
    """)
    return


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
    cdr_viewer(predictions[1].st)
    return


if __name__ == "__main__":
    app.run()
