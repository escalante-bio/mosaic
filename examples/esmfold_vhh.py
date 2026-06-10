import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        "Demo VHH CDR design against PD-L1, roughly recreating the ESMFold2 binder "
        "design algorithm. We fold the binder + target complex with "
        "ESMFold2-Experimental-Fast and add an ESMC-6B pseudo-perplexity term.",
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
    import equinox as eqx

    import mosaic.losses.structure_prediction as sp
    from mosaic.common import TOKENS
    from mosaic.losses.esmc import ESMCPseudoPerplexity, load_esmc
    from mosaic.losses.esmfold2 import StackedEsmFold2PLL
    from mosaic.losses.transformations import (
        NormedGradient,
        SetPositions,
        RandomChoice,
    )
    from mosaic.models.esmfold2 import (
        ESMFold2ExperimentalFast,
        ESMFold2ExperimentalFast2025,
        ESMFold2Fast,
    )
    from mosaic.optimizers import batched_simplex_APGM
    from mosaic.structure_prediction import TargetChain

    return (
        ESMCPseudoPerplexity,
        ESMFold2ExperimentalFast,
        ESMFold2ExperimentalFast2025,
        SetPositions,
        StackedEsmFold2PLL,
        TOKENS,
        TargetChain,
        batched_simplex_APGM,
        eqx,
        jax,
        jnp,
        load_esmc,
        mo,
        np,
        sp,
        time,
    )


@app.cell
def _():
    # def to_cpu(pytree):
    #     static, params = eqx.partition(pytree, lambda v: eqx.is_inexact_array(v) and len(v.shape)>=1)
    #     return eqx.combine(static, jax.tree.map(lambda v: np.array(v), params))
    return


@app.cell
def _(ESMFold2ExperimentalFast2025, eqx):
    # remove esmc models to save vram
    model = eqx.tree_at(lambda m: m.esmc, ESMFold2ExperimentalFast2025(), None)
    return (model,)


@app.cell
def _(ESMFold2ExperimentalFast, eqx):
    model_default_cutoff = eqx.tree_at(
        lambda m: m.esmc, ESMFold2ExperimentalFast(), None
    )
    return (model_default_cutoff,)


@app.cell
def _(load_esmc):
    esmc = load_esmc("biohub/ESMC-6B")
    return (esmc,)


@app.cell
def _(eqx, esmc):
    def add_esmc(model):
        return eqx.tree_at(
            lambda m: m.esmc, model, esmc, is_leaf=lambda x: x is None
        )

    return (add_esmc,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        "Technically speaking we have many more copies of ESMC floating around than we need! This could be refactored to use substantially less VRAM + RAM...",
        kind="danger",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    The target is PD-L1. The VHH is a single domain: four framework regions
    (FR1–FR4) with three CDR loops between them. We keep the framework fixed at
    its real residues and mark every designable CDR position with `X`.
    """)
    return


@app.cell(hide_code=True)
def _():
    PDL1_SEQUENCE = (
        "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQ"
        "HSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA"
    )

    FR1 = "EVQLQESGGGLVQPGGSLRLSCAAS"
    CDR1 = "X" * 9
    FR2 = "YWLRQAPGKGLEWVSS"
    CDR2 = "X" * 8
    FR3 = "YYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYC"
    CDR3 = "X" * 8
    FR4 = "KGQGTQVTVSSD"
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
def _(PDL1_SEQUENCE, TargetChain, VHH_TEMPLATE, add_esmc, model, np):
    target_chains = [TargetChain(PDL1_SEQUENCE, use_msa=False)]

    # Design pack: every binder position is an all-X designed slot (refresh tied on,
    # binder allocated at the 14-atom max internally). The framework isn't pinned in
    # the pack — `SetPositions` does that during optimization.
    pack, _ = add_esmc(model).binder_features(len(VHH_TEMPLATE), target_chains)
    # design_positions are the token indices of the `X`s. The binder is chain 0
    # at token offset 0, so they're exactly the `X` positions in the template.
    design_positions = np.array([i for i, c in enumerate(VHH_TEMPLATE) if c == "X"])
    M = len(design_positions)
    return M, design_positions, pack


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
    SetPositions,
    StackedEsmFold2PLL,
    VHH_TEMPLATE,
    design_positions,
    esmc,
    model,
    model_default_cutoff,
    pack,
    structure_loss,
):
    from mosaic.models.esmfold2 import _staticize_scalar_config

    LOSS_RECYCLING_ITERS = 1


    _build = lambda m: m.build_loss(
        loss=structure_loss,
        features=pack,
        recycling_steps=LOSS_RECYCLING_ITERS,
        msa_max_depth=1024,
        lm_dropout=0.5,  # geom + LM refresh tied to the (design) pack
    )

    loss = SetPositions.from_sequence(
        VHH_TEMPLATE,
        StackedEsmFold2PLL(
            structure_losses=[_build(model), _build(model_default_cutoff)],
            esmc=esmc,
            pll=ESMCPseudoPerplexity(esm=None, design_idx=design_positions),
            pll_weight=0.5,
        ),
    )
    return LOSS_RECYCLING_ITERS, loss


@app.cell
def _():
    B = 4  # batch: parallel designs from different gumbel inits
    SEED = 0
    N_SOFT = 128  # stage-1 soft steps
    N_SHARP = 20  # stage-2 sharpening steps
    GUMBEL = 0.75  # init concentration
    S1_MULT = 0.10  # stage-1 stepsize multiplier (× sqrt(M))
    S2_MULT = 0.05  # stage-2 stepsize multiplier
    S2_SCALE = 1.3  # stage-2 anneal rate
    S1_MOM = 0.5  # stage-1 momentum
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
    t_stage1 = time.time() - t0_soft  # exposed for the speed-comparison plot
    print(f"stage 1 done in {t_stage1:.1f}s  ")
    return pssm, t_stage1


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
    t_stage2 = time.time() - t0_sharp  # exposed for the speed-comparison plot
    print(f"stage 2 done in {t_stage2:.1f}s  ")
    return pssm_sharp, t_stage2


@app.cell
def _(mo):
    mo.md(r"""
    During design the CDR ESMC features and binder atom geometry are refreshed
    each step from `argmax(PSSM)` (`refresh_lm` + `geom_refresh`), so the
    design-time forward already tracks the discrete sequence — the design iPTM is
    trustworthy and we **don't need** a separate `target_only_features`
    reprediction to score it. We still repredict here with native
    `target_only_features` (real sidechains, tight atom packing) as an
    independent judge and to write clean structures, but the geom + LM-refreshed
    design score is already faithful.
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
    VHH_TEMPLATE,
    add_esmc,
    design_positions,
    full_sequence,
    jax,
    jnp,
    model,
    np,
    pack,
    pssm_sharp,
):
    L = len(VHH_TEMPLATE)
    designs = []
    predictions = []
    for i in range(B):
        tokens = np.argmax(np.asarray(pssm_sharp[i]), axis=-1)
        seq = full_sequence(VHH_TEMPLATE, design_positions, tokens)
        cdr = "".join(TOKENS[int(t)] for t in tokens)
        binder_ids = jnp.array([TOKENS.index(c) for c in seq])
        pred = add_esmc(model).predict(
            PSSM=jax.nn.one_hot(binder_ids, 20),
            features=pack,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **`binder_features` design-time featurization vs `target_only_features`, at
    matched depth.** Both score each design at `recycling_steps=20` /
    `sampling_steps=100`, so the *only* difference is the featurization path: here
    the `binder_features` pack with in-loop `geom_refresh + refresh_lm` (the
    design-time geometry/LM; the fixed `predict` writes the structure from the
    layout-correct atom37 view) vs the native `target_only_features` repredict above
    (real sidechains, tight packing). Matching the recycling/sampling depth isolates
    whether the design-time featurization itself biases iPTM — a large Δ flags a
    design that only looks good through its own refreshed forward.
    """)
    return


@app.cell
def _(CDR_RANGES):
    from ipymolstar import PDBeMolstar

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


@app.cell
def _(cdr_viewer, predictions):
    cdr_viewer(predictions[0].model_output.to_structure())
    return


@app.cell(hide_code=True)
def _():
    ESMFOLD2_OFFICIAL_DESIGNS = [
        (
            "vhh_rank01_id18",
            "EVQLQESGGGLVQPGGSLRLSCAASGKISDYNVLYWLRQAPGKGLEWVSSLIQSRGKTYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCAVGARWFDKGQGTQVTVSSD",
        ),
        (
            "vhh_rank02_id01",
            "EVQLQESGGGLVQPGGSLRLSCAASGPIDDYAVEYWLRQAPGKGLEWVSSLVQSRGKTYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCNVIGSSFVKGQGTQVTVSSD",
        ),
        (
            "vhh_rank03_id12",
            "EVQLQESGGGLVQPGGSLRLSCAASKPIDDYYLRYWLRQAPGKGLEWVSSLRLSKNKKYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCYALGYRMIKGQGTQVTVSSD",
        ),
        (
            "vhh_rank04_id02",
            "EVQLQESGGGLVQPGGSLRLSCAASGVTLEEYTNYWLRQAPGKGLEWVSSLLSSKNKTYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCLAGFSISNKGQGTQVTVSSD",
        ),
        (
            "vhh_rank05_id14",
            "EVQLQESGGGLVQPGGSLRLSCAASGPVSEYLVQYWLRQAPGKGLEWVSSLLLHKGKTYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCVIPSSFFNKGQGTQVTVSSD",
        ),
        (
            "vhh_rank06_id06",
            "EVQLQESGGGLVQPGGSLRLSCAASKPIDNSNVEYWLRQAPGKGLEWVSSLVISRGKTYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCVALGHFMLKGQGTQVTVSSD",
        ),
        (
            "vhh_rank07_id16",
            "EVQLQESGGGLVQPGGSLRLSCAASEPIENLSVLYWLRQAPGKGLEWVSSFLMNKDKAYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCGTLSTFDKKGQGTQVTVSSD",
        ),
        (
            "vhh_rank08_id05",
            "EVQLQESGGGLVQPGGSLRLSCAASRPVSSNSVMYWLRQAPGKGLEWVSSLKLRTGKRYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCVTLSTFHKKGQGTQVTVSSD",
        ),
        (
            "vhh_rank09_id17",
            "EVQLQESGGGLVQPGGSLRLSCAASRPVSEHSNEYWLRQAPGKGLEWVSSLSGAGGTPYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCMSSSSTEDKGQGTQVTVSSD",
        ),
        (
            "vhh_rank10_id11",
            "EVQLQESGGGLVQPGGSLRLSCAASGRVSDSNVQYWLRQAPGKGLEWVSSLLVSKGKTYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCVFLGHSALKGQGTQVTVSSD",
        ),
        (
            "vhh_rank11_id19",
            "EVQLQESGGGLVQPGGSLRLSCAASGIKLENYNLYWLRQAPGKGLEWVSSLMRTNGAKYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCEVSLSSYAKGQGTQVTVSSD",
        ),
        (
            "vhh_rank12_id10",
            "EVQLQESGGGLVQPGGSLRLSCAASGPISHPRVQYWLRQAPGKGLEWVSSILVRSGKKYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCVFLSRGFYKGQGTQVTVSSD",
        ),
        (
            "vhh_rank13_id00",
            "EVQLQESGGGLVQPGGSLRLSCAASGKIDEYSVEYWLRQAPGKGLEWVSSLKLRSGKTYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCVVNFFTFFKGQGTQVTVSSD",
        ),
        (
            "vhh_rank14_id13",
            "EVQLQESGGGLVQPGGSLRLSCAASRPISGHHVSYWLRQAPGKGLEWVSSLLAHRGKTYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCVYIGSFLSKGQGTQVTVSSD",
        ),
        (
            "vhh_rank15_id15",
            "EVQLQESGGGLVQPGGSLRLSCAASKPIDDHFVQYWLRQAPGKGLEWVSSIIASKGKTYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCVYPASFSYKGQGTQVTVSSD",
        ),
        (
            "vhh_rank16_id04",
            "EVQLQESGGGLVQPGGSLRLSCAASKPISEYYVQYWLRQAPGKGLEWVSSLRLRTGKTYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCGAPAFGLLKGQGTQVTVSSD",
        ),
        (
            "vhh_rank17_id08",
            "EVQLQESGGGLVQPGGSLRLSCAASGVILEYYLTYWLRQAPGKGLEWVSSLRAYKNKTYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCNIIRSFYYKGQGTQVTVSSD",
        ),
        (
            "vhh_rank18_id03",
            "EVQLQESGGGLVQPGGSLRLSCAASEPISNFYNGYWLRQAPGKGLEWVSSLLESRSATYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCMMASQSIYKGQGTQVTVSSD",
        ),
        (
            "vhh_rank19_id07",
            "EVQLQESGGGLVQPGGSLRLSCAASGPISKNAVEYWLRQAPGKGLEWVSSIFVSSGSTYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCSLLGRGVVKGQGTQVTVSSD",
        ),
        (
            "vhh_rank20_id09",
            "EVQLQESGGGLVQPGGSLRLSCAASGPIENVFLSYWLRQAPGKGLEWVSSLLRHGGSKYYRDSVKGRFTISRDNAKNTLYLQMNSLKSEDTAVYYCNMLSHPLYKGQGTQVTVSSD",
        ),
    ]
    return (ESMFOLD2_OFFICIAL_DESIGNS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **iPTM eval of the official ESMFold2 design method.** `ESMFOLD2_OFFICIAL_DESIGNS`
    above holds 20 VHH binders against PD-L1 produced by the official ESMFold2 design
    pipeline.
    """)
    return


@app.cell
def _(
    ESMFOLD2_OFFICIAL_DESIGNS,
    TOKENS,
    VHH_TEMPLATE,
    add_esmc,
    jax,
    jnp,
    mo,
    model,
    np,
    pack,
):
    _rows = []
    for _did, _s in ESMFOLD2_OFFICIAL_DESIGNS:
        assert len(_s) == len(VHH_TEMPLATE), f"{_did}: len {len(_s)} != pack len"
        _ids = jnp.array([TOKENS.index(c) for c in _s])
        # Design-time forward: the binder_features pack has refresh=True, so geom+LM
        # refresh and the atom37 structure writer follow from the pack (no writer).
        _p = add_esmc(model).predict(
            PSSM=jax.nn.one_hot(_ids, 20),
            features=pack,
            recycling_steps=20,
            sampling_steps=100,
            key=jax.random.key(0),
        )
        _ip = float(_p.iptm)
        _rows.append(
            {
                "design": _did,
                "esmfold2 iptm": round(_ip, 3),
                "pLDDT(b)": round(float(np.asarray(_p.plddt)[: len(_s)].mean()), 3),
            }
        )
        print(f"{_did:18} esmfold2={_ip:.3f}", flush=True)

    # Public so the histogram cell can read it.
    official_eval = sorted(_rows, key=lambda r: r["esmfold2 iptm"], reverse=True)
    mo.ui.table(official_eval, selection=None)
    return (official_eval,)


@app.cell
def _(np, official_eval, plt, predictions):
    # Compare two design *methods*, both scored with our design-time forward:
    #   - the official ESMFold2 design method (ESMFOLD2_OFFICIAL_DESIGNS, n=20)
    #   - the sequences designed in this notebook (the B=4 in `predictions`)
    _other = [r["esmfold2 iptm"] for r in official_eval]  # official method, n=20
    _ours = [float(p.iptm) for p in predictions]  # this notebook's designs
    _all = _other + _ours
    _bins = np.linspace(min(_all) - 0.01, max(_all) + 0.01, 16)

    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.hist(
        _other,
        bins=_bins,
        alpha=0.55,
        color="#999999",
        label=f"official ESMFold2 (n={len(_other)})  median={np.median(_other):.3f}",
    )
    _ax.hist(
        _ours,
        bins=_bins,
        alpha=0.7,
        color="#3b6fd6",
        label=f"this notebook (n={len(_ours)})  median={np.median(_ours):.3f}",
    )
    _ax.set_xlabel("iPTM (esmfold2 design-time forward)")
    _ax.set_ylabel("# designs")
    _ax.set_title("iPTM by design method: official ESMFold2 vs this notebook")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Design speed: JAX (this notebook) vs the official torch ESMFold2 pipeline.**
    Per-design wall-clock for this notebook's optimization (recycling_steps=1` — 2 trunk passes) against the official torch ESMFold2
    design notebook (~117 s/design at `recycling_steps=0`, a single trunk pass). We do roughly twice as much work per iteration.
    """)
    return


@app.cell(hide_code=True)
def _(B, LOSS_RECYCLING_ITERS, jax, plt, t_stage1, t_stage2):
    _our_total_s = t_stage1 + t_stage2
    _our_per_design = _our_total_s / B  # B designs produced per batched run

    _torch_per_design = 117.0  # s/design, recycling_steps=0
    _speedup = _torch_per_design / _our_per_design

    _labels = [
        "torch ESMFold2 (modal H100, no kernels).\nB=5 batched\nrecycle=0",
        f"JAX (this notebook, {jax.devices()[0].device_kind})\nB={B} batched\nrecycle={LOSS_RECYCLING_ITERS} (!)",
    ]
    _vals = [_torch_per_design, _our_per_design]

    _fig, _ax = plt.subplots(figsize=(8, 6))
    _bars = _ax.bar(_labels, _vals, color=["#999999", "#3b6fd6"], width=0.6)
    for _b, _v in zip(_bars, _vals):
        _ax.text(
            _b.get_x() + _b.get_width() / 2,
            _v + 2.5,
            f"{_v:.0f} s",
            ha="center",
            fontweight="bold",
        )
    _ax.set_ylabel("wall-clock seconds per design")
    _ax.set_ylim(0, max(_vals) * 1.2)
    _ax.set_title(
        f"{_speedup:.2f}x faster per design, even with nrecycling={LOSS_RECYCLING_ITERS} v.s. 0"
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
