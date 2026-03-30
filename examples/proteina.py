import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import jax
    import jax.numpy as jnp
    import numpy as np
    import equinox as eqx
    import gemmi
    import marimo as mo

    jax.config.update("jax_compilation_cache_dir", "/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )

    from jproteina_complexa.hub import load_denoiser, load_decoder
    from jproteina_complexa.pdb import load_target_cond, make_structure
    from jproteina_complexa.flow_matching import generate
    from jproteina_complexa.types import DecoderBatch
    from jproteina_complexa.constants import AA_CODES, AA_3LETTER
    from mosaic.common import TOKENS as MOSAIC_ORDER

    JPC_TO_MOSAIC = jnp.array([AA_CODES.index(aa) for aa in MOSAIC_ORDER])

    import mosaic.losses.structure_prediction as sp
    from mosaic.models.protenix import Protenix2025
    from mosaic.structure_prediction import TargetChain
    from mosaic.notebook_utils import pdb_viewer

    from mosaic.models.proteina import beam_search


@app.cell
def _():
    AA_CODES
    return


@app.function
def make_complex(dec_output, target_cond):
    """Build a gemmi Structure with binder (chain A) + target (chain B)."""
    seq = "".join(AA_CODES[j] for j in np.array(dec_output.aatype))
    return make_structure(
        [
            (
                "A",
                [AA_3LETTER[aa] for aa in seq],
                np.array(dec_output.coors),
                np.array(dec_output.atom_mask).astype(np.float32),
            ),
            (
                "B",
                [AA_3LETTER[AA_CODES[i]] for i in np.array(target_cond.seq)],
                np.array(target_cond.coords),
                np.array(target_cond.atom_mask),
            ),
        ]
    )


@app.cell
def _():
    mo.md("""
    # Protein Binder Design with Proteina-Complexa

    1. **Single sample** — generate one binder from noise
    2. **Batched sampling** — generate many binders in parallel with `jax.vmap`
    3. **Scoring** — rank designs by Protenix iPTM
    4. **Beam search** — guided search over denoising trajectories
    5. **Inverse folding** — re-sample sequences for the best backbone
    """)
    return


@app.cell
def _():
    mo.md("""
    ## Load target and models
    """)
    return


@app.cell
def _():
    # ── Load PDL1 target ──
    print("Loading PDL1 target...")
    structure = gemmi.read_structure("PDL1.pdb")
    structure.setup_entities()
    chain = structure[0][0]
    # PD-1 binding interface hotspots (0-indexed into chain):
    # Tyr56, Glu58, Asp61, Asn63, Gln66, Val68, Tyr81, Arg113, Ala121, Asp122, Tyr123
    hotspots = [(36, "ILE"), (38, "TYR"), (48, "GLN"), (97, "MET")]
    for _idx, _name in hotspots:
        assert chain[_idx].name.upper() == _name

    target_cond = load_target_cond(chain, hotspots=[idx for (idx, _) in hotspots])
    target_sequence = gemmi.one_letter_code([r.name for r in chain.get_polymer()])

    # ── Load models ──
    print("Loading denoiser + decoder...")
    _t1 = time.perf_counter()
    denoiser = load_denoiser()
    decoder = load_decoder()
    print(f"  Loaded in {time.perf_counter() - _t1:.1f}s")

    BINDER_LENGTH = 90
    mask = jnp.ones(BINDER_LENGTH, dtype=jnp.bool_)

    print(f"Target: {len(target_cond.seq)} residues — {target_sequence[:40]}...")
    return BINDER_LENGTH, decoder, denoiser, mask, target_cond, target_sequence


@app.cell
def _():
    mo.md("""
    ## 1. Single sample

    The simplest use: draw one binder from the flow matching model.
    `generate` integrates an SDE from pure noise to a backbone + latent,
    then the decoder produces all-atom coordinates and a sequence.
    """)
    return


@app.cell
def _(BINDER_LENGTH, decoder, denoiser, mask, target_cond):
    bb, lat = generate(
        denoiser,
        mask,
        jax.random.PRNGKey(0),
        target=target_cond,
    )
    dec = decoder(DecoderBatch(z_latent=lat, ca_coors=bb, mask=mask))
    single_sequence = "".join(AA_CODES[j] for j in np.array(dec.aatype))
    print(f"Sequence ({BINDER_LENGTH} residues): {single_sequence}")
    return (dec,)


@app.cell
def _(dec, target_cond):
    pdb_viewer(make_complex(dec, target_cond))
    return


@app.cell
def _():
    mo.md("""
    ## 2. Batched sampling

    `jax.vmap` over `generate` to produce many candidates in parallel.
    """)
    return


@app.cell
def _(denoiser, mask, target_cond):
    @eqx.filter_jit
    def sample_batch(key, n):
        """Generate n binders in parallel. Returns (bbs [n,N,3], lats [n,N,D])."""
        keys = jax.random.split(key, n)
        return jax.vmap(lambda k: generate(denoiser, mask, k, target=target_cond))(
            keys
        )

    return (sample_batch,)


@app.cell
def _(decoder, mask, sample_batch):
    N_SAMPLES = 8

    _t0 = time.perf_counter()
    batch_bbs, batch_lats = sample_batch(jax.random.PRNGKey(1), N_SAMPLES)
    jax.block_until_ready(batch_bbs)
    print(f"Generated {N_SAMPLES} samples in {time.perf_counter() - _t0:.1f}s")

    batch_decs = jax.vmap(
        lambda b, l: decoder(DecoderBatch(z_latent=l, ca_coors=b, mask=mask))
    )(batch_bbs, batch_lats)

    print(f"  bb shape: {batch_bbs.shape}, lat shape: {batch_lats.shape}")
    return N_SAMPLES, batch_bbs, batch_decs, batch_lats


@app.cell
def _():
    mo.md("""
    ## 3. Scoring with Protenix

    Score each sample using the Protenix2025 model.
    """)
    return


@app.cell
def _(BINDER_LENGTH, target_sequence):
    print("Building Protenix2025 iPTM loss...")
    _t0 = time.perf_counter()

    protenix = Protenix2025()
    features, atom_array = protenix.binder_features(
        binder_length=BINDER_LENGTH,
        chains=[
            TargetChain(
                target_sequence,
                use_msa=True,
            )
        ],
    )
    features = jax.tree.map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, features
    )
    loss_fn = protenix.build_loss(
        loss=sp.BinderTargetContact()
        + sp.WithinBinderContact()
        + 0.5 * sp.TargetBinderPAE()
        + 0.5 * sp.BinderTargetPAE()
        + 0.025 * sp.IPTMLoss()
        + 0.4 * sp.WithinBinderPAE()
        + 0.025 * sp.pTMEnergy()
        + 0.1 * sp.PLDDTLoss(),
        features=features,
        recycling_steps=3,
    )

    print(f"  Built in {time.perf_counter() - _t0:.1f}s")
    return atom_array, features, loss_fn, protenix


@app.cell
def _():
    import biotite

    return (biotite,)


@app.cell
def _(biotite):
    biotite.__version__
    return


@app.cell
def _(N_SAMPLES, batch_bbs, batch_lats, decoder, loss_fn, mask):
    def _score_one(bb, lat, key):
        dec = decoder(DecoderBatch(z_latent=lat, ca_coors=bb, mask=mask))
        seq_hard = jax.nn.one_hot(dec.seq_logits[..., JPC_TO_MOSAIC].argmax(-1), 20)
        loss, aux = loss_fn(seq_hard, key=key)
        return loss, aux


    _t0 = time.perf_counter()
    batch_losses, batch_aux = jax.vmap(_score_one)(
        batch_bbs,
        batch_lats,
        jax.random.split(jax.random.PRNGKey(1), N_SAMPLES),
    )
    jax.block_until_ready(batch_losses)
    print(f"Scored {N_SAMPLES} samples in {time.perf_counter() - _t0:.1f}s")
    return (batch_aux,)


@app.cell
def _(N_SAMPLES, batch_aux):
    _rows = [
        {
            jax.tree_util.keystr(k, simple=True, separator="."): float(v)
            for (k, v) in jax.tree_util.tree_leaves_with_path(
                jax.tree.map(lambda v: v[i], batch_aux)
            )
        }
        for i in range(N_SAMPLES)
    ]

    mo.ui.table(_rows, label="Samples")
    return


@app.cell
def _(atom_array, batch_decs, features, protenix):
    pred = protenix.predict(
        PSSM=jax.nn.one_hot(
            batch_decs.seq_logits[0][..., JPC_TO_MOSAIC].argmax(-1), 20
        ),
        features=features,
        writer=atom_array,
        key=jax.random.key(0),
        recycling_steps=3,
    )
    return (pred,)


@app.cell
def _(pred):
    pdb_viewer(pred.st)
    return


@app.cell
def _():
    mo.md("""
    ## 4. Beam search

    Guided search over denoising trajectories. At each checkpoint the
    algorithm branches, does a look-ahead rollout, scores, and prunes
    to the top beams. Returns every design ever evaluated — pick the
    best by reward.
    """)
    return


@app.cell
def _():
    config_form = (
        mo.md(
            """
            **Beam search parameters**

            Beam width (beams kept after pruning): {beam_width}

            Branches per beam per checkpoint: {n_branch}

            Number of checkpoint intervals: {n_intervals}

            Random seed: {seed}
            """
        )
        .batch(
            beam_width=mo.ui.slider(
                start=1, stop=8, value=4, step=1, show_value=True
            ),
            n_branch=mo.ui.slider(
                start=1, stop=8, value=4, step=1, show_value=True
            ),
            n_intervals=mo.ui.slider(
                start=1, stop=10, value=4, step=1, show_value=True
            ),
            seed=mo.ui.number(start=0, stop=999, value=0, step=1),
        )
        .form(submit_button_label="Run beam search")
    )
    config_form
    return (config_form,)


@app.cell
def _(config_form, decoder, denoiser, loss_fn, mask, target_cond):
    _v = config_form.value
    mo.stop(_v is None, mo.md("*Submit the form above to run beam search.*"))
    bw = _v["beam_width"]
    br = _v["n_branch"]
    n_iv = _v["n_intervals"]
    seed = _v["seed"]

    step_checkpoints = [int(round(i * 400 / n_iv)) for i in range(n_iv + 1)]

    mo.output.append(
        mo.md(
            f"Running beam search: **W={bw}, B={br}**, "
            f"checkpoints={step_checkpoints}, seed={seed}"
        )
    )

    _t0 = time.perf_counter()
    designs = beam_search(
        model=denoiser,
        decoder=decoder,
        loss_fn=loss_fn,
        mask=mask,
        key=jax.random.PRNGKey(seed),
        target=target_cond,
        step_checkpoints=step_checkpoints,
        beam_width=bw,
        n_branch=br,
    )
    designs = sorted(designs, key=lambda d: d.loss)
    elapsed = time.perf_counter() - _t0
    return designs, elapsed


@app.cell
def _(designs, elapsed):
    design_rows = [
        {
            jax.tree_util.keystr(k, simple=True, separator="."): float(v)
            for (k, v) in jax.tree_util.tree_leaves_with_path(d.aux)
        }
        | {"seq": "".join(MOSAIC_ORDER[i] for i in d.sequence)}
        for d in designs
    ]

    best_design = min(designs, key=lambda d: d.loss)

    mo.output.append(
        mo.md(
            f"Completed in **{elapsed:.1f}s** — "
            f"**{len(designs)}** designs evaluated, "
            f"best loss = **{float(best_design.loss):.4f}**"
        )
    )
    return best_design, design_rows


@app.cell
def _(design_rows):
    mo.ui.table(design_rows, label="All beam search designs (ranked by loss)")
    return


@app.cell
def _():
    mo.md("""
    ## Best beam search design
    """)
    return


@app.cell
def _(best_design, target_cond):
    pdb_viewer(make_complex(best_design.decoder_output, target_cond))
    return


@app.cell
def _():
    mo.md("""
    ## 5. Inverse folding

    Given the best backbone from beam search, re-sample only the latent
    space while keeping the backbone fixed. This is structure-conditioned
    inverse folding — finding new sequences that fold into the same structure.
    """)
    return


@app.cell
def _():
    from jproteina_complexa.flow_matching import (
        DenoiseState,
        init_noise,
        PRODUCTION_SAMPLING,
        predict_x1_from_v,
    )
    from jproteina_complexa.types import DenoiserBatch, NoisyState, Timesteps


    @eqx.filter_jit
    def inverse_fold(denoiser, decoder, bb_ca, mask, target, key):
        """Denoise latents from noise while keeping backbone fixed, then decode.

        Args:
            denoiser: denoiser model
            decoder: decoder model (latent → sequence + all-atom coords)
            bb_ca: backbone CA coordinates in Angstroms [N, 3]
            mask: boolean residue mask [N]
            target: TargetCond for the target protein
            key: PRNG key

        Returns:
            DecoderOutput with the inverse-folded sequence and coordinates.
        """
        bb_ca_nm = bb_ca / 10.0
        cfg = PRODUCTION_SAMPLING
        nsteps = cfg.nsteps
        ts_lat = cfg.local_latents.time_schedule(nsteps)
        mask_f = mask.astype(jnp.float32)

        k_noise, k_run = jax.random.split(key)
        state = init_noise(k_noise, 8, mask, cfg)
        state = DenoiseState(
            bb=bb_ca_nm,
            lat=state.lat,
            sc_bb=bb_ca_nm,
            sc_lat=state.sc_lat,
            key=k_run,
        )

        def body(carry):
            state, key, i = carry
            t_lat = ts_lat[i]
            dt_lat = ts_lat[i + 1] - t_lat

            out = denoiser(
                DenoiserBatch(
                    x_t=NoisyState(bb_ca=bb_ca_nm, local_latents=state.lat),
                    t=Timesteps(bb_ca=jnp.array(1.00), local_latents=t_lat),
                    mask=mask,
                    x_sc=NoisyState(bb_ca=state.sc_bb, local_latents=state.sc_lat),
                    target=target,
                )
            )

            sc_lat = predict_x1_from_v(state.lat, out.local_latents, t_lat)

            key, k_step = jax.random.split(key)
            lat = cfg.local_latents.step(
                state.lat,
                out.local_latents,
                t_lat,
                dt_lat,
                mask_f,
                k_step,
            )

            return (
                DenoiseState(
                    bb=bb_ca_nm,
                    lat=lat,
                    sc_bb=bb_ca_nm,
                    sc_lat=sc_lat,
                    key=key,
                ),
                key,
                i + 1,
            )

        state, _, _ = jax.lax.fori_loop(
            0,
            nsteps,
            lambda i, carry: body(carry),
            (state, k_run, jnp.int32(0)),
        )
        return decoder(
            DecoderBatch(
                z_latent=state.lat,
                ca_coors=state.bb * 10.0,
                mask=mask,
            )
        )

    return (inverse_fold,)


@app.cell
def _(
    BINDER_LENGTH,
    best_design,
    decoder,
    denoiser,
    inverse_fold,
    mask,
    target_cond,
):
    _t0 = time.perf_counter()
    inv_dec = inverse_fold(
        denoiser,
        decoder,
        best_design.bb,
        mask,
        target_cond,
        jax.random.PRNGKey(1),
    )
    jax.block_until_ready(inv_dec.aatype)
    print(f"Inverse folding took {time.perf_counter() - _t0:.1f}s")

    inv_sequence = "".join(AA_CODES[j] for j in np.array(inv_dec.aatype))
    orig_sequence = "".join(MOSAIC_ORDER[i] for i in np.array(best_design.sequence))
    _diff = "".join(
        "·" if a != b else "|" for a, b in zip(orig_sequence, inv_sequence)
    )
    _n_diff = _diff.count("·")

    mo.md(
        f"```\nOriginal:       {orig_sequence}\n"
        f"                {_diff}\n"
        f"Inverse-folded: {inv_sequence}\n```\n\n"
        f"**Mutations:** {_n_diff}/{BINDER_LENGTH}"
    )
    return (inv_dec,)


@app.cell
def _(inv_dec, target_cond):
    pdb_viewer(make_complex(inv_dec, target_cond))
    return


@app.cell
def _(atom_array, best_design, features, inv_dec, protenix):
    _t0 = time.perf_counter()

    _orig_pssm = jax.nn.one_hot(
        best_design.decoder_output.seq_logits[..., JPC_TO_MOSAIC].argmax(-1), 20
    )
    _inv_pssm = jax.nn.one_hot(
        inv_dec.seq_logits[..., JPC_TO_MOSAIC].argmax(-1), 20
    )

    inv_pred = protenix.predict(
        PSSM=_inv_pssm,
        features=features,
        writer=atom_array,
        key=jax.random.key(0),
        recycling_steps=3,
    )

    orig_pred = protenix.predict(
        PSSM=_orig_pssm,
        features=features,
        writer=atom_array,
        key=jax.random.key(0),
        recycling_steps=3,
    )


    print(f"Protenix evaluation took {time.perf_counter() - _t0:.1f}s")

    mo.md(
        f"| | iPTM | pLDDT |\n"
        f"|---|---|---|\n"
        f"| **Original** | {float(orig_pred.iptm):.4f} | {float(orig_pred.plddt.mean()):.4f} |\n"
        f"| **Inverse-folded** | {float(inv_pred.iptm):.4f} | {float(inv_pred.plddt.mean()):.4f} |"
    )
    return (inv_pred,)


@app.cell
def _(inv_pred):
    pdb_viewer(inv_pred.st)
    return


if __name__ == "__main__":
    app.run()
