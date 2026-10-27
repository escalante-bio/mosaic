import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import tempfile
    import time

    import equinox as eqx
    import gemmi
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np

    jax.config.update("jax_compilation_cache_dir", "/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )

    from atomworks.io.parser import parse
    from atomworks.io.utils.io_utils import to_cif_file
    from jrfd4_proteina import convert, inverse_fold, release_sampling_config, sample
    from jrfd4_proteina.search import Checkpoint, beam_search

    import mosaic.losses.structure_prediction as sp
    from mosaic.common import TOKENS
    from mosaic.models.protenix import Protenix2025
    from mosaic.models.rfd4_proteina import (
        LossReward,
        centered_ca,
        design_sequence,
        design_structure,
        inverse_folding_config,
        load_rfd4_proteina,
    )
    from mosaic.notebook_utils import pdb_viewer
    from mosaic.structure_prediction import TargetChain


@app.function
def binder_string(design, binder_length):
    return "".join(TOKENS[i] for i in np.array(design_sequence(design, binder_length).argmax(-1)))


@app.function
def take(tree, i):
    """The `i`-th element of a batched pytree."""
    return jax.tree.map(lambda x: x[i], tree)


@app.function
def aux_row(aux):
    return {
        jax.tree_util.keystr(k, simple=True, separator="."): float(v)
        for (k, v) in jax.tree_util.tree_leaves_with_path(aux)
    }


@app.cell
def _():
    mo.md("""
    # Protein Binder Design with RFD4-Proteina

    1. **Single sample** — generate one binder from noise
    2. **Batched sampling** — generate many binders in parallel with `jax.vmap`
    3. **Scoring** — rank designs by a Protenix loss
    4. **Beam search** — guided search over sampling trajectories
    5. **Inverse folding** — re-sample sequences for the best backbone
    """)
    return


@app.cell
def _():
    mo.md("""
    ## Load target and models

    RFD4-Proteina designs are specified with the reference's condition-spec language and featurized by
    its data pipeline. This mirrors the reference's
    [binder in 10 lines](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/docs/how-to/binder-in-10-lines.md)
    example: the target's backbone geometry (free pose: the model places the target too), its sequence,
    and hotspot atoms. Inputs must be mmCIF, so the PDB is converted with atomworks first.
    """)
    return


@app.cell
def _():
    BINDER_LENGTH = 90
    # PD-1 binding interface hotspots (author numbering), one side-chain tip atom each
    hotspots = [(54, "ILE", "CD1"), (56, "TYR", "OH"), (66, "GLN", "NE2"), (115, "MET", "SD")]

    _chain = gemmi.read_structure("PDL1.pdb")[0][0]
    for _num, _name, _ in hotspots:
        assert _chain[str(_num)][0].name == _name
    target_sequence = gemmi.one_letter_code([r.name for r in _chain])
    _first, _last = _chain[0].seqid.num, _chain[len(_chain) - 1].seqid.num

    _contig = f"{BINDER_LENGTH}, /0, A{_first}-{_last}"
    with tempfile.TemporaryDirectory() as _tmp:
        _target_cif = f"{_tmp}/PDL1.cif"
        to_cif_file(parse("PDL1.pdb")["asym_unit"][0], _target_cif)
        _spec = {
            "pdl1_binder": {
                "input": _target_cif,
                "contig": _contig,
                "conditions": {
                    "C_CRDGRP": {1: [{"select": "sel('A/*/*/[N,CA,C,O]')"}]},
                    "C_SEQ": {True: [{"select": "sel('A')"}]},
                    "C_HOT": {True: [{"select": f"sel('A/*/{n}/[{atom}]')"} for n, _, atom in hotspots]},
                },
            }
        }
        batch = convert.featurize(_spec, seed=0)[0]

        # Inverse folding, as the reference's `generation=inverse_folding`: no coordinate conditions, so the
        # fixed CA trace is the only geometry; the target sequence stays given and the binder's is designed.
        _if_spec = {
            "pdl1_inverse_fold": {
                "input": _target_cif,
                "contig": _contig,
                "conditions": {"C_SEQ": {True: [{"select": "sel('A')"}]}},
            }
        }
        if_batch = convert.featurize(_if_spec, seed=0)[0]

    print("Loading RFD4-Proteina...")
    _t1 = time.perf_counter()
    # bf16 mixed precision in the flow trunk, as the reference's default `bf16-mixed`
    model = load_rfd4_proteina().bfloat16()
    config = release_sampling_config()
    print(f"  Loaded in {time.perf_counter() - _t1:.1f}s")

    print(f"Target: {len(target_sequence)} residues — {target_sequence[:40]}...")
    print(f"Batch: {batch.padding_mask.shape[0]} tokens, {config.nsteps} sampling steps")
    return BINDER_LENGTH, batch, config, if_batch, model, target_sequence


@app.cell
def _():
    mo.md("""
    ## 1. Single sample

    `sample` integrates the flow from noise to CA coordinates + latents for binder and target, then the
    decoder produces all-atom coordinates and a sequence.
    """)
    return


@app.cell
def _(BINDER_LENGTH, batch, config, model):
    _t0 = time.perf_counter()
    single = sample(model, batch, config, key=jax.random.key(0))
    jax.block_until_ready(single.coords)
    print(f"Sampled in {time.perf_counter() - _t0:.1f}s (including compilation)")
    print(f"Sequence ({BINDER_LENGTH} residues): {binder_string(single, BINDER_LENGTH)}")
    return (single,)


@app.cell
def _(batch, single):
    pdb_viewer(design_structure(single, batch))
    return


@app.cell
def _():
    mo.md("""
    ## 2. Batched sampling

    `vmap` over keys to produce many candidates in parallel. The model is an argument of the jitted
    function, not a closure constant.
    """)
    return


@app.cell
def _():
    @eqx.filter_jit
    def sample_batch(model, batch, config, key, n):
        keys = jax.random.split(key, n)
        return jax.vmap(lambda k: sample(model, batch, config, key=k))(keys)

    return (sample_batch,)


@app.cell
def _(batch, config, model, sample_batch):
    N_SAMPLES = 8

    _t0 = time.perf_counter()
    batch_designs = sample_batch(model, batch, config, jax.random.key(1), N_SAMPLES)
    jax.block_until_ready(batch_designs.coords)
    print(f"Generated {N_SAMPLES} samples in {time.perf_counter() - _t0:.1f}s")
    print(f"  coords shape: {batch_designs.coords.shape}")
    return N_SAMPLES, batch_designs


@app.cell
def _():
    mo.md("""
    ## 3. Scoring with Protenix

    Score each sample using the Protenix2025 model. `LossReward` wraps a mosaic loss as
    `design -> (-loss, aux)` on the decoded binder sequence: the same function beam search maximizes.
    """)
    return


@app.cell
def _(BINDER_LENGTH, target_sequence):
    print("Building Protenix2025 loss...")
    _t0 = time.perf_counter()

    protenix = Protenix2025()
    features, atom_array = protenix.binder_features(
        binder_length=BINDER_LENGTH,
        chains=[TargetChain(target_sequence, use_msa=True)],
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
    reward = LossReward(loss_fn, BINDER_LENGTH)

    print(f"  Built in {time.perf_counter() - _t0:.1f}s")
    return atom_array, features, protenix, reward


@app.cell
def _(N_SAMPLES, batch_designs, reward):
    @eqx.filter_jit
    def score_batch(reward, designs, keys):
        return jax.vmap(lambda d, k: reward(d, key=k))(designs, keys)

    _t0 = time.perf_counter()
    batch_rewards, batch_aux = score_batch(
        reward, batch_designs, jax.random.split(jax.random.key(1), N_SAMPLES)
    )
    jax.block_until_ready(batch_rewards)
    print(f"Scored {N_SAMPLES} samples in {time.perf_counter() - _t0:.1f}s")
    return batch_aux, batch_rewards


@app.cell
def _(BINDER_LENGTH, N_SAMPLES, batch_aux, batch_designs, batch_rewards):
    _rows = sorted(
        (
            {"loss": -float(batch_rewards[i])}
            | aux_row(take(batch_aux, i))
            | {"seq": binder_string(take(batch_designs, i), BINDER_LENGTH)}
            for i in range(N_SAMPLES)
        ),
        key=lambda row: row["loss"],
    )
    best_sample = take(batch_designs, int(np.argmax(np.asarray(batch_rewards))))
    print(f"Best unguided sample: loss {_rows[0]['loss']:.4f}")

    mo.ui.table(_rows, label="Samples (ranked by loss)")
    return (best_sample,)


@app.cell
def _(BINDER_LENGTH, atom_array, best_sample, features, protenix):
    pred = protenix.predict(
        PSSM=design_sequence(best_sample, BINDER_LENGTH),
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

    Guided search over sampling trajectories. At each checkpoint every survivor branches, the branches
    are rolled out to t=1, decoded and scored, and the best continue. Every evaluated candidate is kept
    in `result.checkpoints`; pick the best by reward.
    """)
    return


@app.cell
def _():
    BEAM_WIDTH = 4  # beams kept after pruning
    N_BRANCH = 4  # branches per beam per checkpoint
    N_INTERVALS = 4  # checkpoint intervals over the sampling schedule
    SEED = 0
    return BEAM_WIDTH, N_BRANCH, N_INTERVALS, SEED


@app.cell
def _(BEAM_WIDTH, N_BRANCH, N_INTERVALS, SEED, batch, config, model, reward):
    checkpoint_steps = [round((i + 1) * config.nsteps / N_INTERVALS) for i in range(N_INTERVALS)]
    print(
        f"Running beam search: W={BEAM_WIDTH}, B={N_BRANCH}, "
        f"checkpoints={checkpoint_steps}, seed={SEED}"
    )

    _t0 = time.perf_counter()
    result = beam_search(
        model,
        batch,
        config,
        [Checkpoint(s, reward) for s in checkpoint_steps],
        key=jax.random.key(SEED),
        beam_width=BEAM_WIDTH,
        n_branch=N_BRANCH,
    )
    elapsed = time.perf_counter() - _t0
    return elapsed, result


@app.cell
def _(BINDER_LENGTH, elapsed, result):
    candidates = [
        (scored.step, float(scored.rewards[i]), take(scored.aux, i), take(scored.designs, i))
        for scored in result.checkpoints
        for i in range(scored.rewards.shape[0])
    ]
    candidates.sort(key=lambda c: -c[1])
    design_rows = [
        {"step": step, "loss": -r} | aux_row(aux) | {"seq": binder_string(d, BINDER_LENGTH)}
        for step, r, aux, d in candidates
    ]
    best_design = candidates[0][3]

    mo.output.append(
        mo.md(
            f"Completed in **{elapsed:.1f}s** — "
            f"**{len(candidates)}** designs evaluated, "
            f"best loss = **{-candidates[0][1]:.4f}**"
        )
    )
    print(f"Beam search: {elapsed:.1f}s, {len(candidates)} designs, best loss {-candidates[0][1]:.4f}")
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
def _(batch, best_design):
    pdb_viewer(design_structure(best_design, batch))
    return


@app.cell
def _():
    mo.md("""
    ## 5. Inverse folding

    Given the best design's CA trace (binder and target), re-generate only the latents with the trace
    held fixed, then decode: structure-conditioned inverse folding. As in the reference's
    `generation=inverse_folding`, the trace is centered on the structure's heavy-atom centroid, every step
    runs 3 trunk passes (`recycle_strategy=medium/flat`), and the batch carries no coordinate conditions.
    """)
    return


@app.cell
def _(BINDER_LENGTH, best_design, config, if_batch, model):
    _t0 = time.perf_counter()
    inv_design = inverse_fold(
        model,
        if_batch,
        inverse_folding_config(config),
        centered_ca(best_design.coords, best_design.atom_mask),
        key=jax.random.key(1),
    )
    jax.block_until_ready(inv_design.coords)
    print(f"Inverse folding took {time.perf_counter() - _t0:.1f}s")

    inv_sequence = binder_string(inv_design, BINDER_LENGTH)
    orig_sequence = binder_string(best_design, BINDER_LENGTH)
    _diff = "".join("·" if a != b else "|" for a, b in zip(orig_sequence, inv_sequence))
    _n_diff = _diff.count("·")
    print(f"Inverse folding mutations: {_n_diff}/{BINDER_LENGTH}")

    mo.md(
        f"```\nOriginal:       {orig_sequence}\n"
        f"                {_diff}\n"
        f"Inverse-folded: {inv_sequence}\n```\n\n"
        f"**Mutations:** {_n_diff}/{BINDER_LENGTH}"
    )
    return (inv_design,)


@app.cell
def _(if_batch, inv_design):
    pdb_viewer(design_structure(inv_design, if_batch))
    return


@app.cell
def _(BINDER_LENGTH, atom_array, best_design, features, inv_design, protenix):
    _t0 = time.perf_counter()

    inv_pred = protenix.predict(
        PSSM=design_sequence(inv_design, BINDER_LENGTH),
        features=features,
        writer=atom_array,
        key=jax.random.key(0),
        recycling_steps=3,
    )

    orig_pred = protenix.predict(
        PSSM=design_sequence(best_design, BINDER_LENGTH),
        features=features,
        writer=atom_array,
        key=jax.random.key(0),
        recycling_steps=3,
    )

    print(f"Protenix evaluation took {time.perf_counter() - _t0:.1f}s")
    print(f"iPTM original {float(orig_pred.iptm):.4f} inverse-folded {float(inv_pred.iptm):.4f}")

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
