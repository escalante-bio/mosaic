import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import itertools

    import equinox as eqx
    import gemmi
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    from safetensors.numpy import load_file
    from tinyprot.metrics import dockQ
    from tinyprot.structure import Structure

    from mosaic.models.esmfold2 import ESMFold2
    from mosaic.models.opendde import OpenDDE
    from mosaic.notebook_utils import pdb_viewer
    from mosaic.structure_prediction import TargetChain, Trunk


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Mixing OpenDDE's pair embedding into ESMFold2

    Single-sequence ESMFold2 docks this nanobody onto its antigen wrongly; OpenDDE, with
    an MSA, gets it right. A small MLP translates OpenDDE's pair embedding into ESMFold2's
    pair space, and averaging the two after every trunk pass lets ESMFold2 get it right too.

    ## Recycling

    A model embeds its inputs once, then applies its trunk repeatedly to its own output.
    The state after the last pass is what the diffusion module and the heads read.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.mermaid(
        """
        flowchart LR
            F[features] --> E[embed]
            E --> T[trunk pass]
            T -- "state (single s, pair z), recycled" --> T
            T --> D[diffusion module and heads]
            D --> X[structure, confidence]
        """
    )
    return


@app.cell
def _():
    esmfold = ESMFold2.experimental_fast()
    opendde = OpenDDE.v1()
    return esmfold, opendde


@app.cell
def _(esmfold, opendde):
    # 9HO4: a nanobody (H) bound to frataxin (A).
    CHAINS = [
        {
            "id": "H",
            "role": "antibody",
            "sequence": (
                "VQLQESGGGLVQPGGSLRLSCAASGSIFSTDAMGWYRQFPGKNRKLVARISSGDYGDYTNYADSVKGRFTISRDNAKNTV"
                "YLQMNSLKPEDTAVYYCNSVPPAPGEYGRDYWGAGTQVTVSSAAAYPY"
            ),
        },
        {
            "id": "A",
            "role": "antigen",
            "sequence": (
                "TTYERLAEETLDSLAEFFEDLADKPYTFEDYDVSFGSGVLTVKLGGDLGTYVINKQTPNKQIWLSSPSSGPKRYDWTGKN"
                "WVYSHDGVSLHELLAAELTKALKTKLDLSSLAYS"
            ),
        },
    ]
    esmfold_features, _ = esmfold.target_only_features(
        [TargetChain(c["sequence"], use_msa=False) for c in CHAINS], key=jax.random.key(0)
    )
    opendde_features, _ = opendde.target_only_features(
        [TargetChain(c["sequence"], use_msa=True) for c in CHAINS], key=jax.random.key(0)
    )
    return CHAINS, esmfold_features, opendde_features


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Mixing

    OpenDDE recycles on its own. After each of ESMFold2's passes, its pair is replaced by
    the average of itself and that pass's OpenDDE pair, translated by the MLP; the next
    pass recycles the average.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.mermaid(
        """
        flowchart LR
            subgraph OpenDDE["OpenDDE (with MSA)"]
                direction LR
                O1[pass 1] --> O2[pass 2] --> O3[pass ...]
            end
            subgraph ESMFold2["ESMFold2 (single sequence)"]
                direction LR
                E1[pass 1] --> A1((average)) --> E2[pass 2] --> A2((average)) --> E3[pass ...]
            end
            O1 -- MLP --> A1
            O2 -- MLP --> A2
            E3 --> D[ESMFold2's diffusion module and heads]
        """
    )
    return


@app.cell
def _():
    # The MLP (two hidden layers) and the per-channel normalization of both pair spaces.
    pair_map = {
        name: jnp.asarray(array)
        for name, array in load_file(
            mo.notebook_dir() / "weights" / "opendde_v1_to_esmfold2_experimental_fast.safetensors"
        ).items()
    }
    return (pair_map,)


@app.function
def translate(pair_map, z):
    """An OpenDDE pair embedding `[T, T, 384]` in ESMFold2's pair space `[T, T, 256]`."""
    x = (z - pair_map["source_mean"]) / pair_map["source_std"]
    x = jax.nn.relu(x @ pair_map["weight_0"].T + pair_map["bias_0"])
    x = jax.nn.relu(x @ pair_map["weight_1"].T + pair_map["bias_1"])
    x = x @ pair_map["weight_2"].T + pair_map["bias_2"]
    return x * pair_map["destination_std"] + pair_map["destination_mean"]


@app.function
@eqx.filter_jit
def recycle(model, features, passes, key, pair_map=None, teacher_pairs=None):
    """The embedded inputs and the trunk state after each pass. With `teacher_pairs`
    `[passes, T, T, C]`, each pass's pair is averaged with that pass's translated teacher
    pair before the next pass recycles it."""
    k_prepare, k_init, k_passes = jax.random.split(key, 3)
    features, inputs = model.prepare(features, None, key=k_prepare)

    def one_pass(state, key_and_teacher):
        k, teacher = key_and_teacher
        state = model.step(inputs, state, key=k)
        if teacher is not None:
            mixed = (state.z + translate(pair_map, teacher)) / 2
            state = eqx.tree_at(lambda s: s.z, state, mixed)
        return state, state

    keys = jax.random.split(k_passes, passes)
    _, states = jax.lax.scan(one_pass, model.init_state(inputs, key=k_init), (keys, teacher_pairs))
    return inputs, states


@app.cell
def _(esmfold, esmfold_features, opendde, opendde_features, pair_map):
    PASSES = 5
    opendde_inputs, opendde_states = recycle(opendde, opendde_features, PASSES, jax.random.key(1))
    esmfold_inputs, alone_states = recycle(esmfold, esmfold_features, PASSES, jax.random.key(1))
    # The same passes and keys, mixing after each one.
    _, mixed_states = recycle(
        esmfold, esmfold_features, PASSES, jax.random.key(1), pair_map, opendde_states.z
    )
    return alone_states, esmfold_inputs, mixed_states, opendde_inputs, opendde_states


@app.function
@eqx.filter_jit
def decode(model, features, inputs, states, keys):
    """A structure for each key, sampled from the last of a model's trunk `states` (neither
    trunk here has a final step, so that state is what the heads read)."""
    trunk = Trunk(features=features, inputs=inputs, state=jax.tree.map(lambda x: x[-1], states))
    sampler = model.sampler()
    return jax.lax.map(lambda k: model.sample_structure(trunk, sampler=sampler, key=k), keys)


@app.cell
def _(
    alone_states,
    esmfold,
    esmfold_features,
    esmfold_inputs,
    mixed_states,
    opendde,
    opendde_features,
    opendde_inputs,
    opendde_states,
):
    # Five samples each, from the same noise.
    _keys = jax.random.split(jax.random.key(2), 5)
    outputs = {
        "ESMFold2 alone": decode(esmfold, esmfold_features, esmfold_inputs, alone_states, _keys),
        "ESMFold2, mixed after every pass": decode(
            esmfold, esmfold_features, esmfold_inputs, mixed_states, _keys
        ),
        "OpenDDE alone": decode(opendde, opendde_features, opendde_inputs, opendde_states, _keys),
    }
    return (outputs,)


@app.function(hide_code=True)
def dockq(reference_path, structure, chains):
    """Mean antibody-antigen DockQ of a predicted `gemmi.Structure` against the reference
    structure, over the chain assignments that match `chains`' sequences."""
    reference = Structure.from_mmcif(str(reference_path))
    predicted = {
        chain.name: [
            (
                gemmi.find_tabulated_residue(residue.name).one_letter_code.upper(),
                {atom.name: atom.pos.tolist() for atom in residue},
            )
            for residue in chain
        ]
        for chain in structure[0]
    }
    sequences = {name: "".join(code for code, _ in rs) for name, rs in predicted.items()}
    antibody = {c["id"] for c in chains if c["role"] == "antibody"}
    best = 0.0
    for permutation in itertools.permutations(predicted):
        mapping = dict(zip([c["id"] for c in chains], permutation))
        if any(sequences[mapping[c["id"]]] != c["sequence"] for c in chains):
            continue
        model = copy.deepcopy(reference)
        for c in chains:
            target = model.chains[c["id"]]
            target.coords.fill(0)
            target.mask.fill(False)
            residues = predicted[mapping[c["id"]]]
            for i, position in enumerate(target.ridx):
                atoms = residues[int(position) - 1][1]
                for j, atom in enumerate(target.aname[i]):
                    if atom in atoms:
                        target.coords[i, j] = atoms[atom]
                        target.mask[i, j] = True
        scores = dockQ(reference, model, mapping={k: k for k in reference.chains})
        interface = [
            v["DockQ"] for k, v in scores.items() if (k[0] in antibody) != (k[1] in antibody)
        ]
        best = max(best, float(np.mean(interface)))
    return best


@app.cell
def _(CHAINS, outputs):
    # DockQ against the crystal structure.
    _reference = mo.notebook_dir() / "data" / "9HO4_H057KL0000.cif"
    structures = {
        name: [
            jax.tree.map(lambda x: x[i], output).to_structure()
            for i in range(output.plddt.shape[0])
        ]
        for name, output in outputs.items()
    }
    scores = {
        name: np.array([dockq(_reference, structure, CHAINS) for structure in samples])
        for name, samples in structures.items()
    }
    mo.ui.table(
        [
            {
                "": name,
                "mean DockQ": round(float(scores[name].mean()), 2),
                "worst": round(float(scores[name].min()), 2),
                "best": round(float(scores[name].max()), 2),
                "mean pLDDT": round(float(outputs[name].plddt.mean()), 2),
            }
            for name in outputs
        ],
        selection=None,
    )
    return scores, structures


@app.cell
def _(outputs):
    shown = mo.ui.dropdown(
        list(outputs), value="ESMFold2, mixed after every pass", label="Structure"
    )
    shown
    return (shown,)


@app.cell
def _(scores, shown, structures):
    # The best of the five samples.
    pdb_viewer(structures[shown.value][int(scores[shown.value].argmax())])
    return


if __name__ == "__main__":
    app.run()
