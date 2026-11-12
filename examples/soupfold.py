import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import copy

    import equinox as eqx
    import jax
    import marimo as mo
    import numpy as np
    from jaxtyping import Array, Float
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
    # [SoupFold](https://arxiv.org/abs/2609.15552)

    SoupFold is a model stacking/bagging/averaging approach that uses small MLPs to combine the hidden states of multiple folding models.

    In the setup from the paper the pair representations are averaged after each model completes its recycling iterations.

    In `mosaic,` it's very easy to try variations on this approach, for instance averaging model representations after each recycling iteration. In this small example we show variations in prediction and performance depending on when we ensemble hidden states using an MLP trained on a handful of structures.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.mermaid(
        """
        flowchart LR
            F[features] --> E[embed]
            subgraph RECYCLE["trunk pass × (1 + recycling steps)"]
                T[trunk pass] --> S[("state: single s, pair z")]
                S -. recycled .-> T
            end
            E --> T
            subgraph SAMPLES["× samples"]
                D[diffusion] --> X[structure] --> C[confidence]
            end
            S --> D
            S --> C
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
        [TargetChain(c["sequence"], use_msa=False) for c in CHAINS],
        key=jax.random.key(0),
    )
    opendde_features, _ = opendde.target_only_features(
        [TargetChain(c["sequence"], use_msa=True) for c in CHAINS],
        key=jax.random.key(0),
    )
    return CHAINS, esmfold_features, opendde_features


@app.cell(hide_code=True)
def _():
    mo.mermaid(
        """
        flowchart LR
            F1["features (with MSA)"] --> E1[embed]
            subgraph OPENDDE["OpenDDE: trunk pass × (1 + recycling steps)"]
                T1[trunk pass] --> S1[("state: single s, pair z")]
                S1 -. recycled .-> T1
            end
            E1 --> T1

            F2["features (single sequence)"] --> E2[embed]
            subgraph ESMFOLD2["ESMFold2: trunk pass × (1 + recycling steps)"]
                T2[trunk pass] --> S2[("state: single s, pair z")]
                S2 --> A(("average pair z"))
                A -. recycled .-> T2
            end
            E2 --> T2

            S1 -- "MLP (pair z)" --> A

            subgraph SAMPLES["× samples"]
                D[diffusion] --> X[structure] --> C[confidence]
            end
            A --> D
            A --> C
        """
    )
    return


@app.class_definition
class PairMLP(eqx.Module):
    """MLP mapping pair embedding from one model to another"""

    source_mean: Float[Array, "D "]
    source_std: Float[Array, "D "]
    weight_0: Float[Array, "H D"]
    bias_0: Float[Array, "H "]
    weight_1: Float[Array, "H H"]
    bias_1: Float[Array, "H "]
    weight_2: Float[Array, "O H"]
    bias_2: Float[Array, "O "]
    destination_std: Float[Array, "O "]
    destination_mean: Float[Array, "O "]

    def __call__(self, z: Float[Array, "T T H"]):
        x = (z - self.source_mean) / self.source_std
        x = jax.nn.relu(x @ self.weight_0.T + self.bias_0)
        x = jax.nn.relu(x @ self.weight_1.T + self.bias_1)
        x = x @ self.weight_2.T + self.bias_2
        return x * self.destination_std + self.destination_mean


@app.cell
def _():
    pair_map = PairMLP(
        **load_file(
            mo.notebook_dir()
            / "weights"
            / "opendde_v1_to_esmfold2_experimental_fast.safetensors"
        )
    )
    return (pair_map,)


@app.function
@eqx.filter_jit
def recycle(
    model,
    features,
    passes,
    key,
    pair_map=None,
    teacher_pairs=None,
    weights=None,
):
    """The embedded inputs and the trunk state after each pass. With `teacher_pairs`
    `[passes, T, T, C]` and `weights` `[passes]`, each pass's pair `z` becomes
    `(z + w * pair_map(teacher)) / (1 + w)` before the next pass recycles it (`w = 0`
    leaves it unchanged)."""
    k_prepare, k_init, k_passes = jax.random.split(key, 3)
    features, inputs = model.prepare(features, None, key=k_prepare)

    def one_pass(state, key_and_teacher_and_weight):
        k, teacher, w = key_and_teacher_and_weight
        state = model.step(inputs, state, key=k)
        if teacher is not None:
            mixed = (state.z + w * pair_map(teacher)) / (1 + w)
            state = eqx.tree_at(lambda s: s.z, state, mixed)
        return state, state

    keys = jax.random.split(k_passes, passes)
    _, states = jax.lax.scan(
        one_pass,
        model.init_state(inputs, key=k_init),
        (keys, teacher_pairs, weights),
    )
    return inputs, states


@app.cell
def _(esmfold, esmfold_features, opendde, opendde_features, pair_map):
    PASSES = 10
    opendde_inputs, opendde_states = recycle(
        opendde, opendde_features, PASSES, jax.random.key(1)
    )
    esmfold_inputs, alone_states = recycle(
        esmfold, esmfold_features, PASSES, jax.random.key(1)
    )
    # How strongly each of ESMFold2's passes mixes in OpenDDE's pair (translated by the MLP).
    MIXING = {
        "after every pass": np.ones(PASSES),
        "after first pass": jax.nn.one_hot(0, PASSES),
        "after last pass (SoupFold)": jax.nn.one_hot(PASSES - 1, PASSES),
        "after penultimate pass": jax.nn.one_hot(PASSES - 2, PASSES),
    }
    mixed_states = {
        name: recycle(
            esmfold,
            esmfold_features,
            PASSES,
            jax.random.key(1),
            pair_map,
            opendde_states.z,
            weights,
        )[1]
        for name, weights in MIXING.items()
    }
    return (
        alone_states,
        esmfold_inputs,
        mixed_states,
        opendde_inputs,
        opendde_states,
    )


@app.function
@eqx.filter_jit
def decode(model, features, inputs, states, keys):
    """A structure for each key, sampled from the last of a model's trunk `states` (neither
    trunk here has a final step, so that state is what the heads read)."""
    trunk = Trunk(
        features=features,
        inputs=inputs,
        state=jax.tree.map(lambda x: x[-1], states),
    )
    sampler = model.sampler()
    return jax.vmap(
        lambda k: model.sample_structure(trunk, sampler=sampler, key=k)
    )(keys)


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
    NUM_SAMPLES = 2  # structures per variant
    _keys = jax.random.split(jax.random.key(2), NUM_SAMPLES)

    def _esmfold(states):
        return decode(esmfold, esmfold_features, esmfold_inputs, states, _keys)

    outputs = jax.tree.map(
        np.array,
        {
            "ESMFold2 alone": _esmfold(alone_states),
            **{
                f"ESMFold2, mixed {name}": _esmfold(states)
                for name, states in mixed_states.items()
            },
            "OpenDDE alone": decode(
                opendde, opendde_features, opendde_inputs, opendde_states, _keys
            ),
        },
    )
    return (outputs,)


@app.function
def dockq(reference, structure, chains):
    """Mean antibody-antigen DockQ of a predicted `gemmi.Structure` against the reference
    structure. The predicted atoms are copied onto a copy of the reference (which may lack
    residues), so the two have the same atoms."""
    model = copy.deepcopy(reference)
    for c, predicted in zip(chains, structure[0]):
        residues = [
            {atom.name: atom.pos.tolist() for atom in residue}
            for residue in predicted
        ]
        target = model.chains[c["id"]]
        target.coords.fill(0)
        target.mask.fill(False)
        for i, position in enumerate(target.ridx):
            atoms = residues[int(position) - 1]
            for j, name in enumerate(target.aname[i]):
                if name in atoms:
                    target.coords[i, j] = atoms[name]
                    target.mask[i, j] = True

    antibody = {c["id"] for c in chains if c["role"] == "antibody"}
    scores = dockQ(reference, model)
    return float(
        np.mean(
            [
                v["DockQ"]
                for (a, b), v in scores.items()
                if (a in antibody) != (b in antibody)
            ]
        )
    )


@app.cell
def _(CHAINS, outputs):
    # DockQ against the crystal structure.
    _reference = Structure.from_mmcif(
        str(mo.notebook_dir() / "data" / "9HO4_H057KL0000.cif")
    )
    structures = {
        name: [
            jax.tree.map(lambda x: x[i], output).to_structure()
            for i in range(output.plddt.shape[0])
        ]
        for name, output in outputs.items()
    }
    scores = {
        name: np.array(
            [dockq(_reference, structure, CHAINS) for structure in samples]
        )
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
            for name in sorted(outputs, key=lambda name: -scores[name].mean())
        ],
        selection=None,
    )
    return scores, structures


@app.cell
def _(outputs):
    shown = mo.ui.dropdown(
        list(outputs),
        value="ESMFold2, mixed after every pass",
        label="Structure",
    )
    shown
    return (shown,)


@app.cell
def _(scores, shown, structures):
    # The best of the samples.
    pdb_viewer(structures[shown.value][int(scores[shown.value].argmax())])
    return


if __name__ == "__main__":
    app.run()
