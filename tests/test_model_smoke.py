"""Integration smoke tests for model paths used by the example notebooks.

These tests load real checkpoints and may download weights on first use. Run
them explicitly with ``uv run pytest -m model_smoke``.
"""

from __future__ import annotations

import gc
from importlib import import_module
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import gemmi
import optax
import pytest

from mosaic.common import LossTerm
from mosaic.losses.structure_prediction import StructureModelOutput
from mosaic.structure_prediction import (
    StructurePrediction,
    StructurePredictionModel,
    TargetChain,
)


# Imports stay lazy so normal test collection does not initialize every model
# stack. The matrix covers every common-interface factory used in examples.
MODEL_FACTORIES = [
    pytest.param(("mosaic.models.af2", "AlphaFold2"), id="alphafold2"),
    pytest.param(("mosaic.models.boltz1", "Boltz1"), id="boltz1"),
    pytest.param(("mosaic.models.boltz2", "Boltz2"), id="boltz2"),
    pytest.param(("mosaic.models.protenix", "ProtenixMini"), id="protenix-mini"),
    pytest.param(("mosaic.models.protenix", "ProtenixBase"), id="protenix-base"),
    pytest.param(("mosaic.models.protenix", "Protenix2025"), id="protenix-2025"),
    pytest.param(("mosaic.models.protenix", "ProtenixV2"), id="protenix-v2"),
    pytest.param(("mosaic.models.of3", "OF3"), id="openfold3"),
    pytest.param(("mosaic.models.esmfold2", "ESMFold2Fast"), id="esmfold2-fast"),
    pytest.param(
        ("mosaic.models.esmfold2", "ESMFold2ExperimentalFast"),
        id="esmfold2-experimental-fast",
    ),
    pytest.param(
        ("mosaic.models.esmfold2", "ESMFold2ExperimentalFast2025"),
        id="esmfold2-experimental-fast-2025",
    ),
    pytest.param(("mosaic.models.promera", "JPromeraModel"), id="promera"),
    pytest.param(("mosaic.models.opendde", "OpenDDEModelV1"), id="opendde-v1"),
    pytest.param(("mosaic.models.opendde", "OpenDDEModelAbag"), id="opendde-abag"),
]


@pytest.fixture(params=MODEL_FACTORIES, scope="module")
def structure_model(request):
    module_name, factory_name = request.param
    factory = getattr(import_module(module_name), factory_name)
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        model = factory()
        yield model

        # Avoid retaining several large checkpoints in one pytest process.
        del model
    jax.clear_caches()
    gc.collect()


@pytest.fixture(scope="module")
def boltzgen_assets():
    module = import_module("mosaic.models.boltzgen")
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        model = module.load_boltzgen()
        yaml = """
entities:
  - protein:
      id: B
      sequence: 8
      secondary_structure: HHHHHHHH
"""
        features, writer = module.load_features_and_structure_writer(yaml)
        yield model, features, writer

        del model, features, writer
    jax.clear_caches()
    gc.collect()


@pytest.mark.slow
@pytest.mark.model_smoke
def test_boltzgen_loads_and_featurizes(boltzgen_assets):
    model, features, writer = boltzgen_assets

    assert jax.tree.leaves(model)
    assert jax.tree.leaves(features)
    assert callable(writer)


@pytest.mark.slow
@pytest.mark.model_forward
@pytest.mark.xfail(
    jax.default_backend() == "cpu",
    reason="BoltzGen diffusion currently produces non-finite coordinates on CPU",
    strict=True,
)
def test_boltzgen_samples_and_writes_structure(boltzgen_assets):
    module = import_module("mosaic.models.boltzgen")
    model, features, writer = boltzgen_assets
    sampler = module.Sampler.from_features(
        model=model,
        features=features,
        recycling_steps=1,
        deterministic=True,
        key=jax.random.key(10),
    )
    coordinates = sampler(
        structure_module=model.structure_module,
        num_sampling_steps=10,
        step_scale=jnp.asarray(2.0),
        noise_scale=jnp.asarray(0.88),
        key=jax.random.key(11),
    )
    structure = writer(coordinates)
    assert np.asarray(coordinates).shape[-1] == 3
    assert np.isfinite(np.asarray(coordinates)).all()
    assert len(structure) == 1
    assert sum(len(chain) for chain in structure[0]) == 8


@pytest.mark.slow
@pytest.mark.model_smoke
def test_model_loads(structure_model):
    assert isinstance(structure_model, StructurePredictionModel)


@pytest.mark.slow
@pytest.mark.model_smoke
def test_model_featurizes_single_sequence(structure_model):
    chain = TargetChain(sequence="ACDEFGHIKLMNPQRSTVWY", use_msa=False)
    features, _writer = structure_model.target_only_features([chain])

    assert features is not None
    assert jax.tree.leaves(features), "featurization returned an empty pytree"


@pytest.mark.slow
@pytest.mark.model_smoke
def test_model_featurizes_binder_and_target(structure_model):
    target = TargetChain(sequence="ACDEFGHIKLMNPQRSTVWY", use_msa=False)
    features, _writer = structure_model.binder_features(
        binder_length=8,
        chains=[target],
    )

    assert features is not None
    assert jax.tree.leaves(features), "featurization returned an empty pytree"


def _template_chain() -> gemmi.Chain:
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for index, residue_name in enumerate(("ALA", "CYS", "ASP"), start=1):
        residue = gemmi.Residue()
        residue.name = residue_name
        residue.seqid = gemmi.SeqId(index, " ")
        for atom_index, atom_name in enumerate(("N", "CA", "C", "O")):
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element(atom_name[0])
            atom.pos = gemmi.Position(float(index), float(atom_index), 0.0)
            residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    return structure[0][0]


@pytest.mark.slow
@pytest.mark.model_smoke
def test_notebook_models_featurize_template_chain(structure_model):
    supported = {"AlphaFold2", "Boltz2", "OF3", "Protenix"}
    if type(structure_model).__name__ not in supported:
        pytest.skip("this notebook model does not use template-chain features")
    target = TargetChain(
        sequence="ACD", use_msa=False, template_chain=_template_chain()
    )
    features, _writer = structure_model.binder_features(2, [target])
    assert jax.tree.leaves(features)


class MeanConfidenceLoss(LossTerm):
    def __call__(self, sequence, output, key):
        del sequence, key
        value = -jnp.mean(output.plddt)
        return value, {"mean_plddt": -value}


@pytest.fixture(scope="module")
def forward_inputs(structure_model):
    features, writer = structure_model.binder_features(
        binder_length=2,
        chains=[TargetChain(sequence="ACD", use_msa=False)],
    )
    pssm = jax.nn.one_hot(jnp.asarray([0, 7]), 20)
    sampling_steps = None if type(structure_model).__name__ == "AlphaFold2" else 2
    return SimpleNamespace(
        features=features,
        writer=writer,
        pssm=pssm,
        sampling_steps=sampling_steps,
    )


@pytest.fixture(scope="module")
def built_loss(structure_model, forward_inputs):
    return structure_model.build_loss(
        loss=MeanConfidenceLoss(),
        features=forward_inputs.features,
        recycling_steps=1,
        sampling_steps=forward_inputs.sampling_steps,
    )


@pytest.fixture(scope="module")
def model_output_result(structure_model, forward_inputs):
    return structure_model.model_output(
        PSSM=forward_inputs.pssm,
        features=forward_inputs.features,
        recycling_steps=1,
        sampling_steps=forward_inputs.sampling_steps,
        key=jax.random.key(1),
    )


@pytest.fixture(scope="module")
def prediction_result(structure_model, forward_inputs):
    return structure_model.predict(
        PSSM=forward_inputs.pssm,
        features=forward_inputs.features,
        writer=forward_inputs.writer,
        recycling_steps=1,
        sampling_steps=forward_inputs.sampling_steps,
        key=jax.random.key(2),
    )


@pytest.fixture(scope="module")
def evaluated_loss(built_loss, forward_inputs):
    return eqx.filter_jit(built_loss)(forward_inputs.pssm, key=jax.random.key(3))


@pytest.fixture(scope="module")
def loss_and_gradient(built_loss, forward_inputs):
    value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(built_loss, has_aux=True))
    return value_and_grad(forward_inputs.pssm, key=jax.random.key(4))


@pytest.fixture(params=["binder", "target_only"], scope="module")
def multichain_prediction(structure_model, request):
    target_sequences = ("AC", "DEF")
    targets = [
        TargetChain(sequence=sequence, use_msa=False) for sequence in target_sequences
    ]
    if request.param == "binder":
        features, writer = structure_model.binder_features(
            binder_length=2,
            chains=targets,
        )
        pssm = jax.nn.one_hot(jnp.asarray([0, 7]), 20)
        expected_names = (
            ("ALA", "GLY"),
            ("ALA", "CYS"),
            ("ASP", "GLU", "PHE"),
        )
    else:
        features, writer = structure_model.target_only_features(targets)
        pssm = None
        expected_names = (
            ("ALA", "CYS"),
            ("ASP", "GLU", "PHE"),
        )
    sampling_steps = None if type(structure_model).__name__ == "AlphaFold2" else 2
    prediction = structure_model.predict(
        PSSM=pssm,
        features=features,
        writer=writer,
        recycling_steps=1,
        sampling_steps=sampling_steps,
        key=jax.random.key(5),
    )
    return SimpleNamespace(
        prediction=prediction,
        mode=request.param,
        expected_names=expected_names,
    )


def _protein_chains(structure):
    return [list(chain) for chain in structure[0]]


def _atom_coordinates(chains):
    return {
        (chain_index, residue_index, atom.name): np.asarray(
            [atom.pos.x, atom.pos.y, atom.pos.z]
        )
        for chain_index, chain in enumerate(chains)
        for residue_index, residue in enumerate(chain)
        for atom in residue
    }


@pytest.mark.slow
@pytest.mark.model_forward
def test_model_builds_loss(built_loss):
    assert isinstance(built_loss, LossTerm)


@pytest.mark.slow
@pytest.mark.model_forward
def test_model_output_runs(model_output_result):
    assert isinstance(model_output_result, StructureModelOutput)
    assert model_output_result.plddt.shape == (5,)
    assert model_output_result.pae.shape == (5, 5)
    assert model_output_result.atom37_coords.shape == (5, 37, 3)
    assert np.isfinite(np.asarray(model_output_result.plddt)).all()
    assert np.isfinite(np.asarray(model_output_result.pae)).all()
    assert np.isfinite(np.asarray(model_output_result.atom37_coords)).all()


@pytest.mark.slow
@pytest.mark.model_forward
def test_model_predict_runs(prediction_result):
    assert isinstance(prediction_result, StructurePrediction)
    assert prediction_result.st is not None
    assert len(prediction_result.st) > 0
    assert prediction_result.plddt.shape == (5,)
    assert prediction_result.pae.shape == (5, 5)


@pytest.mark.slow
@pytest.mark.model_forward
def test_model_loss_runs(evaluated_loss):
    value, auxiliary = evaluated_loss
    assert np.isfinite(np.asarray(value)).all()
    assert auxiliary is not None


@pytest.mark.slow
@pytest.mark.model_forward
def test_model_loss_gradient_runs(loss_and_gradient, forward_inputs):
    (value, auxiliary), gradient = loss_and_gradient

    assert np.isfinite(np.asarray(value)).all()
    assert auxiliary is not None
    assert gradient.shape == forward_inputs.pssm.shape
    assert np.isfinite(np.asarray(gradient)).all()


@pytest.mark.slow
@pytest.mark.model_forward
def test_model_optimizer_step_runs(loss_and_gradient, forward_inputs):
    (_value, _auxiliary), gradient = loss_and_gradient
    optimizer = optax.adam(learning_rate=0.01)
    state = optimizer.init(forward_inputs.pssm)
    updates, state = optimizer.update(
        gradient,
        state,
        params=forward_inputs.pssm,
    )
    updated_pssm = optax.apply_updates(forward_inputs.pssm, updates)

    assert state is not None
    assert updated_pssm.shape == forward_inputs.pssm.shape
    assert np.isfinite(np.asarray(updated_pssm)).all()


@pytest.mark.slow
@pytest.mark.model_forward
def test_notebook_models_run_multisample_loss(structure_model, forward_inputs):
    if type(structure_model).__name__ not in {"ESMFold2", "OF3", "OpenDDEModel", "Protenix"}:
        pytest.skip("representative notebooks only")
    loss = structure_model.build_multisample_loss(
        loss=MeanConfidenceLoss(),
        features=forward_inputs.features,
        recycling_steps=1,
        sampling_steps=2,
        num_samples=2,
    )
    value, auxiliary = eqx.filter_jit(loss)(forward_inputs.pssm, key=jax.random.key(12))
    assert np.isfinite(np.asarray(value)).all()
    assert auxiliary is not None


@pytest.mark.slow
@pytest.mark.model_forward
def test_promera_backbone_design_and_refold_flow(structure_model):
    if type(structure_model).__name__ != "JPromeraModel":
        pytest.skip("Promera-specific notebook workflow")
    from mosaic.losses.protein_mpnn import inverse_fold
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol

    binder_length = 2
    target = TargetChain("ACD", use_msa=False)
    design_features, _ = structure_model.target_only_features(
        [TargetChain("X" * binder_length, use_msa=False), target]
    )
    backbone = structure_model.model_output(
        features=design_features,
        recycling_steps=1,
        sampling_steps=2,
        key=jax.random.key(13),
    )
    sequence = inverse_fold(
        load_mpnn_sol(),
        binder_length,
        backbone,
        temp=0.1,
        jacobi_iterations=1,
        key=jax.random.key(14),
    )
    refold_features, writer = structure_model.binder_features(binder_length, [target])
    prediction = structure_model.predict(
        PSSM=jax.nn.one_hot(sequence, 20),
        features=refold_features,
        writer=writer,
        recycling_steps=1,
        sampling_steps=2,
        key=jax.random.key(15),
    )
    assert sequence.shape == (binder_length,)
    assert np.isfinite(np.asarray(backbone.atom37_coords)).all()
    assert tuple(map(len, _protein_chains(prediction.st))) == (binder_length, 3)


@pytest.mark.slow
@pytest.mark.model_forward
def test_model_predict_preserves_multichain_boundaries(multichain_prediction):
    prediction = multichain_prediction.prediction
    expected_lengths = tuple(map(len, multichain_prediction.expected_names))
    chains = _protein_chains(prediction.st)
    asym_id = np.asarray(prediction.model_output.asym_id)
    observed_asym_ids = dict.fromkeys(asym_id.tolist())

    assert tuple(map(len, chains)) == expected_lengths
    assert tuple(
        np.count_nonzero(asym_id == chain_id) for chain_id in observed_asym_ids
    ) == (expected_lengths)
    assert prediction.plddt.shape == (sum(expected_lengths),)
    assert prediction.pae.shape == (sum(expected_lengths),) * 2


@pytest.mark.slow
@pytest.mark.model_forward
def test_model_output_structure_matches_prediction(multichain_prediction):
    prediction = multichain_prediction.prediction
    expected_names = multichain_prediction.expected_names
    expected_lengths = tuple(map(len, expected_names))
    converted_chains = _protein_chains(prediction.model_output.to_structure())
    predicted_chains = _protein_chains(prediction.st)

    assert tuple(map(len, converted_chains)) == expected_lengths
    assert (
        tuple(tuple(residue.name for residue in chain) for chain in converted_chains)
        == expected_names
    )
    for chain_index, (converted, predicted) in enumerate(
        zip(converted_chains, predicted_chains)
    ):
        for converted_residue, predicted_residue in zip(converted, predicted):
            allowed_names = {converted_residue.name}
            if multichain_prediction.mode == "binder" and chain_index == 0:
                allowed_names.add("UNK")
            assert predicted_residue.name in allowed_names

    converted_atoms = _atom_coordinates(converted_chains)
    predicted_atoms = _atom_coordinates(predicted_chains)
    shared_atoms = converted_atoms.keys() & predicted_atoms.keys()
    for chain_index, chain in enumerate(converted_chains):
        for residue_index, _residue in enumerate(chain):
            assert {
                (chain_index, residue_index, name) for name in ("N", "CA", "C", "O")
            } <= shared_atoms
    comparable_atoms = {
        key
        for key in shared_atoms
        if not (
            multichain_prediction.mode == "binder"
            and key[0] == 0
            and predicted_chains[key[0]][key[1]].name == "UNK"
            and key[2] not in {"N", "CA", "C", "O"}
        )
    }
    converted_coords = np.asarray([converted_atoms[key] for key in comparable_atoms])
    predicted_coords = np.asarray([predicted_atoms[key] for key in comparable_atoms])
    assert np.sqrt(np.mean(np.square(converted_coords - predicted_coords))) < 0.1


def _binder_terminal_ca_distance(model_output, binder_length: int) -> float:
    ca_index = 1  # atom37 order: N, CA, C, O, ...
    coordinates = np.asarray(model_output.atom37_coords[:binder_length])
    n_terminal_ca = coordinates[0, ca_index]
    c_terminal_ca = coordinates[-1, ca_index]
    return float(np.linalg.norm(n_terminal_ca - c_terminal_ca))


@pytest.mark.slow
@pytest.mark.model_forward
def test_af2_cyclic_binder_forward_pass_runs(structure_model):
    if type(structure_model).__name__ != "AlphaFold2":
        pytest.skip("AF2-specific cyclic-binder test")

    target = TargetChain(sequence="ACDEFGHIKLMNPQRSTVWY", use_msa=False)
    binder_length = 12
    features, writer = structure_model.binder_features(
        binder_length, [target], cyclic=True
    )

    assert "offset" in features
    total_length = binder_length + len(target.sequence)
    assert features["offset"].shape == (total_length, total_length)

    pssm = jax.nn.one_hot(jnp.zeros(binder_length, dtype=int), 20)
    loss = structure_model.build_loss(
        loss=MeanConfidenceLoss(), features=features, recycling_steps=1
    )
    value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))
    (value, auxiliary), gradient = value_and_grad(pssm, key=jax.random.key(0))

    assert np.isfinite(np.asarray(value)).all()
    assert auxiliary is not None
    assert gradient.shape == pssm.shape
    assert np.isfinite(np.asarray(gradient)).all()

    prediction = structure_model.predict(
        PSSM=pssm,
        features=features,
        writer=writer,
        recycling_steps=1,
        sampling_steps=None,
        key=jax.random.key(1),
    )
    assert np.isfinite(np.asarray(prediction.plddt)).all()


@pytest.mark.slow
@pytest.mark.model_forward
def test_af2_cyclic_binder_closes_termini_more_than_linear(structure_model):
    if type(structure_model).__name__ != "AlphaFold2":
        pytest.skip("AF2-specific cyclic-binder test")

    # Smaller than test_af2_cyclic_binder_forward_pass_runs: compute is
    # roughly cubic in sequence length, and this test pays for it once per
    # (model_idx, cyclic) combination below.
    target = TargetChain(sequence="ACDEFGHIKL", use_msa=False)
    binder_length = 8
    pssm = jax.nn.one_hot(jnp.zeros(binder_length, dtype=int), 20)

    num_sub_models = 5 if structure_model.multimer else 2
    margins = []
    for model_idx in range(num_sub_models):
        key = jax.random.key(model_idx)
        distances = {}
        for cyclic in (False, True):
            features, _writer = structure_model.binder_features(
                binder_length, [target], cyclic=cyclic
            )
            output = structure_model.model_output(
                PSSM=pssm,
                features=features,
                recycling_steps=1,
                sampling_steps=None,
                model_idx=model_idx,
                key=key,
            )
            distances[cyclic] = _binder_terminal_ca_distance(output, binder_length)
        margins.append(distances[False] - distances[True])

    # Every independently-trained multimer sub-model should agree, by a real
    # margin (not a coin-flip tie), that cyclization pulls the termini
    # closer together than the linear default. Observed margins across all
    # 5 multimer sub-models are ~9.5-15 A; 3.0 A leaves a wide safety
    # buffer while still ruling out a near-tie passing by chance.
    assert all(margin > 3.0 for margin in margins), margins


@pytest.mark.slow
@pytest.mark.model_forward
def test_af2_cyclic_binder_closes_absolutely_with_target_chain_gap():
    # The multimer/monomer AF2 code paths (modules_multimer.py vs. modules.py)
    # each need their own `batch["offset"]` override; the module-scoped
    # `structure_model` fixture above only exercises the multimer=True
    # default, so it could not catch a monomer-path regression. Callers that
    # run AF2 with multimer=False (e.g. grasp's mosaic design stage) go
    # through modules.py's relpos exclusively.
    #
    # A relative margin (cyclic closer than linear) can pass even when
    # neither prediction is anywhere near actually closed. This also checks
    # the practical claim: with a target chain present, cyclic binder
    # termini end up close in absolute terms, not just closer than the
    # linear control. Regression target for the missing target/binder
    # `residue_index` chain-break gap in `make_af_features` (mosaic's own
    # `multimer_to_monomer_features` already uses this +50 convention;
    # `make_af_features` didn't, so cross-chain relpos features looked like
    # ordinary short-range contacts and destabilized the predicted
    # structure enough that the binder's own cyclic-offset sub-block
    # couldn't pull its termini together).
    from mosaic.models.af2 import AlphaFold2

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        model = AlphaFold2(multimer=False)

        target = TargetChain(sequence="ACDEFGHIKL", use_msa=False)
        binder_length = 8
        pssm = jax.nn.one_hot(jnp.zeros(binder_length, dtype=int), 20)

        margins = []
        cyclic_distances = []
        for model_idx in range(2):
            key = jax.random.key(model_idx)
            distances = {}
            for cyclic in (False, True):
                features, _writer = model.binder_features(
                    binder_length, [target], cyclic=cyclic
                )
                output = model.model_output(
                    PSSM=pssm,
                    features=features,
                    recycling_steps=1,
                    sampling_steps=None,
                    model_idx=model_idx,
                    key=key,
                )
                distances[cyclic] = _binder_terminal_ca_distance(output, binder_length)
            margins.append(distances[False] - distances[True])
            cyclic_distances.append(distances[True])

        del model
    jax.clear_caches()
    gc.collect()

    assert all(margin > 3.0 for margin in margins), margins
    # Real head-to-tail closure puts N/C-terminal CA atoms within a few A of
    # each other; 10 A leaves headroom for prediction noise while still
    # ruling out the ~15-25 A "not actually closed" failure mode observed
    # before the chain-break gap fix.
    assert all(distance < 10.0 for distance in cyclic_distances), cyclic_distances
