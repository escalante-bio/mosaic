"""Integration smoke tests for model paths used by the example notebooks.

These tests load real checkpoints and may download weights on first use. Run
them explicitly with ``uv run pytest -m model_smoke``.
"""

from __future__ import annotations

import gc
from importlib import import_module
from operator import attrgetter
from types import SimpleNamespace

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.common import LossTerm
from mosaic.structure_prediction import (
    DiffusionStructurePredictionModel,
    StructureModelOutput,
    StructurePrediction,
    StructurePredictionModel,
    TargetChain,
)

# Imports stay lazy so normal test collection does not initialize every model
# stack. The matrix covers every common-interface factory used in examples.
MODEL_FACTORIES = [
    pytest.param(("mosaic.models.af2", "AlphaFold2", ()), id="alphafold2"),
    pytest.param(("mosaic.models.boltz2", "Boltz2", ()), id="boltz2"),
    *(
        pytest.param(
            ("mosaic.models.protenix", f"Protenix.{release}", ()),
            id=f"protenix_{release}",
        )
        for release in ("mini", "base", "base_20250630", "v2")
    ),
    pytest.param(("mosaic.models.openfold3", "OpenFold3", ()), id="openfold3"),
    *(
        pytest.param(
            ("mosaic.models.esmfold2", f"ESMFold2.{release}", ()),
            id=f"esmfold2_{release}",
        )
        for release in ("fast", "experimental_fast", "experimental_fast_cutoff2025")
    ),
    pytest.param(("mosaic.models.promera", "Promera", ()), id="promera"),
    pytest.param(("mosaic.models.opendde", "OpenDDE.v1", ()), id="opendde-v1"),
    pytest.param(("mosaic.models.opendde", "OpenDDE.abag", ()), id="opendde-abag"),
    pytest.param(("mosaic.models.mol1", "Mol1", ()), id="mol1"),
    pytest.param(("mosaic.models.bioemu2", "BioEmu2", ()), id="bioemu2"),
    pytest.param(("mosaic.models.atlasfold", "AtlasFold", ()), id="atlasfold-m"),
]


# Cutoff and antibody-specialized weights use the same execution paths as their
# siblings. Keep their loading and featurization checks in the full smoke matrix.
FORWARD_FACTORIES = [
    factory
    for factory in MODEL_FACTORIES
    if factory.id not in {"esmfold2_experimental_fast_cutoff2025", "opendde-abag"}
]


@pytest.fixture(params=MODEL_FACTORIES, scope="module")
def structure_model(request):
    module_name, factory_name, args = request.param
    factory = attrgetter(factory_name)(import_module(module_name))
    model = factory(*args)
    yield model
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
    sampling = import_module("jfold.sampling")
    trunk = module.boltzgen_trunk(
        model,
        features,
        recycling_steps=1,
        deterministic=True,
        key=jax.random.key(10),
    )
    denoise, atom_mask = module.boltzgen_denoiser(model, features, *trunk)
    sampler = module.boltzgen_sampler(
        model.structure_module, 10, step_scale=2.0, noise_scale=0.88
    )
    coordinates = sampling.sample(denoise, atom_mask, sampler, key=jax.random.key(11))[
        None
    ]
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
    features = structure_model.target_only_features([chain], key=jax.random.key(0))

    assert features is not None
    assert jax.tree.leaves(features), "featurization returned an empty pytree"


@pytest.mark.slow
@pytest.mark.model_smoke
def test_model_featurizes_binder_and_target(structure_model):
    target = TargetChain(sequence="ACDEFGHIKLMNPQRSTVWY", use_msa=False)
    features = structure_model.binder_features(8, [target], key=jax.random.key(0))

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
    supported = {"AlphaFold2", "Boltz2", "OpenFold3", "Protenix", "Mol1"}
    if type(structure_model).__name__ not in supported:
        pytest.skip("this notebook model does not use template-chain features")
    target = TargetChain(
        sequence="ACD", use_msa=False, template_chain=_template_chain()
    )
    features = structure_model.binder_features(2, [target], key=jax.random.key(0))
    assert jax.tree.leaves(features)


class MeanConfidenceLoss(LossTerm):
    def __call__(self, sequence, output, key):
        del sequence, key
        value = -jnp.mean(output.plddt)
        return value, {"mean_plddt": -value}


class MeanDistogramLoss(LossTerm):
    def __call__(self, sequence, output, key):
        del sequence, key
        value = jnp.mean(jax.nn.log_softmax(output.distogram_logits)[..., 0])
        return value, {"distogram": value}


def _has_confidence(structure_model) -> bool:
    """Whether the model predicts pLDDT and PAE (BioEmu2 has no confidence head)."""
    return (
        getattr(getattr(structure_model, "model", None), "confidence", True) is not None
    )


def _smoke_loss(structure_model) -> LossTerm:
    return (
        MeanConfidenceLoss()
        if _has_confidence(structure_model)
        else MeanDistogramLoss()
    )


def _two_step_sampling(structure_model):
    """Keyword arguments for a cheap 2-step diffusion, per model family."""
    if isinstance(structure_model, DiffusionStructurePredictionModel):
        return {"sampler": structure_model.sampler(2)}
    return {}


@pytest.fixture(scope="module")
def forward_inputs(structure_model):
    features = structure_model.binder_features(
        2, [TargetChain(sequence="ACD", use_msa=False)], key=jax.random.key(0)
    )
    pssm = jax.nn.one_hot(jnp.asarray([0, 7]), 20)
    return SimpleNamespace(features=features, pssm=pssm)


@pytest.fixture(scope="module")
def built_loss(structure_model, forward_inputs):
    return structure_model.build_loss(
        loss=_smoke_loss(structure_model),
        features=forward_inputs.features,
        recycling_steps=1,
        **_two_step_sampling(structure_model),
    )


@pytest.fixture(params=["binder", "target_only"], scope="module")
def multichain_prediction(structure_model, request):
    target_sequences = ("AC", "DEF")
    targets = [
        TargetChain(sequence=sequence, use_msa=False) for sequence in target_sequences
    ]
    if request.param == "binder":
        features = structure_model.binder_features(2, targets, key=jax.random.key(0))
        pssm = jax.nn.one_hot(jnp.asarray([0, 7]), 20)
        expected_names = (
            ("ALA", "GLY"),
            ("ALA", "CYS"),
            ("ASP", "GLU", "PHE"),
        )
    else:
        features = structure_model.target_only_features(targets, key=jax.random.key(0))
        pssm = None
        expected_names = (
            ("ALA", "CYS"),
            ("ASP", "GLU", "PHE"),
        )
    prediction = structure_model.predict(
        PSSM=pssm,
        features=features,
        recycling_steps=1,
        **_two_step_sampling(structure_model),
        key=jax.random.key(5),
    )
    return SimpleNamespace(prediction=prediction, expected_names=expected_names)


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
@pytest.mark.parametrize("structure_model", FORWARD_FACTORIES, indirect=True)
def test_model_loss_gradient_runs(built_loss, forward_inputs):
    value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(built_loss, has_aux=True))
    (value, auxiliary), gradient = value_and_grad(
        forward_inputs.pssm, key=jax.random.key(4)
    )

    assert np.isfinite(np.asarray(value)).all()
    assert auxiliary is not None
    assert gradient.shape == forward_inputs.pssm.shape
    assert np.isfinite(np.asarray(gradient)).all()


@pytest.mark.slow
@pytest.mark.model_forward
@pytest.mark.parametrize(
    "structure_model", [("mosaic.models.openfold3", "OpenFold3", ())], indirect=True
)
def test_notebook_models_run_multisample_loss(structure_model, forward_inputs):
    loss = structure_model.build_loss(
        loss=_smoke_loss(structure_model),
        features=forward_inputs.features,
        recycling_steps=1,
        sampler=structure_model.sampler(2),
        num_samples=2,
    )
    value, auxiliary = eqx.filter_jit(loss)(forward_inputs.pssm, key=jax.random.key(12))
    assert np.isfinite(np.asarray(value)).all()
    assert auxiliary is not None


@pytest.mark.slow
@pytest.mark.parametrize(
    "structure_model", [("mosaic.models.promera", "Promera", ())], indirect=True
)
def test_promera_backbone_design_and_refold_flow(structure_model):
    from mosaic.losses.protein_mpnn import inverse_fold
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol

    binder_length = 2
    target = TargetChain("ACD", use_msa=False)
    design_features = structure_model.target_only_features(
        [TargetChain("X" * binder_length, use_msa=False), target], key=jax.random.key(0)
    )
    backbone = structure_model.model_output(
        features=design_features,
        recycling_steps=1,
        sampler=structure_model.sampler(2),
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
    refold_features = structure_model.binder_features(
        binder_length, [target], key=jax.random.key(0)
    )
    prediction = structure_model.predict(
        PSSM=jax.nn.one_hot(sequence, 20),
        features=refold_features,
        recycling_steps=1,
        sampler=structure_model.sampler(2),
        key=jax.random.key(15),
    )
    assert sequence.shape == (binder_length,)
    assert np.isfinite(np.asarray(backbone.atom37_coords)).all()
    assert tuple(map(len, _protein_chains(prediction.st))) == (binder_length, 3)


@pytest.mark.slow
@pytest.mark.model_forward
@pytest.mark.parametrize("structure_model", FORWARD_FACTORIES, indirect=True)
def test_model_prediction_contract(structure_model, multichain_prediction):
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

    output = prediction.model_output
    assert isinstance(prediction, StructurePrediction)
    assert isinstance(output, StructureModelOutput)
    assert output.atom37_coords.shape == (sum(expected_lengths), 37, 3)
    assert np.isfinite(output.atom37_coords).all()
    confidence = (prediction.plddt, prediction.pae, prediction.iptm)
    check = np.isfinite if _has_confidence(structure_model) else np.isnan
    assert all(check(array).all() for array in confidence)
    expected_names = multichain_prediction.expected_names

    # A designed binder is written with its sequence, not as a placeholder.
    assert (
        tuple(tuple(residue.name for residue in chain) for chain in chains)
        == expected_names
    )
    for chain in chains:
        assert [residue.seqid.num for residue in chain] == list(
            range(1, len(chain) + 1)
        )

    atoms = _atom_coordinates(chains)
    assert np.isfinite(np.asarray(list(atoms.values()))).all()
    # Every residue has its backbone, at the model output's backbone coordinates.
    backbone = np.asarray(
        [
            [atoms[chain_index, residue_index, name] for name in ("N", "CA", "C", "O")]
            for chain_index, chain in enumerate(chains)
            for residue_index, _residue in enumerate(chain)
        ]
    )
    np.testing.assert_allclose(
        backbone, np.asarray(prediction.model_output.backbone_coordinates), atol=1e-3
    )
