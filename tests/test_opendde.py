"""Test `OpenDDE` with jfold's featurization and forward pass."""


import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jfold.featurize.system import THREE_LETTER
from jfold.featurize.views import opendde as view
from jfold_testing import atom_positions

from mosaic.common import TOKENS
from mosaic.losses.atom37 import ATOM37_NAMES
from mosaic.models import opendde
from mosaic.models.opendde import OpenDDE
from mosaic.structure_prediction import PolymerType, TargetChain

TARGET = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL"
BINDER_LENGTH = 12


@pytest.mark.parametrize(
    "chain",
    [
        TargetChain("ACGT", polymer_type=PolymerType.DNA, use_msa=False),
        TargetChain("ACDZ", use_msa=False),
        TargetChain("ACDE", use_msa=False, template_chain=gemmi.Chain("A")),
    ],
)
def test_unsupported_chains_raise(chain):
    key = jax.random.key(0)
    with pytest.raises(NotImplementedError, match="OpenDDE on jfold"):
        opendde.target_only_features([chain], key=key)
    with pytest.raises(NotImplementedError, match="OpenDDE on jfold"):
        opendde.binder_features(4, [chain], key=key)


def _onehot(sequence: str):
    return jax.nn.one_hot(jnp.array([TOKENS.index(c) for c in sequence]), 20)


def _assert_same_conformers(ours, ref, atom_to_token):
    """Check that each residue's reference conformer is the same up to its pose."""
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    for t in np.unique(atom_to_token):
        a, b = ours[atom_to_token == t], ref[atom_to_token == t]
        np.testing.assert_allclose(
            np.linalg.norm(a[:, None] - a[None], axis=-1),
            np.linalg.norm(b[:, None] - b[None], axis=-1),
            atol=1e-4,
        )


@pytest.mark.parametrize("binder", ["ACDEFHIKLMNP", "QRSTVWYAGCGG"])
def test_rebuilt_binder_is_the_designed_sequence(binder):
    """Check that `set_sequence` gives the view's features of the designed sequence."""
    target = [
        TargetChain("MKTAYIAKQRG", use_msa=False),
        TargetChain("ACDW", use_msa=False),
    ]
    design, writer = opendde.binder_features(
        len(binder), target, key=jax.random.key(0)
    )
    pssm = _onehot(binder)
    ours = opendde.set_binder_sequence(design, pssm, jax.random.key(1)).features
    ref, _ = opendde.target_only_features(
        [TargetChain(binder, use_msa=False), *target], key=jax.random.key(2)
    )
    num_atoms = ref["ref_pos"].shape[0]
    assert int(ours["atom_pad_mask"].sum()) == num_atoms
    assert not ours["atom_pad_mask"][num_atoms:].any()
    real = {
        k: v[:num_atoms] if v.shape[0] == ours["ref_pos"].shape[0] else v
        for k, v in ours.items()
    }
    skip = {"ref_pos"}
    if "G" in binder:
        skip |= {"atom_to_structural_token_idx"}
        skip |= {k for k in view.STRUCTURAL_KEYS if real[k].shape != ref[k].shape}
    for k, v in ref.items():
        if k not in skip:
            np.testing.assert_array_equal(real[k], v, err_msg=k)
    _assert_same_conformers(real["ref_pos"], ref["ref_pos"], ref["atom_to_token_idx"])

    st = writer(ours["ref_pos"], pssm)
    assert [r.name for r in st[0][0]] == [THREE_LETTER[c] for c in binder]
    assert sum(1 for c in st[0] for r in c for _ in r) == num_atoms


@pytest.fixture(scope="module")
def model():
    return OpenDDE.v1()


def _assert_sensible(prediction, target_start: int):
    """Check that a prediction is sensible and written where it places the atoms."""
    output = prediction.model_output
    assert jnp.isfinite(output.structure_coordinates).all()
    assert ((output.plddt >= 0) & (output.plddt <= 1)).all()
    ca = output.backbone_coordinates[target_start:, 1]
    gaps = jnp.linalg.norm(ca[1:] - ca[:-1], axis=-1)
    assert 3.0 < float(jnp.median(gaps)) < 4.5
    assert np.isfinite(atom_positions(prediction.st)).all()


@pytest.mark.slow
@pytest.mark.model_forward
def test_target_only(model):
    """Check a target-only prediction with an MSA."""
    chains = [TargetChain(sequence=TARGET, use_msa=True)]
    features, writer = model.target_only_features(chains, key=jax.random.key(5))

    coords = features["ref_pos"]
    atoms = model.atom37(features, coords)
    (chain,) = writer(coords)[0]
    assert len(chain) == len(TARGET)
    for residue, mask, positions in zip(chain, atoms.mask, atoms.coords, strict=True):
        written = {a.name: a.pos.tolist() for a in residue}
        assert set(written) == {ATOM37_NAMES[i] for i in np.flatnonzero(mask)}
        for name, position in written.items():
            np.testing.assert_allclose(
                position, positions[ATOM37_NAMES.index(name)], atol=1e-3
            )

    prediction = model.predict(features=features, writer=writer, key=jax.random.key(4))
    _assert_sensible(prediction, 0)


@pytest.mark.slow
@pytest.mark.model_forward
@pytest.mark.parametrize("gly", [False, True])
def test_design_prediction(model, gly):
    """Check that a design set to a soft PSSM predicts a sensible structure and iPTM."""
    chains = [TargetChain(sequence=TARGET, use_msa=False)]
    features, writer = model.binder_features(
        BINDER_LENGTH, chains, key=jax.random.key(5)
    )
    logits = 2.0 * jax.random.normal(jax.random.key(1), (BINDER_LENGTH, 20))
    if gly:
        pssm = jax.nn.softmax(logits).at[3].set(jax.nn.one_hot(TOKENS.index("G"), 20))
    else:
        pssm = jax.nn.softmax(logits.at[:, TOKENS.index("W")].add(20.0))
    prediction = model.predict(
        PSSM=pssm,
        features=features,
        writer=writer,
        key=jax.random.key(4),
    )
    _assert_sensible(prediction, BINDER_LENGTH)
    assert 0 <= float(prediction.iptm) <= 1
    names = [THREE_LETTER[TOKENS[k]] for k in np.asarray(pssm).argmax(-1)]
    assert [r.name for r in prediction.st[0][0]] == names


@pytest.mark.slow
@pytest.mark.model_forward
def test_samplers(model):
    """Check upstream's 200-step sampler and recycling counts."""
    sampler = model.upstream_sampler()
    assert len(sampler.sigmas) == 201
    assert (np.diff(sampler.sigmas) < 0).all()
    assert sampler.center.__name__ == "masked_centering"
    assert sampler.align.__name__ == "no_alignment"
    assert (model.design_recycling_steps, model.inference_recycling_steps) == (4, 10)
