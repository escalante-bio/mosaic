"""Test `BioEmu2` with jfold's featurization and with upstream's."""

import string

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jfold_testing import atom_labels

import mosaic.msa
from mosaic.common import TOKENS
from mosaic.losses.structure_prediction import BinderTargetContact
from mosaic.models.bioemu2 import BioEmu2, featurize
from jfold.sampling import sample
from mosaic.structure_prediction import PolymerType, TargetChain

pytestmark = [pytest.mark.slow, pytest.mark.model_forward]

TARGET = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQ"[
        :60
    ]
)
BINDER_LENGTH = 12
DNA = "GACTCG"


def assert_same_features(ours: dict, ref: dict, atol: float = 0.0):
    assert ours.keys() == ref.keys()
    for key, value in ours.items():
        a, b = np.asarray(value, np.float64), np.asarray(ref[key], np.float64)
        np.testing.assert_allclose(a, b, rtol=0, atol=atol, err_msg=key)


def one_hot(sequence: str):
    return jax.nn.one_hot(jnp.array([TOKENS.index(c) for c in sequence]), 20)


def query(sequence: str) -> str:
    return f">query\n{sequence}\n"


def native(entities: list[tuple[str, str]], a3m: dict[str, str] | None = None):
    """Build bioemu2's input for `entities`, given as (entity type, sequence) pairs."""
    from bioemu2.data_interface import System
    from bioemu2.msa.alignment import MSA

    a3m = a3m or {}
    system = System.model_validate(
        {
            "entities": [
                {"entity_type": kind, "sequence": sequence, "asym_ids": [chain_id]}
                for chain_id, (kind, sequence) in zip(string.ascii_uppercase, entities)
            ]
        }
    )
    proteins = {
        chain_id: sequence
        for chain_id, (kind, sequence) in zip(string.ascii_uppercase, entities)
        if kind == "protein"
    }
    msa = MSA(
        unpaired_a3m={i: a3m.get(seq, query(seq)) for i, seq in proteins.items()},
        paired_a3m={i: query(seq) for i, seq in proteins.items()},
    )
    return system, msa


@pytest.fixture(scope="module")
def model():
    return BioEmu2()


@pytest.fixture(scope="module")
def upstream():
    pytest.importorskip("bioemu2")
    from mosaic.models.upstream import bioemu2

    return bioemu2


@pytest.fixture(scope="module")
def chains():
    return [TargetChain(sequence=TARGET, use_msa=False)]


def test_features_match(model, upstream, chains):
    """Check that the design features match upstream's for the same system and binder."""
    binder = ("protein", "X" * BINDER_LENGTH)
    f_up, _ = upstream.features(
        native([binder, ("protein", TARGET)]), mode=model.mode, key=jax.random.key(0)
    )
    f_jf, _ = model.binder_features(BINDER_LENGTH, chains, key=jax.random.key(0))
    assert_same_features(f_jf, f_up)

    pssm = jax.nn.softmax(
        2.0 * jax.random.normal(jax.random.key(1), (BINDER_LENGTH, 20))
    )
    key = jax.random.key(0)
    f_up = model.set_sequence(f_up, pssm, key=key)
    f_jf = model.set_sequence(f_jf, pssm, key=key)
    assert_same_features(f_jf, f_up)
    t_up, t_jf = model.tokens(f_up), model.tokens(f_jf)
    for a, b in zip(jax.tree.leaves(t_jf), jax.tree.leaves(t_up), strict=True):
        np.testing.assert_array_equal(a, b)

    coords = jax.random.normal(jax.random.key(2), (f_jf["atom_pad_mask"].shape[0], 3))
    a_up, a_jf = model.atom37(f_up, coords), model.atom37(f_jf, coords)
    np.testing.assert_array_equal(a_jf.coords, a_up.coords)
    np.testing.assert_array_equal(a_jf.mask, a_up.mask)

    (_, w_up), (_, w_jf) = (
        upstream.features(
            native([("protein", TARGET)]), mode=model.mode, key=jax.random.key(0)
        ),
        model.target_only_features(chains, key=jax.random.key(0)),
    )
    st_up, st_jf = model.structure(w_up, coords), model.structure(w_jf, coords)
    assert atom_labels(st_jf) == atom_labels(st_up)


@pytest.mark.parametrize("mode", ["structure", "equilibrium"])
def test_features_match_msa(model, upstream, mode, monkeypatch):
    """Check features with a shared a3m, to float32 arctan's last ulp in deletions."""
    other = TARGET[::-1]
    a3m = {
        seq: f">query\n{seq}\n"
        f">a\n{seq[:10]}kk{'-' * 5}{seq[15:30]}A{seq[31:]}\n"
        f">b\n{seq[:20]}xyz{seq[20:59]}-\n"
        for seq in (TARGET, other)
    }
    monkeypatch.setattr(mosaic.msa, "fetch_a3m", a3m.__getitem__)
    jf = BioEmu2(mode, model=model.model)
    target = TargetChain(sequence=TARGET, use_msa=True)

    f_up, _ = upstream.features(
        native([("protein", "X" * BINDER_LENGTH), ("protein", TARGET)], a3m),
        mode=mode,
        key=jax.random.key(0),
    )
    f_jf, _ = jf.binder_features(BINDER_LENGTH, [target], key=jax.random.key(0))
    assert f_jf["msa"].shape[0] == 3
    assert_same_features(f_jf, f_up, atol=1e-6)
    for sequences in ([TARGET, TARGET], [TARGET, other]):
        f_up, _ = upstream.features(
            native([("protein", seq) for seq in sequences], a3m),
            mode=mode,
            key=jax.random.key(0),
        )
        f_jf, _ = jf.target_only_features(
            [TargetChain(sequence=seq, use_msa=True) for seq in sequences],
            key=jax.random.key(0),
        )
        assert_same_features(f_jf, f_up, atol=1e-6)


@pytest.mark.parametrize(
    "chain",
    [
        TargetChain(DNA, polymer_type=PolymerType.DNA, use_msa=False),
        TargetChain("MKTAYZ", use_msa=False),
        TargetChain(TARGET, use_msa=False, template_chain=gemmi.Chain("A")),
    ],
    ids=["dna", "residue", "template"],
)
def test_unsupported(chain):
    with pytest.raises(NotImplementedError, match=r"upstream\.bioemu2\.features"):
        featurize([TargetChain(TARGET, use_msa=False), chain], mode="structure")


@pytest.mark.parametrize("mode", ["structure", "equilibrium"])
def test_samplers(model, mode):
    """Check that upstream's presets differ only in `step_scale`."""
    a = model.upstream_sampler()
    b = BioEmu2(mode, model=model.model).upstream_sampler()
    np.testing.assert_array_equal(b.sigmas, a.sigmas)
    assert b.step_scale == {"structure": 1.5, "equilibrium": 1.0}[mode]
    assert (b.gamma_0, b.gamma_min, b.noise_scale) == (
        a.gamma_0,
        a.gamma_min,
        a.noise_scale,
    )


def test_upstream_dna(model, upstream):
    """Check upstream's featurization of a protein-DNA complex from bioemu2's input."""
    binder = "GWYKCPSHDLIFEGMRAQNTVG"  # every residue type
    targets = [("protein", TARGET), ("dna", DNA)]
    design, _ = upstream.features(
        native([("protein", "X" * len(binder)), *targets]),
        mode=model.mode,
        key=jax.random.key(0),
    )
    folded, _ = upstream.features(
        native([("protein", binder), *targets]), mode=model.mode, key=jax.random.key(0)
    )
    designed = model.set_sequence(design, one_hot(binder), key=jax.random.key(0))
    for key in ("res_type", "profile", "msa"):
        np.testing.assert_allclose(designed[key], folded[key], atol=1e-6, err_msg=key)

    sampler = model.sampler(20)
    pred = model.predict(
        PSSM=one_hot(binder), features=design, recycling_steps=2, sampler=sampler,
        key=jax.random.key(0),
    )  # fmt: skip
    assert np.isfinite(pred.model_output.structure_coordinates).all()
    assert np.isfinite(pred.model_output.distogram_logits).all()
    assert len(pred.st[0][0]) == len(binder)  # atom37 view: binder first

    features, writer = upstream.features(
        native(targets), mode=model.mode, key=jax.random.key(0)
    )
    pred = model.predict(
        features=features, writer=writer, recycling_steps=2, sampler=sampler,
        key=jax.random.key(0),
    )  # fmt: skip
    assert np.isfinite(pred.model_output.structure_coordinates).all()
    assert [len(chain) for chain in pred.st[0]] == [len(TARGET), len(DNA)]
    assert pred.st[0][1][0].name == "DG"

    loss = model.build_loss(
        loss=BinderTargetContact(), features=design, recycling_steps=2, num_samples=2
    )
    pssm = jax.nn.softmax(jax.random.normal(jax.random.key(0), (len(binder), 20)))

    @eqx.filter_jit
    def value_and_grad(loss, pssm, key):
        return jax.value_and_grad(lambda p: loss(p, key=key)[0])(pssm)

    value, grad = value_and_grad(loss, pssm, jax.random.key(1))
    assert np.isfinite(value) and np.isfinite(grad).all() and np.abs(grad).max() > 0


def test_upstream_ensemble(model, upstream):
    """Check that vmapping the sampler over keys gives distinct all-atom samples."""
    features, writer = upstream.features(
        native([("protein", TARGET), ("dna", DNA)]),
        mode=model.mode,
        key=jax.random.key(0),
    )
    denoise, atom_mask = model.denoiser(
        features=features, recycling_steps=2, key=jax.random.key(0)
    )
    sampler = model.sampler(20)
    keys = jax.random.split(jax.random.key(1), 3)
    coords = jax.vmap(lambda k: sample(denoise, atom_mask, sampler, key=k))(keys)
    assert np.isfinite(coords).all()
    assert not np.allclose(coords[0], coords[1])
    structures = [model.structure(writer, x) for x in coords]
    assert all(st[0].count_atom_sites() == int(atom_mask.sum()) for st in structures)
