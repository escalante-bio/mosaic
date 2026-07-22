from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.common import LossTerm, TOKENS
from mosaic.losses.opendde import (
    MultiSampleOpenDDELoss,
    OpenDDEAtomTemplates,
    OpenDDEDesignFeatures,
    set_binder_sequence,
)
from mosaic.models.opendde import OpenDDEModel
from mosaic.structure_prediction import TargetChain


class _ZeroLoss(LossTerm):
    def __call__(self, sequence, output, *, key):
        del sequence, output, key
        return jnp.array(0.0), {}


def _empty_atom_templates():
    import mosaic.models.opendde as adapter

    floats = {"ref_pos", "ref_element", "ref_charge", "ref_atom_name_chars"}
    arrays = {
        name: jnp.zeros(shape, dtype=jnp.float32 if name in floats else jnp.int32)
        for name, shape in adapter._atom_template_shapes().items()
    }
    return OpenDDEAtomTemplates(**arrays)


def _adapter_model():
    return OpenDDEModel(
        model=None,
        dense_atom_to_atom37=jnp.zeros((32, 15), dtype=jnp.int32),
        atom_templates=_empty_atom_templates(),
    )


def test_design_loss_rejects_target_only_features():
    model = _adapter_model()

    with pytest.raises(TypeError, match="binder_features"):
        model.build_loss(loss=_ZeroLoss(), features=object())


def test_binder_features_constructs_design_bundle_once(monkeypatch):
    raw = SimpleNamespace(
        asym_id=np.asarray([0, 0, 1, 1, 1]),
        atom_to_token_idx=np.asarray([0, 0, 1, 1, 2, 3, 4]),
    )
    monkeypatch.setattr(
        OpenDDEModel,
        "target_only_features",
        lambda self, chains: (raw, "writer"),
    )
    model = _adapter_model()

    features, writer = model.binder_features(2, [TargetChain("ACD")])

    assert isinstance(features, OpenDDEDesignFeatures)
    assert features.features is raw
    assert features.atom_templates is model.atom_templates
    assert features.binder_length == 2
    assert features.binder_atom_alloc == 4
    assert writer == "writer"


@pytest.mark.parametrize(
    ("shape", "message"),
    [((3, 20), "length"), ((2, 19), "20 columns")],
)
def test_set_binder_sequence_validates_pssm_shape(shape, message):
    features = OpenDDEDesignFeatures(
        features=None,
        atom_templates=_empty_atom_templates(),
        binder_length=2,
        binder_atom_alloc=1,
    )

    with pytest.raises(ValueError, match=message):
        set_binder_sequence(jnp.zeros(shape), features, jax.random.key(0))


def test_featurize_configs_are_isolated_and_follow_cache_root(monkeypatch, tmp_path):
    import mosaic.models.opendde as adapter

    base = SimpleNamespace(use_msa=False, seeds=[])
    roots = []

    def cached(root):
        roots.append(root)
        return base

    monkeypatch.setattr(adapter, "_cached_featurize_config", cached)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    monkeypatch.setenv("MOSAIC_CACHE_DIR", str(first_root))
    first = adapter._featurize_configs()
    first.use_msa = True
    first.seeds.append(1)

    monkeypatch.setenv("MOSAIC_CACHE_DIR", str(second_root))
    second = adapter._featurize_configs()

    assert roots == [str(first_root), str(second_root)]
    assert base.use_msa is False
    assert base.seeds == []
    assert second.use_msa is False
    assert second.seeds == []


def test_atom_template_cache_is_versioned_and_recovers_from_corruption(
    monkeypatch, tmp_path
):
    import mosaic.models.opendde as adapter

    built = []
    templates = _empty_atom_templates()
    monkeypatch.setenv("MOSAIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(adapter, "_jopendde_build_id", lambda: "revision123456")

    def build(_featurize_one):
        built.append(None)
        return templates

    monkeypatch.setattr(adapter, "build_opendde_atom_templates", build)
    adapter._ATOM_TEMPLATE_CACHE.clear()

    first = adapter._get_atom_templates()
    files = list((tmp_path / "opendde" / "atom_templates").glob("*.npz"))

    assert first is templates
    assert len(files) == 1
    assert "v1-revision123" in files[0].name
    assert built == [None]

    adapter._ATOM_TEMPLATE_CACHE.clear()
    loaded = adapter._get_atom_templates()
    assert built == [None]
    assert np.asarray(loaded.ref_pos).shape == np.asarray(templates.ref_pos).shape

    files[0].write_bytes(b"not an npz")
    adapter._ATOM_TEMPLATE_CACHE.clear()
    rebuilt = adapter._get_atom_templates()

    assert rebuilt is templates
    assert built == [None, None]
    with np.load(files[0], allow_pickle=False) as data:
        adapter._validate_atom_template_cache(data, "revision123456")
    assert not list(files[0].parent.glob("*.tmp"))


class _FakeModel(eqx.Module):
    def get_pairformer_output(self, features, num_cycles):
        del features, num_cycles
        return jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)


class _RandomLoss(LossTerm):
    def __call__(self, sequence, output, *, key):
        del sequence
        loss_random = jax.random.uniform(key)
        return output + loss_random, {
            "model_random": output,
            "loss_random": loss_random,
        }


def test_multisample_uses_distinct_model_and_loss_keys(monkeypatch):
    import mosaic.losses.opendde as opendde_loss

    monkeypatch.setattr(
        opendde_loss,
        "set_binder_sequence",
        lambda sequence, features, key: None,
    )

    def fake_forward(model, feat, s_inputs, s, z, key, **kwargs):
        del model, feat, s_inputs, s, z, kwargs
        return jax.random.uniform(key)

    monkeypatch.setattr(opendde_loss, "opendde_forward_from_trunk", fake_forward)
    num_samples = 3
    loss = MultiSampleOpenDDELoss(
        model=_FakeModel(),
        features=None,
        loss=_RandomLoss(),
        dense_atom_to_atom37=jnp.zeros((32, 15), dtype=jnp.int32),
        num_samples=num_samples,
    )
    key = jax.random.key(5)

    value, auxiliary = eqx.filter_jit(loss)(jnp.zeros((2, 20)), key=key)

    samples_key, _ = jax.random.split(key)
    sample_keys = jax.random.split(samples_key, num_samples)
    split_keys = jax.vmap(jax.random.split)(sample_keys)
    model_values = jax.vmap(jax.random.uniform)(split_keys[:, 0])
    loss_values = jax.vmap(jax.random.uniform)(split_keys[:, 1])
    totals = model_values + loss_values
    order = np.argsort(np.asarray(totals))

    assert np.isclose(value, totals.mean())
    assert np.allclose(np.asarray(auxiliary["model_random"]), model_values[order])
    assert np.allclose(np.asarray(auxiliary["loss_random"]), loss_values[order])
    assert not np.allclose(model_values, loss_values)


@pytest.mark.slow
@pytest.mark.opendde_smoke
def test_refreshed_atom_metadata_matches_native_featurization():
    import mosaic.models.opendde as adapter

    target = TargetChain("ACD", use_msa=False)
    placeholder, _ = adapter._featurize(
        [TargetChain("W" * len(TOKENS), use_msa=False), target]
    )
    native, _ = adapter._featurize([TargetChain(TOKENS, use_msa=False), target])
    binder_length, binder_atom_alloc = adapter._binder_extents(placeholder)
    design = OpenDDEDesignFeatures(
        features=placeholder,
        atom_templates=adapter._get_atom_templates(),
        binder_length=binder_length,
        binder_atom_alloc=binder_atom_alloc,
    )
    pssm = jax.nn.one_hot(jnp.arange(20), 20)

    refreshed = eqx.filter_jit(set_binder_sequence)(pssm, design, jax.random.key(11))

    refreshed_mask = np.asarray(refreshed.ref_mask) > 0.5
    native_mask = np.asarray(native.ref_mask) > 0.5
    assert refreshed_mask.sum() == native_mask.sum()
    for field in (
        "ref_element",
        "ref_charge",
        "ref_atom_name_chars",
        "atom_to_token_idx",
        "atom_to_tokatom_idx",
        "distogram_rep_atom_mask",
        "pae_rep_atom_mask",
    ):
        refreshed_value = np.asarray(getattr(refreshed, field))[refreshed_mask]
        native_value = np.asarray(getattr(native, field))[native_mask]
        assert np.array_equal(refreshed_value, native_value), field

    assert np.array_equal(
        np.asarray(refreshed.frame_atom_index),
        np.asarray(native.frame_atom_index),
    )
    for token in range(len(TOKENS)):
        refreshed_pos = np.asarray(refreshed.ref_pos)[
            refreshed_mask & (np.asarray(refreshed.atom_to_token_idx) == token)
        ]
        native_pos = np.asarray(native.ref_pos)[
            native_mask & (np.asarray(native.atom_to_token_idx) == token)
        ]
        refreshed_dist = np.linalg.norm(
            refreshed_pos[:, None] - refreshed_pos[None, :], axis=-1
        )
        native_dist = np.linalg.norm(native_pos[:, None] - native_pos[None, :], axis=-1)
        assert np.allclose(refreshed_dist, native_dist, atol=1e-4), TOKENS[token]
