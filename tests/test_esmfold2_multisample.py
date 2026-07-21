import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.common import LossTerm
import mosaic.losses.esmfold2 as esmfold2_loss


class _OutputLoss(LossTerm):
    def __call__(self, sequence, output, *, key):
        del sequence, key
        return output, {"sample": output}


@pytest.mark.parametrize("num_samples", [1, 3])
def test_esmfold2_loss_runs_trunk_once(monkeypatch, num_samples):
    calls = {"prepare": 0, "trunk": 0}

    def fake_prepare(*args, **kwargs):
        del args, kwargs
        calls["prepare"] += 1
        return (
            object(),
            None,
            jnp.zeros(1, dtype=jnp.int32),
            jnp.zeros((4, 1), dtype=jnp.int32),
            jnp.zeros((1, 20)),
        )

    def fake_trunk(*args, **kwargs):
        del args, kwargs
        calls["trunk"] += 1
        return jnp.array(0.0), jnp.array(0.0)

    def fake_forward(*args, **kwargs):
        key = args[3]
        del kwargs
        sample = jax.random.uniform(key)
        return sample, jnp.zeros((1, 1, 1)), {}

    def fake_to_output(**kwargs):
        return kwargs["sample_atom_coords"]

    monkeypatch.setattr(
        esmfold2_loss,
        "_prepare_esmfold2_inputs",
        fake_prepare,
    )
    monkeypatch.setattr(esmfold2_loss, "esmfold2_trunk", fake_trunk)
    monkeypatch.setattr(
        esmfold2_loss,
        "esmfold2_forward_from_trunk",
        fake_forward,
    )
    monkeypatch.setattr(
        esmfold2_loss,
        "_to_structure_model_output",
        fake_to_output,
    )

    loss = esmfold2_loss.ESMFold2Loss(
        esmf=None,
        pack=None,
        loss=_OutputLoss(),
        res_type_perm=jnp.zeros((20, 33)),
        distogram_bins=jnp.zeros(1),
        num_loops=1,
        num_sampling_steps=2,
        msa_max_depth=None,
        num_samples=num_samples,
    )
    key = jax.random.key(7)

    value, auxiliary = eqx.filter_jit(loss)(jnp.zeros((1, 20)), key=key)

    _, samples_key = jax.random.split(key)
    expected = jax.vmap(jax.random.uniform)(
        jax.random.split(samples_key, num_samples)
    )
    assert calls == {"prepare": 1, "trunk": 1}
    assert np.isclose(value, expected.mean())
    # Scalar aux is always returned as a loss-sorted list (no single-sample
    # squeeze), uniformly across sample counts -- a 1-element list when num_samples == 1.
    assert np.allclose(np.asarray(auxiliary["sample"]), np.sort(expected))
