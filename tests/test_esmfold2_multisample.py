import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.losses.esmfold2 import EsmFold2FeaturePack
from mosaic.models.esmfold2 import ESMFold2


class _Trunk(eqx.Module):
    def language_model(self, hidden):
        assert hidden.dtype == jnp.float32
        return hidden * 2

    def embed(self, features, lm_z, *, key):
        del features, key
        return lm_z


def test_bfloat16_lm_features_enter_structure_trunk_as_detached_float32():
    model = ESMFold2(
        esmf=_Trunk(),
        esmc=None,
        res_type_perm=jnp.zeros((20, 33)),
        distogram_bins=jnp.zeros(1),
        esmc_token_of_aa=jnp.zeros(20, jnp.int32),
        unk_input_id=0,
    )

    def evaluate(hidden):
        pack = EsmFold2FeaturePack(
            features=None, target_lm_hidden=hidden, atom37_idx=jnp.zeros(1, jnp.int32)
        )
        return model.embed(pack, key=jax.random.key(0)).sum()

    value, gradient = jax.value_and_grad(evaluate)(jnp.ones((1, 3, 2, 4), dtype=jnp.bfloat16))
    assert value.dtype == jnp.float32
    np.testing.assert_array_equal(gradient, 0)
