# Very simple *absolute* (delta G, ~not~ delta-delta G) stability model trained on top of frozen ESM2 + boltz trunk on the Megascale dataset
# Specifically this is trained to minimize MSE on the split described here: https://github.com/SimonKitSangChu/EsmTherm?tab=readme-ov-file
# Could almost certainly be improved but seems to work fine.

# NOTE: this was trained with Boltz recycling steps = 0!
# May break if this is changed.

from pathlib import Path

import equinox as eqx
import jax
from jaxtyping import Array, Float, PyTree
from joltz import TrunkOutputs
import jax.numpy as jnp
import numpy as np

from ..esm import ESM2
from .boltz import TrunkLoss
from .esm import boltz_to_esm_matrix


class StabilityModel(TrunkLoss):
    esm: ESM2
    mlp_pre: eqx.nn.MLP
    mlp_post: eqx.nn.MLP  # really linear.

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        trunk_output: TrunkOutputs,
        *, key,
    ):
        boltz_embedding = jax.vmap(
            eqx.nn.LayerNorm(use_bias=False, use_weight=False, shape=384)
        )(trunk_output.s[:sequence.shape[0]])

        # now let's compute the ESM2 embedding, this is a little more involved but not too bad
        esm_toks_unpadded = sequence @ boltz_to_esm_matrix()
        # add cls and eos tokens
        esm_toks = jnp.concatenate(
            [
                jax.nn.one_hot([0], 33),
                esm_toks_unpadded,
                jax.nn.one_hot([2], 33),
            ]
        )
        # run through embedding layer
        esm_embedding = esm_toks @ self.esm.embed_tokens.weight
        # rescale to account for masking during ESM training
        mask_ratio_train = 0.15 * 0.8
        esm_embedding = esm_embedding * (1 - mask_ratio_train)
        # apply ESM trunk
        esm_embedding = self.esm._apply_trunk(
            esm_embedding, np.ones((esm_toks.shape[0], esm_toks.shape[0]))
        )
        # ln
        esm_embedding = jax.vmap(
            eqx.nn.LayerNorm(use_bias=False, use_weight=False, shape=1280)
        )(esm_embedding)

        # cat embeddings 
        print(boltz_embedding.shape, esm_embedding[1:-1].shape)
        embedding = jnp.concatenate((boltz_embedding, esm_embedding[1:-1]), axis=-1)

        # standard deep set reduction: mlp_pre -> mean -> mlp_post
        estimated_delta_g = self.mlp_post((jax.vmap(self.mlp_pre)(embedding)).mean(0))
        # this isn't very accurate, so let's clip it
        estimated_delta_g = estimated_delta_g.clip(-10, 3)
        return -estimated_delta_g, {"delta_g": estimated_delta_g} # sign error?

    @staticmethod
    def from_pretrained(esm: ESM2, path: Path = Path("stability.eqx")):
        # using ESM2 650M
        esm_embedding_dim = 1280
        model = (
            eqx.nn.MLP(
                in_size=esm_embedding_dim + 384,
                out_size=2048,
                width_size=2 * esm_embedding_dim,
                activation=jax.nn.elu,
                depth=1,
                key=jax.random.key(0),
            ),
            eqx.nn.MLP(
                2048,
                out_size="scalar",
                width_size=2 * esm_embedding_dim,
                activation=jax.nn.elu,
                depth=0,
                key=jax.random.key(0),
            ),
        )
        return StabilityModel(esm, *eqx.tree_deserialise_leaves(path, model))
