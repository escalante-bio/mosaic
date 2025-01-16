import torch
import jax
import equinox as eqx
# from jax import numpy as jnp
import numpy as onp
import re
from .esm import ESM2
from ..util import At


def load_pretrained_esm(model_name: str = "esm2_t33_650M_UR50D"):
    # download weights and config
    state = torch.hub.load_state_dict_from_url(
        f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt",
        map_location="cpu",
    )

    # initialize eqx model from config
    cfg = state["cfg"]["model"]
    with jax.default_device(jax.devices("cpu")[0]):
        eqx_model = ESM2(
            cfg.encoder_embed_dim,
            cfg.encoder_layers,
            cfg.encoder_attention_heads,
            key=jax.random.PRNGKey(0),
        )

    # paste in weights and return
    return (
        _load_from_pytorch_state_dict(eqx_model, state["model"]),
        cfg.encoder_embed_dim,
    )


def _translate_state_path(p: str):
    # :-)
    # replace leading "encoder.sentence_encoder." with "."
    p = re.sub(r"^encoder\.sentence_encoder\.", ".", p)
    # replace encoder.lm_head with .lm_head
    p = re.sub(r"^encoder\.lm_head\.", ".lm_head.", p)
    # wild hack here but now replace ".layers.$n." with ".layers[$n]." where $n is a number
    p = re.sub(r"\.layers\.(\d+)\.", r".layers[\1].", p)
    # replace ".$c_proj." with ".$c" where $c is a string
    p = re.sub(r"\.(k|q|v|out)_proj\.", r".\1.", p)
    # replace "out" with "o"
    p = re.sub(r"\.out\.", ".o.", p)
    return p


def _load_from_pytorch_state_dict(esm: ESM2, state_dict: dict):#, dtype=jnp.float32):
    # first convert pytorch key paths to look like jax keypaths
    state_dict = {_translate_state_path(k): v for k, v in state_dict.items()}

    # We do something *very* weird here: we unstack the trunk of our ESM2 model to match the pytorch-style list of layers
    stacked_layers, static_trunk = eqx.partition(esm.layers, eqx.is_array)
    layers = [jax.tree.map(lambda v: v[i], stacked_layers) for i in range(esm.num_layers)]
    # replace stacked layers with layers in model
    esm = At(esm).layers(layers)

    # now let's flatten our eqx model
    params, not_params = eqx.partition(esm, eqx.is_array)
    flattened, treedef = jax.tree_util.tree_flatten_with_path(params)

    # warn about non-matching keys
    pytorch_keys = set(state_dict.keys())
    eqx_keys = {jax.tree_util.keystr(key_path) for key_path, _ in flattened}
    pytorch_not_eqx = pytorch_keys - eqx_keys
    eqx_not_pytorch = eqx_keys - pytorch_keys
    if len(pytorch_not_eqx) > 0:
        print(f"warning: torch statedict contains unused {len(pytorch_not_eqx)} keys.")
        
    if len(eqx_not_pytorch) > 0:
        raise ValueError(f"eqx model contains unexpected parameters: {eqx_not_pytorch}")

    def _convert_and_check_shape(key_path, initial_shape):
        # value = jnp.array(state_dict[jax.tree_util.keystr(key_path)], dtype=dtype)
        value = onp.array(state_dict[jax.tree_util.keystr(key_path)])
        # test that the shapes match
        assert (
            value.shape == initial_shape
        ), f"shape mismatch: {key_path} expected {initial_shape} but got {value.shape}"
        return value

    loaded_parameters = [
        _convert_and_check_shape(key_path, v.shape) for key_path, v in flattened
    ]

    # now unflatten and combine with static portion of pytree
    unstacked_model = eqx.combine(
        jax.tree_util.tree_unflatten(treedef, loaded_parameters), not_params
    )
    # finally stack the trunk again
    stacked_layers = jax.tree.map(lambda *v: onp.stack(v), *unstacked_model.layers)

    return At(unstacked_model).layers(eqx.combine(stacked_layers, static_trunk))
