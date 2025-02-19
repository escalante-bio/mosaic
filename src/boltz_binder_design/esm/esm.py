from esm import pretrained
import esm2quinox
import jax
import equinox as eqx

class ESM2(object):
    esm: esm2quinox.ESM2

    def __init__(self, *args, model=None, **kwargs):
        models = {
                "esm2_t6_8M_UR50D": pretrained.esm2_t6_8M_UR50D,
                "esm2_t12_35M_UR50D": pretrained.esm2_t12_35M_UR50D,
                "esm2_t30_150M_UR50D": pretrained.esm2_t30_150M_UR50D,
                "esm2_t33_650M_UR50D": pretrained.esm2_t33_650M_UR50D,
                "esm2_t36_3B_UR50D": pretrained.esm2_t36_3B_UR50D,
                "esm2_t48_15B_UR50D": pretrained.esm2_t48_15B_UR50D,
            }
        if model:
            if model not in models:
                raise ValueError(f"Unknown ESM2 model {model}")
            else:
                torch_model, _ = models[model]()
                self.esm = esm2quinox.from_torch(torch_model)
        else:
            self.esm = esm2quinox.ESM2(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.esm, attr)

    @eqx.filter_jit
    def _apply_trunk(self, x, is_pad):

        dynamic_layers, static_layer = eqx.partition(self.esm.layers, eqx.is_array)

        def f(x, dynamic_layer):
            layer = eqx.combine(dynamic_layer, static_layer)
            x = layer(x, is_pad=is_pad)
            return x, None

        x, _ = jax.lax.scan(f, x, xs=dynamic_layers)
        return jax.vmap(self.esm.layer_norm)(x)
