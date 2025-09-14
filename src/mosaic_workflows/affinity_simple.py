from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class AffinitySimpleReadout(eqx.Module):
    """Minimal Boltz2-style affinity heads in JAX.

    - Pools cross pairs from z to vector g
    - Two-layer MLP to token_s
    - Two-layer MLP to scalar value
    """

    token_z: int
    token_s: int
    out1_w: jnp.ndarray
    out2_w: jnp.ndarray
    val1_w: jnp.ndarray
    val2_w: jnp.ndarray
    val3_w: jnp.ndarray

    def __init__(self, *, token_z: int, token_s: int, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.token_z = int(token_z)
        self.token_s = int(token_s)
        self.out1_w = 0.02 * jax.random.normal(k1, (token_z, token_z))
        self.out2_w = 0.02 * jax.random.normal(k2, (token_s, token_z))
        self.val1_w = 0.02 * jax.random.normal(k3, (token_s, token_s))
        self.val2_w = 0.02 * jax.random.normal(k4, (token_s, token_s))
        self.val3_w = 0.02 * jax.random.normal(k5, (1, token_s))

    def __call__(
        self,
        *,
        z: jnp.ndarray,                 # [N,N,Z]
        s_inputs: jnp.ndarray,          # [N,S] (unused but kept for parity)
        feats: dict,
    ) -> jnp.ndarray:
        pad = feats["token_pad_mask"][0].astype(bool)
        rec = (feats["mol_type"][0] == 0) & pad
        lig = feats["affinity_token_mask"][0].astype(bool) & pad
        pair = (lig[:, None] & rec[None, :]) | (rec[:, None] & lig[None, :]) | (lig[:, None] & lig[None, :])
        pair = pair & (~jnp.eye(pair.shape[0], dtype=bool))

        denom = jnp.maximum(pair.sum((0, 1)), 1.0)
        g = (z * pair[..., None]).sum((0, 1)) / denom  # [Z]

        # out mlp to token_s
        h = jax.nn.relu(g @ self.out1_w.T)
        h = jax.nn.relu(h @ self.out2_w.T)

        # regression head
        v = jax.nn.relu(h @ self.val1_w.T)
        v = jax.nn.relu(v @ self.val2_w.T)
        v = (v @ self.val3_w.T).squeeze()
        return v


