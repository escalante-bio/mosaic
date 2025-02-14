import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int, PRNGKeyArray

from ._custom_types import AnyArray
from ._layer import TransformerLayer


@jax.jit
def _randint(key: PRNGKeyArray, index: Int[ArrayLike, ""], maxval: Int[ArrayLike, ""]):
    return jr.randint(jr.fold_in(key, index), (), minval=0, maxval=maxval)


_alphabet = {
    "b": 0,  # beginning-of-sequence / 'cls'
    "p": 1,  # pad
    "e": 2,  # end-of-sequence
    "u": 3,  # unknown
    "L": 4,
    "A": 5,
    "G": 6,
    "V": 7,
    "S": 8,
    "E": 9,
    "R": 10,
    "T": 11,
    "I": 12,
    "D": 13,
    "P": 14,
    "K": 15,
    "Q": 16,
    "N": 17,
    "F": 18,
    "Y": 19,
    "M": 20,
    "H": 21,
    "W": 22,
    "C": 23,
    "X": 24,
    "B": 25,
    "U": 26,
    "Z": 27,
    "O": 28,
    ".": 29,
    "-": 30,
    "1": 31,  # null_1
    "m": 32,  # mask
}
assert len(_alphabet) == max(_alphabet.values()) + 1


def tokenise(
    proteins: list[str], length: None | int = None, key: None | PRNGKeyArray = None
) -> Int[np.ndarray, "batch length"]:
    """Tokenises a batch of proteins, each represented as strings. Will include start
    and stop tokens.

    **Arguments:**

    - `proteins`: a list of proteins, each in FASTA format.
    - `length`: the length to pad or truncate to. If not passed then defaults to two
        greater than the maximum length of all `proteins`. (The extra two is to fit the
        start and stop token.) Padding is always done at the end of the protein;
        truncation is done randomly along its length.
    - `key`: a `jax.random.key`; must be provided if truncating due to `length` being
        shorter than some of the input protein sequences. Truncation is deterministic
        with respect to this key.

    **Returns:**

    The tokenised sequences.

    !!! Example

        ```python
        from esm2quinox import tokenise
        proteins = tokenise(["SPIDERMAN", "FOO"])
        proteins = tokenise(["SPIDERMAN", "FOO"], length=4, key=jax.random.key(0))
        ```
    """
    if length is None:
        # +2 for start and stop.
        length = max(map(len, proteins)) + 2
    out = np.full((len(proteins), length), _alphabet["p"])
    for protein_index, protein in enumerate(proteins):
        protein = f"b{protein.upper()}e"
        if len(protein) > length:
            if key is None:
                raise ValueError(
                    "Must pass tokenise(..., key=...) when cropping to lengths shorter "
                    "than the input."
                )
            start = _randint(key, protein_index, len(protein) - length + 1)
            protein = protein[start : start + length]
        for residue_index, residue in enumerate(protein):
            out[protein_index, residue_index] = _alphabet[residue]
    return out


class ESM2Result(eqx.Module):
    """The output result from calling `esm2quinox.ESM2.__call__`. Has the `.hidden`
    representation from the final layer of the model, and the `.logits` (pre-softmax)
    from mapping that hidden layers through a prediction head.
    """

    hidden: Float[Array, "length embed_size"]
    logits: Float[Array, "length alphabet_size"]


class LogitHead(eqx.Module):
    layer_norm: eqx.nn.LayerNorm
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, embed_size: int, alphabet_size: int, key: PRNGKeyArray):
        key1, key2 = jr.split(key)
        self.layer_norm = eqx.nn.LayerNorm(embed_size)
        self.linear1 = eqx.nn.Linear(embed_size, embed_size, key=key1)
        self.linear2 = eqx.nn.Linear(embed_size, alphabet_size, key=key2)

    def __call__(self, hidden: Float[Array, " embed_size"]):
        x = self.linear1(hidden)
        x = jax.nn.gelu(x, approximate=False)
        x = self.layer_norm(x)
        logits = self.linear2(x)
        return logits


class ESM2(eqx.Module):
    """The masked language modelling trunk of ESM2."""

    num_layers: int = eqx.field(static=True)
    embed_size: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    token_dropout: bool = eqx.field(static=True)
    alphabet: dict[str, int] = eqx.field(static=True)

    layers: TransformerLayer
    layer_norm: eqx.nn.LayerNorm
    logit_head: LogitHead

    def __init__(
        self,
        num_layers: int,
        embed_size: int,
        num_heads: int,
        token_dropout: bool,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `num_layers`: the number of transformer layers.
        - `embed_size`: the size of the embedding that is propagated from layer to
            layer.
        - `num_heads`: how many heads to use in the multihead attention.
        - `token_dropout`: whether to scale the input embeddings (before the transformer
            layers) by the number of mask tokens present. If `False` then the embeddings
            are left unchanged. If `True` then embeddings are scaled by
            ```python
            0.88 / (1 - (number_of_masks / number_of_not_pads))
            ```
        - `key`: a random key for initialising each layer.
        """
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.token_dropout = token_dropout
        self.alphabet = _alphabet.copy()

        keys = jr.split(key, num_layers + 1)
        layer_keys = keys[:-1]
        logit_key = keys[-1]

        self.layers = eqx.filter_vmap(TransformerLayer)(
            embed_size, 4 * embed_size, num_heads, layer_keys
        )
        self.layer_norm = eqx.nn.LayerNorm(embed_size)
        self.logit_head = LogitHead(embed_size, len(self.alphabet), logit_key)

    @property
    def embedding(self):
        return eqx.nn.Embedding(
            num_embeddings=len(self.alphabet),
            embedding_size=self.embed_size,
            weight=self.logit_head.linear2.weight,
        )

    def __call__(self, tokens: str | Int[AnyArray, " length"]) -> ESM2Result:
        """**Arguments:**

        - `tokens`: the input tokens. May either be a JAX array of shape `(length,)` or
            (for convenience when hacking around), a raw Python string; in the latter
            case then it will be tokenised before calling the model.

        If you need to process a batch of data then wrap your model in `jax.vmap`.

        **Returns:**

        An `esm2quinox.ESM2Result` object.

        !!! Example

            ```python
            proteins = esm2quinox.tokenise(["SPIDERMAN", "SPUDMAN"])
            out = jax.vmap(model)(proteins)
            ```
        """
        if isinstance(tokens, str):
            # Make it possible to just call it directly for one-offs.
            # +2 to handle the beginning-of-sequence and end-of-sequence tokens.
            [tokens] = tokenise([tokens], length=len(tokens) + 2, key=None)
            assert type(tokens) is np.ndarray
        return self._call(tokens)

    @eqx.filter_jit
    def _call(self, tokens: Int[AnyArray, " length"]) -> ESM2Result:
        x = jax.vmap(self.embedding)(tokens)
        is_pad = tokens == _alphabet["p"]
        not_pad = jnp.logical_not(is_pad)
        if self.token_dropout:
            is_mask = tokens == _alphabet["m"]
            x = jnp.where(is_mask[:, None], 0, x)
            mask_ratio_train = 0.15 * 0.8
            mask_ratio_observed = is_mask.sum() / not_pad.sum()
            factor = (1 - mask_ratio_train) / (1 - mask_ratio_observed)
            x = x * factor
        x = jnp.where(not_pad[:, None], x, 0)

        dynamic_layers, static_layer = eqx.partition(self.layers, eqx.is_array)

        def f(x, dynamic_layer):
            layer = eqx.combine(dynamic_layer, static_layer)
            x = layer(x, is_pad=is_pad)
            return x, None

        x, _ = lax.scan(f, x, xs=dynamic_layers)
        hidden = jax.vmap(self.layer_norm)(x)
        logits = jax.vmap(self.logit_head)(hidden)

        return ESM2Result(hidden=hidden, logits=logits)
