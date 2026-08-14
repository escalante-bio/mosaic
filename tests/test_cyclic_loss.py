import jax
import jax.numpy as jnp
import numpy as np

from mosaic.geometry import cyclic_offset_matrix
from mosaic.losses.structure_prediction import (
    HelixLoss,
    WithinBinderContact,
    StructureModelOutput,
)


def _random_output(binder_len: int, key) -> StructureModelOutput:
    n = binder_len
    bins = jnp.linspace(2.0, 20.0, 8)
    distogram_logits = jax.random.normal(key, (n, n, len(bins)))
    return StructureModelOutput(
        distogram_logits=distogram_logits,
        distogram_bins=bins,
        plddt=jnp.ones(n),
        pae=jnp.zeros((n, n)),
        pae_logits=jnp.zeros((n, n, 1)),
        pae_bins=jnp.zeros(1),
        structure_coordinates=jnp.zeros((n, 37, 3)),
        backbone_coordinates=jnp.zeros((n, 4, 3)),
        full_sequence=jax.nn.one_hot(jnp.zeros(n, dtype=jnp.int32), 20),
        asym_id=jnp.zeros(n, dtype=jnp.int32),
        residue_idx=jnp.arange(n),
        atom37_coords=jnp.zeros((n, 37, 3)),
        atom37_mask=jnp.zeros((n, 37)),
    )


def test_within_binder_contact_cyclic_false_is_regression_safe():
    binder_len = 10
    key = jax.random.PRNGKey(0)
    output = _random_output(binder_len, key)
    sequence = jnp.zeros((binder_len, 20))

    loss_default = WithinBinderContact(min_sequence_separation=5)
    loss_explicit_linear = WithinBinderContact(min_sequence_separation=5, cyclic=False)

    value_default, aux_default = loss_default(sequence, output, key=None)
    value_explicit, aux_explicit = loss_explicit_linear(sequence, output, key=None)

    assert value_default == value_explicit
    assert aux_default == aux_explicit


def test_within_binder_contact_cyclic_true_differs_from_linear():
    binder_len = 10
    key = jax.random.PRNGKey(1)
    output = _random_output(binder_len, key)
    sequence = jnp.zeros((binder_len, 20))

    linear_loss = WithinBinderContact(min_sequence_separation=5, cyclic=False)
    cyclic_loss = WithinBinderContact(min_sequence_separation=5, cyclic=True)

    linear_value, _ = linear_loss(sequence, output, key=None)
    cyclic_value, _ = cyclic_loss(sequence, output, key=None)

    # with min_sequence_separation=5 and binder_len=10, the wraparound pair
    # (0, 9) has cyclic distance 1 (excluded) but linear distance 9 (included),
    # so the two masks genuinely differ and should produce different losses.
    assert linear_value != cyclic_value


def test_helix_loss_cyclic_false_is_regression_safe():
    binder_len = 12
    key = jax.random.PRNGKey(2)
    output = _random_output(binder_len, key)
    sequence = jnp.zeros((binder_len, 20))

    loss_default = HelixLoss()
    loss_explicit_linear = HelixLoss(cyclic=False)

    value_default, aux_default = loss_default(sequence, output, key=None)
    value_explicit, aux_explicit = loss_explicit_linear(sequence, output, key=None)

    assert value_default == value_explicit
    assert aux_default == aux_explicit


def test_helix_loss_cyclic_true_includes_wraparound_i_plus_3_pairs():
    binder_len = 12
    key = jax.random.PRNGKey(3)
    output = _random_output(binder_len, key)
    sequence = jnp.zeros((binder_len, 20))

    linear_loss = HelixLoss(cyclic=False)
    cyclic_loss = HelixLoss(cyclic=True)

    linear_value, _ = linear_loss(sequence, output, key=None)
    cyclic_value, _ = cyclic_loss(sequence, output, key=None)

    # jnp.diagonal(log_contact, 3) only ever sees binder_len - 3 pairs and never
    # the wraparound i,i+3 pairs near the C-terminus, so cyclic mode (which
    # includes those wraparound pairs) should generally produce a different
    # value for a random (non-degenerate) contact map.
    assert linear_value != cyclic_value

    # sanity: the cyclic mask should select exactly 2*binder_len pairs (each
    # residue has both a forward and backward "distance-3" neighbor around
    # the ring, i.e. (i, i+3) and (i, i-3)).
    offset = np.abs(cyclic_offset_matrix(binder_len, offset_type=1))
    assert (offset == 3).sum() == 2 * binder_len
