import jax
import jax.numpy as jnp

from mosaic.losses.structure_prediction import (
    CyclicClosureLoss,
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


def test_cyclic_closure_loss_only_scores_first_last_pair():
    binder_len = 10
    key = jax.random.PRNGKey(4)
    output = _random_output(binder_len, key)
    sequence = jnp.zeros((binder_len, 20))

    loss = CyclicClosureLoss()
    value, aux = loss(sequence, output, key=None)

    from mosaic.losses.structure_prediction import contact_cross_entropy

    expected_log_contact = contact_cross_entropy(
        output.distogram_logits, loss.closure_distance, bins=output.distogram_bins
    )
    expected_value = -expected_log_contact[0, -1]

    assert value == expected_value
    assert aux == {"cyclic_closure": expected_log_contact[0, -1]}


def test_cyclic_closure_loss_rewards_short_terminal_distance():
    binder_len = 6
    bins = jnp.linspace(2.0, 20.0, 8)
    sequence = jnp.zeros((binder_len, 20))
    loss = CyclicClosureLoss(closure_distance=4.0)

    def _output_with_terminal_logits(terminal_logits):
        distogram_logits = jnp.zeros((binder_len, binder_len, len(bins)))
        distogram_logits = distogram_logits.at[0, -1].set(terminal_logits)
        return StructureModelOutput(
            distogram_logits=distogram_logits,
            distogram_bins=bins,
            plddt=jnp.ones(binder_len),
            pae=jnp.zeros((binder_len, binder_len)),
            pae_logits=jnp.zeros((binder_len, binder_len, 1)),
            pae_bins=jnp.zeros(1),
            structure_coordinates=jnp.zeros((binder_len, 37, 3)),
            backbone_coordinates=jnp.zeros((binder_len, 4, 3)),
            full_sequence=jax.nn.one_hot(jnp.zeros(binder_len, dtype=jnp.int32), 20),
            asym_id=jnp.zeros(binder_len, dtype=jnp.int32),
            residue_idx=jnp.arange(binder_len),
            atom37_coords=jnp.zeros((binder_len, 37, 3)),
            atom37_mask=jnp.zeros((binder_len, 37)),
        )

    # all mass on the shortest bin (< closure_distance) vs. all mass on the
    # longest bin (> closure_distance): closure loss should be far lower
    # (more rewarding) in the "close" case.
    close_output = _output_with_terminal_logits(
        jnp.array([10.0] + [0.0] * (len(bins) - 1))
    )
    far_output = _output_with_terminal_logits(
        jnp.array([0.0] * (len(bins) - 1) + [10.0])
    )

    close_value, _ = loss(sequence, close_output, key=None)
    far_value, _ = loss(sequence, far_output, key=None)

    assert close_value < far_value
