import jax.numpy as jnp

from mosaic.losses.structure_prediction import reduce_samples
from mosaic.optimizers import _ranking_leaf


def test_ranking_leaf_found_as_bare_scalar():
    aux = {"structure": {"ranking_loss": jnp.array(0.5), "iptm": jnp.array(-0.5)}}
    assert _ranking_leaf(aux, "ranking_loss") == 0.5


def test_ranking_leaf_found_through_list_wrapping():
    # reduce_samples wraps per-sample scalar metrics in a loss-sorted list, which
    # a terminal-key match would miss. Matching anywhere in the path still finds it.
    aux = {"structure": {"ranking_loss": [jnp.array(0.5)]}}
    assert _ranking_leaf(aux, "ranking_loss") == 0.5


def test_ranking_leaf_multisample_returns_best_sample():
    # Sorted ascending by loss, so index 0 is the best sample.
    aux = {"ranking_loss": [jnp.array(0.1), jnp.array(0.7), jnp.array(0.9)]}
    assert _ranking_leaf(aux, "ranking_loss") == 0.1


def test_ranking_leaf_absent_returns_none():
    aux = {"structure": {"iptm": jnp.array(-0.5)}}
    assert _ranking_leaf(aux, "ranking_loss") is None


def test_ranking_leaf_resolves_through_reduce_samples_output():
    # End-to-end with the real reduce_samples: a per-sample ranking_loss metric
    # survives the list wrapping and resolves to the best (lowest-loss) sample.
    values = jnp.array([0.7, 0.1, 0.9])  # sample 1 is best
    auxiliary = {"ranking_loss": jnp.array([0.7, 0.1, 0.9])}
    _, aux = reduce_samples(values, auxiliary, jnp.min, num_samples=3)
    assert _ranking_leaf(aux, "ranking_loss") == 0.1
