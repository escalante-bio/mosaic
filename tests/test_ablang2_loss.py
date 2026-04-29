import numpy as np
import torch
import jax
import jax.numpy as jnp
from mosaic.losses.ablang2 import load_ablang2, Ablang2PseudoLikelihood
from mosaic.losses.transformations import SetPositions
import ablang2
import mosaic.optimizers as optimizers_module
from mosaic.common import TOKENS

import pytest

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _default_jax_device():
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        return jax.devices("cpu")[0]


@pytest.fixture(autouse=True)
def _use_gpu():
    with jax.default_device(_default_jax_device()):
        yield


@pytest.fixture
def ablang2_jax():
    model, tok = load_ablang2()
    return model, tok


@pytest.fixture
def ablang2_torch():
    return ablang2.pretrained("ablang2-paired", device=TORCH_DEVICE)


@pytest.mark.slow
def test_ablang2_designable_pseudo_likelihood_matches_direct_computation(
    ablang2_jax, ablang2_torch
):
    """Check that Ablang2PseudoLikelihood PPL matches ablang2's
    ``pretrained(mode='pseudo_log_likelihood')`` on a single heavy chain."""
    heavy = (
        "EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQG"
        "RVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS"
    )
    n = len(heavy)

    ref_pll = ablang2_torch([[heavy, ""]], mode="pseudo_log_likelihood")
    expected_ppl = float(np.exp(-ref_pll[0]))

    model, tok = ablang2_jax
    seq_standard_tokens = jax.nn.one_hot(
        jnp.array([TOKENS.index(aa) for aa in heavy], dtype=jnp.int32),
        len(TOKENS),
    )
    loss_term = Ablang2PseudoLikelihood(
        model=model,
        tokenizer=tok,
        heavy_len=n,
        stop_grad=True,
    )
    (loss_value, aux), grad = optimizers_module._eval_loss_and_grad(
        loss_term, seq_standard_tokens, jax.random.key(0)
    )

    assert float(aux["ablang2_ppl"]) == pytest.approx(expected_ppl, rel=1e-3)
    assert float(loss_value) == pytest.approx(float(-ref_pll[0]), rel=1e-3)
    assert grad.shape == seq_standard_tokens.shape
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0)


@pytest.mark.slow
def test_ablang2_designable_pseudo_likelihood_light_only_matches_ablang2(
    ablang2_jax, ablang2_torch
):
    """Check that Ablang2PseudoLikelihood PPL matches ablang2's
    ``pretrained(mode='pseudo_log_likelihood')`` on a single light chain."""
    light = (
        "DIQLTQSPLSLPVTLGQPASISCRSSQSLEASDTNIYLSWFQQRPGQSPRRLIYKISNRDSGVPD"
        "RFSGSGSGTHFTLRISRVEADDVAVYYCMQGTHWPPAFGQGTKVDIK"
    )
    n = len(light)

    ref_pll = ablang2_torch([["", light]], mode="pseudo_log_likelihood")
    expected_ppl = float(np.exp(-ref_pll[0]))

    model, tok = ablang2_jax
    seq_standard_tokens = jax.nn.one_hot(
        jnp.array([TOKENS.index(aa) for aa in light], dtype=jnp.int32),
        len(TOKENS),
    )
    loss_term = Ablang2PseudoLikelihood(
        model=model,
        tokenizer=tok,
        heavy_len=0,
        stop_grad=True,
    )
    (loss_value, aux), grad = optimizers_module._eval_loss_and_grad(
        loss_term, seq_standard_tokens, jax.random.key(0)
    )

    assert float(aux["ablang2_ppl"]) == pytest.approx(expected_ppl, rel=1e-3)
    assert float(loss_value) == pytest.approx(float(-ref_pll[0]), rel=1e-3)
    assert grad.shape == seq_standard_tokens.shape


@pytest.mark.slow
def test_ablang2_designable_pseudo_likelihood_paired_matches_ablang2(
    ablang2_jax, ablang2_torch
):
    """Check that Ablang2PseudoLikelihood PPL matches ablang2's
    ``pretrained(mode='pseudo_log_likelihood')`` on a paired heavy+light input."""
    heavy = (
        "EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQG"
        "RVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS"
    )
    light = (
        "DIQLTQSPLSLPVTLGQPASISCRSSQSLEASDTNIYLSWFQQRPGQSPRRLIYKISNRDSGVPD"
        "RFSGSGSGTHFTLRISRVEADDVAVYYCMQGTHWPPAFGQGTKVDIK"
    )
    full_seq = heavy + light
    n_h = len(heavy)

    ref_pll = ablang2_torch([[heavy, light]], mode="pseudo_log_likelihood")
    expected_ppl = float(np.exp(-ref_pll[0]))

    model, tok = ablang2_jax
    seq_standard_tokens = jax.nn.one_hot(
        jnp.array([TOKENS.index(aa) for aa in full_seq], dtype=jnp.int32),
        len(TOKENS),
    )
    loss_term = Ablang2PseudoLikelihood(
        model=model,
        tokenizer=tok,
        heavy_len=n_h,
        stop_grad=True,
    )
    (loss_value, aux), grad = optimizers_module._eval_loss_and_grad(
        loss_term, seq_standard_tokens, jax.random.key(0)
    )

    assert float(aux["ablang2_ppl"]) == pytest.approx(expected_ppl, rel=1e-3)
    assert float(loss_value) == pytest.approx(float(-ref_pll[0]), rel=1e-3)
    assert grad.shape == seq_standard_tokens.shape


@pytest.mark.slow
def test_ablang2_designable_pseudo_likelihood_matches_per_residue_aggregation(
    ablang2_jax, ablang2_torch
):
    """Verify that scoring a subset of designable positions matches
    aggregating ablang2's per-residue PLLs over that subset."""
    heavy = (
        "EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQG"
        "RVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS"
    )
    n = len(heavy)
    designable = [1, 3, 10, 50]

    labels = ablang2_torch.tokenizer(
        [[heavy, ""]], pad=True, w_extra_tkns=True, device=ablang2_torch.used_device
    )
    idxs = (
        ~torch.isin(labels, torch.tensor(ablang2_torch.tokenizer.all_special_tokens))
    ).nonzero()
    masked_tokens = labels.repeat(len(idxs), 1)
    for num, idx in enumerate(idxs):
        masked_tokens[num, idx[1]] = ablang2_torch.tokenizer.mask_token
    with torch.no_grad():
        logits = ablang2_torch.AbLang(masked_tokens)
    logits[:, :, ablang2_torch.tokenizer.all_special_tokens] = -float("inf")
    logits = torch.stack([logits[num, idx[1]] for num, idx in enumerate(idxs)])
    labels_flat = labels[:, idxs[:, 1:]].squeeze(2)[0]
    per_residue_nll = torch.nn.functional.cross_entropy(
        logits, labels_flat, reduction="none"
    )
    expected_loss = float(per_residue_nll[designable].mean())

    model, tok = ablang2_jax
    seq_standard_tokens = jax.nn.one_hot(
        jnp.array([TOKENS.index(aa) for aa in heavy], dtype=jnp.int32),
        len(TOKENS),
    )
    loss_term = Ablang2PseudoLikelihood(
        model=model,
        tokenizer=tok,
        heavy_len=n,
        designable_positions=jnp.array(designable, dtype=jnp.int32),
        stop_grad=True,
    )
    (loss_value, _), _ = optimizers_module._eval_loss_and_grad(
        loss_term, seq_standard_tokens, jax.random.key(0)
    )

    assert float(loss_value) == pytest.approx(expected_loss, rel=1e-3)


@pytest.mark.slow
def test_setpositions_vs_designable_positions_gradients(ablang2_jax):
    """Check gradient behaviour when using SetPositions as a wrapper vs designable_positions.

    With designable_positions=variable_positions the PLL is averaged over M positions,
    so the gradient at each variable position is proportional to 1/M.

    With SetPositions wrapping and no designable_positions the PLL is averaged over all
    N positions.  Fixed positions are constants in the computation graph so their gradient
    w.r.t. the variable-only input is zero, but the normalization denominator is still N.
    The gradient at variable positions is therefore proportional to 1/N.

    Consequently the two approaches give the same gradient *direction* but differ in
    magnitude by the factor M/N.  This test verifies both facts.
    """
    heavy = (
        "EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQG"
        "RVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS"
    )
    designable = list(range(95, 103))  # 8 positions in CDR3
    n = len(heavy)
    m = len(designable)

    wildtype_with_x = "".join(
        "X" if i in designable else aa for i, aa in enumerate(heavy)
    )

    model, tok = ablang2_jax
    variable_positions = jnp.array(designable, dtype=jnp.int32)

    seq_full = jax.nn.one_hot(
        jnp.array([TOKENS.index(aa) for aa in heavy], dtype=jnp.int32),
        len(TOKENS),
    )

    # --- Approach 1: designable_positions passed explicitly ---
    loss_with_dp = Ablang2PseudoLikelihood(
        model=model,
        tokenizer=tok,
        heavy_len=n,
        designable_positions=variable_positions,
        stop_grad=True,
    )
    _, grad_full = optimizers_module._eval_loss_and_grad(
        loss_with_dp, seq_full, jax.random.key(0)
    )
    grad_at_variable = np.array(grad_full[variable_positions])  # (M, 20)
    grad_at_fixed = np.array(
        grad_full[jnp.array([i for i in range(n) if i not in designable])]
    )

    # Fixed positions should have zero gradient when designable_positions is set
    np.testing.assert_allclose(grad_at_fixed, 0.0, atol=1e-6)

    # --- Approach 2: SetPositions wrapping, no designable_positions ---
    wildtype_tokens = jnp.array(
        [TOKENS.index(aa) if aa != "X" else -1 for aa in wildtype_with_x],
        dtype=jnp.int32,
    )
    loss_no_dp = Ablang2PseudoLikelihood(
        model=model,
        tokenizer=tok,
        heavy_len=n,
        stop_grad=True,
    )
    set_pos_loss = SetPositions(wildtype_tokens, variable_positions, loss_no_dp)

    seq_variable = seq_full[variable_positions]  # (M, 20) — variable positions only
    _, grad_variable = optimizers_module._eval_loss_and_grad(
        set_pos_loss, seq_variable, jax.random.key(0)
    )

    # Approach 2 gradient = Approach 1 gradient * (M / N) due to differing normalisers
    np.testing.assert_allclose(
        np.array(grad_variable),
        grad_at_variable * (m / n),
        rtol=1e-3,
        atol=1e-4,
    )


@pytest.mark.slow
def test_setpositions_combined_with_designable_positions(ablang2_jax):
    """SetPositions + designable_positions together should give identical outputs to
    designable_positions alone.

    SetPositions reconstructs the same full sequence that approach 1 already receives
    (wildtype at fixed positions, optimised values at variable positions), so the inner
    Ablang2PseudoLikelihood sees exactly the same inputs in both cases.  The normali-
    sation denominator is M in both cases, so loss values and gradients must match.
    """
    heavy = (
        "EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQG"
        "RVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS"
    )
    designable = list(range(95, 103))
    n = len(heavy)

    wildtype_with_x = "".join(
        "X" if i in designable else aa for i, aa in enumerate(heavy)
    )

    model, tok = ablang2_jax
    variable_positions = jnp.array(designable, dtype=jnp.int32)

    seq_full = jax.nn.one_hot(
        jnp.array([TOKENS.index(aa) for aa in heavy], dtype=jnp.int32),
        len(TOKENS),
    )
    seq_variable = seq_full[variable_positions]  # (M, 20)

    # --- Approach 1: designable_positions, full sequence input ---
    loss_dp = Ablang2PseudoLikelihood(
        model=model,
        tokenizer=tok,
        heavy_len=n,
        designable_positions=variable_positions,
        stop_grad=True,
    )
    (loss_value_dp, _), grad_full = optimizers_module._eval_loss_and_grad(
        loss_dp, seq_full, jax.random.key(0)
    )

    # --- Approach 3: SetPositions + designable_positions, variable-only input ---
    wildtype_tokens = jnp.array(
        [TOKENS.index(aa) if aa != "X" else -1 for aa in wildtype_with_x],
        dtype=jnp.int32,
    )
    loss_combined = SetPositions(
        wildtype_tokens,
        variable_positions,
        Ablang2PseudoLikelihood(
            model=model,
            tokenizer=tok,
            heavy_len=n,
            designable_positions=variable_positions,
            stop_grad=True,
        ),
    )
    (loss_value_combined, _), grad_variable = optimizers_module._eval_loss_and_grad(
        loss_combined, seq_variable, jax.random.key(0)
    )

    # Loss values must be identical
    assert float(loss_value_combined) == pytest.approx(float(loss_value_dp), rel=1e-3)

    # Gradients at variable positions must be identical
    np.testing.assert_allclose(
        np.array(grad_variable),
        np.array(grad_full[variable_positions]),
        rtol=1e-3,
        atol=1e-4,
    )
