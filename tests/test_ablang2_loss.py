import numpy as np
import torch
import jax
import jax.numpy as jnp
from mosaic.losses import ablang2 as ablang2_loss_mod
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


@pytest.mark.slow
def test_ablang2_designable_pseudo_likelihood_matches_direct_computation():
    """Check that Ablang2PseudoLikelihood PPL matches ablang2's
    ``pretrained(mode='pseudo_log_likelihood')`` on a single heavy chain."""
    heavy = (
        "EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQG"
        "RVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS"
    )
    n = len(heavy)

    ab2 = ablang2.pretrained("ablang2-paired", device=TORCH_DEVICE)
    ref_pll = ab2([[heavy, ""]], mode="pseudo_log_likelihood")
    expected_ppl = float(np.exp(-ref_pll[0]))

    model, tok = ablang2_loss_mod._cached_ablang2_model()
    seq_standard_tokens = jax.nn.one_hot(
        jnp.array([TOKENS.index(aa) for aa in heavy], dtype=jnp.int32),
        len(TOKENS),
    )
    loss_term = ablang2_loss_mod.Ablang2PseudoLikelihood(
        model=model,
        tokenizer=tok,
        chain_slices=(("H", 0, n),),
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
def test_ablang2_designable_pseudo_likelihood_light_only_matches_ablang2():
    """Check that Ablang2PseudoLikelihood PPL matches ablang2's
    ``pretrained(mode='pseudo_log_likelihood')`` on a single light chain."""
    light = (
        "DIQLTQSPLSLPVTLGQPASISCRSSQSLEASDTNIYLSWFQQRPGQSPRRLIYKISNRDSGVPD"
        "RFSGSGSGTHFTLRISRVEADDVAVYYCMQGTHWPPAFGQGTKVDIK"
    )
    n = len(light)

    ab2 = ablang2.pretrained("ablang2-paired", device=TORCH_DEVICE)
    ref_pll = ab2([["", light]], mode="pseudo_log_likelihood")
    expected_ppl = float(np.exp(-ref_pll[0]))

    model, tok = ablang2_loss_mod._cached_ablang2_model()
    seq_standard_tokens = jax.nn.one_hot(
        jnp.array([TOKENS.index(aa) for aa in light], dtype=jnp.int32),
        len(TOKENS),
    )
    loss_term = ablang2_loss_mod.Ablang2PseudoLikelihood(
        model=model,
        tokenizer=tok,
        chain_slices=(("L", 0, n),),
        stop_grad=True,
    )
    (loss_value, aux), grad = optimizers_module._eval_loss_and_grad(
        loss_term, seq_standard_tokens, jax.random.key(0)
    )

    assert float(aux["ablang2_ppl"]) == pytest.approx(expected_ppl, rel=1e-3)
    assert float(loss_value) == pytest.approx(float(-ref_pll[0]), rel=1e-3)
    assert grad.shape == seq_standard_tokens.shape


@pytest.mark.slow
def test_ablang2_designable_pseudo_likelihood_paired_matches_ablang2():
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
    n = len(full_seq)
    n_h = len(heavy)

    ab2 = ablang2.pretrained("ablang2-paired", device=TORCH_DEVICE)
    ref_pll = ab2([[heavy, light]], mode="pseudo_log_likelihood")
    expected_ppl = float(np.exp(-ref_pll[0]))

    model, tok = ablang2_loss_mod._cached_ablang2_model()
    seq_standard_tokens = jax.nn.one_hot(
        jnp.array([TOKENS.index(aa) for aa in full_seq], dtype=jnp.int32),
        len(TOKENS),
    )
    loss_term = ablang2_loss_mod.Ablang2PseudoLikelihood(
        model=model,
        tokenizer=tok,
        chain_slices=(("H", 0, n_h), ("L", n_h, n)),
        stop_grad=True,
    )
    (loss_value, aux), grad = optimizers_module._eval_loss_and_grad(
        loss_term, seq_standard_tokens, jax.random.key(0)
    )

    assert float(aux["ablang2_ppl"]) == pytest.approx(expected_ppl, rel=1e-3)
    assert float(loss_value) == pytest.approx(float(-ref_pll[0]), rel=1e-3)
    assert grad.shape == seq_standard_tokens.shape


@pytest.mark.slow
def test_ablang2_designable_pseudo_likelihood_matches_per_residue_aggregation():
    """Verify that scoring a subset of designable positions matches
    aggregating ablang2's per-residue PLLs over that subset."""

    heavy = (
        "EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQG"
        "RVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS"
    )
    n = len(heavy)
    designable = [1, 3, 10, 50]

    ab2 = ablang2.pretrained("ablang2-paired", device=TORCH_DEVICE)
    labels = ab2.tokenizer(
        [[heavy, ""]], pad=True, w_extra_tkns=True, device=ab2.used_device
    )
    idxs = (
        ~torch.isin(labels, torch.tensor(ab2.tokenizer.all_special_tokens))
    ).nonzero()
    masked_tokens = labels.repeat(len(idxs), 1)
    for num, idx in enumerate(idxs):
        masked_tokens[num, idx[1]] = ab2.tokenizer.mask_token
    with torch.no_grad():
        logits = ab2.AbLang(masked_tokens)
    logits[:, :, ab2.tokenizer.all_special_tokens] = -float("inf")
    logits = torch.stack([logits[num, idx[1]] for num, idx in enumerate(idxs)])
    labels_flat = labels[:, idxs[:, 1:]].squeeze(2)[0]
    per_residue_nll = torch.nn.functional.cross_entropy(
        logits, labels_flat, reduction="none"
    )
    expected_loss = float(per_residue_nll[designable].mean())

    model, tok = ablang2_loss_mod._cached_ablang2_model()
    seq_standard_tokens = jax.nn.one_hot(
        jnp.array([TOKENS.index(aa) for aa in heavy], dtype=jnp.int32),
        len(TOKENS),
    )
    loss_term = ablang2_loss_mod.Ablang2PseudoLikelihood(
        model=model,
        tokenizer=tok,
        chain_slices=(("H", 0, n),),
        designable_positions=jnp.array(designable, dtype=jnp.int32),
        stop_grad=True,
    )
    (loss_value, _), _ = optimizers_module._eval_loss_and_grad(
        loss_term, seq_standard_tokens, jax.random.key(0)
    )

    assert float(loss_value) == pytest.approx(expected_loss, rel=1e-3)
