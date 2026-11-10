"""`ESMFold2` (featurized by `jfold.featurize`, jfold's forward pass; jfold's own
tests check the view against upstream's featurization and the converted checkpoints
against upstream's torch models): a design pack's binder, set to a sequence, is what the
view featurizes for that sequence; the defaults, the samplers, refusals, and a whole
design prediction and the structure written from it."""

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jfold.blocks.inputs import probabilities
from jfold.models.esmfold2 import RecyclingPairTrunk
from jfold.nn import Dropout
from jfold_testing import atom_positions, rel_err

from mosaic.common import TOKENS
from mosaic.models.esmfold2 import NUM_CLASSES, ESMFold2
from mosaic.structure_prediction import PolymerType, TargetChain

pytestmark = [pytest.mark.slow, pytest.mark.model_forward]

TARGET = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQCGVWPAACRERLGLKL"[
    :60
]
BINDER = "ACDEFGHIKLMW"
BINDER_LENGTH = len(BINDER)
# The release type's full checkpoint (an MSA encoder) and the experimental type's -Fast
# one (none).
CASES = ("base", "experimental_fast")


@pytest.fixture(scope="module", params=CASES)
def model(request):
    return getattr(ESMFold2, request.param)()


@pytest.fixture(scope="module")
def design(model):
    """A design pack for `TARGET` and its writer (ESMC's states at full precision)."""
    with jax.default_matmul_precision("highest"):
        return model.binder_features(
            BINDER_LENGTH,
            [TargetChain(sequence=TARGET, use_msa=False)],
            key=jax.random.key(0),
        )


def _onehot(sequence: str):
    return jax.nn.one_hot(jnp.array([TOKENS.index(c) for c in sequence]), 20)


def test_rebuilt_binder_is_the_designed_sequence(model, design):
    """`set_sequence` on a design pack gives the view's features of the designed sequence
    on its real atoms (the binder's packed at the front, the target's after them), the
    same atom37 slots, and ESMC's states for it (the binder's run on it alone)."""
    features, _ = design
    with jax.default_matmul_precision("highest"):
        ours = model.set_sequence(features, _onehot(BINDER), key=jax.random.key(0))
        ref, _ = model.target_only_features(
            [
                TargetChain(sequence=BINDER, use_msa=False),
                TargetChain(sequence=TARGET, use_msa=False),
            ],
            key=jax.random.key(0),
        )
    f, f_ref = ours.features, ref.features
    num_atoms = int(f_ref["atom_attention_mask"].sum())
    assert int(f["atom_attention_mask"].sum()) == num_atoms
    assert not f["atom_attention_mask"][num_atoms:].any()
    assert f.keys() == f_ref.keys()
    for k, v in f_ref.items():
        ours_k = f[k]
        if k in ("res_type", "msa"):
            v = probabilities(v, NUM_CLASSES)
        if k == "lm_hidden_states":
            error = rel_err(ours_k, v)
            print(f"ESMC states: {error:.2e}")
            assert error < 1e-5
            continue
        if v.shape[0] == f_ref["ref_pos"].shape[0]:  # by atom
            ours_k, v = ours_k[:num_atoms], v[:num_atoms]
        np.testing.assert_array_equal(np.asarray(ours_k), np.asarray(v), err_msg=k)
    np.testing.assert_array_equal(
        ours.atom37_slot[:num_atoms], ref.atom37_slot[:num_atoms]
    )


def test_unsupported(model):
    """What the view does not featurize raises; so does an MSA on a checkpoint without an
    MSA encoder."""
    for chains in (
        [TargetChain(sequence="ACGU", polymer_type=PolymerType.RNA, use_msa=False)],
        [TargetChain(sequence="MKZ", use_msa=False)],
    ):
        with pytest.raises(NotImplementedError):
            model.target_only_features(chains, key=jax.random.key(0))
    if model.model.trunk.msa is None:
        with pytest.raises(ValueError, match="no MSA encoder"):
            model.target_only_features(
                [TargetChain(sequence=TARGET, use_msa=True)], key=jax.random.key(0)
            )


def test_defaults(model):
    """Upstream `fold`'s: LM dropout 0.3, on at inference; the MSA profile zeroed for
    checkpoints with `disable_msa_features` (the experimental -Fast ones); 4 recycling
    steps in design, 21 at inference."""
    jmodel = model.model
    experimental = isinstance(jmodel.trunk, RecyclingPairTrunk)
    dropout = (jmodel.trunk_init if experimental else jmodel.trunk).lm_dropout
    assert (
        isinstance(dropout, Dropout) and dropout.rate == 0.3 and not dropout.inference
    )
    assert jmodel.input_embedder.msa_profile == (not experimental)
    assert (model.design_recycling_steps, model.inference_recycling_steps) == (4, 21)


def test_samplers(model):
    """Upstream's sampler and the default few-step one: decreasing noise levels down to
    zero, centred on the real atoms, rigidly aligned."""
    upstream, default = model.upstream_sampler(), model.sampler()
    assert len(upstream.sigmas) > len(default.sigmas)
    for sampler in (upstream, default):
        assert sampler.sigmas[-1] == 0
        assert (np.diff(sampler.sigmas) < 0).all()
        assert sampler.center.__name__ == "masked_centering"
        assert sampler.align.__name__ == "rigid_align"


def test_predicts(model, design):
    """A full design-pack prediction (default LM dropout, jitted, the binder's atoms and ESMC
    states rebuilt in the trace) runs and gives a sensible structure and confidence, written
    with the binder's atoms. With the upstream sampler: the default few-step sampler
    compresses this target's backbone (median CA gap ~2.9 Å)."""
    features, writer = design
    pssm = jax.nn.one_hot(
        jax.random.randint(jax.random.key(3), (BINDER_LENGTH,), 0, 20), 20
    )
    prediction = model.predict(
        PSSM=pssm,
        features=features,
        writer=writer,
        sampler=model.upstream_sampler(),
        key=jax.random.key(4),
    )
    output = prediction.model_output
    assert jnp.isfinite(output.structure_coordinates).all()
    assert ((output.plddt >= 0) & (output.plddt <= 1)).all()
    # Consecutive CA atoms of the target sit ~3.8 Å apart.
    ca = output.backbone_coordinates[BINDER_LENGTH:, 1]
    gaps = jnp.linalg.norm(ca[1:] - ca[:-1], axis=-1)
    assert 3.0 < float(jnp.median(gaps)) < 4.5
    # The binder written as `argmax(pssm)`, with that sequence's atoms, then the target.
    written = prediction.st
    names = [
        gemmi.expand_one_letter(TOKENS[k], gemmi.ResidueKind.AA)
        for k in pssm.argmax(-1)
    ]
    assert [r.name for r in written[0][0]] == names
    set_features = model.set_sequence(features, pssm, key=jax.random.key(5))
    real = set_features.features["atom_attention_mask"]
    assert written[0].count_atom_sites() == int(real.sum())
    assert np.allclose(
        atom_positions(written), output.structure_coordinates[real], atol=1e-3
    )
