"""Mol-1 (jol1) wrapper: binder repacking, sequence injection, losses and prediction.

Loads the converted ``mol-1-base`` weights (downloaded and converted on first use).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.common import TOKENS
from mosaic.losses.mol1 import prepare_features, set_binder_sequence
from mosaic.losses.structure_prediction import BinderTargetContact, PLDDTLoss
from mosaic.models.mol1 import Mol1Base
from mosaic.structure_prediction import PolymerType, TargetChain

pytestmark = pytest.mark.slow

TARGET = "MAEVIRSSAFWRSFPIFEEFDSETLCELSGIASYRKWSAGTVIFQRGD"
BINDER = "GWYKCPSHDLIFEGMRAQNTVG"  # every residue type, Gly at both ends


@pytest.fixture(scope="module")
def model():
    return Mol1Base()


@pytest.fixture(scope="module")
def chains():
    return [
        TargetChain(TARGET, use_msa=False),
        TargetChain("GACTCG", polymer_type=PolymerType.DNA, use_msa=False),
    ]


def one_hot(seq):
    return jax.nn.one_hot(jnp.array([TOKENS.index(c) for c in seq]), 20)


def test_repack_matches_featurization(model, chains):
    """Repacking the all-Trp design pack to a sequence gives exactly the features of
    featurizing that sequence, plus atom padding at the end."""
    L = len(BINDER)
    design, _ = model.binder_features(L, chains)
    repacked = prepare_features(design, one_hot(BINDER), jax.random.key(0))
    native, _ = model.target_only_features([TargetChain(BINDER, use_msa=False), *chains])

    f, g = repacked.features, native.features
    n = g.num_atoms
    assert f.num_tokens == g.num_tokens and f.num_atoms > n
    assert bool(f.atoms.atom_mask[:n].all()) and not bool(f.atoms.atom_mask[n:].any())
    for name in ("ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars",
                 "ref_space_uid", "atom_to_token", "atom_to_tokatom"):  # fmt: skip
        np.testing.assert_array_equal(getattr(f.atoms, name)[:n], getattr(g.atoms, name), name)
    np.testing.assert_array_equal(repacked.atom37_idx[:n], native.atom37_idx)
    assert bool((repacked.atom37_idx[n:] == -1).all())
    np.testing.assert_array_equal(f.tokens.rep_atom, g.tokens.rep_atom)
    np.testing.assert_array_equal(repacked.plddt_atom_idx, native.plddt_atom_idx)
    for name in ("restype", "profile"):
        np.testing.assert_array_equal(getattr(f.tokens, name), getattr(g.tokens, name), name)
    np.testing.assert_array_equal(f.msa.msa[0], g.msa.msa[0])

    # Reference conformers come in a random pose per residue: same internal geometry.
    def distances(pos, uid):
        x = np.asarray(pos)[np.asarray(g.atoms.ref_space_uid) == uid]
        return np.linalg.norm(x[:, None] - x[None], axis=-1)

    for uid in np.unique(np.asarray(g.atoms.ref_space_uid)):
        np.testing.assert_allclose(
            distances(f.atoms.ref_pos[:n], uid), distances(g.atoms.ref_pos, uid), atol=1e-4
        )
    # ... and a fresh pose every step.
    other = prepare_features(design, one_hot(BINDER), jax.random.key(1)).features
    binder = np.asarray(g.atoms.atom_to_token) < L
    assert not np.allclose(other.atoms.ref_pos[:n][binder], f.atoms.ref_pos[:n][binder])

    # With the same poses, the model sees the same system.
    g = eqx.tree_at(lambda g: g.atoms.ref_pos, g, f.atoms.ref_pos[:n])
    embed = eqx.filter_jit(lambda m, f: m.embed(f).s_inputs)
    s_repacked, s_native = embed(model.mol1, f), embed(model.mol1, g)
    err = np.abs(s_repacked - s_native).max() / np.abs(s_native).max()
    assert err < 1e-5, err


def test_soft_sequence_reaches_every_channel(model, chains):
    L = len(BINDER)
    design, _ = model.binder_features(L, chains)
    pssm = jax.nn.softmax(jax.random.normal(jax.random.key(0), (L, 20)))
    f = set_binder_sequence(pssm, design.features)
    for x in (f.tokens.restype[:L, :20], f.tokens.profile[:L, :20], f.msa.msa[0, :L, :20]):
        np.testing.assert_allclose(x, pssm, rtol=1e-6)


def test_profile_with_binder_msa(model, chains, tmp_path, monkeypatch):
    """With an MSA on the binder, setting its sequence gives the features of featurizing
    the new sequence with the same MSA hits."""
    rng = np.random.default_rng(0)
    hits = ["".join(rng.choice(list(TOKENS), len(BINDER))) for _ in range(5)]
    designed = BINDER[::-1]

    def fetch_msa(sequence):
        path = tmp_path / f"{sequence}.a3m"
        path.write_text("".join(f">{i}\n{s}\n" for i, s in enumerate([sequence, *hits])))
        return path

    monkeypatch.setattr("mosaic.models.mol1._fetch_msa", fetch_msa)
    pack, _ = model.target_only_features([TargetChain(BINDER), *chains])
    native, _ = model.target_only_features([TargetChain(designed), *chains])
    f, g = set_binder_sequence(one_hot(designed), pack.features), native.features
    for name in ("restype", "profile"):
        np.testing.assert_allclose(getattr(f.tokens, name), getattr(g.tokens, name), atol=1e-6)
    np.testing.assert_array_equal(f.msa.msa, g.msa.msa)


def test_loss_gradient(model, chains):
    L = len(BINDER)
    design, _ = model.binder_features(L, chains)
    loss = model.build_loss(
        loss=BinderTargetContact() + PLDDTLoss(), features=design, recycling_steps=2
    )
    pssm = jax.nn.softmax(jax.random.normal(jax.random.key(0), (L, 20)))

    @eqx.filter_jit
    def value_and_grad(loss, pssm, key):
        return jax.value_and_grad(lambda p: loss(p, key=key)[0])(pssm)

    value, grad = value_and_grad(loss, pssm, jax.random.key(1))
    assert np.isfinite(value) and np.isfinite(grad).all() and np.abs(grad).max() > 0


def test_predict(model, chains):
    L = len(BINDER)
    design, _ = model.binder_features(L, chains)
    pred = model.predict(
        PSSM=one_hot(BINDER), features=design, recycling_steps=2, key=jax.random.key(0)
    )
    assert pred.plddt.shape == (L + len(TARGET) + 6,)
    assert len(pred.st[0][0]) == L  # atom37 view: binder first

    native, atom_array = model.target_only_features(chains)
    pred = model.predict(
        features=native, writer=atom_array, recycling_steps=2, key=jax.random.key(0)
    )
    assert sum(len(ch) for ch in pred.st[0]) == len(TARGET) + 6
    assert np.isfinite(pred.iptm)
