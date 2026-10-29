"""A Protenix design pack with a PSSM reproduces Protenix's own featurization of
the designed binder (CPU featurization only; no weights)."""

import equinox as eqx
import jax
import numpy as np
import pytest

pytest.importorskip("protenix")

from protenix.atom_padding import ATOM_FEATURE_AXES
from protenix.data.template import ChainInput, featurize

from mosaic.common import TOKENS
from mosaic.losses.protenix import binder_design_features, set_binder_sequence
from mosaic.models.protenix import atom_templates

TARGET = [ChainInput("MKTAYIAKQRQISFVKSHFSRQ", compute_msa=False), ChainInput("VVGADGVGK", compute_msa=False)]


def residue_distances(ref_pos, a2t):
    """Intra-residue distance matrices: `ref_pos` up to each residue's random rigid augmentation."""
    return [np.linalg.norm(ref_pos[a2t == t][:, None] - ref_pos[a2t == t][None], axis=-1) for t in np.unique(a2t)]


def mismatches(native, refreshed):
    """Fields of `refreshed` (real-atom prefix) that differ from `native`."""
    n = len(native["atom_to_token_idx"])
    assert int(refreshed["atom_pad_mask"].sum()) == n and refreshed["atom_pad_mask"][:n].all()
    bad = []
    for name, value in native.items():
        new = jax.tree.map(np.asarray, refreshed[name])
        if name in ATOM_FEATURE_AXES:
            new, tail = (new[:n, :n], new[n:]) if name == "bond_mask" else (new[:n], new[n:])
            if np.any(tail):
                bad.append(f"{name} (padding)")
        if name == "ref_pos":
            d = zip(residue_distances(value, native["atom_to_token_idx"]), residue_distances(new, native["atom_to_token_idx"]))
            same = all(np.allclose(a, b, atol=1e-4) for a, b in d)
        else:
            same = jax.tree.all(jax.tree.map(lambda a, b: a.shape == b.shape and np.allclose(a, b), value, new))
        if not same:
            bad.append(name)
    return bad


@pytest.mark.slow
@pytest.mark.parametrize("sequence", ["GCPWKDEGRW", "PPGGCCWWYG", "".join(TOKENS[i % 20] for i in range(7, 17))])
def test_design_pack_matches_native_featurization(sequence):
    pack = binder_design_features(
        featurize([ChainInput("W" * len(sequence), compute_msa=False), *TARGET])[0], len(sequence), atom_templates()
    )
    native = featurize([ChainInput(sequence, compute_msa=False), *TARGET])[0]
    pssm = jax.nn.one_hot(np.array([TOKENS.index(a) for a in sequence]), 20)
    refreshed = eqx.filter_jit(set_binder_sequence)(pssm, pack, jax.random.key(0))
    assert mismatches(native, refreshed) == []


def msa_dir(path, query, hits):
    """A precomputed-MSA directory (unpaired hits, query-only pairing) for `ChainInput`."""
    path.mkdir()
    (path / "non_pairing.a3m").write_text("".join(f">{i}\n{s}\n" for i, s in enumerate([query, *hits])))
    (path / "pairing.a3m").write_text(f">query\n{query}\n")
    return str(path)


T = "MKTAYIAKQRQISFVKSHFSRQ"
T_HITS = ["MKTAYLAKQRQISFVKSHFSRQ", "MKSAYIAKQRQ-SFVKSHFSRQ", "MRTAYIAKERQISFVKSHFSRA"]


@pytest.mark.slow
@pytest.mark.parametrize("placeholder", ["X", "W"])
def test_plain_features_sequence_identity(tmp_path, placeholder):
    """Plain dict, binder without an MSA, target with one: `restype`, `profile`
    (the binder's own, depth 1) and the MSA query row match native featurization."""
    sequence = "GCPWKDEGRW"
    target = ChainInput(T, compute_msa=False, precomputed_msa_dir=msa_dir(tmp_path / "t", T, T_HITS))
    plain = featurize([ChainInput(placeholder * len(sequence), compute_msa=False), target])[0]
    native = featurize([ChainInput(sequence, compute_msa=False), target])[0]
    pssm = jax.nn.one_hot(np.array([TOKENS.index(a) for a in sequence]), 20)
    new = jax.jit(set_binder_sequence)(pssm, plain, jax.random.key(0))
    for name in ("restype", "profile", "msa"):
        np.testing.assert_allclose(np.asarray(new[name]), native[name], atol=1e-6, err_msg=name)


@pytest.mark.slow
def test_plain_features_binder_msa(tmp_path):
    """Plain dict, binder with its own MSA: swapping the query keeps the hits'
    contribution to the profile (rank-one update by the binder's own depth)."""
    old, new_sequence, hits = "GCPWK", "ACPWR", ["GCPWR", "ACPWK"]
    target = ChainInput(T, compute_msa=False, precomputed_msa_dir=msa_dir(tmp_path / "t", T, T_HITS))
    binder = lambda q, name: ChainInput(q, compute_msa=False, precomputed_msa_dir=msa_dir(tmp_path / name, q, hits))
    plain = featurize([binder(old, "old"), target])[0]
    native = featurize([binder(new_sequence, "new"), target])[0]
    pssm = jax.nn.one_hot(np.array([TOKENS.index(a) for a in new_sequence]), 20)
    new = jax.jit(set_binder_sequence)(pssm, plain, jax.random.key(0))
    for name in ("restype", "profile", "msa"):
        np.testing.assert_allclose(np.asarray(new[name]), native[name], atol=1e-6, err_msg=name)
