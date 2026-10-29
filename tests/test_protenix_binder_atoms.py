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
