"""Test `Promera` with jfold's view and with ProMERA's own featurizer."""


import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jfold_testing import atom_labels, atom_positions

from mosaic import msa
from mosaic.models.promera import Promera, view_features
from mosaic.structure_prediction import PolymerType, TargetChain

pytestmark = pytest.mark.slow

TARGET = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK"
BINDER_LENGTH = 12
CHAINS = [TargetChain(sequence=TARGET, use_msa=False)]
# The binder's placeholder, and the target, as tinyprot's schema entities.
PLACEHOLDER = {"type": "protein", "sequence": "X" * BINDER_LENGTH, "use_msa": False}
TARGET_ENTITY = {"type": "protein", "sequence": TARGET, "use_msa": False}


@pytest.fixture(scope="module")
def model():
    return Promera()


@pytest.fixture(scope="module")
def upstream():
    pytest.importorskip("tinyprot")
    from mosaic.models.upstream import promera

    return promera


def featurizations(model, upstream, chains, schema, **kwargs):
    """Return the binder-design features and writer of one system from both paths."""
    key = jax.random.key(5)
    return (
        model.binder_features(BINDER_LENGTH, chains, key=key),
        model.design_pack(*upstream.features(schema, key=key, **kwargs), BINDER_LENGTH),
    )


def with_pose_of(features, reference):
    """Return `features` with the reference-conformer poses of `reference`."""
    return eqx.tree_at(
        lambda f: (f.features["ref_pos"], f.pack.target["ref_pos"]),
        features,
        (reference.features["ref_pos"], reference.pack.target["ref_pos"]),
    )


def assert_rigidly_posed(ours, ref):
    """Check that each residue's conformer in `ours` is a rigid motion of `ref`'s."""
    a, b = ours.features, ref.features
    real = np.asarray(a["atom_pad_mask"]) > 0
    uid = np.asarray(a["ref_space_uid"])
    for residue in np.unique(uid[real]):
        atoms = real & (uid == residue)
        x, y = np.asarray(a["ref_pos"])[atoms], np.asarray(b["ref_pos"])[atoms]
        dx = np.linalg.norm(x[:, None] - x[None], axis=-1)
        dy = np.linalg.norm(y[:, None] - y[None], axis=-1)
        np.testing.assert_allclose(dx, dy, atol=1e-4)


def assert_same_features(ours, ref, *, pose: bool = True):
    """Check that the arrays, atom37 slots and design pack are the same."""
    assert set(ours.features) == set(ref.features)
    for name, value in ours.features.items():
        other = ref.features[name]
        assert value.shape == other.shape, name
        if name != "ref_pos" or pose:
            assert jnp.array_equal(value, other), name
    if not pose:
        assert_rigidly_posed(ours, ref)
    assert jnp.array_equal(ours.atom37_index, ref.atom37_index)
    for name, value in ours.pack.target.items():
        if name != "ref_pos" or pose:
            assert jnp.array_equal(value, ref.pack.target[name]), name
    for a, b in zip(jax.tree.leaves(ours.pack), jax.tree.leaves(ref.pack)):
        assert a.shape == b.shape


def soft_sequence():
    return jax.nn.softmax(
        2.0 * jax.random.normal(jax.random.key(1), (BINDER_LENGTH, 20))
    )


@pytest.mark.model_forward
def test_featurizations_agree(model, upstream):
    """Check that upstream's features equal the view's up to the conformers' pose."""
    schema = {"A": PLACEHOLDER, "B": TARGET_ENTITY, "C": TARGET_ENTITY}
    (f_view, w_view), (f_up, w_up) = featurizations(
        model, upstream, CHAINS + CHAINS[:1], schema
    )
    assert_same_features(f_up, f_view, pose=False)

    model = eqx.tree_at(lambda m: m.mask_fraction, model, 0.25)
    pssm, key = soft_sequence(), jax.random.key(2)
    s_view = model.set_sequence(f_view, pssm, key=key)
    s_up = model.set_sequence(with_pose_of(f_up, f_view), pssm, key=key)
    assert_same_features(s_up, s_view)
    for a, b in zip(
        jax.tree.leaves(model.tokens(s_up)), jax.tree.leaves(model.tokens(s_view))
    ):
        assert jnp.array_equal(a, b)

    coords = 10.0 * jax.random.normal(jax.random.key(3), s_up.features["ref_pos"].shape)
    a37, a37_view = model.atom37(s_up, coords), model.atom37(s_view, coords)
    assert jnp.array_equal(a37.mask, a37_view.mask)
    assert jnp.array_equal(a37.coords, a37_view.coords)
    st, st_view = w_up(coords, binder_pssm=pssm), w_view(coords, binder_pssm=pssm)
    assert atom_labels(st) == atom_labels(st_view)
    np.testing.assert_allclose(atom_positions(st), atom_positions(st_view), atol=1e-3)

    coords = coords[: int(f_up.features["atom_pad_mask"].sum())]
    st, st_view = w_up(coords), w_view(coords)
    assert atom_labels(st) == atom_labels(st_view)
    np.testing.assert_allclose(atom_positions(st), atom_positions(st_view), atol=1e-3)


def test_upstream_pose_follows_key(upstream):
    schema = {"A": {"type": "protein", "sequence": "GSHM", "use_msa": False}}
    a, b, c = (upstream.features(schema, key=jax.random.key(k))[0] for k in (0, 0, 1))
    assert jnp.array_equal(a.features["ref_pos"], b.features["ref_pos"])
    assert not jnp.array_equal(a.features["ref_pos"], c.features["ref_pos"])


def a3m(sequence: str) -> str:
    """Build a small a3m for `sequence`."""

    def sub(i, letter):
        return sequence[:i] + letter + sequence[i + 1 :]

    hits = {
        "101": sequence,
        "UniRef100_P01468": sub(3, "W"),
        "UniRef100_P01469": sequence[:20] + "kk" + sub(25, "A")[20:],
        "env_1": sequence,
        "UniRef100_P01470": "-----" + sequence[5:],
        "env_2": sub(40, "W"),
        "UniRef100_P01463": sub(50, "D")[:-4] + "----",
        "UniRef100_P01458": sub(10, "Y"),
    }
    return "".join(f">{name}\n{seq}\n" for name, seq in hits.items())


def test_msa_features(model, upstream, monkeypatch, tmp_path):
    """Check that the same a3m gives the same features on both paths."""
    from tinyprot.msa import hash_sequence

    text = a3m(TARGET)
    monkeypatch.setattr(msa, "fetch_a3m", lambda sequence: text)
    (tmp_path / f"{hash_sequence(TARGET)}.a3m").write_text(text)
    chains = [TargetChain(sequence=TARGET, use_msa=True)]
    schema = {"A": PLACEHOLDER, "B": TARGET_ENTITY | {"use_msa": True}}
    (f_view, _), (f_up, _) = featurizations(
        model, upstream, chains, schema, msa_dir=tmp_path
    )
    assert_same_features(f_up, f_view, pose=False)
    assert f_view.features["msa"].shape[0] == 7
    assert int(f_view.features["msa_paired"][:, -1].sum()) == 5


def test_unsupported_inputs():
    template = gemmi.read_pdb_string(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )[0]["A"]
    for chain in [
        TargetChain(sequence="ACGU", polymer_type=PolymerType.RNA, use_msa=False),
        TargetChain(sequence="MKZ", use_msa=False),
        TargetChain(sequence=TARGET, use_msa=False, template_chain=template),
    ]:
        with pytest.raises(
            NotImplementedError, match="mosaic.models.upstream.promera.features"
        ):
            view_features([chain], key=jax.random.key(0))


def test_design_pack_needs_placeholder(model):
    for binder, binder_length in (("G" * BINDER_LENGTH, BINDER_LENGTH), ("X" * 12, 10)):
        chains = [TargetChain(binder, use_msa=False), *CHAINS]
        features, writer = model.target_only_features(chains, key=jax.random.key(0))
        with pytest.raises(ValueError, match="placeholder"):
            model.design_pack(features, writer, binder_length)


NUCLEIC_SCHEMA = {
    "A": {"type": "protein", "sequence": "MKTAYIAKQRQISFVKSHFSRQ", "use_msa": False},
    "B": {"type": "rna", "sequence": "GGACUUCGGUCC", "use_msa": False},
    "C": {"type": "dna", "sequence": "GCGATCGC", "use_msa": False},
}


def residue_names(st: gemmi.Structure) -> list[list[str]]:
    return [[residue.name for residue in chain] for chain in st[0]]


def three_letter(entity: dict) -> list[str]:
    if entity["type"] == "protein":
        return [
            gemmi.expand_one_letter(c, gemmi.ResidueKind.AA) for c in entity["sequence"]
        ]
    prefix = "D" if entity["type"] == "dna" else ""
    return [prefix + c for c in entity["sequence"]]


@pytest.mark.model_forward
def test_upstream_nucleic_acids(model, upstream):
    """Check the upstream path on protein, RNA and DNA chains."""
    features, writer = upstream.features(NUCLEIC_SCHEMA, key=jax.random.key(0))
    prediction = model.predict(
        features=features,
        writer=writer,
        recycling_steps=1,
        sampler=model.upstream_sampler(20),
        key=jax.random.key(1),
    )
    out = prediction.model_output
    assert jnp.isfinite(out.structure_coordinates).all()
    assert ((out.plddt >= 0) & (out.plddt <= 1)).all()
    assert residue_names(prediction.st) == [
        three_letter(e) for e in NUCLEIC_SCHEMA.values()
    ]
    num_atoms = int(features.features["atom_pad_mask"].sum())
    assert prediction.st[0].count_atom_sites() == num_atoms

    schema = {"A": PLACEHOLDER} | dict(zip("BCD", NUCLEIC_SCHEMA.values()))
    features, writer = model.design_pack(
        *upstream.features(schema, key=jax.random.key(2)), BINDER_LENGTH
    )
    pssm = jax.nn.one_hot(
        jax.random.randint(jax.random.key(3), (BINDER_LENGTH,), 0, 20), 20
    )
    prediction = model.predict(
        PSSM=pssm,
        features=features,
        writer=writer,
        recycling_steps=1,
        sampler=model.upstream_sampler(20),
        key=jax.random.key(4),
    )
    out = prediction.model_output
    assert jnp.isfinite(out.structure_coordinates).all()
    assert jnp.isfinite(out.pae).all() and np.isfinite(float(prediction.iptm))
    binder = {
        "type": "protein",
        "sequence": "".join("ARNDCQEGHILKMFPSTWYV"[i] for i in pssm.argmax(-1)),
    }
    assert residue_names(prediction.st) == [
        three_letter(e) for e in [binder, *NUCLEIC_SCHEMA.values()]
    ]
    designed = model.set_sequence(features, pssm, key=jax.random.key(4))
    num_atoms = int(designed.features["atom_pad_mask"].sum())
    assert prediction.st[0].count_atom_sites() == num_atoms


@pytest.mark.model_forward
def test_predicts(model):
    """Check that a full prediction gives a sensible structure and confidence."""
    features, writer = model.binder_features(
        BINDER_LENGTH, CHAINS, key=jax.random.key(5)
    )
    pssm = jax.nn.one_hot(
        jax.random.randint(jax.random.key(3), (BINDER_LENGTH,), 0, 20), 20
    )
    prediction = model.predict(
        PSSM=pssm, features=features, writer=writer, key=jax.random.key(4)
    )
    output = prediction.model_output
    assert jnp.isfinite(output.structure_coordinates).all()
    assert ((output.plddt >= 0) & (output.plddt <= 1)).all()
    # Consecutive CA atoms of the target should be ~3.8 Å apart.
    ca = output.backbone_coordinates[BINDER_LENGTH:, 1]
    gaps = jnp.linalg.norm(ca[1:] - ca[:-1], axis=-1)
    assert 3.0 < float(jnp.median(gaps)) < 4.5
    assert [len(chain) for chain in prediction.st[0]] == [BINDER_LENGTH, len(TARGET)]
