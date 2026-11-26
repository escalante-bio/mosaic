"""Test `Boltz2` with jfold's view and with boltz's own featurization."""

from pathlib import Path

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mosaic.msa
from mosaic.atom37 import ATOM37_INDEX
from mosaic.models.boltz2 import (
    Boltz2,
    boltz2_atom37,
    set_binder_sequence,
    target_only_features,
)
from mosaic.structure_prediction import PolymerType, TargetChain

TARGET = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQCGVWPAACRERLGLKL"[
    :60
]
BINDER_LENGTH = 12
UBIQUITIN = Path(__file__).parent / "data" / "1ubq.cif"


@pytest.fixture(scope="module")
def model():
    return Boltz2()


@pytest.fixture(scope="module")
def boltz_features():
    """Import `mosaic.models.upstream.boltz2.features`, skipping without boltz."""
    pytest.importorskip("boltz")
    from mosaic.models.upstream.boltz2 import features

    return features


@pytest.fixture(scope="module")
def chains():
    return [TargetChain(sequence=TARGET, use_msa=False)]


@pytest.fixture(scope="module")
def upstream(boltz_features):
    """Return boltz's binder-design features and writer, from its native input."""
    return boltz_features(binder_input(TARGET), key=jax.random.key(5))


@pytest.fixture(scope="module")
def jfold(model, chains):
    """Return the view's features for the same binder design."""
    return model.binder_features(BINDER_LENGTH, chains, key=jax.random.key(5))


def binder_input(sequence: str, msa: str = "empty", **extra) -> dict:
    """Build boltz's input for a binder placeholder and a target `sequence` with `msa`."""
    return {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": "X" * BINDER_LENGTH, "msa": "empty"}},
            {"protein": {"id": "B", "sequence": sequence, "msa": msa}},
        ],
        **extra,
    }


def chain_atoms(st: gemmi.Structure, chain: str) -> dict[tuple[str, int, str], list]:
    """Return the position of each atom of `chain`, by residue name and number and atom name."""
    return {
        (r.name, r.seqid.num, a.name): a.pos.tolist() for r in st[0][chain] for a in r
    }


def upstream_pose(features: dict, upstream_features: dict) -> dict:
    """Return the view's `features` with boltz's random reference-conformer pose."""
    return features | {"ref_pos": upstream_features["ref_pos"]}


def assert_same_features(ours: dict, ref: dict):
    """Check that `ours` equals boltz's dict, up to the conformers' pose."""
    for k, v in ours.items():
        assert v.shape == ref[k].shape and v.dtype == ref[k].dtype, k
        assert k == "ref_pos" or jnp.array_equal(v, ref[k]), k


def a3m(sequence: str) -> str:
    """Build a small a3m for `sequence`."""
    mutant = "A" + sequence[1:10] + "-" + sequence[11:]
    inserted = sequence[:20] + "gk" + sequence[20:]
    return f">query\n{sequence}\n>mutant\n{mutant}\n>inserted\n{inserted}\n"


def ubiquitin_chain() -> gemmi.Chain:
    st = gemmi.read_structure(str(UBIQUITIN))
    st.remove_ligands_and_waters()
    return st[0]["A"]


def random_pssm(seed: int):
    return jax.nn.one_hot(
        jax.random.randint(jax.random.key(seed), (BINDER_LENGTH,), 0, 20), 20
    )


def test_unsupported():
    """Check that inputs the view cannot featurize raise before anything is fetched."""
    key = jax.random.key(0)
    for chains in [
        [TargetChain(sequence="ACGT", polymer_type=PolymerType.DNA, use_msa=False)],
        [TargetChain(sequence="MKZT", use_msa=False)],
        [TargetChain(sequence=TARGET), TargetChain(sequence=TARGET[:30])],
        [TargetChain(sequence=TARGET), TargetChain(sequence=TARGET, use_msa=False)],
    ]:
        with pytest.raises(NotImplementedError, match="upstream.boltz2.features"):
            target_only_features(chains, key=key)


@pytest.mark.slow
def test_features_match_upstream_msa_template(
    model, boltz_features, monkeypatch, tmp_path
):
    """Check that the same a3m and template give boltz's features, up to the pose."""
    monkeypatch.setattr(mosaic.msa, "fetch_a3m", a3m)
    sequence = gemmi.one_letter_code([r.name for r in ubiquitin_chain()])
    msa = tmp_path / "target.a3m"
    msa.write_text(a3m(sequence))
    template = {"cif": str(UBIQUITIN), "chain_id": "B", "template_id": "A"}
    key = jax.random.key(5)
    f_up, _ = boltz_features(
        binder_input(sequence, msa=str(msa), templates=[template]), key=key
    )
    chains = [TargetChain(sequence, template_chain=ubiquitin_chain())]
    f_jf = model.binder_features(BINDER_LENGTH, chains, key=key)
    assert_same_features(f_jf, f_up)
    assert f_jf["msa"].shape[0] == 3
    assert int(f_jf["template_mask"].sum()) == len(sequence)


@pytest.mark.slow
def test_design_pack_matches_upstream(model, upstream, jfold):
    """Check that a soft binder sequence gives the same features on either path."""
    pssm = jax.nn.softmax(
        2.0 * jax.random.normal(jax.random.key(1), (BINDER_LENGTH, 20))
    )
    key = jax.random.key(0)
    f_up = model.set_sequence(upstream[0], pssm, key=key)
    f_jf = model.set_sequence(upstream_pose(jfold, upstream[0]), pssm, key=key)
    assert_same_features(f_jf, f_up)
    coords = jax.random.normal(jax.random.key(2), (f_up["atom_pad_mask"].shape[0], 3))
    a37_up, a37_jf = model.atom37(f_up, coords), model.atom37(f_jf, coords)
    assert jnp.array_equal(a37_jf.mask, a37_up.mask)
    assert jnp.array_equal(a37_jf.coords, a37_up.coords)
    for a, b in zip(
        jax.tree.leaves(model.tokens(f_jf)), jax.tree.leaves(model.tokens(f_up))
    ):
        assert jnp.array_equal(a, b)


@pytest.mark.slow
def test_structure_matches_upstream_writer(model, upstream, jfold):
    """Check that `structure` writes the target's atoms as boltz's writer does."""
    f_up, w_up = upstream
    coords = jax.random.normal(jax.random.key(4), (f_up["atom_pad_mask"].shape[0], 3))
    ours, ref = (
        chain_atoms(model.structure(jfold, coords), "B"),
        chain_atoms(w_up(coords), "B"),
    )
    assert ours.keys() == ref.keys()
    for atom, position in ours.items():
        np.testing.assert_allclose(position, ref[atom], atol=2e-3)


@pytest.mark.slow
@pytest.mark.model_forward
def test_upstream_nucleic_acids_and_ligand(model, boltz_features):
    """Check boltz's featurization of a protein with DNA and RNA chains and a ligand."""
    features, writer = boltz_features(
        {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": TARGET[:30], "msa": "empty"}},
                {"dna": {"id": "B", "sequence": "ACGTACGT"}},
                {"rna": {"id": "C", "sequence": "ACGUACGU"}},
                {"ligand": {"id": "D", "ccd": "ATP"}},
            ],
        },
        key=jax.random.key(0),
    )
    prediction = model.predict(features=features, key=jax.random.key(1))
    output = prediction.model_output
    assert jnp.isfinite(output.structure_coordinates).all()
    assert jnp.isfinite(output.pae).all()
    assert ((output.plddt >= 0) & (output.plddt <= 1)).all()
    # The upstream writer writes every atom, including the nucleic acids and the ligand.
    st = writer(output.structure_coordinates)
    assert [(c.name, len(c)) for c in st[0]] == [
        ("A", 30),
        ("B", 8),
        ("C", 8),
        ("D", 1),
    ]
    assert [r.name for r in st[0]["B"]][:2] == ["DA", "DC"]
    assert [r.name for r in st[0]["C"]][:2] == ["A", "C"]
    assert st[0]["D"][0].name == "ATP"
    num_atoms = int(features["atom_pad_mask"].sum())
    assert sum(len(r) for c in st[0] for r in c) == num_atoms
    written = np.array([a.pos.tolist() for c in st[0] for r in c for a in r])
    np.testing.assert_allclose(
        written, output.structure_coordinates[:num_atoms], atol=2e-3
    )


@pytest.mark.parametrize("soft", [False, True])
def test_atom37_preserves_reference_layout(soft):
    features = target_only_features(
        [TargetChain("GXW", use_msa=False)], key=jax.random.key(0)
    )
    sequence = jax.nn.one_hot(jnp.array([17, 17, 7]), 20)
    if soft:
        sequence = 0.8 * sequence + 0.2 / 20
    updated = set_binder_sequence(sequence, features)
    # Decode actual input names independently. Include the W side chain even after
    # replacement by G, and never manufacture a side chain for the original G.
    chars = np.asarray(features["ref_atom_name_chars"]).argmax(-1)
    token = np.asarray(features["atom_to_token"]).argmax(-1)
    coords = jnp.arange(3 * len(chars), dtype=jnp.float32).reshape(-1, 3)
    expected = np.zeros((3, 37, 3))
    mask = np.zeros((3, 37))
    for i in np.flatnonzero(features["atom_pad_mask"]):
        name = "".join(chr(32 + c) for c in chars[i]).rstrip()
        if name in ATOM37_INDEX:
            slot = ATOM37_INDEX[name]
            expected[token[i], slot] = coords[i]
            mask[token[i], slot] = 1
    assert mask[0, 3] == 0 and mask[2, 28] == 1
    actual, present = jax.jit(boltz2_atom37)(updated, coords)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(present, mask)


def test_atom37_keeps_modified_protein_and_drops_ligand_padding_and_unknown_names():
    names = ["CA", "CA", "CA", "CA", "Q"]
    inputs = {
        "ref_atom_name_chars": jax.nn.one_hot(
            jnp.array([[ord(c) - 32 for c in n.ljust(4)] for n in names]), 64
        ),
        "atom_to_token": jnp.eye(5),
        "mol_type": jnp.array([0, 3, 0, 0, 0]),
        "modified": jnp.array([0, 0, 1, 0, 0]),
        "atom_pad_mask": jnp.array([1, 1, 1, 0, 1]),
        "res_type": jnp.zeros((5, 33)),
    }
    coords = jnp.ones((5, 3))
    _, mask = boltz2_atom37(inputs, coords)
    assert mask.sum() == 2
    assert mask[0, 1] == mask[2, 1] == 1
    grad = jax.grad(lambda x: boltz2_atom37(inputs, x)[0].sum())(coords)
    np.testing.assert_array_equal(
        grad, jnp.zeros_like(coords).at[jnp.array([0, 2])].set(1)
    )
