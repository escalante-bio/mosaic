"""`Boltz2` on its two featurizations: jfold's view (torch-free, proteins only) and
boltz's own from its native input (`mosaic.models.upstream.boltz2.features`): the same
features for proteins (up to the reference conformers' random pose), the same predictions
and written structures from them, and boltz's for the systems the view does not take
(DNA, RNA, ligands)."""

from pathlib import Path

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jfold_testing import atom_labels, atom_positions

import mosaic.msa
from mosaic.models.boltz2 import Boltz2
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
    """`mosaic.models.upstream.boltz2.features`."""
    pytest.importorskip("boltz")
    from mosaic.models.upstream.boltz2 import features

    return features


@pytest.fixture(scope="module")
def chains():
    return [TargetChain(sequence=TARGET, use_msa=False)]


@pytest.fixture(scope="module")
def upstream(boltz_features):
    """boltz's features and writer for a design pack, from its native input."""
    return boltz_features(binder_input(TARGET), key=jax.random.key(5))


@pytest.fixture(scope="module")
def jfold(model, chains):
    """The view's features and writer for the same design pack."""
    return model.binder_features(BINDER_LENGTH, chains, key=jax.random.key(5))


def binder_input(sequence: str, msa: str = "empty", **extra) -> dict:
    """boltz's input for a binder placeholder and a target `sequence` (with `msa`), laid
    out as `binder_features` lays them out; `extra`: more top-level entries."""
    return {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": "X" * BINDER_LENGTH, "msa": "empty"}},
            {"protein": {"id": "B", "sequence": sequence, "msa": msa}},
        ],
        **extra,
    }


def upstream_pose(features: dict, upstream_features: dict) -> dict:
    """The view's `features` with boltz's random reference-conformer pose."""
    return features | {"ref_pos": upstream_features["ref_pos"]}


def assert_same_features(ours: dict, ref: dict):
    """`ours` (the view's) is boltz's dict on its keys, up to the conformers' pose."""
    for k, v in ours.items():
        assert v.shape == ref[k].shape and v.dtype == ref[k].dtype, k
        assert k == "ref_pos" or jnp.array_equal(v, ref[k]), k


def a3m(sequence: str) -> str:
    """A small a3m for `sequence`: the query, a point mutant with a gap, and a hit with an
    insertion (lowercase)."""
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


def test_unsupported(model):
    """What the view does not featurize raises, pointing to boltz's featurization, before
    anything is fetched."""
    key = jax.random.key(0)
    for chains in [
        [TargetChain(sequence="ACGT", polymer_type=PolymerType.DNA, use_msa=False)],
        [TargetChain(sequence="MKZT", use_msa=False)],
        [TargetChain(sequence=TARGET), TargetChain(sequence=TARGET[:30])],
        [TargetChain(sequence=TARGET), TargetChain(sequence=TARGET, use_msa=False)],
    ]:
        with pytest.raises(NotImplementedError, match="upstream.boltz2.features"):
            model.target_only_features(chains, key=key)


@pytest.mark.slow
def test_features_match_upstream_msa_template(
    model, boltz_features, monkeypatch, tmp_path
):
    """With the same a3m on both paths (the view reads `mosaic.msa.fetch_a3m`'s, boltz a
    file) and the same template: boltz's dict up to the conformers' pose."""
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
    f_jf, _ = model.binder_features(BINDER_LENGTH, chains, key=key)
    assert_same_features(f_jf, f_up)
    assert f_jf["msa"].shape[0] == 3
    assert int(f_jf["template_mask"].sum()) == len(sequence)


@pytest.mark.slow
def test_design_pack_matches_upstream(model, upstream, jfold):
    """A soft binder sequence set on either path's features gives the same features,
    tokens and atom37."""
    pssm = jax.nn.softmax(
        2.0 * jax.random.normal(jax.random.key(1), (BINDER_LENGTH, 20))
    )
    key = jax.random.key(0)
    f_up = model.set_sequence(upstream[0], pssm, key=key)
    f_jf = model.set_sequence(upstream_pose(jfold[0], upstream[0]), pssm, key=key)
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
def test_writers_agree(upstream, jfold):
    """Both writers write the same atoms, labelled the same, at the same positions."""
    (f_up, w_up), (_, w_jf) = upstream, jfold
    coords = jax.random.normal(jax.random.key(4), (f_up["atom_pad_mask"].shape[0], 3))
    ours, ref = w_jf(coords), w_up(coords)
    assert atom_labels(ours) == atom_labels(ref)
    np.testing.assert_allclose(atom_positions(ours), atom_positions(ref), atol=2e-3)


@pytest.mark.slow
@pytest.mark.model_forward
def test_upstream_nucleic_acids_and_ligand(model, boltz_features):
    """A protein with DNA and RNA chains and a ligand, featurized by boltz from its native
    input: the prediction is finite and the written structure has every entity, each with
    its residues and atoms."""
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
    prediction = model.predict(features=features, writer=writer, key=jax.random.key(1))
    output = prediction.model_output
    assert jnp.isfinite(output.structure_coordinates).all()
    assert jnp.isfinite(output.pae).all()
    assert ((output.plddt >= 0) & (output.plddt <= 1)).all()
    st = prediction.st
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
