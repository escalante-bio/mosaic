"""`OpenFold3` on its two featurizations: jfold's view (torch-free, proteins only) and
OpenFold3's own pipeline from its own query (`mosaic.models.upstream.openfold3.features`):
the same features for proteins (up to the reference conformers, which each draws at
random), predictions and written structures that agree, and upstream's for the systems the
view does not take (DNA, RNA, ligands)."""

import shutil
from pathlib import Path

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jfold.featurize.views import openfold3 as view
from jfold.models import openfold3 as jfold_of3
from jfold_testing import atom_labels

import mosaic.msa
from mosaic.models import openfold3
from mosaic.models.openfold3 import OpenFold3
from mosaic.structure_prediction import PolymerType, TargetChain

TARGET = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQCGVWPAACRERLGLKL"[
    :60
]
BINDER_LENGTH = 12
UBIQUITIN = Path(__file__).parent / "data" / "1ubq.cif"
UNK = ("N", "CA", "C", "O", "CB", "CG")  # the placeholder binder's atoms


def placeholder(chains: list[TargetChain]) -> list[TargetChain]:
    """`chains` after the poly-X binder placeholder, as `binder_features` builds them."""
    return [TargetChain(sequence="X" * BINDER_LENGTH, use_msa=False), *chains]


def protein(chain_id: str, sequence: str, **fields) -> dict:
    """A protein chain of OpenFold3's query."""
    return {
        "molecule_type": "protein",
        "chain_ids": chain_id,
        "sequence": sequence,
    } | fields


def design_query(*chains: dict) -> dict:
    """OpenFold3's query of `chains` (ids B, C, ...) after the poly-X binder placeholder
    (A), as `binder_features` lays it out."""
    return {"chains": [protein("A", "X" * BINDER_LENGTH), *chains]}


def upstream_module():
    """`mosaic.models.upstream.openfold3`, skipped without jopenfold3."""
    pytest.importorskip("jopenfold3")
    from mosaic.models.upstream import openfold3 as upstream

    return upstream


def upstream_features(query: dict, key):
    return upstream_module().features(query, key=key)


@pytest.fixture(scope="module")
def chains():
    return [TargetChain(sequence=TARGET, use_msa=False)]


@pytest.fixture(scope="module")
def jfold(chains):
    """The view's features and writer for a design pack."""
    return openfold3.target_only_features(placeholder(chains), key=jax.random.key(5))


@pytest.fixture(scope="module")
def upstream():
    """OpenFold3's features and writer for the same system, from its query."""
    return upstream_features(design_query(protein("B", TARGET)), jax.random.key(5))


@pytest.fixture(scope="module")
def model():
    return OpenFold3()


def assert_same_features(ours: dict, ref: dict):
    """`ours` is `ref` on every key, up to the reference conformers (`ref_pos`: the same
    shape) and float32 rounding of the template unit vectors."""
    assert ours.keys() == ref.keys()
    for k, v in ours.items():
        r = ref[k]
        assert v.shape == r.shape and v.dtype == r.dtype, k
        if k == "ref_pos":
            continue
        tolerance = 1e-5 if k == "template_unit_vector" else 0
        np.testing.assert_allclose(v, r, rtol=0, atol=tolerance, err_msg=k)


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


def zeros(features: dict) -> np.ndarray:
    return np.zeros((features["atom_mask"].shape[0], 3))


def test_unsupported():
    """What the view does not featurize raises before anything is fetched, pointing to
    OpenFold3's own featurization."""
    key = jax.random.key(0)
    for chains in [
        [TargetChain(sequence="ACGT", polymer_type=PolymerType.DNA, use_msa=False)],
        [TargetChain(sequence="MKZT", use_msa=False)],
        [TargetChain(sequence=TARGET), TargetChain(sequence=TARGET[:30])],
        [TargetChain(sequence=TARGET), TargetChain(sequence=TARGET, use_msa=False)],
    ]:
        with pytest.raises(NotImplementedError, match="upstream.openfold3.features"):
            openfold3.target_only_features(chains, key=key)


def test_set_sequence(jfold):
    """The binder's residue types, profile and query MSA row are the soft sequence, its
    other MSA rows gaps; its atoms (the placeholder's six) are named by the sequence's
    argmax in atom37, its representative atoms the placeholder's."""
    features = jfold[0]
    pssm = jax.nn.softmax(jax.random.normal(jax.random.key(1), (BINDER_LENGTH, 20)))
    designed = openfold3.set_binder_sequence(pssm, features)
    for k in ("restype", "profile"):
        np.testing.assert_array_equal(designed[k][:BINDER_LENGTH, :20], pssm)
        np.testing.assert_array_equal(
            designed[k][BINDER_LENGTH:], features[k][BINDER_LENGTH:]
        )
    np.testing.assert_array_equal(designed["msa"][0, :BINDER_LENGTH, :20], pssm)
    assert (designed["msa"][1:, :BINDER_LENGTH].argmax(-1) == 31).all()
    np.testing.assert_array_equal(designed["rep_atom"], features["rep_atom"])

    coords = jax.random.normal(jax.random.key(2), (features["atom_mask"].shape[0], 3))
    atom37, mask = openfold3.of3_atom37(designed, coords)
    start = features["start_atom_index"]
    residues = view.residues()
    for i, aa in enumerate(np.asarray(pssm.argmax(-1))):
        atoms = residues[jfold_of3.RESIDUES[aa]].atoms
        assert int(mask[i].sum()) == min(len(atoms), len(UNK))
        np.testing.assert_array_equal(atom37[i, 1], coords[start[i] + 1])  # CA
    assert jnp.array_equal(
        mask[BINDER_LENGTH:], openfold3.of3_atom37(features, coords)[1][BINDER_LENGTH:]
    )


@pytest.mark.slow
def test_features_match_upstream(jfold, upstream):
    """The view's features are upstream's, up to the reference conformers, and its writer
    writes the same atoms."""
    assert_same_features(jfold[0], upstream[0])
    jfold_st, upstream_st = jfold[1](zeros(jfold[0])), upstream[1](zeros(upstream[0]))
    assert atom_labels(jfold_st) == atom_labels(upstream_st)
    binder = [a for a in atom_labels(upstream_st) if a[0] == "A" and a[2] == 1]
    assert [a[3] for a in binder] == list(UNK)


@pytest.mark.slow
def test_features_match_upstream_msa_template(monkeypatch, tmp_path):
    """With an MSA (the same a3m on both paths) and a template (upstream realigns its
    `.m8` hit with Kalign; jfold's view as upstream does), and the binder's set sequence."""
    upstream = upstream_module()
    monkeypatch.setattr(mosaic.msa, "fetch_a3m", a3m)
    sequence = gemmi.one_letter_code([r.name for r in ubiquitin_chain()])
    key = jax.random.key(5)
    chains = placeholder([TargetChain(sequence, template_chain=ubiquitin_chain())])
    f_jf, _ = openfold3.target_only_features(chains, key=key)

    # Upstream reads the template's structure, `<entry id>.cif`, from its structure
    # directory, and the MSA by its file stem.
    structures = tmp_path / "structures"
    structures.mkdir()
    shutil.copy(UBIQUITIN, structures / "1ubq.cif")
    msa = tmp_path / "msa" / "colabfold_main.a3m"
    msa.parent.mkdir()
    msa.write_text(a3m(sequence))
    hits = tmp_path / "hits.m8"
    n = len(sequence)
    hits.write_text(f"query\t1ubq_A\t1.0\t{n}\t0\t0\t1\t{n}\t1\t{n}\t0.001\t100\n")
    query = design_query(
        protein(
            "B",
            sequence,
            main_msa_file_paths=str(msa),
            template_alignment_file_path=str(hits),
        )
    )
    f_up, _ = upstream.features(query, key=key, template_structures=structures)
    assert_same_features(f_jf, f_up)
    # Query row twice, then the two hits; the target's template, the binder without one.
    assert f_up["msa"].shape[0] == 4
    assert int(f_up["template_pseudo_beta_mask"].sum()) == len(sequence)
    assert (f_up["template_restype"][0, :BINDER_LENGTH].argmax(-1) == 31).all()

    pssm = jax.nn.softmax(jax.random.normal(jax.random.key(1), (BINDER_LENGTH, 20)))
    assert_same_features(
        openfold3.set_binder_sequence(pssm, f_jf),
        openfold3.set_binder_sequence(pssm, f_up),
    )


@pytest.mark.slow
@pytest.mark.model_forward
def test_predicts(model, jfold):
    """A full prediction runs and gives a sensible structure and confidence; the binder's
    tokens are the designed sequence."""
    pssm = jax.nn.one_hot(
        jax.random.randint(jax.random.key(3), (BINDER_LENGTH,), 0, 20), 20
    )
    output = model.model_output(PSSM=pssm, features=jfold[0], key=jax.random.key(4))
    assert jnp.isfinite(output.structure_coordinates).all()
    assert ((output.plddt >= 0) & (output.plddt <= 1)).all()
    # Consecutive CA atoms of the target sit ~3.8 Å apart.
    ca = output.backbone_coordinates[BINDER_LENGTH:, 1]
    gaps = jnp.linalg.norm(ca[1:] - ca[:-1], axis=-1)
    assert 3.0 < float(jnp.median(gaps)) < 4.5

    tokens = model.tokens(model.set_sequence(jfold[0], pssm, key=jax.random.key(0)))
    np.testing.assert_array_equal(tokens.sequence[:BINDER_LENGTH], pssm)
    assert tokens.mask.all() and (tokens.asym_id[:BINDER_LENGTH] == 1).all()


@pytest.mark.slow
@pytest.mark.model_forward
def test_upstream_nucleic_acids_ligand(model):
    """Upstream's featurization of a binder design against a protein, a DNA and an RNA
    chain and ATP: featurizes, designs and predicts (finite outputs), and writes every
    entity."""
    dna, rna = "ACGTAGCT", "GGCUAGCC"
    query = design_query(
        protein("B", TARGET[:40]),
        {"molecule_type": "dna", "chain_ids": "C", "sequence": dna},
        {"molecule_type": "rna", "chain_ids": "D", "sequence": rna},
        {"molecule_type": "ligand", "chain_ids": "E", "ccd_codes": "ATP"},
    )
    features, writer = upstream_features(query, jax.random.key(5))
    atp_atoms = 31  # heavy atoms, a token each
    num_tokens = BINDER_LENGTH + 40 + len(dna) + len(rna) + atp_atoms
    assert features["restype"].shape == (num_tokens, 32)
    assert int(features["is_dna"].sum()) == len(dna)
    assert int(features["is_rna"].sum()) == len(rna)
    assert int(features["is_ligand"].sum()) == atp_atoms

    pssm = jax.nn.one_hot(
        jax.random.randint(jax.random.key(3), (BINDER_LENGTH,), 0, 20), 20
    )
    prediction = model.predict(
        PSSM=pssm, features=features, writer=writer, key=jax.random.key(4)
    )
    output = prediction.model_output
    assert jnp.isfinite(output.structure_coordinates).all()
    assert jnp.isfinite(output.distogram_logits).all()
    assert prediction.plddt.shape == (num_tokens,)
    assert ((prediction.plddt >= 0) & (prediction.plddt <= 1)).all()
    assert jnp.isfinite(prediction.pae).all() and np.isfinite(float(prediction.iptm))

    written = prediction.st[0]
    assert [c.name for c in written] == ["A", "B", "C", "D", "E"]
    assert [r.name for r in written["A"]] == ["UNK"] * BINDER_LENGTH  # the placeholder
    assert gemmi.one_letter_code([r.name for r in written["B"]]) == TARGET[:40]
    assert [r.name for r in written["C"]] == [f"D{n}" for n in dna]
    assert [r.name for r in written["D"]] == list(rna)
    assert [r.name for r in written["E"]] == ["ATP"]
    assert len(written["E"][0]) == atp_atoms
    assert sum(len(r) for c in written for r in c) == features["atom_mask"].shape[0]
