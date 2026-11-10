"""`Mol1`: jfold's featurization against Mol-1's own from its native input
(`mosaic.models.upstream.mol1.features`, molence) for the same protein systems; the design
pack's binder repack, sequence channels and writer on both paths; predictions; and Mol-1's
own featurization of DNA and RNA chains. Both paths read the same cached MSAs
(`mosaic.msa.fetch_a3m`; as a3m files in the native input)."""

import dataclasses
import json
from pathlib import Path

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jfold_testing import atom_labels, atom_positions

from mosaic import msa
from mosaic.common import TOKENS
from mosaic.losses.structure_prediction import BinderTargetContact, PLDDTLoss
from mosaic.models.mol1 import ATOM_KEYS, Mol1
from mosaic.structure_prediction import PolymerType, TargetChain

pytestmark = [pytest.mark.slow, pytest.mark.model_forward]

TARGET = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL"
BINDER = "GWYKCPSHDLIF"
BINDER_LENGTH = len(BINDER)
PLACEHOLDER = "W" * BINDER_LENGTH
CHAINS = [TargetChain(sequence=TARGET, use_msa=True)]
DNA = {"dnaSequence": {"sequence": "GACTCG", "count": 1}}
RNA = {"rnaSequence": {"sequence": "ACGU", "count": 1}}


def one_hot(sequence: str):
    return jax.nn.one_hot(jnp.array([TOKENS.index(c) for c in sequence]), 20)


def ubiquitin() -> tuple[str, gemmi.Chain]:
    """1UBQ's chain A (no waters) and its sequence."""
    st = gemmi.read_structure(str(Path(__file__).parent / "data" / "1ubq.cif"))
    st.remove_ligands_and_waters()
    chain = st[0]["A"]
    return gemmi.one_letter_code([r.name for r in chain]), chain


def native_input(*sequences: dict) -> dict:
    """An entry of Mol-1's input JSON."""
    return {"name": "mosaic", "sequences": list(sequences)}


def protein(sequence: str, **fields) -> dict:
    """A protein chain of Mol-1's input; `fields`: its MSA and template paths."""
    return {"proteinChain": {"sequence": sequence, "count": 1, **fields}}


def a3m_file(path: Path, a3m: str) -> dict:
    """`a3m` written to `path`, as a protein chain's unpaired MSA."""
    path.write_text(a3m)
    return {"unpairedMsaPath": str(path)}


def target(tmp_path: Path) -> dict:
    """`TARGET` with its MSA (the cached `msa.fetch_a3m`, as `CHAINS`)."""
    return protein(TARGET, **a3m_file(tmp_path / "target.a3m", msa.fetch_a3m(TARGET)))


def template_file(chain: gemmi.Chain, path: Path) -> dict:
    """`chain` as Mol-1's template JSON, written to `path`: its amino acids as the mmCIF,
    residue `i` standing for the query's residue `i`."""
    st = gemmi.Structure()
    st.add_model(gemmi.Model(1))
    st[0].add_chain(chain.clone())
    st.remove_ligands_and_waters()
    st.setup_entities()
    indices = list(range(len(st[0][0])))
    template = {
        "mmcif": st.make_mmcif_document().as_string(),
        "queryIndices": indices,
        "templateIndices": indices,
    }
    path.write_text(json.dumps([template]))
    return {"templatesPath": str(path)}


@pytest.fixture(scope="module")
def model():
    return Mol1()


@pytest.fixture(scope="module")
def upstream():
    """`mosaic.models.upstream.mol1` (molence)."""
    pytest.importorskip("mol.data.inference.infer_dataloader")
    from mosaic.models.upstream import mol1

    return mol1


def distances(pos, uid, mask):
    """The pairwise distances within each reference conformer (`ref_space_uid`), which
    upstream poses at random."""
    pos, uid, mask = np.asarray(pos), np.asarray(uid), np.asarray(mask)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    return np.where((uid[:, None] == uid[None]) & mask[:, None] & mask[None], d, 0.0)


def assert_same_features(ours: dict, ref: dict):
    """Every key equal, `ref_pos` up to each conformer's pose."""
    assert set(ours) == set(ref), set(ours) ^ set(ref)
    for name, value in ref.items():
        a, b = np.asarray(ours[name]), np.asarray(value)
        assert a.shape == b.shape, (name, a.shape, b.shape)
        if name != "ref_pos":
            np.testing.assert_array_equal(a, b.astype(a.dtype), name)
    mask = np.asarray(ref["atom_pad_mask"])
    uid = np.asarray(ref["ref_space_uid"])
    np.testing.assert_allclose(
        distances(ours["ref_pos"], uid, mask),
        distances(ref["ref_pos"], uid, mask),
        atol=1e-4,
    )


def target_systems(tmp_path: Path) -> dict[str, tuple[list[TargetChain], dict]]:
    """Each system as jfold's chains and as Mol-1's native input."""
    sequence, template = ubiquitin()
    templated = template_file(template, tmp_path / "template.json")
    return {
        "msa": (CHAINS, native_input(target(tmp_path))),
        "homodimer": (
            [*CHAINS, *CHAINS],
            native_input(target(tmp_path), target(tmp_path)),
        ),
        "template": (
            [TargetChain(sequence, use_msa=False, template_chain=template)],
            native_input(protein(sequence, **templated)),
        ),
    }


@pytest.mark.parametrize("name", ["msa", "homodimer", "template"])
def test_features_match_upstream(model, upstream, name, tmp_path):
    """jfold's view gives Mol-1's own features (up to the conformers' poses) and the same
    atoms to write."""
    chains, native = target_systems(tmp_path)[name]
    ours, writer = model.target_only_features(chains, key=jax.random.key(0))
    ref, ref_writer = upstream.features(native, key=jax.random.key(0))
    assert_same_features(ours.features, ref.features)
    coords = np.asarray(ref.features["ref_pos"])
    st, st_ref = writer(coords), ref_writer(coords)
    assert atom_labels(st) == atom_labels(st_ref)
    np.testing.assert_allclose(atom_positions(st), atom_positions(st_ref), atol=1e-3)


def test_design_pack_matches_upstream(model, upstream, tmp_path):
    """The design packs agree (up to the poses); given the same poses, setting a soft
    sequence repacks the binder and fills the sequence channels identically. The binder
    writers write the same atoms."""
    ours, writer = model.binder_features(BINDER_LENGTH, CHAINS, key=jax.random.key(0))
    native = native_input(protein(PLACEHOLDER), target(tmp_path))
    ref, ref_writer = model.design_pack(
        *upstream.features(native, key=jax.random.key(1)), BINDER_LENGTH
    )
    assert_same_features(ours.features, ref.features)

    ref = dataclasses.replace(
        ref, features=ref.features | {"ref_pos": ours.features["ref_pos"]}
    )
    pssm = jax.nn.softmax(
        2.0 * jax.random.normal(jax.random.key(2), (BINDER_LENGTH, 20))
    )
    a = model.set_sequence(ours, pssm, key=jax.random.key(3)).features
    b = model.set_sequence(ref, pssm, key=jax.random.key(3)).features
    for name, value in b.items():
        np.testing.assert_array_equal(a[name], value, name)

    coords = np.asarray(a["ref_pos"])
    st, st_ref = writer(coords, pssm), ref_writer(coords, pssm)
    assert atom_labels(st) == atom_labels(st_ref)
    np.testing.assert_allclose(atom_positions(st), atom_positions(st_ref), atol=1e-3)


@pytest.mark.parametrize("featurization", ["jfold", "upstream"])
def test_repack_matches_featurization(model, featurization, request, tmp_path):
    """Repacking the all-Trp design pack to a sequence gives the features of featurizing
    that sequence (up to the poses), plus atom padding at the end; and the writer writes
    the repacked atoms. Against a protein target, and on Mol-1's own featurization, a
    protein and a DNA chain."""
    if featurization == "jfold":
        design, writer = model.binder_features(
            BINDER_LENGTH, CHAINS, key=jax.random.key(5)
        )
        native, native_writer = model.target_only_features(
            [TargetChain(BINDER, use_msa=False), *CHAINS], key=jax.random.key(1)
        )
    else:
        upstream = request.getfixturevalue("upstream")
        design, writer = model.design_pack(
            *upstream.features(
                native_input(protein(PLACEHOLDER), target(tmp_path), DNA),
                key=jax.random.key(5),
            ),
            BINDER_LENGTH,
        )
        native, native_writer = upstream.features(
            native_input(protein(BINDER), target(tmp_path), DNA), key=jax.random.key(1)
        )
    repacked = model.set_sequence(design, one_hot(BINDER), key=jax.random.key(0))
    f, g = repacked.features, native.features
    n = g["ref_pos"].shape[0]
    assert f["ref_pos"].shape[0] > n
    assert f["atom_pad_mask"][:n].all() and not f["atom_pad_mask"][n:].any()
    assert (f["atom37_idx"][n:] == -1).all()
    for name, value in g.items():
        if name not in ("ref_pos", "msa"):
            ours = f[name][:n] if name in ATOM_KEYS else f[name]
            np.testing.assert_array_equal(ours, value, name)
    np.testing.assert_array_equal(f["msa"][0], g["msa"][0])

    # A fresh pose per residue, each step, of the same conformers.
    uid, mask = g["ref_space_uid"], g["atom_pad_mask"]
    np.testing.assert_allclose(
        distances(f["ref_pos"][:n], uid, mask),
        distances(g["ref_pos"], uid, mask),
        atol=1e-4,
    )
    other = model.set_sequence(design, one_hot(BINDER), key=jax.random.key(2)).features
    binder = np.asarray(g["atom_to_token_idx"]) < BINDER_LENGTH
    assert not np.allclose(other["ref_pos"][:n][binder], f["ref_pos"][:n][binder])

    coords = np.asarray(f["ref_pos"])
    st, st_native = writer(coords, one_hot(BINDER)), native_writer(coords)
    assert atom_labels(st) == atom_labels(st_native)
    np.testing.assert_allclose(atom_positions(st), atom_positions(st_native), atol=1e-3)
    assert gemmi.one_letter_code([r.name for r in st[0][0]]) == BINDER


@pytest.mark.parametrize("binder", [BINDER, "W" * (BINDER_LENGTH + 1)])
def test_design_pack_refuses_other_binders(model, binder):
    """A design pack's first chain must be the all-Trp placeholder of the binder's
    length."""
    chains = [TargetChain(binder, use_msa=False), TargetChain(TARGET, use_msa=False)]
    features, writer = model.target_only_features(chains, key=jax.random.key(0))
    with pytest.raises(ValueError, match="all-Trp placeholder"):
        model.design_pack(features, writer, BINDER_LENGTH)


@pytest.mark.parametrize("featurization", ["jfold", "upstream"])
def test_profile_with_binder_msa(model, featurization, request, monkeypatch, tmp_path):
    """With an MSA on the binder, setting its sequence gives the features of featurizing
    the new sequence with the same MSA hits."""
    rng = np.random.default_rng(0)
    hits = ["".join(rng.choice(list(TOKENS), BINDER_LENGTH)) for _ in range(5)]
    designed = BINDER[::-1]

    def binder_a3m(sequence: str) -> str:
        return "".join(f">{i}\n{s}\n" for i, s in enumerate([sequence, *hits]))

    key = jax.random.key(0)
    if featurization == "jfold":
        fetched = msa.fetch_a3m
        monkeypatch.setattr(
            msa,
            "fetch_a3m",
            lambda s: binder_a3m(s) if s in (BINDER, designed) else fetched(s),
        )
        pack, _ = model.target_only_features([TargetChain(BINDER), *CHAINS], key=key)
        native, _ = model.target_only_features(
            [TargetChain(designed), *CHAINS], key=key
        )
    else:
        upstream = request.getfixturevalue("upstream")

        def featurize(sequence: str):
            binder_msa = a3m_file(tmp_path / f"{sequence}.a3m", binder_a3m(sequence))
            entry = native_input(protein(sequence, **binder_msa), target(tmp_path))
            return upstream.features(entry, key=key)[0]

        pack, native = featurize(BINDER), featurize(designed)
    f = model.set_sequence(pack, one_hot(designed), key=key).features
    g = native.features
    for name in ("restype", "profile", "msa"):
        np.testing.assert_allclose(f[name], g[name], atol=1e-6, err_msg=name)


def chain_sequences(st: gemmi.Structure) -> list[str]:
    return [gemmi.one_letter_code([r.name for r in chain]) for chain in st[0]]


def test_upstream_predicts_nucleic_acids(model, upstream):
    """Mol-1's own featurization of protein, DNA and RNA chains: a prediction with finite
    coordinates and pLDDT in [0, 1] for every token, every chain written with all its
    atoms."""
    native = native_input(protein(TARGET), DNA, RNA)
    features, writer = upstream.features(native, key=jax.random.key(0))
    pred = model.predict(
        features=features, writer=writer, recycling_steps=2, key=jax.random.key(1)
    )
    output = pred.model_output
    num_tokens = len(TARGET) + len("GACTCG") + len("ACGU")
    assert jnp.isfinite(output.structure_coordinates).all()
    assert output.plddt.shape == (num_tokens,)
    assert ((output.plddt >= 0) & (output.plddt <= 1)).all()
    assert np.isfinite(pred.iptm)
    assert chain_sequences(pred.st) == [TARGET, "GACTCG", "ACGU"]
    assert sum(1 for _ in pred.st[0].all()) == features.features["ref_pos"].shape[0]


def test_upstream_design_with_dna(model, upstream, tmp_path):
    """A design pack against a protein and a DNA chain from Mol-1's native input: finite
    loss gradients, and a prediction written as the binder's argmax plus every target
    chain."""
    native = native_input(protein(PLACEHOLDER), target(tmp_path), DNA)
    design, writer = model.design_pack(
        *upstream.features(native, key=jax.random.key(0)), BINDER_LENGTH
    )
    loss = model.build_loss(
        loss=BinderTargetContact() + PLDDTLoss(), features=design, recycling_steps=2
    )
    pssm = jax.nn.softmax(jax.random.normal(jax.random.key(1), (BINDER_LENGTH, 20)))

    @eqx.filter_jit
    def value_and_grad(loss, pssm, key):
        return jax.value_and_grad(lambda p: loss(p, key=key)[0])(pssm)

    value, grad = value_and_grad(loss, pssm, jax.random.key(2))
    assert np.isfinite(value) and np.isfinite(grad).all() and np.abs(grad).max() > 0

    pred = model.predict(
        PSSM=one_hot(BINDER),
        features=design,
        writer=writer,
        recycling_steps=2,
        key=jax.random.key(3),
    )
    assert jnp.isfinite(pred.model_output.structure_coordinates).all()
    assert chain_sequences(pred.st) == [BINDER, TARGET, "GACTCG"]


def test_template(model):
    """A template of its chain's length: slot 0 holds the template's residues and
    geometry, slots 1-3 are empty."""
    sequence, template = ubiquitin()
    chains = [TargetChain(sequence, use_msa=False, template_chain=template)]
    pack, _ = model.target_only_features(chains, key=jax.random.key(0))
    f = pack.features
    assert jnp.array_equal(f["template_aatype"][0], f["restype"].argmax(-1))
    assert f["template_pseudo_beta_mask"][0].all()
    assert not f["template_aatype"][1:].any()
    assert not f["template_distogram"][1:].any()


@pytest.mark.parametrize(
    "chain",
    [
        TargetChain("GACTCG", polymer_type=PolymerType.DNA, use_msa=False),
        TargetChain("MKTBYIAK", use_msa=False),
    ],
)
def test_unsupported(model, chain):
    """DNA/RNA and residues outside the 20 amino acids and X raise, pointing to Mol-1's
    own featurization."""
    with pytest.raises(NotImplementedError, match="upstream.mol1.features"):
        model.binder_features(BINDER_LENGTH, [*CHAINS, chain], key=jax.random.key(0))
