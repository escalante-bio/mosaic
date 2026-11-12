"""Test `Protenix` with jfold's featurization and with upstream's."""

from pathlib import Path

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jfold_testing import atom_labels, atom_positions

import mosaic.msa
from mosaic.common import TOKENS
from mosaic.models.protenix import (
    _INFERENCE_RECYCLING_STEPS,
    Protenix,
    featurize,
    set_binder_sequence,
)
from mosaic.structure_prediction import PolymerType, TargetChain

TARGET = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL"
BINDER_LENGTH = 12
DESIGNED = "GCPWKDEGRWAY"  # a binder sequence of `BINDER_LENGTH`
PLACEHOLDER = "W" * BINDER_LENGTH
ATP = {"ligand": {"ligand": "CCD_ATP", "count": 1}}
MG = {"ion": {"ion": "MG", "count": 1}}


@pytest.fixture(scope="module")
def upstream():
    """Import `mosaic.models.upstream.protenix` if the `protenix` package is installed."""
    pytest.importorskip("protenix")
    from mosaic.models.upstream import protenix

    return protenix


def ubiquitin() -> tuple[str, gemmi.Chain]:
    """Return 1UBQ's chain A without waters, and its sequence."""
    st = gemmi.read_structure(str(Path(__file__).parent / "data" / "1ubq.cif"))
    st.remove_ligands_and_waters()
    chain = st[0]["A"]
    return gemmi.one_letter_code([r.name for r in chain]), chain


def toy_a3m(s: str) -> str:
    """Build an a3m for `s` with substitutions, gaps, insertions and a duplicate row."""
    rows = [
        s,
        "A" + s[1:],
        s[:10] + "-" * 5 + s[15:],
        s[:20] + "kw" + "A" + s[21:30] + "g" + s[30:],
        "---" + s[3:-3] + "LLL",
        "A" + s[1:],
    ]
    return "".join(f">row{i}\n{row}\n" for i, row in enumerate(rows))


def msa_dir(path: Path, query: str, a3m: str) -> str:
    """Write a precomputed MSA directory as Protenix's input JSON takes it."""
    path.mkdir()
    (path / "non_pairing.a3m").write_text(a3m)
    (path / "pairing.a3m").write_text(f">query\n{query}\n")
    return str(path)


def protein(sequence: str, msa: str | None = None) -> dict:
    """Build a `proteinChain` entity of Protenix's input JSON."""
    chain = {"sequence": sequence, "count": 1}
    if msa is not None:
        chain["msa"] = {"precomputed_msa_dir": msa, "pairing_db": "uniref100"}
    return {"proteinChain": chain}


def native(*entities: dict) -> dict:
    """Build Protenix's input for `entities`, as one entry of its input JSON."""
    return {"name": "test", "sequences": list(entities)}


def one_hot(sequence: str) -> jax.Array:
    return jax.nn.one_hot(jnp.array([TOKENS.index(a) for a in sequence]), 20)


def jfold_pack(chains: list[TargetChain], *, key):
    """Build what `Protenix.binder_features` builds, without loading weights."""
    binder = TargetChain(PLACEHOLDER, use_msa=False)
    return Protenix.design_pack(*featurize([binder, *chains], key=key), BINDER_LENGTH)


def assert_same_features(ours: dict, theirs: dict):
    assert ours.keys() == theirs.keys()
    for k, v in ours.items():
        u = theirs[k]
        assert v.shape == u.shape and v.dtype == u.dtype, k
        assert jnp.array_equal(v, u), k


@pytest.mark.parametrize(
    "chains",
    [
        [TargetChain("ACGT", use_msa=False, polymer_type=PolymerType.DNA)],
        [TargetChain("MKZL", use_msa=False)],
        [TargetChain(TARGET, use_msa=True), TargetChain("GSGSGS", use_msa=True)],
    ],
    ids=["dna", "residue", "paired-msa"],
)
def test_unsupported(chains):
    with pytest.raises(NotImplementedError, match="Protenix's own input JSON"):
        featurize(chains, key=jax.random.key(0))


def test_unsupported_template():
    """Check that a template residue jfold does not read (selenomethionine) raises."""
    sequence, template = ubiquitin()
    template[0].name = "MSE"
    chains = [TargetChain(sequence, use_msa=False, template_chain=template)]
    with pytest.raises(NotImplementedError, match="templates"):
        featurize(chains, key=jax.random.key(0))


def test_template_length():
    """Check that a template must have its chain's length, as upstream requires."""
    sequence, template = ubiquitin()
    longer = TargetChain(sequence + "G", use_msa=False, template_chain=template)
    with pytest.raises(ValueError, match="the template has 76 residues"):
        featurize([longer], key=jax.random.key(0))


@pytest.mark.parametrize(
    "binder, length", [("A" * BINDER_LENGTH, BINDER_LENGTH), (PLACEHOLDER, 10)]
)
def test_design_pack_refuses(binder, length):
    """Check that `design_pack` needs a Trp placeholder of the binder's length first."""
    chains = [TargetChain(binder, use_msa=False), TargetChain(TARGET, use_msa=False)]
    features, writer = featurize(chains, key=jax.random.key(0))
    with pytest.raises(ValueError, match="the binder"):
        Protenix.design_pack(features, writer, length)


def test_recycling_steps(upstream):
    """Check that each release's number of trunk passes is upstream's `N_cycle`."""
    from jfold.pretrained.recipes.protenix import NAMES
    from protenix.configs.configs_model_type import model_configs

    for name, upstream_name in NAMES.items():
        n_cycle = model_configs[upstream_name]["model"]["N_cycle"]
        assert _INFERENCE_RECYCLING_STEPS[name] == n_cycle, name


@pytest.mark.slow
@pytest.mark.parametrize("system", ["binder-target", "msa", "homodimer-msa"])
def test_features_match(upstream, system, monkeypatch, tmp_path):
    """Check that on protein systems jfold's view gives exactly upstream's features."""
    monkeypatch.setattr(mosaic.msa, "fetch_a3m", toy_a3m)
    sequence, _ = ubiquitin()
    target = TargetChain(sequence, use_msa=True)
    binder = TargetChain(PLACEHOLDER, use_msa=False)
    with_msa = protein(sequence, msa_dir(tmp_path / "ubq", sequence, toy_a3m(sequence)))
    chains, input = {
        "binder-target": (
            [binder, TargetChain(TARGET, use_msa=False)],
            native(protein(PLACEHOLDER), protein(TARGET)),
        ),
        "msa": ([binder, target], native(protein(PLACEHOLDER), with_msa)),
        "homodimer-msa": ([target, target], native(with_msa, with_msa)),
    }[system]
    key = jax.random.key(7)
    ours, _ = featurize(chains, key=key)
    theirs, _ = upstream.features(input, key=key)
    if system != "binder-target":
        assert ours["msa"].shape[0] > 1 and ours["has_deletion"].any()
    assert_same_features(ours, theirs)


@pytest.mark.slow
def test_templates_match(upstream):
    """Check that jfold's view places templates as the fork's template featurizer does."""
    from protenix.data.template import ChainInput, build_templates_from_chains

    sequence, template = ubiquitin()
    chains = [
        TargetChain(PLACEHOLDER, use_msa=False),
        TargetChain(sequence, use_msa=False, template_chain=template),
    ]
    ours, _ = featurize(chains, key=jax.random.key(0))
    theirs = build_templates_from_chains(
        [ChainInput(PLACEHOLDER), ChainInput(sequence, template=template)]
    ).as_protenix_dict()
    assert ours["template_pseudo_beta_mask"].any()
    assert_same_features(
        {k: ours[k] for k in theirs}, jax.tree.map(jnp.asarray, theirs)
    )


@pytest.mark.slow
def test_writers_agree(upstream):
    """Check that both writers label the atoms alike, also with a designed binder."""
    key = jax.random.key(0)
    pack, writer = jfold_pack([TargetChain(TARGET, use_msa=False)], key=key)
    _, upstream_writer = Protenix.design_pack(
        *upstream.features(native(protein(PLACEHOLDER), protein(TARGET)), key=key),
        BINDER_LENGTH,
    )
    coords = pack.features["ref_pos"]
    pssm = one_hot(DESIGNED)
    for args in [(coords,), (coords, pssm)]:
        ours, theirs = writer(*args), upstream_writer(*args)
        assert atom_labels(ours) == atom_labels(theirs)
        np.testing.assert_allclose(
            atom_positions(ours), atom_positions(theirs), atol=1e-3
        )
    binder = [r.name for r in writer(coords, pssm)[0][0]]
    assert gemmi.one_letter_code(binder) == DESIGNED


def residue_distances(ref_pos, a2t):
    """Return intra-residue distance matrices, which ignore each residue's pose."""
    return [
        np.linalg.norm(ref_pos[a2t == t][:, None] - ref_pos[a2t == t][None], axis=-1)
        for t in np.unique(a2t)
    ]


def mismatches(native: dict, refreshed: dict) -> list[str]:
    """List fields of `refreshed` differing from `native` on its first `n` atoms."""
    n = len(native["atom_to_token_idx"])
    pad = np.asarray(refreshed["atom_pad_mask"])
    assert int(pad.sum()) == n and pad[:n].all()
    bad = []
    for name, value in native.items():
        value, new = np.asarray(value), np.asarray(refreshed[name])
        if name == "bond_mask":
            new, tail = new[:n, :n], [new[n:], new[:, n:]]
        elif new.shape[:1] == pad.shape:  # per atom
            new, tail = new[:n], [new[n:]]
        else:
            tail = []
        if any(np.any(t) for t in tail):
            bad.append(f"{name} (padding)")
        if name == "ref_pos":
            a2t = np.asarray(native["atom_to_token_idx"])
            distances = zip(residue_distances(value, a2t), residue_distances(new, a2t))
            same = all(np.allclose(a, b, atol=1e-4) for a, b in distances)
        else:
            same = value.shape == new.shape and np.allclose(value, new)
        if not same:
            bad.append(name)
    return bad


@pytest.mark.slow
@pytest.mark.parametrize("featurization", ["jfold", "upstream"])
@pytest.mark.parametrize(
    "sequence",
    ["GCPWKDEGRWAY", "PPGGCCWWYGAA", "".join(TOKENS[i % 20] for i in range(7, 19))],
)
def test_design_pack_matches_upstream(upstream, featurization, sequence):
    """Check that a design with a sequence set equals upstream's featurization of it."""
    targets = ["MKTAYIAKQRQISFVKSHFSRQ", "VVGADGVGK"]
    key = jax.random.key(0)
    if featurization == "jfold":
        others = [protein(t) for t in targets]
        pack, _ = jfold_pack([TargetChain(t, use_msa=False) for t in targets], key=key)
    else:
        others = [protein(t) for t in targets] + [ATP, MG]
        features, writer = upstream.features(
            native(protein(PLACEHOLDER), *others), key=key
        )
        pack, _ = Protenix.design_pack(features, writer, BINDER_LENGTH)
    reference, _ = upstream.features(native(protein(sequence), *others), key=key)
    refreshed = eqx.filter_jit(set_binder_sequence)(one_hot(sequence), pack, key)
    assert mismatches(reference, refreshed.features) == []


T = "MKTAYIAKQRQISFVKSHFSRQ"
T_HITS = ["MKTAYLAKQRQISFVKSHFSRQ", "MKSAYIAKQRQ-SFVKSHFSRQ", "MRTAYIAKERQISFVKSHFSRA"]


def a3m(query: str, hits: list[str]) -> str:
    return "".join(f">{i}\n{s}\n" for i, s in enumerate([query, *hits]))


@pytest.mark.slow
@pytest.mark.parametrize("placeholder", ["X", "W"])
def test_plain_features_sequence_identity(upstream, tmp_path, placeholder):
    """Check a plain dict whose binder has no MSA and whose target has one."""
    sequence = "GCPWKDEGRW"
    target = protein(T, msa_dir(tmp_path / "t", T, a3m(T, T_HITS)))
    key = jax.random.key(0)
    binder = protein(placeholder * len(sequence))
    plain, _ = upstream.features(native(binder, target), key=key)
    reference, _ = upstream.features(native(protein(sequence), target), key=key)
    new = jax.jit(set_binder_sequence)(one_hot(sequence), plain, key)
    for name in ("restype", "profile", "msa"):
        np.testing.assert_allclose(new[name], reference[name], atol=1e-6, err_msg=name)


@pytest.mark.slow
def test_plain_features_binder_msa(upstream, tmp_path):
    """Check a plain dict whose binder has its own MSA."""
    old, new_sequence, hits = "GCPWK", "ACPWR", ["GCPWR", "ACPWK"]
    target = protein(T, msa_dir(tmp_path / "t", T, a3m(T, T_HITS)))

    def binder(query):
        return protein(query, msa_dir(tmp_path / query, query, a3m(query, hits)))

    key = jax.random.key(0)
    plain, _ = upstream.features(native(binder(old), target), key=key)
    reference, _ = upstream.features(native(binder(new_sequence), target), key=key)
    new = jax.jit(set_binder_sequence)(one_hot(new_sequence), plain, key)
    for name in ("restype", "profile", "msa"):
        np.testing.assert_allclose(new[name], reference[name], atol=1e-6, err_msg=name)


@pytest.fixture(scope="module")
def model():
    """Load Protenix-Mini."""
    return Protenix.mini()


@pytest.mark.slow
@pytest.mark.model_forward
def test_predicts(model):
    """Check that a design prediction gives a sensible structure and confidence."""
    chains = [TargetChain(TARGET, use_msa=False)]
    pack, writer = model.binder_features(BINDER_LENGTH, chains, key=jax.random.key(5))
    pssm = jax.nn.one_hot(
        jax.random.randint(jax.random.key(3), (BINDER_LENGTH,), 0, 20), 20
    )
    prediction = model.predict(
        PSSM=pssm, features=pack, writer=writer, key=jax.random.key(4)
    )
    output = prediction.model_output
    assert jnp.isfinite(output.structure_coordinates).all()
    assert ((output.plddt >= 0) & (output.plddt <= 1)).all()
    # Consecutive CA atoms of the target should be ~3.8 Å apart.
    ca = output.backbone_coordinates[BINDER_LENGTH:, 1]
    gaps = jnp.linalg.norm(ca[1:] - ca[:-1], axis=-1)
    assert 3.0 < float(jnp.median(gaps)) < 4.5
    binder = gemmi.one_letter_code([r.name for r in prediction.st[0][0]])
    assert binder == "".join(TOKENS[i] for i in pssm.argmax(-1))
    assert gemmi.one_letter_code([r.name for r in prediction.st[0][1]]) == TARGET


def assert_writes_every_entity(st: gemmi.Structure, dna: str):
    """Check that `st` has the protein, DNA, ATP and Mg, each with its atoms."""
    chains = {c.name: c for c in st[0]}
    names = {name: [r.name for r in chain] for name, chain in chains.items()}
    assert gemmi.one_letter_code(names["B"]) == TARGET
    assert names["C"] == [f"D{a}" for a in dna]
    assert names["D"] == ["ATP"] and len(chains["D"][0]) == 31
    assert names["E"] == ["MG"]
    assert chains["D"][0].het_flag == "H"
    entity_types = {e.name: e.entity_type for e in st.entities}
    assert entity_types["ATP!"] == gemmi.EntityType.NonPolymer


@pytest.mark.slow
@pytest.mark.model_forward
def test_upstream_ligand_dna(model, upstream):
    """Check upstream's featurization of a protein with DNA, ATP and Mg."""
    dna = "ACGTAC"
    targets = [
        protein(TARGET),
        {"dnaSequence": {"sequence": dna, "count": 1}},
        ATP,
        MG,
    ]
    key = jax.random.key(0)
    features, writer = upstream.features(native(protein("GSWEEK"), *targets), key=key)
    prediction = model.predict(features=features, writer=writer, key=key)
    assert jnp.isfinite(prediction.model_output.structure_coordinates).all()
    assert np.isfinite(prediction.plddt).all() and np.isfinite(prediction.pae).all()
    assert_writes_every_entity(prediction.st, dna)
    assert len(prediction.plddt) == len(features["asym_id"])

    features, writer = upstream.features(
        native(protein(PLACEHOLDER), *targets), key=key
    )
    pack, writer = model.design_pack(features, writer, BINDER_LENGTH)
    pssm = one_hot(DESIGNED)
    prediction = model.predict(PSSM=pssm, features=pack, writer=writer, key=key)
    assert jnp.isfinite(prediction.model_output.structure_coordinates).all()
    assert np.isfinite(prediction.plddt).all() and np.isfinite(float(prediction.iptm))
    assert_writes_every_entity(prediction.st, dna)
    binder = gemmi.one_letter_code([r.name for r in prediction.st[0]["A"]])
    assert binder == DESIGNED
