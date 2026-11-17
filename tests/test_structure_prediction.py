import jax
import jax.numpy as jnp
import numpy as np

from mosaic.structure_prediction import StructureModelOutput, predicted_tm_score


def test_interface_tm_score_and_gradient_ignore_padding():
    n = 24
    logits = jax.random.normal(jax.random.key(0), (n, n, 3))
    bins = jnp.array([0.25, 2.5, 12.0])

    def score(logits, padding):
        padded = jnp.pad(logits, ((0, padding), (0, padding), (0, 0)), constant_values=jnp.nan)
        indices = jnp.arange(n + padding)
        binder = indices < 4
        return predicted_tm_score(
            padded, bins, binder[:, None] != binder[None, :], token_mask=indices < n
        ).max()

    value_grad = jax.jit(jax.value_and_grad(score), static_argnums=1)
    value, grad = value_grad(logits, 0)
    padded_value, padded_grad = value_grad(logits, 16)
    np.testing.assert_allclose(padded_value, value, rtol=1e-6)
    np.testing.assert_allclose(padded_grad, grad, rtol=1e-5, atol=1e-8)


def test_to_structure_preserves_discontiguous_residue_indices():
    n_residues = 4
    atom37_mask = jnp.zeros((n_residues, 37)).at[:, 1].set(1)
    output = StructureModelOutput(
        distogram_logits=jnp.zeros((n_residues, n_residues, 1)),
        distogram_bins=jnp.zeros(1),
        plddt=jnp.ones(n_residues),
        pae=jnp.zeros((n_residues, n_residues)),
        pae_logits=jnp.zeros((n_residues, n_residues, 1)),
        pae_bins=jnp.zeros(1),
        structure_coordinates=jnp.zeros((n_residues, 37, 3)),
        backbone_coordinates=jnp.zeros((n_residues, 4, 3)),
        full_sequence=jax.nn.one_hot(jnp.zeros(n_residues, dtype=jnp.int32), 20),
        asym_id=jnp.zeros(n_residues, dtype=jnp.int32),
        residue_idx=jnp.array([4, 5, 10, 11]),
        atom37_coords=jnp.zeros((n_residues, 37, 3)),
        atom37_mask=atom37_mask,
    )

    structure = output.to_structure()

    assert [residue.seqid.num for residue in structure[0][0]] == [4, 5, 10, 11]


def test_target_chain_msa_must_agree_with_use_msa():
    import pytest

    from mosaic.structure_prediction import TargetChain

    assert TargetChain("ACD", msa=">query\nAC\nD\n>hit\nAaC-\n").msa is not None
    with pytest.raises(ValueError, match="use_msa=False"):
        TargetChain("ACD", use_msa=False, msa=">query\nACD\n")


def test_jfold_system_takes_a_chain_s_own_msa_and_template(monkeypatch):
    from jfold.featurize import Template

    from mosaic import msa
    from mosaic.models.jfold_featurize import jfold_system
    from mosaic.structure_prediction import TargetChain

    def fetch(sequence):
        return f">query\n{sequence}\n>fetched\n{sequence}\n"

    monkeypatch.setattr(msa, "fetch_a3m", fetch)
    own = ">query\nACD\n>hit\nA-D\n"
    template = Template("ACD", ({"CA": (0.0, 0.0, 0.0)}, {}, {}))
    system = jfold_system(
        [
            TargetChain("ACD", msa=own, template_chain=template),
            TargetChain("GG"),
            TargetChain("W", use_msa=False),
        ]
    )
    assert [c.msa for c in system.chains] == [own, fetch("GG"), None]
    assert system.chains[0].template is template
