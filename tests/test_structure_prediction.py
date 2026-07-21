import jax
import jax.numpy as jnp

from mosaic.losses.structure_prediction import StructureModelOutput


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
