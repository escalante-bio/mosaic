import equinox as eqx
import gemmi
import numpy as np
from jax import tree
from jaxtyping import Array, Bool, Float, Int

from ..alphafold.common import residue_constants
from ..alphafold.data import (
    feature_processing,
    msa_pairing,
    parsers,
    pipeline,
    pipeline_multimer,
)

class AFFeatures(eqx.Module):
    aatype: Float[Array, "N 21"]
    all_atom_mask: Bool[Array, "N 37"]
    all_atom_positions: Float[Array, "N 37 3"]
    assembly_num_chains: Int
    asym_id: Int[Array, "N"]
    bert_mask: Bool[Array, "512 N"]
    cluster_bias_mask: Bool[Array, "512"]
    deletion_matrix: Int[Array, "512 N"]
    deletion_mean: Float[Array, "N"]
    entity_id: Int[Array, "N"]
    entity_mask: Bool[Array, "N"]
    msa: Float[Array, "512 N 21"]
    msa_mask: Bool[Array, "512 N"]
    num_alignments: Int
    num_templates: Int
    residue_index: Int[Array, "N"]
    seq_length: Int
    seq_mask: Bool[Array, "N"]
    sym_id: Int[Array, "N"]
    template_aatype: Float[Array, "4 N"]
    template_all_atom_mask: Bool[Array, "4 N 37"]
    template_all_atom_positions: Float[Array, "4 N 37 3"]


def af2_atom_positions(chain: gemmi.Chain) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance(chain, gemmi.Chain)
    all_residues = list(chain)
    num_res = len(all_residues)
    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros(
        [num_res, residue_constants.atom_type_num], dtype=np.int64
    )

    for res_idx, res in enumerate(all_residues):
        for atom in res:
            atom_name = atom.name
            x, y, z = atom.pos.x, atom.pos.y, atom.pos.z
            if atom_name in residue_constants.atom_order.keys():
                all_positions[res_idx, residue_constants.atom_order[atom_name]] = [
                    x,
                    y,
                    z,
                ]
                all_positions_mask[res_idx, residue_constants.atom_order[atom_name]] = (
                    1.0
                )
            elif atom_name.upper() == "SE" and res.name() == "MSE":
                # Put the coordinates of the selenium atom in the sulphur column.
                all_positions[res_idx, residue_constants.atom_order["SD"]] = [x, y, z]
                all_positions_mask[res_idx, residue_constants.atom_order["SD"]] = 1.0

    return all_positions, all_positions_mask


def af2_get_atom_positions_gemmi(st) -> tuple[np.ndarray, np.ndarray]:
    return tree.map(
        lambda *v: np.concatenate(v), *[af2_atom_positions(chain) for chain in st[0]]
    )


def aa_code(c: gemmi.Chain):
    return gemmi.one_letter_code([r.name for r in c])


def empty_placeholder_template_features(num_templates: int, num_res: int):
    return {
        "template_aatype": np.zeros(
            (num_templates, num_res, len(residue_constants.restypes_with_x_and_gap)),
            dtype=np.float32,
        ),
        "template_all_atom_masks": np.zeros(
            (num_templates, num_res, residue_constants.atom_type_num), dtype=np.float32
        ),
        "template_all_atom_positions": np.zeros(
            (num_templates, num_res, residue_constants.atom_type_num, 3),
            dtype=np.float32,
        ),
        "template_domain_names": np.zeros([num_templates], dtype=object),
        "template_sequence": np.zeros([num_templates], dtype=object),
        "template_sum_probs": np.zeros([num_templates], dtype=np.float32),
    }


def chain_template_features(chain: gemmi.Chain):
    sequence = gemmi.one_letter_code([r.name for r in chain])
    all_atom, all_atom_masks = af2_atom_positions(chain)

    # # mask out non-backbone + CB atoms
    # all_atom_masks[:, 4:] = 0
    # all_atom[:, 4:] = 0

    return {
        "template_aatype": residue_constants.sequence_to_onehot(
            sequence, residue_constants.HHBLITS_AA_TO_ID
        )[None],
        "template_all_atom_positions": all_atom[None],
        "template_all_atom_masks": all_atom_masks[None],
        "template_domain_names": np.array([f"{chain.name}".encode()]),
        "template_sequence": np.array([sequence.encode()]),
        "template_sum_probs": np.array([len(chain)], dtype=np.float32),  # ?
    }


def build_features(sequences_and_templates: list[tuple[str, any]]) -> AFFeatures:
    """Compute input feature dictionary for single AF2 input"""
    features_for_chain = {}

    for sequence_idx, (seq, template) in enumerate(sequences_and_templates):
        feature_dict = {}
        msa = parsers.Msa(
            sequences=[seq],
            deletion_matrix=[
                [0] * len(seq),
            ],
            descriptions=["none"],
        )
        feature_dict.update(
            pipeline.make_sequence_features(
                sequence=seq, description="query", num_res=len(seq)
            )
        )
        feature_dict.update(pipeline.make_msa_features(msas=[msa]))

        feature_dict.update(template)
        valid_feats = msa_pairing.MSA_FEATURES + ("msa_species_identifiers",)
        all_seq_features = {
            f"{k}_all_seq": v
            for k, v in pipeline.make_msa_features([msa]).items()
            if k in valid_feats
        }
        feature_dict.update(all_seq_features)

        features_for_chain["ABCDEFGHI"[sequence_idx]] = feature_dict

    all_chain_features = {}
    for chain_id, chain_features in features_for_chain.items():
        all_chain_features[chain_id] = pipeline_multimer.convert_monomer_features(
            chain_features, chain_id
        )

    all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)

    features = feature_processing.pair_and_merge(all_chain_features=all_chain_features)

    # Pad MSA to avoid zero-sized extra_msa.
    features = pipeline_multimer.pad_msa(features, min_num_seq=1)


    #### we need to one-hot the sequence and MSA
    return AFFeatures(**features)
