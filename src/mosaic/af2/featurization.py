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


def mask_templates_except_positions(features: AFFeatures, keep_positions: np.ndarray) -> AFFeatures:
    """Return a copy of features with template info masked out except at keep_positions.

    Only the first template slice is considered; others are preserved.
    """
    import numpy as _np
    keep = _np.asarray(keep_positions, dtype=_np.int32)
    N = features.all_atom_mask.shape[0]
    mask_rows = _np.zeros((N,), dtype=features.template_all_atom_mask.dtype)
    mask_rows[keep] = 1
    # broadcast to atom dimension
    new_mask = features.template_all_atom_mask.copy()
    new_mask[0] = new_mask[0] * mask_rows[:, None]
    new_pos = features.template_all_atom_positions.copy()
    # zero out masked rows to avoid leaking coordinates
    row_mask = (mask_rows == 1)[:, None, None]
    new_pos[0] = _np.where(row_mask, new_pos[0], 0.0)

    return eqx.tree_at(
        lambda f: (f.template_all_atom_mask, f.template_all_atom_positions),
        features,
        (new_mask, new_pos),
    )


def inject_partial_template_from_pdb(
    features: AFFeatures,
    keep_positions: np.ndarray,
    pdb_path: str,
    chain_id: str,
    pdb_resnums: np.ndarray,
) -> AFFeatures:
    """Fill template arrays only at keep_positions using atoms from PDB residues.

    - Expects len(keep_positions) == len(pdb_resnums)
    - Writes into template slice 0 for positions in keep_positions
    - Leaves other positions masked/zeroed
    """
    import numpy as _np
    import gemmi as _gemmi
    from ..alphafold.common import residue_constants as _rc

    keep = _np.asarray(keep_positions, dtype=_np.int32)
    resnums = _np.asarray(pdb_resnums, dtype=_np.int32)
    assert keep.shape[0] == resnums.shape[0]

    st = _gemmi.read_structure(str(pdb_path))
    chain = st[0][str(chain_id)]
    # Build look-up for residue seqid -> gemmi residue
    res_by_num = {r.seqid.num: r for r in chain}

    N = features.all_atom_mask.shape[0]
    new_mask = features.template_all_atom_mask.copy()
    new_pos = features.template_all_atom_positions.copy()

    # Zero everything for template 0 initially
    new_mask[0, :, :] = 0
    new_pos[0, :, :, :] = 0.0

    for p_idx, resnum in zip(keep.tolist(), resnums.tolist()):
        if not (0 <= p_idx < N):
            continue
        res = res_by_num.get(int(resnum))
        if res is None:
            continue
        for atom in res:
            an = str(atom.name)
            if an in _rc.atom_order:
                ai = _rc.atom_order[an]
                new_pos[0, p_idx, ai, 0] = float(atom.pos.x)
                new_pos[0, p_idx, ai, 1] = float(atom.pos.y)
                new_pos[0, p_idx, ai, 2] = float(atom.pos.z)
                new_mask[0, p_idx, ai] = 1

    return eqx.tree_at(
        lambda f: (f.template_all_atom_mask, f.template_all_atom_positions),
        features,
        (new_mask, new_pos),
    )

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
