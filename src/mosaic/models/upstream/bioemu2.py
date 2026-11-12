"""Featurize BioEmu2 inputs with the upstream `bioemu2` package, which needs torch.

This supports every entity type bioemu2 handles, including DNA and RNA. `features` takes
the `(System, MSA)` pair that `bioemu2.structure.preparation.prepare_structure` takes and
returns the same feature dict as `BioEmu2.target_only_features`, with a structure writer.

For design, put the binder first as a protein chain of "X" residues, since `set_sequence`
overwrites the first `PSSM.shape[0]` tokens. Give each protein chain its query sequence as
its paired MSA, as jfold does. Without paired MSAs, upstream cuts all unpaired MSAs of a
heteromer to the depth of the shallowest one, which is a single row when the binder has no
MSA::

    from bioemu2.data_interface import DNA, Protein, System
    from bioemu2.msa.alignment import MSA, get_msa

    from mosaic.losses.structure_prediction import BinderTargetContact
    from mosaic.models.bioemu2 import BioEmu2
    from mosaic.models.upstream import bioemu2 as upstream

    binder, target = "X" * 60, "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    system = System(
        entities=[
            Protein(sequence=binder, asym_ids=["A"]),
            Protein(sequence=target, asym_ids=["B"]),
            DNA(sequence="GACTCG", asym_ids=["C"]),
        ]
    )
    target_msa = get_msa(System(entities=[Protein(sequence=target, asym_ids=["B"])]))
    msa = MSA(
        unpaired_a3m={"A": f">query\\n{binder}\\n", "B": target_msa.a3m("B")},
        paired_a3m={"A": f">query\\n{binder}\\n", "B": f">query\\n{target}\\n"},
    )
    model = BioEmu2()
    features, writer = upstream.features((system, msa), mode=model.mode, key=key)
    loss = model.build_loss(loss=BinderTargetContact(), features=features)
"""

import tempfile
from pathlib import Path

import gemmi
import mdtraj
import numpy as np
from bioemu2.biomoa.utils.feature_converter import convert_densebatch_to_dictionary
from bioemu2.conditioning.method import SamplingMode
from bioemu2.data_interface import System
from bioemu2.msa.alignment import MSA
from bioemu2.structure.batch import batch_collate
from bioemu2.structure.inference_inputs import topology_from_structure
from bioemu2.structure.preparation import prepare_structure
from bioemu2.trajectory_io import save_mdtraj_cif_frame
from jaxtyping import Array, PRNGKeyArray

from mosaic.models.bioemu2 import feature_dict

# The upstream features that jfold's featurizer also returns: those that
# `jfold.models.bioemu2.features` reads, plus `mol_type`.
KEYS = (
    "template_coords",
    "charge",
    "element",
    "name_encoding",
    "ref_space_uid",
    "token_idx_at_atom",
    "atom_pad_mask",
    "atom_mask",
    "mol_type",
    "token_pad_mask",
    "token_center_atom_index",
    "res_type",
    "asym_id",
    "entity_id",
    "sym_id",
    "residue_index",
    "token_bonds",
    "msa",
    "msa_mask",
    "msa_paired",
    "has_deletion",
    "deletion_value",
    "profile",
    "deletion_mean",
    "pocket_feature",
    "method_feature",
    "contact_conditioning",
    "contact_threshold",
    "coords_cond",
    "dt",
    "is_dynamics_conditioned",
)


class CifWriter:
    """Convert coordinates of shape `[A, 3]` to a `gemmi.Structure` with upstream's writer.

    Padding atoms must come after the real ones and are dropped."""

    def __init__(self, topology: mdtraj.Topology):
        self.topology = topology

    def __call__(self, coords) -> gemmi.Structure:
        coords = np.asarray(coords).reshape(-1, 3)[: self.topology.n_atoms]
        # mdtraj works in nanometers.
        frame = mdtraj.Trajectory(xyz=coords[None] / 10.0, topology=self.topology)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.cif"
            save_mdtraj_cif_frame(frame, path)
            return gemmi.read_structure(str(path))


def features(
    input: tuple[System, MSA], *, mode: SamplingMode, key: PRNGKeyArray
) -> tuple[dict[str, Array], CifWriter]:
    """Featurize a `System` and the `MSA` of its protein chains with upstream's pipeline.

    `mode` should match the `BioEmu2` model's mode. Return the features and a structure
    writer. Featurization is deterministic, so `key` is unused."""
    del key
    system, msa = input
    structure = prepare_structure(system, msa=msa, mode=mode, contacts=[])
    batch = convert_densebatch_to_dictionary(batch_collate([structure]))
    topology = topology_from_structure(structure, renamed_asym_ids=system.chain_ids)
    return feature_dict({k: batch[k][0].numpy() for k in KEYS}), CifWriter(topology)
