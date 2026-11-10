"""BioEmu2 featurized by upstream's `bioemu2` package (torch) from its own input: the
`(System, MSA)` pair `bioemu2.structure.preparation.prepare_structure` takes, so every entity
bioemu2 builds (protein, DNA, RNA, ...) with its MSA handling. `features` returns the feature
dict `BioEmu2.target_only_features` returns and a writer of every entity, ready for
`BioEmu2.predict`, `BioEmu2.denoiser` and the losses.

To design, put the binder first as a protein placeholder of "X" residues: `set_sequence`
overwrites the first `PSSM.shape[0]` tokens. Give each protein chain its query as its
paired MSA, as jfold's featurization does: without paired MSAs upstream cuts a heteromer's
unpaired MSAs to the shallowest chain's depth (one row next to a single-sequence binder),
while with them every chain keeps its own rows::

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

# What jfold's view returns of upstream's dict: the entries `jfold.models.bioemu2.features`
# reads, and `mol_type`.
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
    """Writes coordinates `[A, 3]` (padding atoms after the real ones) as a
    `gemmi.Structure`, every entity, with upstream's topology and mmCIF writer."""

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
    """Upstream's features for `input`, a `System` and the `MSA` of its protein chains, in
    `mode` ("structure" or "equilibrium", the backend's `mode`), as `BioEmu2` takes them,
    and their writer. Upstream's featurization is deterministic: `key` is unused."""
    del key
    system, msa = input
    structure = prepare_structure(system, msa=msa, mode=mode, contacts=[])
    batch = convert_densebatch_to_dictionary(batch_collate([structure]))
    topology = topology_from_structure(structure, renamed_asym_ids=system.chain_ids)
    return feature_dict({k: batch[k][0].numpy() for k in KEYS}), CifWriter(topology)
