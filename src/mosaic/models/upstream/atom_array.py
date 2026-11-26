import io

import gemmi
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io import pdbx


def atom_array_to_gemmi(atoms: AtomArray, coords, *, infer_entities: bool = False) -> gemmi.Structure:
    """Write the real atoms in feature order, dropping trailing coordinate padding."""
    atoms = atoms.copy()
    atoms.coord = np.asarray(coords, np.float32).reshape(-1, 3)[: len(atoms)]
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, atoms)
    if infer_entities:
        # Protenix uses Gemmi's entity names (e.g. ATP!), Biotite writes numeric IDs.
        del cif.block["atom_site"]["label_entity_id"]
    buffer = io.StringIO()
    cif.write(buffer)
    st = gemmi.make_structure_from_block(gemmi.cif.read_string(buffer.getvalue())[0])
    st.setup_entities()
    return st
