from ipymolstar import PDBeMolstar
import gemmi

def pdb_viewer(st: gemmi.Structure):
    """Display a PDB file using Molstar"""
    custom_data = {
        "data": st.make_pdb_string(),
        "format": "pdb",
        "binary": False,
    }
    return PDBeMolstar(custom_data=custom_data, theme="dark")