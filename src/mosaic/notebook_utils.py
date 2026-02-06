from ipymolstar import PDBeMolstar
import gemmi

def pdb_viewer(st: gemmi.Structure):
    """Display a PDB file using Molstar"""
    custom_data = {
        "data": st.make_pdb_string(),
        "format": "pdb",
        "binary": False,
    }
    return PDBeMolstar(custom_data=custom_data, visual_style="cartoon")

def monomer_ca_rmsd(a_chain: gemmi.Chain, b_chain: gemmi.Chain):
    return monomer_ca_alignment(a_chain, b_chain).rmsd

def monomer_ca_alignment(a_chain: gemmi.Chain, b_chain: gemmi.Chain):
    assert len(a_chain) == len(b_chain)  
    return gemmi.superpose_positions([r.get_ca().pos for r in a_chain], [r.get_ca().pos for r in b_chain])

def gemmi_structure_from_models(name: str, models: list[gemmi.Model], chain_idx = 1) -> gemmi.Structure:
    """ Align models using chain `chain_idx` and stack into a single gemmi Structure."""
    target_zero = models[0][chain_idx]
    st = gemmi.Structure()
    st.name = name
    st.add_model(models[0])
    for model in models[1:]:
        superposition = monomer_ca_alignment(target_zero, model[chain_idx])
        for c in model:
            c.get_polymer().transform_pos_and_adp(superposition.transform)
        st.add_model(model)

    st.renumber_models()
    st.assign_label_seq_id()
    st.setup_entities()
    st.ensure_entities()
    st.add_entity_types()
    st.assign_subchains()
    return st