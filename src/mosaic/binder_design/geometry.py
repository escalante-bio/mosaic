"""BioPython-based structural metrics for binder design filtering.

Migrated from ``ddcraft.utils.biopython`` (DdCraft py3.10 pipeline) to run
natively inside Mosaic. Numerical behaviour is preserved exactly; only
imports/logging were adapted to remove the dependency on the ``ddcraft``
package.
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
from collections import defaultdict

import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import Chain, DSSP, PDBIO, PDBParser, Polypeptide, Select, Selection, Superimposer
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Selection import unfold_entities
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def get_chain_length(pdb_file, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb_structure", pdb_file)
    residues = []

    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if residue.id[0] == ' ':  # Ignore heteroatoms and waters
                        residues.append(residue)

                return len(residues)
    return 0  # If chain not found

# analyze sequence composition of design
def validate_design_sequence(sequence, num_clashes, advanced_settings):
    note_array = []

    # Check if protein contains clashes after relaxation
    if num_clashes is not None and num_clashes > 0:
        note_array.append('Relaxed structure contains clashes.')

    # Check if the sequence contains disallowed amino acids
    if advanced_settings["omit_AAs"]:
        restricted_AAs = advanced_settings["omit_AAs"].split(',')
        for restricted_AA in restricted_AAs:
            if restricted_AA in sequence:
                note_array.append('Contains: '+restricted_AA+'!')

    # Analyze the protein
    analysis = ProteinAnalysis(sequence)

    # Calculate the reduced extinction coefficient per 1% solution
    extinction_coefficient_reduced = analysis.molar_extinction_coefficient()[0]
    molecular_weight = round(analysis.molecular_weight() / 1000, 2)
    extinction_coefficient_reduced_1 = round(extinction_coefficient_reduced / molecular_weight * 0.01, 2)

    # Check if the absorption is high enough
    if extinction_coefficient_reduced_1 <= 2:
        note_array.append(f'Absorption value is {extinction_coefficient_reduced_1}, consider adding tryptophane to design.')

    # Join the notes into a single string
    notes = ' '.join(note_array)

    return notes

# temporary function, calculate RMSD of input PDB and trajectory target
def chain_rmsd(reference_pdb, mobile_pdb, chain_id):
    """CA RMSD of one chain between two structures, after superposing on that same chain.

    Reports internal conformational deviation (e.g. did the target distort during prediction?)
    independently of any rigid-body movement. Returns None if the chain cannot be compared.
    """
    parser = PDBParser(QUIET=True)
    try:
        ref = parser.get_structure('reference', reference_pdb)[0]
        mob = parser.get_structure('mobile', mobile_pdb)[0]
    except Exception:
        return None

    chain_id = chain_id.split(',')[0].strip()
    if chain_id not in ref or chain_id not in mob:
        return None

    def ca_by_resid(model):
        return {
            residue.id[1]: residue['CA']
            for residue in model[chain_id]
            if is_aa(residue, standard=True) and 'CA' in residue
        }

    ref_ca, mob_ca = ca_by_resid(ref), ca_by_resid(mob)
    shared = sorted(set(ref_ca) & set(mob_ca))
    if len(shared) < 3:
        return None

    sup = Superimposer()
    sup.set_atoms([ref_ca[i] for i in shared], [mob_ca[i] for i in shared])
    return round(float(sup.rms), 2)


def hotspot_rmsd(trajectory_pdb, prediction_pdb, target_chain, binder_chain):
    """Binder RMSD between a predicted complex and its trajectory after superposing on the target.

    Superposing on the target chain puts both structures in the target's frame, so the residual
    binder RMSD reports whether the predicted design still engages the same epitope in the same
    pose as the trajectory that was optimised against the hotspots. A design that slides to a
    different surface patch yields a large value.

    Returns None if the structures cannot be compared (missing chains / no shared residues).
    """
    parser = PDBParser(QUIET=True)
    try:
        ref = parser.get_structure('trajectory', trajectory_pdb)[0]
        mob = parser.get_structure('prediction', prediction_pdb)[0]
    except Exception:
        return None

    target_chain = target_chain.split(',')[0].strip()

    def ca_by_resid(model, chain_id):
        if chain_id not in model:
            return {}
        return {
            residue.id[1]: residue['CA']
            for residue in model[chain_id]
            if is_aa(residue, standard=True) and 'CA' in residue
        }

    ref_target, mob_target = ca_by_resid(ref, target_chain), ca_by_resid(mob, target_chain)
    shared_target = sorted(set(ref_target) & set(mob_target))
    ref_binder, mob_binder = ca_by_resid(ref, binder_chain), ca_by_resid(mob, binder_chain)
    shared_binder = sorted(set(ref_binder) & set(mob_binder))

    if len(shared_target) < 3 or not shared_binder:
        return None

    sup = Superimposer()
    sup.set_atoms([ref_target[i] for i in shared_target],
                  [mob_target[i] for i in shared_target])
    # Apply the target-frame transform to the mobile binder without altering the reference.
    sup.apply([mob_binder[i] for i in shared_binder])

    diff = np.array([mob_binder[i].coord - ref_binder[i].coord for i in shared_binder])
    return round(float(np.sqrt((diff ** 2).sum(axis=1).mean())), 2)


def target_pdb_rmsd(trajectory_pdb, starting_pdb, chain_ids_string):
    # Parse the PDB files
    parser = PDBParser(QUIET=True)
    structure_trajectory = parser.get_structure('trajectory', trajectory_pdb)
    structure_starting = parser.get_structure('starting', starting_pdb)
    
    # Extract chain A from trajectory_pdb
    chain_trajectory = structure_trajectory[0]['A']
    
    # Extract the specified chains from starting_pdb
    chain_ids = chain_ids_string.split(',')
    residues_starting = []
    for chain_id in chain_ids:
        chain_id = chain_id.strip()
        chain = structure_starting[0][chain_id]
        for residue in chain:
            if is_aa(residue, standard=True):
                residues_starting.append(residue)
    
    # Extract residues from chain A in trajectory_pdb
    residues_trajectory = [residue for residue in chain_trajectory if is_aa(residue, standard=True)]
    
    # Ensure that both structures have the same number of residues
    min_length = min(len(residues_starting), len(residues_trajectory))
    residues_starting = residues_starting[:min_length]
    residues_trajectory = residues_trajectory[:min_length]
    
    # Collect CA atoms from the two sets of residues
    atoms_starting = [residue['CA'] for residue in residues_starting if 'CA' in residue]
    atoms_trajectory = [residue['CA'] for residue in residues_trajectory if 'CA' in residue]
    
    # Calculate RMSD using structural alignment
    sup = Superimposer()
    sup.set_atoms(atoms_starting, atoms_trajectory)
    rmsd = sup.rms
    
    return round(rmsd, 2)

# detect C alpha clashes for deformed trajectories
def calculate_clash_score(pdb_file, threshold=2.4, only_ca=False):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    atoms = []
    atom_info = []  # Detailed atom info for debugging and processing

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element == 'H':  # Skip hydrogen atoms
                        continue
                    if only_ca and atom.get_name() != 'CA':
                        continue
                    atoms.append(atom.coord)
                    atom_info.append((chain.id, residue.id[1], atom.get_name(), atom.coord))

    tree = cKDTree(atoms)
    pairs = tree.query_pairs(threshold)

    valid_pairs = set()
    for (i, j) in pairs:
        chain_i, res_i, name_i, coord_i = atom_info[i]
        chain_j, res_j, name_j, coord_j = atom_info[j]

        # Exclude clashes within the same residue
        if chain_i == chain_j and res_i == res_j:
            continue

        # Exclude directly sequential residues in the same chain for all atoms
        if chain_i == chain_j and abs(res_i - res_j) == 1:
            continue

        # If calculating sidechain clashes, only consider clashes between different chains
        if not only_ca and chain_i == chain_j:
            continue

        valid_pairs.add((i, j))

    return len(valid_pairs)

three_to_one_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# identify interacting residues at the binder interface
def hotspot_residues(trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0):
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    # Get the specified chain
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], 'A')
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # Dynamically determine target chain (the chain that's not the binder)
    available_chains = [chain.id for chain in structure[0]]
    target_chain = None
    for chain_id in available_chains:
        if chain_id != binder_chain:
            target_chain = chain_id
            break
    
    if target_chain is None:
        raise ValueError(f"Could not find target chain. Available chains: {available_chains}, binder chain: {binder_chain}")
    
    # Get atoms and coords for the target chain
    target_atoms = Selection.unfold_entities(structure[0][target_chain], 'A')
    target_coords = np.array([atom.coord for atom in target_atoms])

    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Prepare to collect interacting residues
    interacting_residues = {}

    # Query the tree for pairs of atoms within the distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Process each binder atom's interactions
    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        # Convert three-letter code to single-letter code using the manual dictionary
        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues

class FirstModelSelect(Select):
    def accept_model(self, model):
        return model.id == 0  # Only first model

def calc_ss_percentage(pdb_file, advanced_settings, chain_id="B", atom_distance_cutoff=4.0):
    # Parse the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]  # Consider only the first model

    # Save the first model to a temp file (removes MODEL/ENDMDL etc.)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
        tmp_pdb_path = tmp.name
    io = PDBIO()
    io.set_structure(model)
    io.save(tmp_pdb_path, select=FirstModelSelect())

    # Inject a dummy CRYST1 line at the top
    with open(tmp_pdb_path, "r") as f:
        lines = f.readlines()
    
    # If the first line is not CRYST1, insert a dummy one
    if not lines[0].startswith("CRYST1"):
        lines.insert(0, "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1\n")
        with open(tmp_pdb_path, "w") as f:
            f.writelines(lines)
            
    # Calculate DSSP for the cleaned model
    dssp = DSSP(model, tmp_pdb_path, dssp=advanced_settings["dssp_path"])

    # Prepare to count residues
    ss_counts = defaultdict(int)
    ss_interface_counts = defaultdict(int)
    plddts_interface = []
    plddts_ss = []

    # Get chain and interacting residues once
    chain = model[chain_id]
    interacting_residues = set(hotspot_residues(pdb_file, chain_id, atom_distance_cutoff).keys())

    for residue in chain:
        residue_id = residue.id[1]
        if (chain_id, residue_id) in dssp:
            ss = dssp[(chain_id, residue_id)][2]  # Get the secondary structure
            ss_type = 'loop'
            if ss in ['H', 'G', 'I']:
                ss_type = 'helix'
            elif ss == 'E':
                ss_type = 'sheet'

            ss_counts[ss_type] += 1

            if ss_type != 'loop':
                avg_plddt_ss = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_ss.append(avg_plddt_ss)

            if residue_id in interacting_residues:
                ss_interface_counts[ss_type] += 1
                avg_plddt_residue = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_interface.append(avg_plddt_residue)

    # Clean up the temporary file
    os.remove(tmp_pdb_path)

    # Calculate percentages
    total_residues = sum(ss_counts.values())
    total_interface_residues = sum(ss_interface_counts.values())

    percentages = calculate_percentages(total_residues, ss_counts['helix'], ss_counts['sheet'])
    interface_percentages = calculate_percentages(total_interface_residues, ss_interface_counts['helix'], ss_interface_counts['sheet'])

    i_plddt = round(sum(plddts_interface) / len(plddts_interface) / 100, 2) if plddts_interface else 0
    ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0

    return (*percentages, *interface_percentages, i_plddt, ss_plddt)

def mean_ca_bfactor(pdb_file):
    """Mean CA B-factor of the first model, or ``None`` when there are no CAs.

    Predicted structures carry per-residue pLDDT in the B-factor column, so this
    ranks the five AF2 models of a design without re-reading their metrics.
    """
    structure = PDBParser(QUIET=True).get_structure("model", pdb_file)
    values = [
        float(residue["CA"].bfactor)
        for chain in next(structure.get_models())
        for residue in chain
        if residue.id[0] == " " and "CA" in residue
    ]
    return sum(values) / len(values) if values else None


def calculate_percentages(total, helix, sheet):
    helix_percentage = round((helix / total) * 100,2) if total > 0 else 0
    sheet_percentage = round((sheet / total) * 100,2) if total > 0 else 0
    loop_percentage = round(((total - helix - sheet) / total) * 100,2) if total > 0 else 0

    return helix_percentage, sheet_percentage, loop_percentage


def clean_pdb(pdb_file):
    """Strip a PDB file down to structural record lines only.

    Migrated from ``ddcraft.utils.generic.clean_pdb``; used by :mod:`.rosetta`
    after PyRosetta writes out a PDB (e.g. after alignment or relax) to
    remove Rosetta-specific header/footer noise.
    """
    # Read the pdb file and filter relevant lines
    with open(pdb_file, 'r') as f_in:
        relevant_lines = [line for line in f_in if line.startswith(('ATOM', 'HETATM', 'MODEL', 'TER', 'END', 'LINK'))]

    # Write the cleaned lines back to the original pdb file
    with open(pdb_file, 'w') as f_out:
        f_out.writelines(relevant_lines)
