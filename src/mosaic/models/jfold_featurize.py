"""Featurizing with `jfold.featurize` (torch-free; each view builds the same dict as its
upstream pipeline): what the jfold-featurized backends share. A backend refuses what its view
does not take with `check_chains` (and `check_unpaired_msas` where upstream pairs MSAs),
builds a `jfold.featurize.System` with `jfold_system`, runs its view, and writes structures
with `AtomSitesWriter` or `BinderAtomSitesWriter`. `residue_type_features` gives a design
pack its per-amino-acid atoms; MSAs come from `mosaic.msa`."""

import dataclasses
from collections.abc import Callable

import gemmi
import numpy as np
from jfold.featurize import AtomSites, Chain, System, Template
from jfold.featurize.system import ONE_LETTER, THREE_LETTER

from mosaic import msa
from mosaic.common import TOKENS
from mosaic.structure_prediction import PolymerType, TargetChain

CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _is_amino_acid(residue: gemmi.Residue) -> bool:
    info = gemmi.find_tabulated_residue(residue.name)
    return info is not None and info.is_amino_acid()


def template_from_gemmi(chain: gemmi.Chain | Template) -> Template | None:
    """`chain` as a jfold `Template` (one already is): its amino acids with coordinates, in
    order (waters and ligands left out), each with its heavy atoms (the first alternate
    location of each); None if it has an amino acid jfold does not take (anything but the
    20 and UNK)."""
    if isinstance(chain, Template):
        return chain
    residues = [residue for residue in chain if _is_amino_acid(residue)]
    if any(residue.name not in ONE_LETTER for residue in residues):
        return None
    atoms = []
    for residue in residues:
        positions = {}
        for atom in residue:
            if not atom.is_hydrogen():
                positions.setdefault(atom.name, (atom.pos.x, atom.pos.y, atom.pos.z))
        atoms.append(positions)
    return Template("".join(ONE_LETTER[r.name] for r in residues), tuple(atoms))


def check_chains(
    chains: list[TargetChain],
    model: str,
    *,
    templates: bool,
    also: str = "",
    upstream: str = "",
) -> None:
    """Raise `NotImplementedError` for what `model`'s jfold view does not take: chains
    other than protein, residues other than the 20 amino acids and X (and the letters in
    `also`, e.g. a design placeholder), and templates, unless `templates`, and then only
    those jfold reads (amino acids, without modified ones). `upstream`: how to featurize
    such systems instead (the backend's upstream featurization), added to the message."""
    hint = f"; {upstream}" if upstream else ""
    for chain in chains:
        if chain.polymer_type != PolymerType.PROTEIN:
            raise NotImplementedError(
                f"{model} on jfold takes protein chains only, not {chain.polymer_type}{hint}"
            )
        unknown = set(chain.sequence) - set(THREE_LETTER) - set(also)
        if unknown:
            raise NotImplementedError(
                f"{model} on jfold takes the 20 amino acids and X only, not "
                f"{sorted(unknown)}{hint}"
            )
        if chain.template_chain is None:
            continue
        if not templates:
            raise NotImplementedError(f"{model} on jfold takes no templates{hint}")
        if template_from_gemmi(chain.template_chain) is None:
            raise NotImplementedError(
                f"{model} on jfold takes templates of the 20 amino acids and UNK only{hint}"
            )


def check_unpaired_msas(chains: list[TargetChain], model: str, *, upstream: str = "") -> None:
    """Raise `NotImplementedError` unless upstream's MSAs for `chains` are unpaired, as
    jfold's views build them: at most one sequence with an MSA (upstream pairs the MSAs of
    several), used for all of its chains or none (upstream looks MSAs up by sequence).
    `upstream`: as for `check_chains`."""
    with_msa = {c.sequence for c in chains if c.use_msa}
    without_msa = {c.sequence for c in chains if not c.use_msa}
    if len(with_msa) > 1 or with_msa & without_msa:
        raise NotImplementedError(
            f"{model} on jfold takes unpaired MSAs: at most one sequence with an MSA "
            f"(use_msa), on all of its chains{'; ' + upstream if upstream else ''}"
        )


def jfold_system(chains: list[TargetChain]) -> System:
    """`chains` as a jfold `System`, ids A, B, ... in order, each with its MSA (`use_msa`:
    the chain's own `msa`, else a fetched one) and its template."""
    return System(
        tuple(
            Chain(
                id=chain_id,
                sequence=c.sequence,
                msa=_a3m(c),
                template=None
                if c.template_chain is None
                else template_from_gemmi(c.template_chain),
            )
            for chain_id, c in zip(CHAIN_IDS, chains)
        )
    )


def _a3m(chain: TargetChain) -> str | None:
    if not chain.use_msa:
        return None
    return msa.fetch_a3m(chain.sequence) if chain.msa is None else chain.msa


def residue_type_features(
    featurize: Callable[[System], dict[str, np.ndarray]], context: str = "{}"
) -> dict[str, dict[str, np.ndarray]]:
    """`featurize` of each of the 20 amino acids (`TOKENS` order) in a one-chain system whose
    sequence is `context` with the amino acid in place of `{}` (e.g. `"G{}G"`, away from the
    chain's ends): what a design pack rebuilds a binder residue's atoms from."""
    return {aa: featurize(System((Chain("A", context.format(aa)),))) for aa in TOKENS}


class AtomSitesWriter:
    """Writes coordinates `[A, 3]` (padding atoms after the real ones) as a
    `gemmi.Structure`, from a view's atom sites."""

    def __init__(self, sites: AtomSites):
        self.sites = sites

    def __call__(self, coords) -> gemmi.Structure:
        coords = np.asarray(coords).reshape(-1, 3)[: len(self.sites)]
        st = gemmi.read_pdb_string(self.sites.to_pdb(coords))
        st.setup_entities()
        return st


class BinderAtomSitesWriter:
    """Writes coordinates `[A, 3]` as a `gemmi.Structure`, for features whose binder (the
    first chain) carries the atoms of its sequence's argmax: `binder_pssm` (`[N, 20]`)
    gives that sequence; without it, `system` is written as featurized. `atom_sites` is
    the view's."""

    def __init__(self, system: System, atom_sites: Callable[[System], AtomSites]):
        self.system = system
        self.atom_sites = atom_sites

    def __call__(self, coords, binder_pssm=None) -> gemmi.Structure:
        chains = self.system.chains
        if binder_pssm is not None:
            binder = "".join(TOKENS[i] for i in np.asarray(binder_pssm).argmax(-1))
            chains = (dataclasses.replace(chains[0], sequence=binder), *chains[1:])
        return AtomSitesWriter(self.atom_sites(System(chains)))(coords)
