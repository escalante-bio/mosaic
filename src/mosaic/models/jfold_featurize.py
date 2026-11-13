"""Helpers shared by the backends that featurize with `jfold.featurize`.

Each jfold featurizer builds the same feature dict as the model's upstream pipeline, without
torch. A backend checks its inputs with `check_chains`, builds a `jfold.featurize.System`
with `jfold_system`, featurizes it, and writes structures with `AtomSitesWriter` or
`BinderAtomSitesWriter`."""

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
    """Convert a gemmi chain's amino acids and their heavy atoms to a jfold `Template`.

    Waters and ligands are dropped and only the first alternate location of each atom is
    kept. Return None if the chain has an amino acid other than the standard 20 and UNK."""
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
    """Raise `NotImplementedError` if jfold cannot featurize `chains` for `model`.

    Only protein chains made of the 20 amino acids, X and the extra letters in `also` are
    accepted. Templates are accepted only if `templates` is true and they contain only
    standard amino acids. `upstream` is a hint, appended to the error message, on how to
    featurize such inputs instead."""
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
    """Raise `NotImplementedError` if the upstream pipeline would pair MSAs for `chains`.

    jfold only builds unpaired MSAs, which match upstream when at most one distinct sequence
    uses an MSA and every chain with that sequence uses it. `upstream` is a hint appended to
    the error message."""
    with_msa = {c.sequence for c in chains if c.use_msa}
    without_msa = {c.sequence for c in chains if not c.use_msa}
    if len(with_msa) > 1 or with_msa & without_msa:
        raise NotImplementedError(
            f"{model} on jfold takes unpaired MSAs: at most one sequence with an MSA "
            f"(use_msa), on all of its chains{'; ' + upstream if upstream else ''}"
        )


def jfold_system(chains: list[TargetChain]) -> System:
    """Build a jfold `System` from `chains`, naming them A, B, ... in order.

    Chains with `use_msa` get their own `msa` if set, otherwise one fetched from the MSA
    server."""
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
    """Featurize each of the 20 amino acids on its own, keyed by amino acid.

    Each one is featurized as a single chain whose sequence is `context` with the amino acid
    substituted for `{}`, for example `"G{}G"` to keep it away from the chain ends. Design
    backends use these features to rebuild the atoms of each binder residue."""
    return {aa: featurize(System((Chain("A", context.format(aa)),))) for aa in TOKENS}


class AtomSitesWriter:
    """Convert predicted coordinates of shape `[A, 3]` to a `gemmi.Structure`.

    Padding atoms must come after the real ones and are dropped."""

    def __init__(self, sites: AtomSites):
        self.sites = sites

    def __call__(self, coords) -> gemmi.Structure:
        coords = np.asarray(coords).reshape(-1, 3)[: len(self.sites)]
        st = gemmi.read_pdb_string(self.sites.to_pdb(coords))
        st.setup_entities()
        return st


class BinderAtomSitesWriter:
    """Convert predicted coordinates of shape `[A, 3]` to a `gemmi.Structure` for a design.

    The binder is the first chain. If `binder_pssm` (shape `[N, 20]`) is given, the binder's
    sequence is taken as its argmax, otherwise `system` is written unchanged."""

    def __init__(self, system: System, atom_sites: Callable[[System], AtomSites]):
        self.system = system
        self.atom_sites = atom_sites

    def __call__(self, coords, binder_pssm=None) -> gemmi.Structure:
        chains = self.system.chains
        if binder_pssm is not None:
            binder = "".join(TOKENS[i] for i in np.asarray(binder_pssm).argmax(-1))
            chains = (dataclasses.replace(chains[0], sequence=binder), *chains[1:])
        return AtomSitesWriter(self.atom_sites(System(chains)))(coords)
