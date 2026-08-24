"""Column layouts for the design CSVs.

Kept byte-compatible with DdCraft's ``generate_dataframe_labels`` so existing
analysis scripts and filter configs read the output unchanged.
"""

from __future__ import annotations

__all__ = [
    "CORE_LABELS",
    "TRAJECTORY_LABELS",
    "design_labels",
    "final_labels",
    "trajectory_labels",
]

TRAJECTORY_LABELS = [
    "Design", "Protocol", "Length", "Seed", "Helicity", "Target_Hotspot",
    "Sequence", "InterfaceResidues", "pLDDT", "pTM", "i_pTM", "IPSAE", "pAE",
    "i_pAE", "i_pLDDT", "ss_pLDDT", "Unrelaxed_Clashes", "Relaxed_Clashes",
    "Binder_Energy_Score", "Surface_Hydrophobicity", "ShapeComplementarity",
    "PackStat", "dG", "dSASA", "dG/dSASA", "Interface_SASA_%",
    "Interface_Hydrophobicity", "n_InterfaceResidues", "n_InterfaceHbonds",
    "InterfaceHbondsPercentage", "n_InterfaceUnsatHbonds",
    "InterfaceUnsatHbondsPercentage", "Interface_Helix%", "Interface_BetaSheet%",
    "Interface_Loop%", "Binder_Helix%", "Binder_BetaSheet%", "Binder_Loop%",
    "InterfaceAAs", "Target_RMSD", "TrajectoryTime", "Notes", "TargetSettings",
    "Filters", "AdvancedSettings",
]

CORE_LABELS = [
    "pLDDT", "pTM", "i_pTM", "IPSAE", "pAE", "i_pAE", "i_pLDDT", "ss_pLDDT",
    "Unrelaxed_Clashes", "Relaxed_Clashes", "Binder_Energy_Score",
    "Surface_Hydrophobicity", "ShapeComplementarity", "PackStat", "dG", "dSASA",
    "dG/dSASA", "Interface_SASA_%", "Interface_Hydrophobicity",
    "n_InterfaceResidues", "n_InterfaceHbonds", "InterfaceHbondsPercentage",
    "n_InterfaceUnsatHbonds", "InterfaceUnsatHbondsPercentage",
    "Interface_Helix%", "Interface_BetaSheet%", "Interface_Loop%",
    "Binder_Helix%", "Binder_BetaSheet%", "Binder_Loop%", "InterfaceAAs",
    "Hotspot_RMSD", "Target_RMSD", "Binder_pLDDT", "Binder_pTM", "Binder_pAE",
    "Binder_RMSD",
]

_BASE_DESIGN_LABELS = [
    "Design", "Protocol", "Length", "Seed", "Helicity", "Target_Hotspot",
    "Sequence", "InterfaceResidues", "MPNN_score", "MPNN_seq_recovery",
]


def trajectory_labels() -> list[str]:
    return list(TRAJECTORY_LABELS)


def design_labels() -> list[str]:
    labels = list(_BASE_DESIGN_LABELS)
    for label in CORE_LABELS:
        labels += [f"Average_{label}"] + [f"{i}_{label}" for i in range(1, 6)]
    labels += ["DesignTime", "Notes", "TargetSettings", "Filters", "AdvancedSettings"]
    return labels


def final_labels() -> list[str]:
    return ["Rank"] + design_labels()
