#!/usr/bin/env python3
"""Stage 1: structure-model hallucination trajectories.

Generates a binder backbone against a target by optimising a composite Mosaic
objective, then folding the result. Runs standalone as a CLI (one job per
process, which is how the DdCraft bridge drives it) or via ``run_job``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

POSITION_RE = re.compile(r"^([A-Za-z]?)(-?\d+)(?:-(-?\d+))?$")


def _normalized_symmetric_pae(pae: Any, array_module: Any) -> Any:
    pae = array_module.asarray(pae)
    return (pae + array_module.swapaxes(pae, -1, -2)) / 62.0


def load_job(path: Path) -> dict[str, Any]:
    job = json.loads(path.read_text())
    validate_job(job)
    return job


def validate_job(job: dict[str, Any]) -> None:
    required = {
        "design_name",
        "seed",
        "length",
        "starting_pdb",
        "target_chains",
        "target_settings",
        "advanced_settings",
        "output_pdb",
    }
    missing = sorted(required - set(job))
    if missing:
        raise ValueError(f"Mosaic job is missing required fields: {', '.join(missing)}")
    if int(job["length"]) < 1:
        raise ValueError("Mosaic binder length must be positive")
    if not Path(job["starting_pdb"]).is_file():
        raise FileNotFoundError(f"Starting PDB not found: {job['starting_pdb']}")
    if not job["target_chains"]:
        raise ValueError("At least one target chain is required")
    if job["advanced_settings"].get("design_algorithm", "4stage") != "4stage":
        raise ValueError(
            "The Mosaic backend currently supports DdCraft's 4stage design algorithm"
        )
    if job["advanced_settings"].get("optimise_beta", False):
        raise ValueError(
            "optimise_beta is not implemented in the Mosaic backend; the native engine "
            "extends the soft/temporary schedules and changes recycling when a beta "
            "sheeted trajectory is detected. Set optimise_beta to false to use --engine mosaic."
        )
    from mosaic.binder_design.components import model_selection

    selection = model_selection(job["advanced_settings"], "hallucination")
    if (
        selection.name == "af2"
        and not job.get("af_params_dir")
        and not job["advanced_settings"].get("af_params_dir")
        and not selection.options.get("data_dir")
        and not selection.options.get("params_dir")
    ):
        raise ValueError("AF2 hallucination requires af_params_dir")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2) + "\n")
    temporary_path.replace(path)


def _chain_sequence(chain: Any) -> str:
    import gemmi

    sequence = gemmi.one_letter_code([residue.name for residue in chain])
    if not sequence or "X" in sequence:
        raise ValueError(
            f"Chain {chain.name} contains missing or unsupported protein residues"
        )
    return sequence


def _position_index_map(chain: Any) -> dict[int, int]:
    return {int(residue.seqid.num): index for index, residue in enumerate(chain)}


def _expand_positions(
    specification: Any,
    default_chain: str,
    chain_maps: dict[str, dict[int, int]],
    chain_offsets: dict[str, int] | None = None,
) -> list[int]:
    if specification in (None, "", []):
        return []
    if isinstance(specification, (list, tuple)):
        tokens = [str(item).strip() for item in specification if str(item).strip()]
    else:
        tokens = [token.strip() for token in str(specification).split(",") if token.strip()]

    positions: list[int] = []
    for token in tokens:
        match = POSITION_RE.match(token)
        if not match:
            raise ValueError(f"Invalid residue position '{token}'")
        chain_id = match.group(1) or default_chain
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) is not None else start
        if chain_id not in chain_maps:
            raise ValueError(f"Unknown chain '{chain_id}' in residue position '{token}'")
        step = 1 if end >= start else -1
        for residue_number in range(start, end + step, step):
            if residue_number not in chain_maps[chain_id]:
                raise ValueError(
                    f"Residue {chain_id}{residue_number} is missing from the starting PDB"
                )
            index = chain_maps[chain_id][residue_number]
            if chain_offsets is not None:
                index += chain_offsets[chain_id]
            positions.append(index)
    return sorted(set(positions))


def _pseudo_beta_coordinates(chain: Any) -> tuple[list[list[float]], list[bool]]:
    coordinates: list[list[float]] = []
    mask: list[bool] = []
    for residue in chain:
        atom = residue.find_atom("CB", "\0")
        if atom is None:
            atom = residue.find_atom("CA", "\0")
        if atom is None:
            coordinates.append([0.0, 0.0, 0.0])
            mask.append(False)
        else:
            coordinates.append([atom.pos.x, atom.pos.y, atom.pos.z])
            mask.append(True)
    return coordinates, mask


def _align_stage_trajectory(entry_logits, trajectory):
    """Pair each recorded loss with the sequence that produced it.

    ``colabdesign_stage`` applies the gradient update before calling
    ``trajectory_fn``, so every entry holds a loss measured on the pre-update
    sequence alongside the post-update sequence. Shifting the sequences by one
    step recovers the iterate that actually produced each loss, which is what
    ColabDesign's ``save_best`` records.
    """
    aligned = []
    previous_logits = entry_logits
    for step in trajectory:
        aligned.append(
            {
                "loss": step["loss"],
                "plddt": step["plddt"],
                "logits": previous_logits,
            }
        )
        previous_logits = step["logits"]
    return aligned


def _apply_template_masks(
    features: dict[str, Any],
    binder_length: int,
    advanced: dict[str, Any],
    stage: str = "design",
) -> dict[str, Any]:
    """Mask the template the way ColabDesign does.

    ``stage`` selects which pair of settings to read: the trajectory stage uses
    ``rm_template_{seq,sc}_design`` and validation uses the ``_predict`` pair.
    """
    import jax.numpy as jnp

    from mosaic.alphafold.common import residue_constants

    required = {"template_aatype", "template_all_atom_mask"}
    if not required <= set(features):
        requested = any(
            advanced.get(f"rm_template_{kind}_{stage}", False)
            for kind in ("seq", "sc")
        )
        if requested:
            warnings.warn(
                f"Template seq/sidechain masks are AF2-specific and are not "
                f"available for this model's {stage} feature pack; the model's "
                "native template representation is being used unchanged.",
                stacklevel=2,
            )
        return features

    features = dict(features)
    binder_slice = slice(0, binder_length)
    target_slice = slice(binder_length, None)
    if advanced.get(f"rm_template_seq_{stage}", False):
        template_aatype = jnp.asarray(features["template_aatype"])
        features["template_aatype"] = template_aatype.at[:, target_slice].set(20)
    backbone_atoms = {
        residue_constants.atom_order[name] for name in ("N", "CA", "C", "O")
    }
    template_mask = jnp.asarray(features["template_all_atom_mask"])
    atom_mask = jnp.zeros(template_mask.shape[-1], dtype=template_mask.dtype)
    atom_mask = atom_mask.at[jnp.array(sorted(backbone_atoms))].set(1)
    template_mask = template_mask.at[:, binder_slice].multiply(atom_mask)
    if advanced.get(f"rm_template_sc_{stage}", False):
        features["template_all_atom_mask"] = template_mask.at[:, target_slice].multiply(
            atom_mask
        )
    else:
        features["template_all_atom_mask"] = template_mask
    return features


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import equinox as eqx
    import gemmi
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jaxtyping import Array, Float, Int

    import mosaic.losses.structure_prediction as sp
    from mosaic.alphafold.common import residue_constants
    from mosaic.binder_design.components import (
        create_structure_model,
        model_selection,
    )
    from mosaic.common import TOKENS, LossTerm, tokenize
    from mosaic.models.af2 import af2_atom_positions
    from mosaic.optimizers import colabdesign_stage
    from mosaic.structure_prediction import TargetChain

    # ColabDesign derives two different bin conventions from AF2's 64-bin
    # distogram, and Mosaic's AF2_DISTOGRAM_BINS (a 64-point linspace) matches
    # neither:
    #   get_dgram_loss uses the 63 bin *edges* and assigns a distance by
    #     interval membership, (d^2 > edges^2).sum(-1);
    #   get_dgram_bins prepends a 0 to those edges and thresholds the result
    #     with `bins < cutoff` for the contact and helix losses.
    # Using the linspace as if it were bin centres shifts assignments by up to
    # one bin, so reproduce both conventions explicitly.
    DGRAM_BIN_EDGES = jnp.linspace(2.3125, 21.6875, 63)
    DGRAM_CONTACT_BINS = jnp.append(0.0, DGRAM_BIN_EDGES)

    class SequenceConstraintLoss(LossTerm):
        loss: Any
        wildtype: Float[Array, "N 20"]
        variable_positions: Int[Array, " M"]
        allowed_indices: Int[Array, " K"]

        def sequence(self, variable_sequence: Float[Array, "M K"]):
            expanded = jnp.zeros((variable_sequence.shape[0], len(TOKENS)))
            expanded = expanded.at[:, self.allowed_indices].set(variable_sequence)
            return self.wildtype.at[self.variable_positions].set(expanded)

        def __call__(self, variable_sequence: Float[Array, "M K"], *, key):
            return self.loss(self.sequence(variable_sequence), key=key)

    class DdCraftPLDDTLoss(LossTerm):
        binder_mask: Float[Array, " B"]

        def __call__(self, sequence, output, key):
            losses = 1 - output.plddt[: sequence.shape[0]]
            value = (losses * self.binder_mask).sum() / (
                self.binder_mask.sum() + 1e-8
            )
            return value, {"plddt": 1 - value}

    class BinderPLDDTMetric(LossTerm):
        """Unweighted binder pLDDT used for DdCraft's stage continuation gates."""

        def __call__(self, sequence, output, key):
            value = output.plddt[: sequence.shape[0]].mean()
            return 0.0 * value, {"binder_plddt": value}

    class DdCraftIPTMLoss(LossTerm):
        def __call__(self, sequence, output, key):
            iptm = -sp.IPTMLoss()(sequence, output, key=key)[0]
            return 1 - iptm, {"i_ptm": iptm}

    class DdCraftBinderPAE(LossTerm):
        binder_mask: Float[Array, " B"]

        def __call__(self, sequence, output, key):
            binder_length = sequence.shape[0]
            pae = _normalized_symmetric_pae(output.pae, jnp)[:binder_length]
            pair_mask = self.binder_mask[:, None] * jnp.ones_like(pae)
            value = (pae * pair_mask).sum() / (pair_mask.sum() + 1e-8)
            return value, {"pae": value}

    class DdCraftInterfacePAE(LossTerm):
        binder_mask: Float[Array, " B"]
        target_mask: Float[Array, " T"]

        def __call__(self, sequence, output, key):
            binder_length = sequence.shape[0]
            pae = _normalized_symmetric_pae(output.pae, jnp)[
                :binder_length, binder_length:
            ]
            pair_mask = self.binder_mask[:, None] * self.target_mask[None, :]
            value = (pae * pair_mask).sum() / (pair_mask.sum() + 1e-8)
            return value, {"i_pae": value}

    class DdCraftWithinBinderContact(LossTerm):
        binder_mask: Float[Array, " B"]
        contact_distance: float
        af2_convention: bool = eqx.field(static=True)
        sequence_separation: int = eqx.field(static=True)
        num_contacts: int = eqx.field(static=True)

        def __call__(self, sequence, output, key):
            binder_length = sequence.shape[0]
            log_contact = sp.contact_cross_entropy(
                output.distogram_logits[:binder_length, :binder_length],
                self.contact_distance,
                DGRAM_CONTACT_BINS if self.af2_convention else output.distogram_bins,
            )
            positions = jnp.arange(binder_length)
            pair_mask = (
                jnp.abs(positions[:, None] - positions[None, :])
                >= self.sequence_separation
            )
            pair_mask &= self.binder_mask[:, None].astype(bool)
            pair_mask &= self.binder_mask[None, :].astype(bool)
            sorted_log_contact = jnp.sort(
                jnp.where(pair_mask, log_contact, -jnp.inf),
                descending=True,
                axis=-1,
            )[:, : self.num_contacts]
            valid = jnp.isfinite(sorted_log_contact)
            per_position = jnp.where(valid, sorted_log_contact, 0).sum(-1) / (
                valid.sum(-1) + 1e-8
            )
            value = -(
                (per_position * self.binder_mask).sum()
                / (self.binder_mask.sum() + 1e-8)
            )
            return value, {"intra_contact": -value}

    class DdCraftBinderTargetContact(LossTerm):
        binder_mask: Float[Array, " B"]
        target_mask: Float[Array, " T"]
        contact_distance: float
        af2_convention: bool = eqx.field(static=True)
        num_contacts: int = eqx.field(static=True)
        target_hotspot_mode: bool = eqx.field(static=True)

        def __call__(self, sequence, output, key):
            binder_length = sequence.shape[0]
            log_contact = sp.contact_cross_entropy(
                output.distogram_logits[:binder_length, binder_length:],
                self.contact_distance,
                DGRAM_CONTACT_BINS if self.af2_convention else output.distogram_bins,
            )
            if self.target_hotspot_mode:
                log_contact = log_contact.T
                row_mask = self.target_mask
                column_mask = self.binder_mask
            else:
                row_mask = self.binder_mask
                column_mask = self.target_mask
            pair_mask = row_mask[:, None].astype(bool) & column_mask[None, :].astype(
                bool
            )
            sorted_log_contact = jnp.sort(
                jnp.where(pair_mask, log_contact, -jnp.inf),
                descending=True,
                axis=-1,
            )[:, : self.num_contacts]
            valid = jnp.isfinite(sorted_log_contact)
            per_position = jnp.where(valid, sorted_log_contact, 0).sum(-1) / (
                valid.sum(-1) + 1e-8
            )
            value = (per_position * row_mask).sum() / (row_mask.sum() + 1e-8)
            return -value, {"target_contact": value}

    class TemplateDistogramCrossEntropy(LossTerm):
        reference_distances: Float[Array, "N M"]
        pair_mask: Float[Array, "N M"]
        region: str = eqx.field(static=True)
        name: str = eqx.field(static=True)
        af2_convention: bool = eqx.field(static=True)

        def __call__(self, sequence, output, key):
            binder_length = sequence.shape[0]
            if self.region == "binder":
                logits = output.distogram_logits[:binder_length, :binder_length]
            elif self.region == "target":
                logits = output.distogram_logits[binder_length:, binder_length:]
            else:
                logits = output.distogram_logits[:binder_length, binder_length:]
            if self.af2_convention:
                bin_index = (
                    jnp.square(self.reference_distances)[..., None]
                    > jnp.square(DGRAM_BIN_EDGES)
                ).sum(-1)
            else:
                bin_index = jnp.abs(
                    self.reference_distances[..., None] - output.distogram_bins
                ).argmin(-1)
            nll = -jnp.take_along_axis(
                jax.nn.log_softmax(logits, axis=-1),
                bin_index[..., None],
                axis=-1,
            )[..., 0]
            if self.region == "interface":
                reverse_logits = output.distogram_logits[
                    binder_length:, :binder_length
                ].swapaxes(0, 1)
                reverse_nll = -jnp.take_along_axis(
                    jax.nn.log_softmax(reverse_logits, axis=-1),
                    bin_index[..., None],
                    axis=-1,
                )[..., 0]
                nll = (nll + reverse_nll) / 2
            value = (nll * self.pair_mask).sum() / (self.pair_mask.sum() + 1e-8)
            return value, {self.name: value}

    class FixedModelLoss(LossTerm):
        model: Any
        features: Any
        loss: Any
        recycling_steps: int = eqx.field(static=True)
        sampling_steps: int | None = eqx.field(static=True)
        member: int = eqx.field(static=True)
        use_dropout: bool = eqx.field(static=True)

        def __call__(self, sequence, *, key):
            model_kwargs = self.model.design_member_kwargs(
                self.member, use_dropout=self.use_dropout
            )
            if self.sampling_steps is not None:
                model_kwargs["sampling_steps"] = self.sampling_steps
            output = self.model.model_output(
                PSSM=sequence,
                features=self.features,
                recycling_steps=self.recycling_steps,
                key=key,
                **model_kwargs,
            )
            value, auxiliary = self.loss(sequence, output=output, key=key)
            return value, {
                "ddcraft": auxiliary,
                "ddcraft/model_idx": jnp.asarray(self.member),
                "ddcraft/loss": value,
            }

    class SampledAlphaFoldLoss(LossTerm):
        model: Any
        features: Any
        loss: Any
        member_indices: Int[Array, " M"]
        stack_indices: Int[Array, " M"]
        recycling_steps: int = eqx.field(static=True)
        use_dropout: bool = eqx.field(static=True)

        def __call__(self, sequence, *, key):
            sample_key, model_key = jax.random.split(key)
            choice = jax.random.randint(
                sample_key, (), 0, self.member_indices.shape[0]
            )
            member_index = self.member_indices[choice]
            stack_index = self.stack_indices[choice]
            output = self.model.model_output(
                PSSM=sequence,
                features=self.features,
                recycling_steps=self.recycling_steps,
                model_idx=stack_index,
                use_dropout=self.use_dropout,
                key=model_key,
            )
            value, auxiliary = self.loss(sequence, output=output, key=model_key)
            return value, {
                "ddcraft": auxiliary,
                "ddcraft/model_idx": member_index,
                "ddcraft/loss": value,
            }

    class TerminiDistanceLoss(LossTerm):
        threshold: float = 7.0

        def __call__(self, sequence, output, key):
            binder_length = sequence.shape[0]
            ca = output.atom37_coords[:binder_length, residue_constants.atom_order["CA"]]
            distance = jnp.linalg.norm(ca[0] - ca[-1])
            value = jax.nn.relu(jax.nn.elu(distance - self.threshold))
            return value, {"termini_distance": distance}

    class DdCraftRadiusOfGyration(LossTerm):
        def __call__(self, sequence, output, key):
            binder_length = sequence.shape[0]
            ca = output.atom37_coords[
                :binder_length, residue_constants.atom_order["CA"]
            ]
            radius = jnp.sqrt(
                jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8
            )
            threshold = 2.38 * binder_length**0.365
            value = jax.nn.elu(radius - threshold)
            return value, {"radius_of_gyration": radius}

    class DdCraftHelixLoss(LossTerm):
        af2_convention: bool = eqx.field(static=True)

        def __call__(self, sequence, output, key):
            binder_length = sequence.shape[0]
            log_contact = sp.contact_log_probability(
                output.distogram_logits[:binder_length, :binder_length],
                6.0,
                bins=(
                    DGRAM_CONTACT_BINS
                    if self.af2_convention
                    else output.distogram_bins
                ),
            )
            value = -jnp.diagonal(log_contact, 3).mean()
            return value, {"helix": value}

    class TemplateSidechainLoss(LossTerm):
        positions: Int[Array, " P"]
        template_coords: Float[Array, "P 37 3"]
        template_mask: Float[Array, "P 37"]
        mode: str = eqx.field(static=True)

        def _local_coordinates(self, coords):
            n = coords[:, residue_constants.atom_order["N"]]
            ca = coords[:, residue_constants.atom_order["CA"]]
            c = coords[:, residue_constants.atom_order["C"]]
            x = c - ca
            x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
            y0 = n - ca
            y = y0 - (y0 * x).sum(-1, keepdims=True) * x
            y = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)
            z = jnp.cross(x, y)
            basis = jnp.stack((x, y, z), axis=-1)
            centered = coords - ca[:, None, :]
            return jnp.einsum("paj,pjk->pak", centered, basis)

        def __call__(self, sequence, output, key):
            predicted = output.atom37_coords[self.positions]
            difference = predicted - self.template_coords
            squared = (difference**2).sum(-1)
            rmsd = jnp.sqrt(
                (squared * self.template_mask).sum()
                / (self.template_mask.sum() + 1e-8)
            )
            if self.mode == "rmsd":
                return rmsd, {"template_sc_rmsd": rmsd}
            predicted_local = self._local_coordinates(predicted)
            template_local = self._local_coordinates(self.template_coords)
            fape_distance = jnp.sqrt(
                ((predicted_local - template_local) ** 2).sum(-1) + 1e-8
            )
            fape = (
                jnp.minimum(fape_distance, 10.0) * self.template_mask
            ).sum() / (self.template_mask.sum() + 1e-8)
            if self.mode == "fape":
                return fape, {"template_sc_fape": fape}
            return rmsd + fape, {
                "template_sc_rmsd": rmsd,
                "template_sc_fape": fape,
            }

    target = job["target_settings"]
    advanced = job["advanced_settings"]
    design_selection = model_selection(advanced, "hallucination")
    model_settings = dict(advanced)
    if job.get("af_params_dir") and not model_settings.get("af_params_dir"):
        model_settings["af_params_dir"] = job["af_params_dir"]
    design_model = create_structure_model(design_selection, model_settings)
    uses_af2_distogram = design_selection.name == "af2"
    structure = gemmi.read_structure(job["starting_pdb"])
    if len(structure) == 0:
        raise ValueError(f"No model found in {job['starting_pdb']}")
    model = structure[0]
    chain_lookup = {chain.name: chain for chain in model}
    target_chain_ids = list(job["target_chains"])
    binder_chain_id = str(job.get("binder_chain", "")).strip()

    for chain_id in target_chain_ids:
        if chain_id not in chain_lookup:
            raise ValueError(f"Target chain '{chain_id}' is missing from starting PDB")

    structure_conditioned = bool(binder_chain_id)
    binder_length = int(job["length"])
    if structure_conditioned:
        if binder_chain_id not in chain_lookup:
            raise ValueError(
                f"Binder chain '{binder_chain_id}' is missing from starting PDB"
            )
        binder_chain = chain_lookup[binder_chain_id]
        binder_sequence = _chain_sequence(binder_chain)
        if len(binder_sequence) != binder_length:
            raise ValueError(
                f"Configured binder length {binder_length} does not match "
                f"template chain length {len(binder_sequence)}"
            )
    else:
        binder_chain = None
        binder_sequence = "A" * binder_length

    target_sequences = {
        chain_id: _chain_sequence(chain_lookup[chain_id])
        for chain_id in target_chain_ids
    }
    supports_templates = design_model.supports_template_chains()
    target_specs = [
        TargetChain(
            sequence=target_sequences[chain_id],
            use_msa=False,
            template_chain=chain_lookup[chain_id] if supports_templates else None,
        )
        for chain_id in target_chain_ids
    ]

    if structure_conditioned and supports_templates:
        binder_spec = TargetChain(
            sequence=binder_sequence,
            use_msa=False,
            template_chain=binder_chain,
        )
        features, writer = design_model.target_only_features(
            [binder_spec] + target_specs
        )
    else:
        if structure_conditioned and not supports_templates:
            warnings.warn(
                f"{design_selection.name} does not accept template chains; fixed "
                "binder residues will be sequence-constrained but its starting "
                "coordinates cannot condition hallucination.",
                stacklevel=2,
            )
        features, writer = design_model.binder_features(
            binder_length=binder_length,
            chains=target_specs,
        )
    features = _apply_template_masks(features, binder_length, advanced)

    binder_map = (
        _position_index_map(binder_chain)
        if structure_conditioned
        else {index: index - 1 for index in range(1, binder_length + 1)}
    )
    target_maps = {
        chain_id: _position_index_map(chain_lookup[chain_id])
        for chain_id in target_chain_ids
    }
    target_offsets: dict[str, int] = {}
    offset = 0
    for chain_id in target_chain_ids:
        target_offsets[chain_id] = offset
        offset += len(target_sequences[chain_id])

    epitope_indices = _expand_positions(
        target.get("target_hotspot_residues", ""),
        target_chain_ids[0],
        target_maps,
        target_offsets,
    )
    target_coldspot_indices = _expand_positions(
        target.get("target_coldspot_residues", ""),
        target_chain_ids[0],
        target_maps,
        target_offsets,
    )
    paratope_indices = _expand_positions(
        target.get("binder_hotspot_residues", ""),
        binder_chain_id or "B",
        {binder_chain_id or "B": binder_map},
    )
    binder_coldspot_indices = _expand_positions(
        target.get("binder_coldspot_residues", ""),
        binder_chain_id or "B",
        {binder_chain_id or "B": binder_map},
    )
    if binder_coldspot_indices:
        binder_active_indices = sorted(
            set(range(binder_length)) - set(binder_coldspot_indices)
        )
    elif paratope_indices:
        binder_active_indices = paratope_indices
    else:
        binder_active_indices = list(range(binder_length))
    target_length = sum(len(sequence) for sequence in target_sequences.values())
    target_active_indices = (
        epitope_indices if epitope_indices else list(range(target_length))
    )
    target_active_indices = sorted(
        set(target_active_indices) - set(target_coldspot_indices)
    )
    binder_loss_mask = jnp.zeros(binder_length, dtype=jnp.float32).at[
        jnp.asarray(binder_active_indices, dtype=jnp.int32)
    ].set(1)
    target_loss_mask = jnp.zeros(target_length, dtype=jnp.float32).at[
        jnp.asarray(target_active_indices, dtype=jnp.int32)
    ].set(1)

    terms: list[Any] = []

    def add_term(weight: Any, loss: Any) -> None:
        numeric_weight = float(weight or 0)
        if numeric_weight != 0:
            terms.append(numeric_weight * loss)

    add_term(
        advanced.get("weights_plddt", 0),
        DdCraftPLDDTLoss(binder_mask=binder_loss_mask),
    )
    add_term(
        advanced.get("weights_pae_intra", 0),
        DdCraftBinderPAE(binder_mask=binder_loss_mask),
    )
    add_term(
        advanced.get("weights_pae_inter", 0),
        DdCraftInterfacePAE(
            binder_mask=binder_loss_mask,
            target_mask=target_loss_mask,
        ),
    )
    add_term(
        advanced.get("weights_con_intra", 0),
        DdCraftWithinBinderContact(
            binder_mask=binder_loss_mask,
            contact_distance=float(advanced.get("intra_contact_distance", 14.0)),
            af2_convention=uses_af2_distogram,
            sequence_separation=9,
            num_contacts=max(1, int(advanced.get("intra_contact_number", 2))),
        ),
    )
    add_term(
        advanced.get("weights_con_inter", 0),
        DdCraftBinderTargetContact(
            binder_mask=binder_loss_mask,
            target_mask=target_loss_mask,
            contact_distance=float(advanced.get("inter_contact_distance", 6.0)),
            af2_convention=uses_af2_distogram,
            num_contacts=max(1, int(advanced.get("inter_contact_number", 3))),
            target_hotspot_mode=bool(epitope_indices),
        ),
    )
    if advanced.get("use_i_ptm_loss", False):
        add_term(advanced.get("weights_iptm", 0), DdCraftIPTMLoss())
    if advanced.get("use_rg_loss", False):
        add_term(
            advanced.get("weights_rg", 0),
            DdCraftRadiusOfGyration(),
        )
    add_term(
        job.get("helicity_value", 0),
        DdCraftHelixLoss(af2_convention=uses_af2_distogram),
    )
    if advanced.get("use_termini_distance_loss", False):
        add_term(
            advanced.get("weights_termini_loss", 0),
            TerminiDistanceLoss(),
        )

    if structure_conditioned:
        binder_coords, binder_coord_mask = _pseudo_beta_coordinates(binder_chain)
        binder_coords_array = jnp.asarray(binder_coords)
        binder_mask_array = jnp.asarray(binder_coord_mask, dtype=jnp.float32)
        target_coordinates = []
        target_coordinate_masks = []
        for chain_id in target_chain_ids:
            coordinates, coordinate_mask = _pseudo_beta_coordinates(
                chain_lookup[chain_id]
            )
            target_coordinates.extend(coordinates)
            target_coordinate_masks.extend(coordinate_mask)
        target_coords_array = jnp.asarray(target_coordinates)
        target_mask_array = jnp.asarray(target_coordinate_masks, dtype=jnp.float32)

        binder_distances = jnp.linalg.norm(
            binder_coords_array[:, None, :] - binder_coords_array[None, :, :],
            axis=-1,
        )
        binder_pair_mask = (
            binder_mask_array[:, None] * binder_mask_array[None, :]
        ) * (1 - jnp.eye(binder_length))
        add_term(
            advanced.get("weight_dgram_binder", 0),
            TemplateDistogramCrossEntropy(
                reference_distances=binder_distances,
                pair_mask=binder_pair_mask,
                region="binder",
                name="dgram_binder",
                af2_convention=uses_af2_distogram,
            ),
        )
        target_distances = jnp.linalg.norm(
            target_coords_array[:, None, :] - target_coords_array[None, :, :],
            axis=-1,
        )
        target_pair_mask = (
            target_mask_array[:, None] * target_mask_array[None, :]
        ) * (1 - jnp.eye(target_coords_array.shape[0]))
        add_term(
            advanced.get("weight_dgram_target", 0),
            TemplateDistogramCrossEntropy(
                reference_distances=target_distances,
                pair_mask=target_pair_mask,
                region="target",
                name="dgram_target",
                af2_convention=uses_af2_distogram,
            ),
        )
        interface_distances = jnp.linalg.norm(
            binder_coords_array[:, None, :] - target_coords_array[None, :, :],
            axis=-1,
        )
        interface_pair_mask = binder_mask_array[:, None] * target_mask_array[None, :]
        add_term(
            advanced.get("weight_dgram_interface", 0),
            TemplateDistogramCrossEntropy(
                reference_distances=interface_distances,
                pair_mask=interface_pair_mask,
                region="interface",
                name="dgram_interface",
                af2_convention=uses_af2_distogram,
            ),
        )

        if advanced.get("use_template_sidechain_loss", False):
            sidechain_positions = _expand_positions(
                target.get("template_sidechain_positions")
                or target.get("fix_pos", ""),
                binder_chain_id,
                {binder_chain_id: binder_map},
            )
            if not sidechain_positions:
                raise ValueError(
                    "Template sidechain loss requires template_sidechain_positions "
                    "or fix_pos"
                )
            template_coords, template_mask = af2_atom_positions(binder_chain)
            template_coords = jnp.asarray(template_coords[0])[sidechain_positions]
            template_mask = jnp.asarray(template_mask[0])[sidechain_positions]
            excluded = {
                str(atom).strip().upper()
                for atom in advanced.get(
                    "template_sidechain_atoms_exclude", ["N", "C", "O"]
                )
            }
            excluded_indices = [
                residue_constants.atom_order[name]
                for name in excluded
                if name in residue_constants.atom_order
            ]
            template_mask = template_mask.at[:, jnp.asarray(excluded_indices)].set(0)
            mode = str(
                advanced.get("template_sidechain_loss_mode", "rmsd")
            ).lower()
            if mode not in {"rmsd", "fape", "combined"}:
                raise ValueError(
                    "template_sidechain_loss_mode must be rmsd, fape, or combined"
                )
            sidechain_kwargs = {
                "positions": jnp.asarray(sidechain_positions, dtype=jnp.int32),
                "template_coords": template_coords,
                "template_mask": template_mask,
            }
            if mode in {"rmsd", "combined"}:
                add_term(
                    advanced.get("weights_template_sidechain", 0),
                    TemplateSidechainLoss(mode="rmsd", **sidechain_kwargs),
                )
            if mode in {"fape", "combined"}:
                add_term(
                    advanced.get("weights_template_sidechain_fape", 0),
                    TemplateSidechainLoss(mode="fape", **sidechain_kwargs),
                )

    if not terms:
        raise ValueError("Mosaic objective has no enabled loss terms")
    # Always evaluated (at zero weight) so the stage gates can read an unweighted
    # binder pLDDT regardless of which loss terms the configuration enables.
    terms.append(0.0 * BinderPLDDTMetric())
    objective = terms[0]
    for term in terms[1:]:
        objective = objective + term

    # ColabDesign runs num_recycles + 1 forward passes (af/design.py: `cycles =
    # (num_recycles + 1)`), whereas Mosaic's recycling scan length *is* the number of
    # passes and starts from a zeroed recycling state. Passing num_recycles_design
    # straight through therefore gave the Mosaic backend half of native's recycling,
    # which showed up as systematically lower stage-1 binder pLDDT.
    recycling_steps = int(advanced.get("num_recycles_design", 1)) + 1
    configured_members = design_selection.members
    if configured_members is not None:
        design_model_indices = list(configured_members)
    elif design_selection.name == "af2":
        configured_numbers = design_selection.options.get("model_numbers")
        if configured_numbers is not None:
            design_model_indices = [
                int(model_number) - 1 for model_number in configured_numbers
            ]
        else:
            design_model_indices = [
                int(model_index) for model_index in job.get("design_models", [0])
            ] or [0]
    else:
        design_model_indices = list(design_model.ensemble_members())

    def build_model_loss(use_dropout: bool):
        if (
            design_selection.name == "af2"
            and advanced.get("sample_models", True)
            and len(design_model_indices) > 1
        ):
            stack_indices = [
                design_model.member_kwargs(member)["model_idx"]
                for member in design_model_indices
            ]
            return SampledAlphaFoldLoss(
                model=design_model,
                features=features,
                loss=objective,
                member_indices=jnp.asarray(design_model_indices, dtype=jnp.int32),
                stack_indices=jnp.asarray(stack_indices, dtype=jnp.int32),
                recycling_steps=recycling_steps,
                use_dropout=use_dropout,
            )
        return FixedModelLoss(
            model=design_model,
            features=features,
            loss=objective,
            recycling_steps=recycling_steps,
            sampling_steps=design_selection.sampling_steps,
            member=design_model_indices[0],
            use_dropout=use_dropout,
        )

    gradient_model_loss = build_model_loss(use_dropout=True)
    deterministic_model_loss = build_model_loss(use_dropout=False)

    fixed_positions = (
        _expand_positions(
            target.get("fix_pos", ""),
            binder_chain_id,
            {binder_chain_id: binder_map},
        )
        if structure_conditioned
        else []
    )
    variable_positions = [
        index for index in range(binder_length) if index not in set(fixed_positions)
    ]
    if not variable_positions:
        raise ValueError("All binder positions are fixed; there is nothing to design")

    omitted = {
        amino_acid.strip().upper()
        for amino_acid in str(advanced.get("omit_AAs", "") or "").split(",")
        if amino_acid.strip()
    }
    allowed_indices = [
        index for index, amino_acid in enumerate(TOKENS) if amino_acid not in omitted
    ]
    if not allowed_indices:
        raise ValueError("omit_AAs excludes the entire amino-acid alphabet")

    wildtype = jax.nn.one_hot(tokenize(binder_sequence), len(TOKENS))
    constraint_kwargs = {
        "wildtype": wildtype,
        "variable_positions": jnp.asarray(variable_positions, dtype=jnp.int32),
        "allowed_indices": jnp.asarray(allowed_indices, dtype=jnp.int32),
    }
    gradient_constrained_loss = SequenceConstraintLoss(
        loss=gradient_model_loss,
        **constraint_kwargs,
    )
    deterministic_constrained_loss = SequenceConstraintLoss(
        loss=deterministic_model_loss,
        **constraint_kwargs,
    )

    seed = int(job["seed"])
    key = jax.random.key(seed)
    optimization_key, prediction_key = jax.random.split(key)
    logits = 0.01 * jax.random.normal(
        optimization_key,
        (len(variable_positions), len(allowed_indices)),
    )
    learning_rate = float(advanced.get("mosaic_learning_rate", 0.1))

    def find_auxiliary_metric(value, name):
        if isinstance(value, dict):
            if name in value:
                candidate = value[name]
                if hasattr(candidate, "shape") and candidate.shape == ():
                    return float(candidate)
                if isinstance(candidate, (int, float)):
                    return float(candidate)
            for nested in value.values():
                candidate = find_auxiliary_metric(nested, name)
                if candidate is not None:
                    return candidate
        elif isinstance(value, (list, tuple)):
            for nested in value:
                candidate = find_auxiliary_metric(nested, name)
                if candidate is not None:
                    return candidate
        return None

    def collect_stage_step(auxiliary, stage_logits):
        return {
            "loss": float(auxiliary["loss"]),
            "plddt": find_auxiliary_metric(auxiliary, "binder_plddt"),
            "logits": stage_logits,
        }

    stage_best_logits = None

    def run_stage(*, dropout: bool, **stage_kwargs):
        nonlocal logits, optimization_key, stage_best_logits
        optimization_key, stage_key = jax.random.split(optimization_key)
        entry_logits = logits
        logits, trajectory = colabdesign_stage(
            loss_function=(
                gradient_constrained_loss
                if dropout
                else deterministic_constrained_loss
            ),
            x=logits,
            lr=learning_rate,
            key=stage_key,
            trajectory_fn=collect_stage_step,
            **stage_kwargs,
        )
        # colabdesign_stage applies the gradient update *before* invoking
        # trajectory_fn, so each entry pairs a loss measured on the pre-update
        # sequence with the post-update sequence. Re-align them so that the
        # recorded logits are the ones that actually produced the loss.
        steps = _align_stage_trajectory(entry_logits, trajectory)
        best_step = min(steps, key=lambda step: step["loss"])
        if best_step["plddt"] is None:
            raise RuntimeError("Mosaic stage did not report binder pLDDT")
        # ColabDesign runs every stage with save_best=True and calls clear_best()
        # at each stage boundary, so the lowest-loss iterate of the most recent
        # stage is what save_pdb()/get_seqs() return. Optimisation itself is not
        # rewound, so `logits` deliberately keeps the last iterate.
        stage_best_logits = best_step["logits"]
        # DdCraft's get_best_plddt rounds before comparing against the 0.65 gate.
        return round(best_step["plddt"], 2)

    initial_logits_steps = min(50, int(advanced.get("soft_iterations", 100)))
    stage_plddt = run_stage(
        dropout=True,
        n_steps=initial_logits_steps,
        soft_start=0.0,
        soft_end=0.9,
        temp_start=1.0,
        temp_end=1.0,
        hard=False,
    )
    print(f"logits binder pLDDT: {stage_plddt:.4f}", flush=True)
    status = "ok"
    failure_stage = None
    if stage_plddt <= 0.65:
        status = "low_confidence"
        failure_stage = "logits"
    else:
        remaining_logits = max(
            0, int(advanced.get("soft_iterations", 100)) - initial_logits_steps
        )
        if remaining_logits:
            stage_plddt = run_stage(
                dropout=True,
                n_steps=remaining_logits,
                soft_start=0.9,
                soft_end=1.0,
                temp_start=1.0,
                temp_end=1.0,
                hard=False,
            )
            print(f"additional logits binder pLDDT: {stage_plddt:.4f}", flush=True)

        temporary_iterations = int(advanced.get("temporary_iterations", 50))
        if temporary_iterations:
            stage_plddt = run_stage(
                dropout=True,
                n_steps=temporary_iterations,
                soft_start=1.0,
                soft_end=1.0,
                temp_start=1.0,
                temp_end=1e-2,
                hard=False,
            )
        print(f"softmax binder pLDDT: {stage_plddt:.4f}", flush=True)
        if stage_plddt <= 0.65:
            status = "low_confidence"
            failure_stage = "softmax"
        else:
            hard_iterations = int(advanced.get("hard_iterations", 50))
            if hard_iterations:
                stage_plddt = run_stage(
                    dropout=False,
                    n_steps=hard_iterations,
                    soft_start=1.0,
                    soft_end=1.0,
                    temp_start=1e-2,
                    temp_end=1e-2,
                    hard=True,
                )
            print(f"one-hot binder pLDDT: {stage_plddt:.4f}", flush=True)
            if stage_plddt <= 0.65:
                status = "low_confidence"
                failure_stage = "one-hot"

    # The semigreedy walk continues from the last iterate, matching ColabDesign,
    # but the stage's best-loss iterate stays eligible for the final output.
    variable_ids = jax.nn.softmax(logits).argmax(-1)
    stage_best_ids = (
        jax.nn.softmax(stage_best_logits).argmax(-1)
        if stage_best_logits is not None
        else variable_ids
    )
    if status == "ok" and int(advanced.get("greedy_iterations", 0)) > 0:
        greedy_tries = max(
            1,
            math.ceil(
                binder_length
                * (float(advanced.get("greedy_percentage", 30)) / 100.0)
            ),
        )
        design_models = design_model_indices
        rng = np.random.default_rng(seed)

        def evaluate_discrete(ids, model_index, evaluation_key):
            pssm = deterministic_constrained_loss.sequence(
                jax.nn.one_hot(ids, len(allowed_indices))
            )
            prediction_kwargs = design_model.member_kwargs(model_index)
            if design_selection.sampling_steps is not None:
                prediction_kwargs["sampling_steps"] = design_selection.sampling_steps
            prediction = design_model.predict(
                PSSM=pssm,
                features=features,
                writer=writer,
                recycling_steps=recycling_steps,
                key=evaluation_key,
                **prediction_kwargs,
            )
            value, _ = objective(
                pssm,
                output=prediction.model_output,
                key=evaluation_key,
            )
            return float(value), prediction

        initial_model = design_models[0]
        optimization_key, evaluation_key = jax.random.split(optimization_key)
        best_greedy_score, current_prediction = evaluate_discrete(
            variable_ids,
            initial_model,
            evaluation_key,
        )
        best_variable_ids = variable_ids
        # ColabDesign does not clear_best() before the semigreedy stage, so the
        # preceding stage's best iterate competes with the greedy candidates.
        if not bool(jnp.array_equal(stage_best_ids, variable_ids)):
            optimization_key, stage_best_key = jax.random.split(optimization_key)
            stage_best_score, _ = evaluate_discrete(
                stage_best_ids,
                initial_model,
                stage_best_key,
            )
            if stage_best_score < best_greedy_score:
                best_greedy_score = stage_best_score
                best_variable_ids = stage_best_ids
        for iteration in range(int(advanced.get("greedy_iterations", 0))):
            if advanced.get("sample_models", True):
                model_index = int(rng.choice(design_models))
            else:
                model_index = design_models[0]
            binder_plddt = np.asarray(current_prediction.plddt)[:binder_length]
            position_weights = np.maximum(
                1.0 - binder_plddt[np.asarray(variable_positions)],
                0.0,
            )
            if not np.isfinite(position_weights).all() or position_weights.sum() <= 0:
                position_weights = np.ones(len(variable_positions))
            position_weights = position_weights / position_weights.sum()

            candidates = []
            optimization_key, iteration_key = jax.random.split(optimization_key)
            candidate_keys = jax.random.split(iteration_key, greedy_tries)
            for candidate_index in range(greedy_tries):
                candidate = np.asarray(variable_ids).copy()
                position = int(
                    rng.choice(len(variable_positions), p=position_weights)
                )
                probabilities = np.ones(len(allowed_indices), dtype=float)
                probabilities[int(candidate[position])] = 0
                probabilities /= probabilities.sum()
                candidate[position] = int(
                    rng.choice(len(allowed_indices), p=probabilities)
                )
                score, prediction = evaluate_discrete(
                    jnp.asarray(candidate),
                    model_index,
                    candidate_keys[candidate_index],
                )
                candidates.append((score, candidate, prediction))

            score, selected_ids, current_prediction = min(
                candidates, key=lambda candidate: candidate[0]
            )
            variable_ids = jnp.asarray(selected_ids)
            if score < best_greedy_score:
                best_greedy_score = score
                best_variable_ids = variable_ids
            print(
                f"semigreedy {iteration}: loss={score:.4f} "
                f"model={model_index} tries={greedy_tries}",
                flush=True,
            )
        variable_ids = best_variable_ids
    else:
        # No semigreedy stage ran, so the output is the last stage's best-loss
        # iterate, matching ColabDesign's save_pdb()/get_seqs() defaults.
        variable_ids = stage_best_ids

    variable_pssm = jax.nn.one_hot(variable_ids, len(allowed_indices))

    final_pssm = deterministic_constrained_loss.sequence(variable_pssm)
    candidate_models = design_model_indices
    best_prediction = None
    best_score = float("inf")
    best_model_index = candidate_models[0]
    for model_index in candidate_models:
        prediction_key, model_key = jax.random.split(prediction_key)
        prediction_kwargs = design_model.member_kwargs(model_index)
        if design_selection.sampling_steps is not None:
            prediction_kwargs["sampling_steps"] = design_selection.sampling_steps
        prediction = design_model.predict(
            PSSM=final_pssm,
            features=features,
            writer=writer,
            recycling_steps=recycling_steps,
            key=model_key,
            **prediction_kwargs,
        )
        value, _ = objective(
            final_pssm,
            output=prediction.model_output,
            key=model_key,
        )
        score = float(value)
        if score < best_score:
            best_score = score
            best_prediction = prediction
            best_model_index = model_index

    if best_prediction is None:
        raise RuntimeError("Mosaic final prediction did not produce a candidate")

    output_pdb = Path(job["output_pdb"])
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    best_prediction.st.write_pdb(str(output_pdb))

    model_output = best_prediction.model_output
    binder_metric_mask = np.zeros(binder_length, dtype=np.float32)
    binder_metric_mask[binder_active_indices] = 1
    target_metric_mask = np.zeros(target_length, dtype=np.float32)
    target_metric_mask[target_active_indices] = 1
    binder_plddt = np.asarray(best_prediction.plddt)[:binder_length]
    plddt = float(
        (binder_plddt * binder_metric_mask).sum()
        / (binder_metric_mask.sum() + 1e-8)
    )
    raw_pae = np.asarray(best_prediction.pae)
    pae = _normalized_symmetric_pae(raw_pae, np)
    pair_mask = jnp.ones(pae.shape, dtype=bool)
    ptm = float(
        sp.predicted_tm_score(
            logits=model_output.pae_logits,
            bin_centers=model_output.pae_bins,
            pair_mask=pair_mask,
        ).max()
    )
    # Mosaic gives every chain its own asym_id, so a multi-chain target would
    # count target-target pairs as interface. ColabDesign's binder mode uses
    # _lengths = [target_len, binder_len], i.e. exactly two groups, so score the
    # binder against the rest of the complex the way IPTMLoss already does.
    iptm_asym_id = jnp.concatenate(
        (
            jnp.zeros(binder_length),
            jnp.ones(pae.shape[0] - binder_length),
        )
    ).astype(jnp.int32)
    iptm = float(
        sp.predicted_tm_score(
            logits=model_output.pae_logits,
            bin_centers=model_output.pae_bins,
            pair_mask=iptm_asym_id[:, None] != iptm_asym_id[None, :],
        ).max()
    )
    ipsae_config = advanced.get("ipsae", {})
    ipsae = None
    if isinstance(ipsae_config, dict) and ipsae_config.get("enabled", False):
        ipsae_values = sp.interaction_prediction_score(
            logits=model_output.pae_logits,
            bin_centers=model_output.pae_bins,
            asym_id=iptm_asym_id,
            pae_cutoff=float(ipsae_config.get("cutoff", 10.0)),
        )
        ipsae = float(jnp.max(ipsae_values))

    sequence_ids = np.asarray(final_pssm).argmax(-1)
    sequence = "".join(TOKENS[index] for index in sequence_ids)
    return {
        "status": status,
        "failure_stage": failure_stage,
        "design_name": job["design_name"],
        "selected_model": best_model_index,
        "objective": best_score,
        "metrics": {
            "sequence": sequence,
            "plddt": plddt,
            "ptm": ptm,
            "i_ptm": iptm,
            "pae": float(
                (pae[:binder_length] * binder_metric_mask[:, None]).sum()
                / (
                    (
                        binder_metric_mask[:, None]
                        * np.ones_like(pae[:binder_length])
                    ).sum()
                    + 1e-8
                )
            ),
            "i_pae": float(
                (
                    pae[:binder_length, binder_length:]
                    * binder_metric_mask[:, None]
                    * target_metric_mask[None, :]
                ).sum()
                / (
                    (
                        binder_metric_mask[:, None]
                        * target_metric_mask[None, :]
                    ).sum()
                    + 1e-8
                )
            ),
            "ipsae": ipsae,
            "mosaic_objective": best_score,
        },
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the job without importing Mosaic/JAX",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        job = load_job(args.job)
        if args.validate_only:
            print("Mosaic job configuration is valid")
            return 0
        if args.result is None:
            raise ValueError("--result is required unless --validate-only is used")
        result = run_job(job)
        _write_json(args.result, result)
        print(
            f"Completed {result['design_name']} with status={result['status']} "
            f"pLDDT={result['metrics']['plddt']:.4f} "
            f"iPTM={result['metrics']['i_ptm']:.4f}"
        )
        return 0
    except Exception as exc:
        if args.result is not None:
            _write_json(
                args.result,
                {
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        print(f"Mosaic job failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
