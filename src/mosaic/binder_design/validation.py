"""Stage 3: fold the designed sequences and score their confidence.

DdCraft runs this stage through ColabDesign; here it runs on a Mosaic
``StructurePredictionModel``.  The predictor is deliberately model-agnostic --
anything implementing ``target_only_features``/``binder_features``/``predict``
(AlphaFold2 today, Boltz / ESMFold / Protenix / OpenFold3 tomorrow) can be
dropped in, which is the whole point of moving the pipeline into Mosaic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import gemmi
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.binder_design.metrics import (
    ConfidenceMetrics,
    complex_metrics,
    monomer_metrics,
)
from mosaic.structure_prediction import TargetChain

__all__ = ["ComplexPredictor", "ModelPrediction", "MonomerPredictor", "clean_sequence"]

_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
_AA_INDEX = {aa: i for i, aa in enumerate(_AA_ORDER)}


def clean_sequence(sequence: str) -> str:
    """Strip chain separators and anything that is not a residue letter."""
    return re.sub("[^A-Z]", "", sequence.upper())


def _one_hot(sequence: str) -> jnp.ndarray:
    idx = np.array([_AA_INDEX[aa] for aa in sequence], dtype=np.int32)
    return jax.nn.one_hot(jnp.asarray(idx), len(_AA_ORDER))


@dataclass
class ModelPrediction:
    model_index: int
    metrics: ConfidenceMetrics
    structure: gemmi.Structure

    def write_pdb(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.structure.write_pdb(str(path))
        return path


def _chain_sequence(chain: gemmi.Chain) -> str:
    return gemmi.one_letter_code([residue.name for residue in chain]).upper()


class ComplexPredictor:
    """Predicts binder + target complexes with the target supplied as template.

    Features are built once and reused for every sequence, which is what makes
    validation cheap: only the binder one-hot changes between designs.
    """

    def __init__(
        self,
        model,
        *,
        target_pdb: str | Path,
        target_chains: list[str] | str,
        binder_length: int,
        recycling_steps: int = 3,
        model_indices: list[int] | None = None,
        sampling_steps: int | None = None,
        ipsae_cutoff: float | None = None,
        advanced_settings: dict | None = None,
    ):
        if isinstance(target_chains, str):
            target_chains = [c.strip() for c in target_chains.split(",") if c.strip()]

        structure = gemmi.read_structure(str(target_pdb))
        structure.setup_entities()
        structure.remove_ligands_and_waters()
        structure.remove_hydrogens()
        lookup = {chain.name: chain for chain in structure[0]}
        missing = [c for c in target_chains if c not in lookup]
        if missing:
            raise ValueError(f"{target_pdb}: target chains {missing} not found")

        self.model = model
        self.binder_length = binder_length
        self.recycling_steps = recycling_steps
        self.ipsae_cutoff = ipsae_cutoff
        self.sampling_steps = sampling_steps
        self.target_lengths = [len(_chain_sequence(lookup[c])) for c in target_chains]
        self.target_length = sum(self.target_lengths)

        target_specs = [
            TargetChain(
                sequence=_chain_sequence(lookup[c]),
                use_msa=False,
                template_chain=(
                    lookup[c] if model.supports_template_chains() else None
                ),
            )
            for c in target_chains
        ]
        self.target_specs = target_specs
        if model.prediction_features_depend_on_sequence():
            self.features, self.writer = None, None
        else:
            self.features, self.writer = model.binder_features(
                binder_length=binder_length, chains=target_specs
            )
        if advanced_settings and self.features is not None:
            from mosaic.binder_design.trajectory import _apply_template_masks

            self.features = _apply_template_masks(
                self.features, binder_length, advanced_settings, stage="predict"
            )
        self.model_indices = list(
            model.ensemble_members() if model_indices is None else model_indices
        )

    def _prediction_inputs(self, sequence: str):
        if self.model.prediction_features_depend_on_sequence():
            features, writer = self.model.target_only_features(
                [TargetChain(sequence=sequence, use_msa=False), *self.target_specs]
            )
            return None, features, writer
        return _one_hot(sequence), self.features, self.writer

    def predict(
        self,
        sequence: str,
        *,
        key,
        binder_mask: np.ndarray | None = None,
        target_mask: np.ndarray | None = None,
        model_indices: list[int] | None = None,
    ) -> list[ModelPrediction]:
        sequence = clean_sequence(sequence)
        if len(sequence) != self.binder_length:
            raise ValueError(
                f"sequence length {len(sequence)} != binder length {self.binder_length}"
            )
        pssm, features, writer = self._prediction_inputs(sequence)

        results = []
        for model_index in model_indices or self.model_indices:
            key, sub = jax.random.split(key)
            model_kwargs = self.model.member_kwargs(model_index)
            if self.sampling_steps is not None:
                model_kwargs["sampling_steps"] = self.sampling_steps
            prediction = self.model.predict(
                PSSM=pssm,
                features=features,
                writer=writer,
                recycling_steps=self.recycling_steps,
                key=sub,
                **model_kwargs,
            )
            results.append(
                ModelPrediction(
                    model_index=model_index,
                    metrics=complex_metrics(
                        prediction,
                        binder_length=self.binder_length,
                        binder_mask=binder_mask,
                        target_mask=target_mask,
                        ipsae_cutoff=self.ipsae_cutoff,
                    ),
                    structure=prediction.st,
                )
            )
        return results


class MonomerPredictor:
    """Predicts the binder on its own, with no target and no templates."""

    def __init__(
        self,
        model,
        *,
        binder_length: int,
        recycling_steps: int = 3,
        model_indices: list[int] | None = None,
        sampling_steps: int | None = None,
    ):
        self.model = model
        self.binder_length = binder_length
        self.recycling_steps = recycling_steps
        self.sampling_steps = sampling_steps
        if model.prediction_features_depend_on_sequence():
            self.features, self.writer = None, None
        else:
            self.features, self.writer = model.target_only_features(
                [TargetChain(sequence="G" * binder_length, use_msa=False)]
            )
        self.model_indices = list(
            model.ensemble_members() if model_indices is None else model_indices
        )

    def _prediction_inputs(self, sequence: str):
        if self.model.prediction_features_depend_on_sequence():
            features, writer = self.model.target_only_features(
                [TargetChain(sequence=sequence, use_msa=False)]
            )
            return None, features, writer
        return _one_hot(sequence), self.features, self.writer

    def predict(
        self, sequence: str, *, key, model_indices: list[int] | None = None
    ) -> list[ModelPrediction]:
        sequence = clean_sequence(sequence)
        pssm, features, writer = self._prediction_inputs(sequence)
        results = []
        for model_index in model_indices or self.model_indices:
            key, sub = jax.random.split(key)
            model_kwargs = self.model.member_kwargs(model_index)
            if self.sampling_steps is not None:
                model_kwargs["sampling_steps"] = self.sampling_steps
            prediction = self.model.predict(
                PSSM=pssm,
                features=features,
                writer=writer,
                recycling_steps=self.recycling_steps,
                key=sub,
                **model_kwargs,
            )
            results.append(
                ModelPrediction(
                    model_index=model_index,
                    metrics=monomer_metrics(prediction),
                    structure=prediction.st,
                )
            )
        return results
