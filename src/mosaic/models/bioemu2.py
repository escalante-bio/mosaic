"""BioEmu2 (via the ``biojemu2`` JAX port) behind mosaic's structure-prediction interface.

BioEmu2 is a generative model of structure ensembles without confidence heads: every
`StructureModelOutput` / `StructurePrediction` carries NaN `plddt`, `pae` and `iptm`.
Use it for coordinate- and distogram-based losses, and for sampling.

`mode` selects upstream's method conditioning and sampler preset: ``"structure"``
(structure prediction) or ``"equilibrium"`` (equilibrium ensembles).
`recycling_steps` counts trunk passes, one more than upstream's ``num_recycles``.

Sampling an ensemble: run the trunk once with `BioEmu2.denoiser`, then `jax.vmap`
`mosaic.sampling.sample` over keys::

    model = BioEmu2Preview(mode="equilibrium")
    features, system = model.target_only_features([TargetChain(sequence, use_msa=False)])
    denoise, atom_mask = model.denoiser(features=features)
    coords = jax.vmap(lambda k: sample(denoise, atom_mask, model.inference_sampler(), key=k))(keys)
    structures = [model.structure(system, x) for x in coords]
"""

from __future__ import annotations

import dataclasses
import string
import tempfile
from pathlib import Path

import biojemu2
import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

import mosaic.structure_model as sm
from mosaic.cache import cache_dir
from mosaic.losses.atom37 import scatter_atom37
from mosaic.losses.bioemu2 import (
    DISTOGRAM_BINS,
    FIRST_AMINO_ACID,
    BioEmu2FeaturePack,
    build_pack,
    set_binder_sequence,
)
from mosaic.losses.structure_prediction import StructureModelOutput
from mosaic.sampling import Denoise, Sampler
from mosaic.sampling.bioemu2 import bioemu2_denoiser, bioemu2_sampler
from mosaic.sampling.schedules import Schedule
from mosaic.structure_model import (
    AbstractStructureModel,
    Atom37,
    Distogram,
    TokenInfo,
    TrunkState,
)
from mosaic.structure_prediction import (
    PolymerType,
    StructurePrediction,
    TargetChain,
)

_CHECKPOINT = "bioemu2-preview"
_DESIGN_SAMPLING_STEPS = 20
# Upstream inference defaults (`num_recycles=3`).
_INFERENCE_SAMPLING_STEPS = 200
_INFERENCE_RECYCLING_STEPS = 4


def load_model(name: str = _CHECKPOINT) -> biojemu2.BioEmu2:
    """The converted checkpoint, converted from ``microsoft/bioemu2`` (and cached under
    mosaic's cache) on first use."""
    converted = cache_dir("bioemu2", name, "model")
    if not converted.with_suffix(".eqx").exists():
        from biojemu2.convert import convert_released

        converted.parent.mkdir(parents=True, exist_ok=True)
        biojemu2.save_model(convert_released(name), converted)
    return biojemu2.load_model(converted)


def _system(chains: list[TargetChain]):
    from bioemu2.data_interface import DNA, RNA, Protein, System

    entity = {PolymerType.PROTEIN: Protein, PolymerType.DNA: DNA, PolymerType.RNA: RNA}
    return System(
        entities=[
            entity[chain.polymer_type](sequence=chain.sequence, asym_ids=[chain_id])
            for chain_id, chain in zip(string.ascii_uppercase, chains)
        ]
    )


def _msa(chains: list[TargetChain]):
    """Unpaired MSAs: ColabFold (upstream's cached `get_msa`) for ``use_msa`` protein
    chains, the sequence alone otherwise.

    The paired MSAs hold only the query rows. Upstream needs them anyway: without
    paired MSAs a heteromer's unpaired MSAs are cut to the shallowest chain's depth
    (one row with a single-sequence binder), while with them every chain keeps its
    own rows."""
    from bioemu2.msa.alignment import MSA, get_msa

    def a3m(chain):
        if chain.use_msa:
            return get_msa(_system([chain])).a3m("A")
        return f">query\n{chain.sequence}\n"

    proteins = {
        chain_id: chain
        for chain_id, chain in zip(string.ascii_uppercase, chains)
        if chain.polymer_type == PolymerType.PROTEIN
    }
    return MSA(
        unpaired_a3m={chain_id: a3m(chain) for chain_id, chain in proteins.items()},
        paired_a3m={chain_id: f">query\n{chain.sequence}\n" for chain_id, chain in proteins.items()},
    )


class BioEmu2(AbstractStructureModel[BioEmu2FeaturePack, biojemu2.TrunkInit]):
    """BioEmu2. Construct with `BioEmu2Preview()`."""

    bioemu2: biojemu2.BioEmu2
    mode: str = eqx.field(static=True)  # "structure" or "equilibrium"
    design_sampling_steps: int = eqx.field(static=True)
    inference_sampling_steps: int = eqx.field(static=True)
    inference_recycling_steps: int = eqx.field(static=True)

    @property
    def design_passes(self) -> int:
        return 1

    @property
    def inference_passes(self) -> int:
        return self.inference_recycling_steps

    @property
    def backprop_all_passes(self) -> bool:
        return False

    def target_only_features(self, chains: list[TargetChain]):
        """Featurize the chains as given. Returns ``(pack, system)``; the upstream
        ``System`` is the writer for `structure` and `predict`."""
        from bioemu2.biomoa.utils.feature_converter import (
            convert_densebatch_to_dictionary,
        )
        from bioemu2.structure.batch import batch_collate
        from bioemu2.structure.preparation import prepare_structure

        if any(chain.template_chain is not None for chain in chains):
            raise NotImplementedError("BioEmu2 does not take templates")
        system = _system(chains)
        structure = prepare_structure(system, msa=_msa(chains), mode=self.mode, contacts=[])
        feats = convert_densebatch_to_dictionary(batch_collate([structure]))
        return build_pack({k: v.numpy() for k, v in feats.items()}), system

    def binder_features(self, binder_length: int, chains: list[TargetChain]):
        """Design pack: a single-sequence binder of unknown residues (chain 0) plus the
        targets; the PSSM replaces its sequence."""
        binder = TargetChain("X" * binder_length, use_msa=False)
        return self.target_only_features([binder, *chains])

    def design_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> Sampler:
        """Upstream's sampler for `mode` (schedule and step scale) without churn or
        random augmentation: deterministic given the initial noise."""
        return dataclasses.replace(
            self.inference_sampler(num_steps or self.design_sampling_steps, schedule=schedule),
            gamma_0=0.0,
            augment=False,
        )

    def inference_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> Sampler:
        return bioemu2_sampler(
            self.bioemu2,
            num_steps or self.inference_sampling_steps,
            mode=self.mode,
            schedule=schedule,
        )

    @eqx.filter_jit
    def denoiser(
        self,
        *,
        features: BioEmu2FeaturePack,
        PSSM: Float[Array, "N 20"] | None = None,
        recycling_steps: int | None = None,
    ) -> tuple[Denoise, Float[Array, " A"]]:
        """Run the trunk once: the denoiser and atom mask for `mosaic.sampling.sample`."""
        key = jax.random.key(0)  # the trunk is deterministic
        feats, inputs = sm.prepare(self, features, PSSM, key=key)
        state = sm.run_trunk(
            self, feats, inputs, num_passes=recycling_steps or self.inference_recycling_steps, key=key
        )
        return self.structure_denoiser(feats, inputs, state)

    @staticmethod
    def structure(writer, coords: Float[Array, "N 3"]) -> gemmi.Structure:
        """One sample as a gemmi structure, all atoms, with the featurized system
        (``writer``) as topology."""
        from bioemu2.trajectory_io import save_mdtraj_cif_frame
        from biojemu2.reference import to_trajectory

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.cif"
            save_mdtraj_cif_frame(to_trajectory(writer, np.asarray(coords)[None]), path)
            return gemmi.read_structure(str(path))

    # -- the structure-model contract -----------------------------------------

    def set_sequence(
        self, features: BioEmu2FeaturePack, sequence: Float[Array, "L 20"], *, key: PRNGKeyArray
    ) -> BioEmu2FeaturePack:
        del key
        return dataclasses.replace(features, features=set_binder_sequence(sequence, features.features))

    def embed(self, features: BioEmu2FeaturePack, *, key: PRNGKeyArray) -> biojemu2.TrunkInit:
        del key
        return self.bioemu2.embed_inputs(features.features)

    def init_state(
        self, features: BioEmu2FeaturePack, inputs: biojemu2.TrunkInit, *, key: PRNGKeyArray
    ) -> TrunkState:
        del features, key
        state = self.bioemu2.init_state(inputs)
        return TrunkState(pair=state.pair, single=state.single)

    def step(
        self,
        features: BioEmu2FeaturePack,
        inputs: biojemu2.TrunkInit,
        state: TrunkState,
        *,
        key: PRNGKeyArray,
    ) -> TrunkState:
        del key
        out = self.bioemu2.step(features.features, inputs, _biojemu2_state(state))
        return TrunkState(pair=out.pair, single=out.single)

    def structure_denoiser(
        self, features: BioEmu2FeaturePack, inputs: biojemu2.TrunkInit, state: TrunkState
    ) -> tuple[Denoise, Float[Array, " A"]]:
        trunk = self.bioemu2.trunk_embeddings(features.features, inputs, _biojemu2_state(state))
        return bioemu2_denoiser(self.bioemu2, features.features, trunk)

    def distogram(
        self, features: BioEmu2FeaturePack, inputs: biojemu2.TrunkInit, state: TrunkState
    ) -> Distogram:
        del features, inputs
        return Distogram(
            logits=self.bioemu2.distogram_module(state.pair),
            bins=jnp.array(DISTOGRAM_BINS, dtype=jnp.float32),
        )

    def confidence(self, features, inputs, state, coords) -> None:
        """No confidence heads: `plddt` / `pae` come out NaN."""
        del features, inputs, state, coords
        return None

    def tokens(self, features: BioEmu2FeaturePack) -> TokenInfo:
        feats = features.features
        return TokenInfo(
            sequence=feats.tokens.res_type[:, FIRST_AMINO_ACID : FIRST_AMINO_ACID + 20],
            asym_id=feats.trunk.asym_id,
            residue_index=feats.trunk.residue_index,
            mask=feats.trunk.token_valid.astype(bool),
        )

    def atom37(self, features: BioEmu2FeaturePack, coords: Float[Array, "N 3"]) -> Atom37:
        feats = features.features
        atom37_coords, atom37_mask = scatter_atom37(
            atom_coords=coords,
            atom_to_token=feats.atoms.atom_to_token,
            atom37_idx=features.atom37_idx,
            n_token=feats.tokens.res_type.shape[0],
        )
        return Atom37(coords=atom37_coords, mask=atom37_mask)

    def to_structure(
        self,
        features: BioEmu2FeaturePack,
        writer,
        output: StructureModelOutput,
        sequence: Float[Array, "L 20"] | None,
    ) -> gemmi.Structure:
        """With a designed sequence, the atom37 view (protein chains; NaN B-factors),
        since the featurized system no longer matches it; without, all atoms from
        ``writer``."""
        del features
        if sequence is not None:
            return output.to_structure()
        return self.structure(writer, output.structure_coordinates)

    # -- public API (forwards to `mosaic.structure_model`) ----------------------

    def build_loss(
        self,
        *,
        loss,
        features: BioEmu2FeaturePack,
        recycling_steps: int = 1,
        sampler: Sampler | None = None,
        backprop_all_passes: bool | None = None,
    ) -> sm.MultiSampleLoss:
        return self.build_multisample_loss(
            loss=loss,
            features=features,
            recycling_steps=recycling_steps,
            num_samples=1,
            sampler=sampler,
            backprop_all_passes=backprop_all_passes,
        )

    def build_multisample_loss(
        self,
        *,
        loss,
        features: BioEmu2FeaturePack,
        recycling_steps: int = 1,
        num_samples: int = 4,
        reduction=jnp.mean,
        sampler: Sampler | None = None,
        backprop_all_passes: bool | None = None,
    ) -> sm.MultiSampleLoss:
        return sm.build_loss(
            self,
            loss=loss,
            features=features,
            num_passes=recycling_steps,
            num_samples=num_samples,
            reduction=reduction,
            sampler=sampler,
            backprop_all_passes=backprop_all_passes,
        )

    def model_output(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: BioEmu2FeaturePack,
        recycling_steps: int = 1,
        sampler: Sampler | None = None,
        key,
    ) -> StructureModelOutput:
        return sm.model_output(
            self, features, sequence=PSSM, num_passes=recycling_steps, sampler=sampler, key=key
        )

    def predict(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: BioEmu2FeaturePack,
        writer=None,
        recycling_steps: int | None = None,
        sampler: Sampler | None = None,
        key,
    ) -> StructurePrediction:
        """One sample with upstream's inference settings (`iptm` is NaN); see
        `to_structure` for how it is written."""
        return sm.predict(
            self,
            features,
            writer,
            sequence=PSSM,
            num_passes=recycling_steps,
            sampler=sampler,
            key=key,
        )


def _biojemu2_state(state: TrunkState) -> biojemu2.TrunkState:
    return biojemu2.TrunkState(single=state.single, pair=state.pair)


def BioEmu2Preview(mode: str = "structure") -> BioEmu2:
    return BioEmu2(
        bioemu2=load_model(_CHECKPOINT),
        mode=mode,
        design_sampling_steps=_DESIGN_SAMPLING_STEPS,
        inference_sampling_steps=_INFERENCE_SAMPLING_STEPS,
        inference_recycling_steps=_INFERENCE_RECYCLING_STEPS,
    )
