#####################
#
#   Uniform interface for structure prediction models: generating features, building losses, and running structure prediction.

import dataclasses
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from equinox import AbstractVar
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree
from jfold.featurize import Template
from jfold.model import StructureModel
from jfold.sampling import (
    Denoise,
    SamplerSettings,
    StructureSampler,
    exponential_update,
    schedules,
)
from jfold.sampling.schedules import Schedule
from jfold.slots import TrunkInputs, TrunkState

from mosaic.common import LinearCombination, LossTerm
from mosaic.losses.structure_prediction import (
    PAE_BINS,
    IPTMLoss,
    StructureModelOutput,
    reduce_samples,
)


class PolymerType:
    PROTEIN = "PROTEIN"
    RNA = "RNA"
    DNA = "DNA"

def _a3m_query(text: str) -> str:
    """The first record of a3m `text`, without its insertions (lowercase letters)."""
    records, lines = 0, []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            records += 1
            if records == 2:
                break
        elif records == 1 and not line.startswith("#"):
            lines.append(line)
    return "".join(c for c in "".join(lines) if not c.islower())


@dataclass(frozen=True, eq=True, slots=True)
class TargetChain:
    sequence: str
    polymer_type: str = PolymerType.PROTEIN
    use_msa: bool = True
    template_chain: gemmi.Chain | Template | None = None
    # The chain's MSA as a3m text (query first), used instead of fetching one.
    msa: str | None = None

    def __post_init__(self):
        if self.msa is None:
            return
        if not self.use_msa:
            raise ValueError("TargetChain: `msa` given with `use_msa=False`")
        if _a3m_query(self.msa) != self.sequence:
            raise ValueError("TargetChain: the `msa`'s first record is not `sequence`")


class StructurePrediction(eqx.Module):
    st: gemmi.Structure
    plddt: Float[Array, " N"]
    pae: Float[Array, "N N"]
    iptm: float
    model_output: StructureModelOutput


class StructurePredictionModel(eqx.Module):
    @abstractmethod
    def target_only_features(
        self, chains: list[TargetChain], *, key: PRNGKeyArray
    ) -> tuple[PyTree, any]:
        """
        Generate model features and postprocessor for the target chains only.

        Args:
            chains: List of TargetChain objects representing the target chains.
            key: Randomness for featurization (e.g. reference-conformer poses); models
                whose featurization is deterministic ignore it.

        Returns:
            tuple of (PyTree, StructureWriter) containing the generated features and an object for turning a prediction into a gemmi.Structure.

        """

    @abstractmethod
    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ) -> tuple[PyTree, any]:
        """
        Generate model features and postprocessor for a binder of given length and the target chains.

        Args:
            binder_length: Length of the binder chain.
            chains: List of TargetChain objects representing the target chains.
            key: As for `target_only_features`.

        Returns:
            tuple of (PyTree, StructureWriter) containing the generated features and an object for turning a prediction into a gemmi.Structure.

        """


    @abstractmethod
    def predict(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: PyTree,
        writer: any,
        recycling_steps: int = 1,
        key,
    ) -> StructurePrediction:
        pass
       
    @abstractmethod
    def model_output(self, *, PSSM: Float[Array, "N 20"] | None = None,
        features: PyTree,
        recycling_steps: int = 1,
        key) -> StructureModelOutput:
        pass


    @abstractmethod
    def build_loss(self, *, loss: LossTerm | LinearCombination, features: PyTree,  recycling_steps: int = 1) -> LossTerm:
        pass


Writer = Any  # turns coordinates into a structure (model-specific)

# Diffusion steps of the default samplers.
DEFAULT_SAMPLING_STEPS = 8


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------


class TokenInfo(eqx.Module):
    """Per-token identity; `sequence` is in `common.TOKENS` order."""

    sequence: Float[Array, "T 20"]
    asym_id: Int[Array, " T"]
    residue_index: Int[Array, " T"]
    mask: Bool[Array, " T"]  # non-padding tokens


class Atom37(eqx.Module):
    """Coordinates in AlphaFold's atom37 layout."""

    coords: Float[Array, "T 37 3"]
    mask: Float[Array, "T 37"]


class Confidence(eqx.Module):
    plddt: Float[Array, " T"]
    pae_logits: Float[Array, "T T B"]
    pae_bins: Float[Array, " B"]

    @classmethod
    def unavailable(cls, num_tokens: int) -> "Confidence":
        """NaN pLDDT and PAE, for models without a confidence head."""
        return cls(
            plddt=jnp.full((num_tokens,), jnp.nan),
            pae_logits=jnp.full((num_tokens, num_tokens, len(PAE_BINS)), jnp.nan),
            pae_bins=jnp.array(PAE_BINS, dtype=jnp.float32),
        )


class Distogram(eqx.Module):
    logits: Float[Array, "T T B"]
    bins: Float[Array, " B"]


class Inputs(eqx.Module):
    """What `embed` returns: jfold's features (built once per sequence) and trunk inputs."""

    features: eqx.Module
    trunk: TrunkInputs


class Trunk[Features](eqx.Module):
    """What the heads read: the features, the embedded inputs and the last trunk state
    (jfold's `TrunkState`: single `s`, pair `z`)."""

    features: Features
    inputs: Inputs
    state: TrunkState


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def _float32(denoise, x, sigma):
    with jax.default_matmul_precision("float32"):
        return denoise(x, sigma)


def _step(model, features, inputs, state, key):
    return model.trunk(features, inputs, state, key=key)


class DiffusionStructurePredictionModel[Features](StructurePredictionModel):
    """A trunk + diffusion structure model on a jfold `StructureModel` (`model`), which
    runs the forward pass. A backend supplies what is model-specific: featurization of
    its `Features` (and the jfold features built from them), setting the binder sequence,
    tokens, atom37 and the structure writer; and, where its head differs, `token_plddt`."""

    model: AbstractVar[StructureModel]
    distogram_bins: ClassVar[np.ndarray]
    # Rematerialize each trunk pass on the backward pass (large trunks).
    remat_steps: ClassVar[bool] = False
    # `upstream_sampler`'s default step count, where it differs from the converted
    # model's (`model.sampling.num_steps`).
    upstream_sampling_steps: ClassVar[int | None] = None
    # Number of trunk passes (counting the first) for design and for inference.
    design_recycling_steps: AbstractVar[int]
    inference_recycling_steps: AbstractVar[int]
    # Backpropagate through every trunk pass, or only the last.
    backprop_all_passes: AbstractVar[bool]

    # -- shared logic -------------------------------------------------------

    def prepare(
        self, features: Features, sequence: Float[Array, "L 20"] | None, *, key: PRNGKeyArray
    ) -> tuple[Features, Inputs]:
        """Set the binder sequence (if given) and embed."""
        k_sequence, k_embed = jax.random.split(key)
        if sequence is not None:
            features = self.set_sequence(features, sequence, key=k_sequence)
        return features, self.embed(features, key=k_embed)

    def run_trunk(
        self,
        features: Features,
        inputs: Inputs,
        *,
        recycling_steps: int,
        key: PRNGKeyArray,
        initial_state: TrunkState | None = None,
    ) -> Trunk[Features]:
        """Run `recycling_steps` trunk passes from `initial_state` (default
        `init_state`; e.g. the `state` of an earlier `run_trunk` on the same
        tokens). Gradients flow through every pass if `backprop_all_passes`,
        otherwise through the last only."""
        if recycling_steps < 1:
            raise ValueError(f"recycling_steps must be at least 1, got {recycling_steps}")
        k_init, k_passes = jax.random.split(key)
        keys = jax.random.split(k_passes, recycling_steps)

        def one_pass(state, k):
            return self.step(inputs, state, key=k), None

        state = initial_state
        if state is None:
            state = self.init_state(inputs, key=k_init)
        if not self.backprop_all_passes:
            state = jax.lax.stop_gradient(jax.lax.scan(one_pass, state, keys[:-1])[0])
            keys = keys[-1:]
        state = jax.lax.scan(one_pass, state, keys)[0]
        return Trunk(features=features, inputs=inputs, state=state)

    def sample_structure(
        self, trunk: Trunk[Features], *, sampler: SamplerSettings, key: PRNGKeyArray
    ) -> StructureModelOutput:
        """Sample one structure from the trunk and evaluate the heads."""
        coords = self.structure_sampler(trunk, sampler).sample(key)
        tokens = self.tokens(trunk.features)
        distogram = self.distogram(trunk)
        confidence = self.confidence(trunk, coords)
        atoms = self.atom37(trunk.features, coords)
        pae = jnp.sum(jax.nn.softmax(confidence.pae_logits, axis=-1) * confidence.pae_bins, axis=-1)
        return StructureModelOutput(
            distogram_logits=distogram.logits,
            distogram_bins=distogram.bins,
            plddt=confidence.plddt,
            pae=pae,
            pae_logits=confidence.pae_logits,
            pae_bins=confidence.pae_bins,
            structure_coordinates=coords,
            backbone_coordinates=atoms.coords[:, _BACKBONE_ATOM37],
            full_sequence=tokens.sequence,
            asym_id=tokens.asym_id,
            residue_idx=tokens.residue_index,
            atom37_coords=atoms.coords,
            atom37_mask=atoms.mask,
        )

    @eqx.filter_jit
    def model_output(
        self,
        *,
        PSSM: Float[Array, "L 20"] | None = None,
        features: Features,
        recycling_steps: int | None = None,
        sampler: SamplerSettings | None = None,
        initial_state: TrunkState | None = None,
        key: PRNGKeyArray,
    ) -> StructureModelOutput:
        """Predict one structure: `design_recycling_steps` trunk passes and the
        default `sampler()` unless given."""
        if recycling_steps is None:
            recycling_steps = self.design_recycling_steps
        sampler = self.sampler() if sampler is None else sampler
        k_prepare, k_trunk, k_sample = jax.random.split(key, 3)
        features, inputs = self.prepare(features, PSSM, key=k_prepare)
        trunk = self.run_trunk(
            features, inputs, recycling_steps=recycling_steps, key=k_trunk, initial_state=initial_state
        )
        return self.sample_structure(trunk, sampler=sampler, key=k_sample)

    def predict(
        self,
        *,
        PSSM: Float[Array, "L 20"] | None = None,
        features: Features,
        writer: Writer = None,
        recycling_steps: int | None = None,
        sampler: SamplerSettings | None = None,
        initial_state: TrunkState | None = None,
        key: PRNGKeyArray,
    ) -> StructurePrediction:
        """Predict one structure: `inference_recycling_steps` trunk passes and the
        default `sampler()` unless given (pass `upstream_sampler()` for the
        model's own sampler)."""
        if recycling_steps is None:
            recycling_steps = self.inference_recycling_steps
        output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampler=sampler,
            initial_state=initial_state,
            key=key,
        )
        seq = PSSM if PSSM is not None else jnp.zeros((0, 20))
        iptm = -IPTMLoss()(seq, output, key=None)[0]  # deterministic: reads no key
        return StructurePrediction(
            st=self.to_structure(features, writer, output, PSSM),
            plddt=output.plddt,
            pae=output.pae,
            iptm=iptm,
            model_output=output,
        )

    def build_loss(
        self,
        *,
        loss: LossTerm | LinearCombination,
        features: Features,
        recycling_steps: int | None = None,
        num_samples: int = 1,
        reduction: Callable = jnp.mean,
        sampler: SamplerSettings | None = None,
        initial_state: TrunkState | None = None,
    ) -> "MultiSampleLoss[Features]":
        """A design loss: run the trunk once (from `initial_state`, if given) and
        reduce `loss` over `num_samples` diffusion samples."""
        return MultiSampleLoss(
            model=self,
            features=features,
            loss=loss,
            sampler=self.sampler() if sampler is None else sampler,
            recycling_steps=self.design_recycling_steps if recycling_steps is None else recycling_steps,
            num_samples=num_samples,
            reduction=reduction,
            initial_state=initial_state,
        )

    def upstream_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> SamplerSettings:
        """The model's own sampler as upstream defines it (upstream's step count
        by default)."""
        if num_steps is None:
            num_steps = self.upstream_sampling_steps
        return self.model.sampling.settings(num_steps, schedule=schedule)

    def sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> SamplerSettings:
        """The default sampler: the model's own, with the exponential update on a
        geometric schedule, `DEFAULT_SAMPLING_STEPS` steps (fewer for models whose
        sampler caps the schedule, e.g. ESMFold2)."""
        sampler = self.upstream_sampler(
            DEFAULT_SAMPLING_STEPS if num_steps is None else num_steps,
            schedule=schedules.Geometric() if schedule is None else schedule,
        )
        return dataclasses.replace(sampler, update=exponential_update)

    # -- host side --------------------------------------------------------

    @abstractmethod
    def target_only_features(
        self, chains: Sequence[TargetChain], *, key: PRNGKeyArray
    ) -> tuple[Features, Writer]:
        """Featurize the chains as given."""

    @abstractmethod
    def binder_features(
        self, binder_length: int, chains: Sequence[TargetChain], *, key: PRNGKeyArray
    ) -> tuple[Features, Writer]:
        """Features for a placeholder binder (chain 0) followed by `chains`."""

    @abstractmethod
    def to_structure(
        self,
        features: Features,
        writer: Writer,
        output: StructureModelOutput,
        sequence: Float[Array, "L 20"] | None,
    ) -> gemmi.Structure:
        """Convert `output` to a structure."""

    # -- traced -----------------------------------------------------------

    @abstractmethod
    def set_sequence(
        self, features: Features, sequence: Float[Array, "L 20"], *, key: PRNGKeyArray
    ) -> Features:
        """Set the first `L` tokens to the (soft) `sequence`."""

    @abstractmethod
    def jfold_features(self, features: Features) -> eqx.Module:
        """jfold's features from the backend's (traceable: runs inside `jit`)."""

    @abstractmethod
    def tokens(self, features: Features) -> TokenInfo: ...

    @abstractmethod
    def atom37(self, features: Features, coords: Float[Array, "A 3"]) -> Atom37: ...

    # -- the forward pass, on `model` ---------------------------------------

    def embed(self, features: Features, *, key: PRNGKeyArray) -> Inputs:
        """Inputs shared by every trunk pass and the heads."""
        jfeatures = self.jfold_features(features)
        return Inputs(features=jfeatures, trunk=self.model.embed(jfeatures, key=key))

    def init_state(self, inputs: Inputs, *, key: PRNGKeyArray) -> TrunkState:
        """The state before the first pass."""
        return self.model.trunk.init_state(inputs.trunk, key=key)

    def step(self, inputs: Inputs, state: TrunkState, *, key: PRNGKeyArray) -> TrunkState:
        """One trunk pass."""
        step = eqx.filter_checkpoint(_step) if self.remat_steps else _step
        return step(self.model, inputs.features, inputs.trunk, state, key)

    def head_state(self, trunk: Trunk[Features]) -> TrunkState:
        """The state the heads read: the trunk's last, through `model.trunk.finish`."""
        return self.model.trunk.finish(trunk.inputs.features, trunk.state)

    def structure_sampler(self, trunk: Trunk[Features], sampler: SamplerSettings) -> StructureSampler:
        """A sampler for structures from `trunk` with `sampler`'s settings: the conditioned
        structure module (built once, denoising at full float32 matmul precision) and
        the atoms it moves."""
        jfold_sampler = self.model.structure_sampler(
            trunk.inputs.features, trunk.inputs.trunk, self.head_state(trunk)
        )
        return StructureSampler(
            denoise=eqx.Partial(_float32, jfold_sampler.denoise),
            atom_mask=jfold_sampler.atom_mask,
            settings=sampler,
        )

    def structure_denoiser(self, trunk: Trunk[Features]) -> tuple[Denoise, Float[Array, " A"]]:
        """The structure module conditioned on the trunk, and the atom mask."""
        structure_sampler = self.structure_sampler(trunk, self.upstream_sampler())
        return structure_sampler.denoise, structure_sampler.atom_mask

    def distogram(self, trunk: Trunk[Features]) -> Distogram:
        return Distogram(
            logits=self.model.distogram(self.head_state(trunk).z),
            bins=jnp.array(self.distogram_bins),
        )

    def confidence(self, trunk: Trunk[Features], coords: Float[Array, "A 3"]) -> Confidence:
        """pLDDT per token (`token_plddt`) and PAE logits; `Confidence.unavailable` for
        models without a confidence head."""
        if self.model.confidence is None:
            return Confidence.unavailable(self.tokens(trunk.features).mask.shape[0])
        logits = self.model.confidence_logits(
            trunk.inputs.features, trunk.inputs.trunk, self.head_state(trunk), coords
        )
        return Confidence(
            plddt=self.token_plddt(self.plddt(logits.plddt), trunk),
            pae_logits=logits.pae,
            pae_bins=jnp.array(PAE_BINS),
        )

    def plddt(self, logits: Float[Array, "... B"]) -> Float[Array, "..."]:
        """pLDDT in [0, 1]: the expected value over evenly spaced bins."""
        num_bins = logits.shape[-1]
        return jax.nn.softmax(logits, axis=-1) @ ((jnp.arange(num_bins) + 0.5) / num_bins)

    def token_plddt(self, plddt: Float[Array, " N"], trunk: Trunk[Features]) -> Float[Array, " T"]:
        """Per-token pLDDT from the head's: as it is for a per-token head, else each
        token's representative atom's."""
        features = trunk.inputs.features
        if plddt.shape[0] == features.token_mask.shape[0]:
            return plddt
        return plddt[features.rep_atom]


_BACKBONE_ATOM37 = (0, 1, 2, 4)  # N, CA, C, O


class MultiSampleLoss[Features](LossTerm):
    """Run the trunk once and reduce `loss` over `num_samples` diffusion samples."""

    model: DiffusionStructurePredictionModel[Features]
    features: Features
    loss: LossTerm | LinearCombination
    sampler: SamplerSettings
    recycling_steps: int = eqx.field(static=True)
    num_samples: int = eqx.field(static=True, default=1)
    reduction: Callable = eqx.field(static=True, default=jnp.mean)
    # Trunk state before the first pass; `None`: the model's `init_state`.
    initial_state: TrunkState | None = None

    def __call__(self, sequence: Float[Array, "L 20"], *, key: PRNGKeyArray):
        k_prepare, k_trunk, k_samples = jax.random.split(key, 3)
        features, inputs = self.model.prepare(self.features, sequence, key=k_prepare)
        trunk = self.model.run_trunk(
            features,
            inputs,
            recycling_steps=self.recycling_steps,
            key=k_trunk,
            initial_state=self.initial_state,
        )

        def one_sample(k):
            k_sample, k_loss = jax.random.split(k)
            output = self.model.sample_structure(trunk, sampler=self.sampler, key=k_sample)
            return self.loss(sequence, output, key=k_loss)

        values, aux = jax.vmap(one_sample)(jax.random.split(k_samples, self.num_samples))
        return reduce_samples(values, aux, self.reduction, self.num_samples)
