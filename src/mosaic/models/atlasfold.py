"""AtlasFold-M backend running on jfold.

Inputs are featurized with `jfold.featurize`, which builds the same features as upstream's
multimer runner. The forward pass uses jfold's converted AtlasFold-M and AtlasLM-3B weights
from `jfold.pretrained`. Only protein chains of the 20 amino acids and X are supported,
without MSAs or templates.

AtlasFold has no reference conformers. Each residue has 14 atom slots whose embeddings and
geometry the model derives from the residue probabilities, so a soft sequence is set
through `aatype` alone, and its gradient also flows through AtlasLM. Which atom slots exist
and which one is the pseudo-beta atom are discrete, so for a binder they are taken from
`argmax(sequence)` without gradient."""

from pathlib import Path

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray
from jfold import pretrained
from jfold.featurize import Chain, System
from jfold.featurize.system import THREE_LETTER
from jfold.featurize.views import atlasfold as view
from jfold.model import StructureModel
from jfold.models import atlasfold as jatlasfold

from mosaic.cache import cache_dir
from mosaic.common import TOKENS
from mosaic.models.jfold_featurize import CHAIN_IDS, check_chains
from mosaic.structure_prediction import (
    Atom37,
    DiffusionStructurePredictionModel,
    StructureModelOutput,
    TargetChain,
    TokenInfo,
    scatter_atom37,
)

# Upstream's 64 distogram bins between 2 and 22 Å.
DISTOGRAM_BINS = np.linspace(2.0, 22.0, 64)
# Number of trunk passes for design and for `predict`. Upstream's multimer runner uses 10
# recycles.
DESIGN_RECYCLING_STEPS = 2
INFERENCE_RECYCLING_STEPS = 11


def _residue_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-residue-class tables for AtlasFold's 21 classes.

    Return which atom14 slots exist (`[21, 14]`), each slot's atom37 index or -1
    (`[21, 14]`), and the pseudo-beta slot (`[21]`)."""
    table = view.residues()
    exists = np.zeros((len(view.RESTYPES), view.ATOM14), bool)
    atom37 = -np.ones((len(view.RESTYPES), view.ATOM14), np.int32)
    for i, letter in enumerate(view.RESTYPES):
        slots = table[THREE_LETTER[letter]].extra["atom37"]
        exists[i, : len(slots)] = True
        atom37[i, : len(slots)] = slots
    pseudo_beta = np.where(np.array(list(view.RESTYPES)) == "G", view.CA, view.CB)
    return exists, atom37, pseudo_beta


ATOM14_EXISTS, ATOM14_TO_ATOM37, PSEUDO_BETA = _residue_tables()
assert view.RESTYPES[:20] == TOKENS  # AtlasFold uses mosaic's order, with X last


def _system(chains: list[TargetChain]) -> System:
    """Build a jfold `System` from the chain sequences. AtlasFold ignores `use_msa`."""
    return System(tuple(Chain(i, c.sequence) for i, c in zip(CHAIN_IDS, chains)))


class AtlasFoldWriter:
    """Convert atom14 positions of shape `[14 T, 3]` to a `gemmi.Structure` of `system`.

    If `binder_pssm` is given, the first chain's residues are set to its argmax."""

    def __init__(self, system: System):
        self.system = system

    def __call__(self, coords, binder_pssm=None) -> gemmi.Structure:
        chains = self.system.chains
        if binder_pssm is not None:
            binder = "".join(TOKENS[i] for i in np.asarray(binder_pssm).argmax(-1))
            chains = (Chain(chains[0].id, binder), *chains[1:])
        system = System(chains)
        residues = sum(len(c.sequence) for c in chains)
        exists = view.features(system)["atom14_mask"][:residues].reshape(-1)
        coords = np.asarray(coords).reshape(-1, 3)[exists]
        st = gemmi.read_pdb_string(view.atom_sites(system).to_pdb(coords))
        st.setup_entities()
        return st


def target_only_features(chains: list[TargetChain]):
    """Featurize `chains` and return the features with a structure writer.

    jfold pads the features as upstream's runner does. The padding is removed here so that
    outputs and losses cover only real residues. The model masks padding, so this changes
    the results only by float rounding."""
    check_chains(chains, "AtlasFold", templates=False)
    system = _system(chains)
    residues = sum(len(c.sequence) for c in chains)
    lm_tokens = residues + 2 * len(chains)  # each chain adds <cls> and <eos>
    features = {
        k: v[:lm_tokens] if k.startswith("lm.") else v[:residues]
        for k, v in view.features(system).items()
    }
    return jax.tree.map(jnp.asarray, features), AtlasFoldWriter(system)


def set_binder_sequence(sequence: Float[Array, "N 20"], features: dict) -> dict:
    """Set the binder, the first `N` residues, to the soft `sequence`.

    The residue probabilities are set to `sequence`, and the atom14 mask and pseudo-beta
    slot are taken from `argmax(sequence)`."""
    n = sequence.shape[0]
    residues = jax.lax.stop_gradient(jnp.argmax(sequence, axis=-1))
    aatype = features["aatype"].astype(jnp.float32).at[:n].set(jnp.pad(sequence, ((0, 0), (0, 1))))
    return features | {
        "aatype": aatype,
        "atom14_mask": features["atom14_mask"].at[:n].set(jnp.asarray(ATOM14_EXISTS)[residues]),
        "pseudo_beta": features["pseudo_beta"].at[:n].set(jnp.asarray(PSEUDO_BETA)[residues]),
    }


class AtlasFold(DiffusionStructurePredictionModel[dict]):
    """AtlasFold-M with the recycling and 200-step sampler of upstream's multimer runner.

    Each trunk pass, which includes AtlasLM-3B, is rematerialized to save memory."""

    remat_steps = True

    model: StructureModel
    design_recycling_steps: int = eqx.field(static=True)
    inference_recycling_steps: int = eqx.field(static=True)
    backprop_all_passes: bool = eqx.field(static=True)
    # Whether the sequence gradient flows through AtlasLM. AtlasLM reads the sequence
    # either way.
    lm_gradient: bool = eqx.field(static=True)
    distogram_bins = DISTOGRAM_BINS

    def __init__(
        self,
        *,
        model: StructureModel | None = None,
        cache_path: Path | None = None,
        design_recycling_steps: int = DESIGN_RECYCLING_STEPS,
        inference_recycling_steps: int = INFERENCE_RECYCLING_STEPS,
        backprop_all_passes: bool = False,
        lm_gradient: bool = True,
    ):
        """Wrap `model`, or load the converted AtlasFold-M checkpoint if it is None.

        The checkpoint is downloaded on first use into `cache_path`, which defaults to
        mosaic's `jfold` cache."""
        if model is None:
            # Load on the CPU and move to the accelerator in one transfer.
            with jax.default_device(jax.devices("cpu")[0]):
                model = pretrained.load("atlasfold_m", cache=cache_path or cache_dir("jfold"))
            model = eqx.filter_shard(
                model, jax.sharding.SingleDeviceSharding(jax.devices()[0])
            )
        self.model = model
        self.design_recycling_steps = design_recycling_steps
        self.inference_recycling_steps = inference_recycling_steps
        self.backprop_all_passes = backprop_all_passes
        self.lm_gradient = lm_gradient

    def target_only_features(self, chains: list[TargetChain], *, key: PRNGKeyArray):
        """Featurize `chains`. `key` is unused because AtlasFold has no reference
        conformers."""
        del key
        return target_only_features(chains)

    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ):
        return self.target_only_features(
            [TargetChain(sequence="X" * binder_length, use_msa=False)] + chains, key=key
        )

    def set_sequence(
        self, features: dict, sequence: Float[Array, "N 20"], *, key: PRNGKeyArray
    ) -> dict:
        del key
        return set_binder_sequence(sequence, features)

    def jfold_features(self, features: dict) -> jatlasfold.AtlasFoldFeatures:
        jfeatures = jatlasfold.features(features)
        if self.lm_gradient:
            return jfeatures
        tokens = jax.lax.stop_gradient(jfeatures.lm.tokens)
        return eqx.tree_at(lambda f: f.lm.tokens, jfeatures, tokens)

    def tokens(self, features: dict) -> TokenInfo:
        return TokenInfo(
            sequence=features["aatype"][:, :20],
            asym_id=features["asym_id"],
            residue_index=features["res_idx"],
            mask=features["seq_mask"].astype(bool),
        )

    def atom37(self, features: dict, coords: Float[Array, "A 3"]) -> Atom37:
        """Scatter atom14 positions to atom37 using the argmax of each residue type."""
        num_tokens = features["aatype"].shape[0]
        residues = jnp.argmax(features["aatype"], axis=-1)
        slots = jnp.where(
            features["atom14_mask"].astype(bool), jnp.asarray(ATOM14_TO_ATOM37)[residues], -1
        )
        coords37, mask = scatter_atom37(
            atom_coords=coords,
            atom_to_token=jnp.repeat(jnp.arange(num_tokens), view.ATOM14),
            atom37_idx=slots.reshape(-1),
            n_token=num_tokens,
        )
        return Atom37(coords=coords37, mask=mask)

    def to_structure(
        self, features: dict, writer: AtlasFoldWriter, output: StructureModelOutput, sequence
    ) -> gemmi.Structure:
        del features
        return writer(output.structure_coordinates, sequence)
