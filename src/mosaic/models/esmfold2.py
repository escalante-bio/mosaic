"""ESMFold2 backend running on jfold.

Inputs are featurized with `jfold.featurize`, and the forward pass uses jfold's converted
ESMFold2 and ESMC weights from `jfold.pretrained`. `esmfold2_fast` has no MSA encoder and is
the default for design.

ESMC's hidden states are computed once at featurization, with designed positions given to
ESMC as X. For binder design, `set_sequence` rebuilds the binder's atoms and ESMC states
from `argmax(sequence)`, so the atoms the trunk sees match the current sequence. Only
protein chains of the 20 amino acids and X are supported, without templates.
"""

import dataclasses
import functools

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from equinox.nn import Identity
from jaxtyping import Array, Float, Int, PRNGKeyArray
from jfold import pretrained
from jfold.blocks.atoms import mean_atoms_to_tokens
from jfold.blocks.inputs import probabilities
from jfold.featurize import System
from jfold.featurize.views import esmfold2 as view
from jfold.model import StructureModel
from jfold.models import esmfold2 as jesm
from jfold.models.esmfold2 import ESMC, RecyclingPairTrunk, TrackedMSA
from jfold.nn import Dropout
from jfold.slots import Features as JFeatures

from mosaic.cache import cache_dir
from mosaic.common import TOKENS
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37
from mosaic.losses.structure_prediction import StructureModelOutput
from mosaic.models.jfold_featurize import (
    AtomSitesWriter,
    BinderAtomSitesWriter,
    check_chains,
    jfold_system,
    residue_type_features,
)
from mosaic.structure_prediction import (
    Atom37,
    DiffusionStructurePredictionModel,
    TargetChain,
    TokenInfo,
    Trunk,
)

DESIGN_RECYCLING_STEPS = 4
# Defaults for `predict` match upstream `fold`: 20 recycles, 0.3 language-model dropout
# and 200 sampling steps.
INFERENCE_RECYCLING_STEPS = 21
UPSTREAM_SAMPLING_STEPS = 200
LM_DROPOUT = 0.3
# Upstream reads at most this many sequences from each a3m file.
MSA_MAX_SEQUENCES = 8192
NUM_CLASSES = 33  # ESMFold2's residue classes
MAX_ATOMS = 14  # atom slots per binder residue, enough for Trp

# ESMFold2 classes as a `[20, 33]` one-hot, and ESMC token ids, in `TOKENS` order.
RES_TYPE = np.eye(NUM_CLASSES, dtype=np.float32)[[view.TOKEN[c] for c in TOKENS]]
ESMC_ID = np.array([view.ESMC_TOKEN[c] for c in TOKENS])

# Features the model does not use: ESMC's inputs, which are replaced by
# `lm_hidden_states`, and training targets.
_UNREAD = (
    "input_ids",
    "lm_input_ids",
    "lm_sequence_id",
    "lm_positions",
    "frames_idx",
    "gt_coords",
    "is_resolved",
    "disto_cond",
    "disto_cond_mask",
    "pocket_feature",
)
# Per-atom features that are rebuilt for the binder during design.
_ATOM_FIELDS = ("ref_pos", "ref_element", "ref_charge", "ref_atom_name_chars")


class ESMFold2Features(eqx.Module):
    """ESMFold2 input features.

    `features` is jfold's feature dict plus ESMC's per-token hidden states
    (`lm_hidden_states`, shape `[T, layers, D]`). `atom37_slot` is each atom's atom37 index,
    or -1. For binder design, `residue_atoms` holds the atoms of every amino acid so that
    `set_sequence` can rebuild the binder, which is the first chain."""

    features: dict[str, Array]
    atom37_slot: Int[Array, " A"]
    residue_atoms: dict[str, Array] | None = None


def _designed_system(
    chains: list[TargetChain], design_char: str | None, design_geometry: str
) -> System:
    """Build a jfold `System` with every `design_char` residue set to `design_geometry`."""
    system = jfold_system(chains)
    if design_char is None:
        return system
    return System(
        tuple(
            dataclasses.replace(
                c, sequence=c.sequence.replace(design_char, design_geometry)
            )
            for c in system.chains
        )
    )


def atom37_slots(f: dict[str, np.ndarray]) -> np.ndarray:
    """Return each atom's atom37 index in the feature dict `f`, or -1 if it has none."""
    names = [
        "".join(chr(c + 32) for c in chars).strip()
        for chars in f["ref_atom_name_chars"]
    ]
    slots = [ATOM37_INDEX.get(name, -1) for name in names]
    return np.where(f["atom_attention_mask"], slots, -1).astype(np.int32)


@eqx.filter_jit
def esmc_states(
    esmc: ESMC,
    ids: Int[Array, " L"],
    sequence_id: Int[Array, " L"],
    positions: Int[Array, " T"],
) -> Float[Array, "T layers D"]:
    """Run ESMC on `ids` and return its hidden states at each token's `positions`.

    `sequence_id` gives the chain of each id."""
    tokens = jax.nn.one_hot(ids, esmc.embed.weight.shape[1])
    hidden = esmc(tokens, sequence_id, jnp.ones(ids.shape, bool))
    return jesm.token_hidden_states(hidden, positions)


def featurize(
    esmc: ESMC,
    chains: list[TargetChain],
    *,
    design_char: str | None = None,
    design_geometry: str = "G",
) -> tuple[ESMFold2Features, System]:
    """Featurize `chains` and return the features with their jfold `System`.

    Residues equal to `design_char` are designed positions. They are built as
    `design_geometry` residues and given to ESMC as X."""
    check_chains(chains, "ESMFold2", templates=False, also=design_char or "")
    system = _designed_system(chains, design_char, design_geometry)
    f = view.features(system, max_sequences=MSA_MAX_SEQUENCES)
    designed = np.array([c == design_char for chain in chains for c in chain.sequence])
    lm = view.lm_inputs(
        f["asym_id"], np.where(designed, view.ESMC_TOKEN["X"], f["input_ids"])
    )
    states = esmc_states(
        esmc, lm["lm_input_ids"], lm["lm_sequence_id"], lm["lm_positions"]
    )
    arrays = {k: jnp.asarray(v) for k, v in f.items() if k not in _UNREAD}
    features = ESMFold2Features(
        features=arrays | {"lm_hidden_states": states},
        atom37_slot=jnp.asarray(atom37_slots(f)),
    )
    return features, system


@functools.cache
def residue_atoms() -> dict[str, np.ndarray]:
    """Return the atom features of each of the 20 amino acids, featurized alone.

    Each amino acid has `MAX_ATOMS` slots with its real atoms first. The result also holds
    the atom mask, the atom37 indices and the index of each residue's distogram atom."""
    singles = residue_type_features(view.features).values()
    atoms = {
        k: np.stack([f[k][:MAX_ATOMS] for f in singles])
        for k in (*_ATOM_FIELDS, "atom_attention_mask")
    }
    atoms["atom37_slot"] = np.stack([atom37_slots(f)[:MAX_ATOMS] for f in singles])
    atoms["distogram_atom_idx"] = np.array(
        [f["distogram_atom_idx"][0] for f in singles]
    )
    return atoms


def set_binder_sequence(
    f: dict[str, Array], sequence: Float[Array, "N 20"]
) -> dict[str, Array]:
    """Set the residue classes and MSA query row of the binder, the first `N` tokens.

    The binder's columns in the other MSA rows are masked, so its MSA profile equals
    `sequence`."""
    n = sequence.shape[0]
    soft = sequence @ RES_TYPE
    return f | {
        "res_type": probabilities(f["res_type"], NUM_CLASSES).at[:n].set(soft),
        "msa": probabilities(f["msa"], NUM_CLASSES).at[0, :n].set(soft),
        "msa_attention_mask": f["msa_attention_mask"].at[1:, :n].set(False),
    }


def refresh_binder_atoms(
    f: dict[str, Array],
    atom37_slot: Int[Array, " A"],
    atoms: dict[str, Array],
    residues: Int[Array, " N"],
) -> tuple[dict[str, Array], Int[Array, " A"]]:
    """Replace the binder's atoms by those of amino acids `residues`.

    The binder's `N` tokens have `MAX_ATOMS` atom slots each in `f`, and `atoms` comes from
    `residue_atoms`. The new binder atoms are packed at the front, followed by the target's
    atoms and then padding. Shapes are static, so this works under `jit`. Return the new
    features and atom37 indices."""
    n = residues.shape[0]
    allocated = n * MAX_ATOMS
    num_atoms = f["ref_pos"].shape[0]
    count = atoms["atom_attention_mask"][residues].sum(-1)
    first = jnp.cumsum(count) - count
    local = jnp.arange(MAX_ATOMS)
    binder_slot = jnp.where(local < count[:, None], first[:, None] + local, num_atoms)
    binder_slot = binder_slot.reshape(-1)  # unused slots point past the end
    target_slot = count.sum() + jnp.arange(num_atoms - allocated)

    def pack(binder, current, fill=0):
        binder = binder.reshape(-1, *current.shape[1:]).astype(current.dtype)
        packed = jnp.full_like(current, fill).at[binder_slot].set(binder, mode="drop")
        return packed.at[target_slot].set(current[allocated:], mode="drop")

    binder_token = jnp.repeat(jnp.arange(n), MAX_ATOMS)
    real = count.sum() + f["atom_attention_mask"][allocated:].sum()
    distogram_atom = jnp.concatenate(
        [
            first + atoms["distogram_atom_idx"][residues],
            f["distogram_atom_idx"][n:] - allocated + count.sum(),
        ]
    )
    refreshed = {k: pack(atoms[k][residues], f[k]) for k in _ATOM_FIELDS} | {
        "atom_to_token": pack(binder_token, f["atom_to_token"]),
        "ref_space_uid": pack(binder_token, f["ref_space_uid"]),
        "atom_attention_mask": jnp.arange(num_atoms) < real,
        "distogram_atom_idx": distogram_atom.astype(f["distogram_atom_idx"].dtype),
    }
    return f | refreshed, pack(atoms["atom37_slot"][residues], atom37_slot, fill=-1)


def distogram_bin_centers(num_bins: int, max_distance: float) -> np.ndarray:
    """Return the distogram bin centres in Å.

    The bins are evenly spaced from 2 Å to `max_distance`, except for the experimental
    checkpoints' 128 bins, which span 1 to 57 Å with evenly spaced inner edges from 2 to
    52 Å."""
    if num_bins == 128:
        edges = np.concatenate([[1.0], np.linspace(2.0, 52.0, 127), [57.0]])
    else:
        edges = np.linspace(2.0, max_distance, num_bins + 1, dtype=np.float32)
    return 0.5 * (edges[:-1] + edges[1:])


def _configure(
    model: StructureModel,
    *,
    lm_dropout: float,
    msa_max_depth: int | None,
    msa_column_mask_rate: float,
    msa_profile: bool | None = None,
) -> StructureModel:
    """Return `model` with the given language-model dropout and MSA settings.

    The dropout is also applied at inference. Release checkpoints apply it on every
    recycling pass, experimental ones once before recycling. If `msa_profile` is None, the
    checkpoint's own setting is kept."""
    dropout = Dropout(lm_dropout, (), inference=False) if lm_dropout else Identity()
    experimental = isinstance(model.trunk, RecyclingPairTrunk)
    trunk_init = dataclasses.replace(
        model.trunk_init,
        msa_column_mask_rate=msa_column_mask_rate,
        lm_dropout=dropout if experimental else None,
    )
    trunk = model.trunk if experimental else dataclasses.replace(model.trunk, lm_dropout=dropout)
    msa = trunk.msa
    if isinstance(msa, TrackedMSA):
        msa = dataclasses.replace(msa, module=dataclasses.replace(msa.module, max_rows=msa_max_depth))
    elif msa is not None:
        msa = dataclasses.replace(msa, max_rows=msa_max_depth)
    input_embedder = model.input_embedder
    if msa_profile is not None:
        input_embedder = dataclasses.replace(input_embedder, msa_profile=msa_profile)
    return dataclasses.replace(
        model,
        input_embedder=input_embedder,
        trunk_init=trunk_init,
        trunk=dataclasses.replace(trunk, msa=msa),
    )


class ESMFold2(DiffusionStructurePredictionModel[ESMFold2Features]):
    """ESMFold2 with the language-model dropout and sampler of upstream `fold`.

    Each trunk pass is rematerialized to save memory."""

    remat_steps = True
    upstream_sampling_steps = UPSTREAM_SAMPLING_STEPS

    model: StructureModel  # includes ESMC as `model.trunk_init.esmc`
    design_recycling_steps: int = eqx.field(static=True)
    inference_recycling_steps: int = eqx.field(static=True)
    backprop_all_passes: bool = eqx.field(static=True)

    def __init__(
        self,
        model: StructureModel,
        *,
        lm_dropout: float = LM_DROPOUT,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.0,
        msa_profile: bool | None = None,
        design_recycling_steps: int = DESIGN_RECYCLING_STEPS,
        inference_recycling_steps: int = INFERENCE_RECYCLING_STEPS,
        backprop_all_passes: bool = True,
    ):
        """Wrap a converted ESMFold2 model that includes ESMC.

        Use `ESMFold2.fast()` and the other class methods to load released checkpoints.

        `lm_dropout` is the dropout rate on the language-model pair features. As in
        upstream `fold`, it is also applied at inference, and 0 disables it.
        `msa_max_depth` is the number of MSA rows sampled per trunk pass, always including
        the query. If it is None, all rows are used. `msa_column_mask_rate` is the fraction
        of non-query MSA columns masked once per forward pass, which upstream `fold` sets
        to 0.1. `msa_profile` sets whether the token features include the MSA profile and
        deletion mean. If it is None, the checkpoint's own setting is used."""
        self.model = _configure(
            model,
            lm_dropout=lm_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            msa_profile=msa_profile,
        )
        self.design_recycling_steps = design_recycling_steps
        self.inference_recycling_steps = inference_recycling_steps
        self.backprop_all_passes = backprop_all_passes

    @classmethod
    def _release(cls, name: str, **settings) -> "ESMFold2":
        """Load the `jfold.pretrained` checkpoint `name`, downloading it on first use."""
        # Load on the CPU and then move to the accelerator. Loading ESMC-6B directly onto
        # the accelerator fragments its memory pool, and later multi-GB allocations fail.
        with jax.default_device(jax.devices("cpu")[0]):
            model = pretrained.load(name, cache=cache_dir("jfold"))
        model = eqx.filter_shard(model, jax.sharding.SingleDeviceSharding(jax.devices()[0]))
        return cls(model, **settings)

    @classmethod
    def base(cls, **settings) -> "ESMFold2":
        """Load ESMFold2, which has an MSA encoder. `settings` go to `__init__`."""
        return cls._release("esmfold2", **settings)

    @classmethod
    def fast(cls, **settings) -> "ESMFold2":
        """Load ESMFold2-Fast. `settings` go to `__init__`."""
        return cls._release("esmfold2_fast", **settings)

    @classmethod
    def experimental(cls, **settings) -> "ESMFold2":
        """Load ESMFold2-Experimental. `settings` go to `__init__`."""
        return cls._release("esmfold2_experimental", **settings)

    @classmethod
    def experimental_cutoff2025(cls, **settings) -> "ESMFold2":
        """Load ESMFold2-Experimental with a 2025 data cutoff."""
        return cls._release("esmfold2_experimental_cutoff2025", **settings)

    @classmethod
    def experimental_fast(cls, **settings) -> "ESMFold2":
        """Load ESMFold2-Experimental-Fast."""
        return cls._release("esmfold2_experimental_fast", **settings)

    @classmethod
    def experimental_fast_cutoff2025(cls, **settings) -> "ESMFold2":
        """Load ESMFold2-Experimental-Fast with a 2025 data cutoff."""
        return cls._release("esmfold2_experimental_fast_cutoff2025", **settings)

    @classmethod
    def experimental_fast_base300m(cls, **settings) -> "ESMFold2":
        """Load ESMFold2-Experimental-Fast built on ESMC-300M."""
        return cls._release("esmfold2_experimental_fast_base300m", **settings)

    @classmethod
    def experimental_fast_base600m(cls, **settings) -> "ESMFold2":
        """Load ESMFold2-Experimental-Fast built on ESMC-600M."""
        return cls._release("esmfold2_experimental_fast_base600m", **settings)

    @classmethod
    def experimental_fast_base300m_confidence(cls, **settings) -> "ESMFold2":
        """Load ESMFold2-Experimental-Fast on ESMC-300M with FastPLMs' confidence head."""
        return cls._release("esmfold2_experimental_fast_base300m_confidence", **settings)

    @classmethod
    def experimental_fast_base600m_confidence(cls, **settings) -> "ESMFold2":
        """Load ESMFold2-Experimental-Fast on ESMC-600M with FastPLMs' confidence head."""
        return cls._release("esmfold2_experimental_fast_base600m_confidence", **settings)

    @classmethod
    def experimental_fast_base6b(cls, **settings) -> "ESMFold2":
        """Load ESMFold2-Experimental-Fast built on ESMC-6B, trained for 1.5M steps."""
        return cls._release("esmfold2_experimental_fast_base6b", **settings)

    @property
    def distogram_bins(self) -> np.ndarray:
        num_bins = self.model.distogram.linear.weight.shape[0]
        experimental = isinstance(self.model.trunk, RecyclingPairTrunk)
        return distogram_bin_centers(num_bins, 52.0 if experimental else 22.0)

    def _featurize(
        self, chains: list[TargetChain], design_char: str | None, design_geometry: str
    ) -> tuple[ESMFold2Features, System]:
        if self.model.trunk.msa is None and any(c.use_msa for c in chains):
            raise ValueError(
                "this checkpoint has no MSA encoder: set use_msa=False on all chains, or use "
                "a full checkpoint (esmfold2, esmfold2_experimental)."
            )
        return featurize(
            self.model.trunk_init.esmc,
            chains,
            design_char=design_char,
            design_geometry=design_geometry,
        )

    def target_only_features(
        self,
        chains: list[TargetChain],
        *,
        key: PRNGKeyArray,
        design_char: str | None = None,
        design_geometry: str = "G",
    ) -> tuple[ESMFold2Features, AtomSitesWriter]:
        """Featurize `chains` and return the features with a structure writer.

        Residues equal to `design_char` are designed positions, built as `design_geometry`
        residues and given to ESMC as X. `key` is unused."""
        del key
        features, system = self._featurize(chains, design_char, design_geometry)
        return features, AtomSitesWriter(view.atom_sites(system))

    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ) -> tuple[ESMFold2Features, BinderAtomSitesWriter]:
        """Featurize a designed binder of `binder_length` residues followed by `chains`.

        The binder's residues get atom slots for Trp, the largest residue, and
        `set_sequence` rebuilds its atoms and ESMC states from `argmax(sequence)`. `key` is
        unused."""
        del key
        chains = [TargetChain("X" * binder_length, use_msa=False), *chains]
        features, system = self._featurize(chains, "X", "W")
        features = dataclasses.replace(
            features,
            residue_atoms={k: jnp.asarray(v) for k, v in residue_atoms().items()},
        )
        return features, BinderAtomSitesWriter(system, view.atom_sites)

    def to_structure(
        self,
        features: ESMFold2Features,
        writer: AtomSitesWriter | BinderAtomSitesWriter,
        output: StructureModelOutput,
        sequence: Float[Array, "L 20"] | None,
    ) -> gemmi.Structure:
        """Build a `gemmi.Structure` from the predicted coordinates.

        Binder design features require `sequence`, which sets the binder's atoms. For other
        features, a given `sequence` only renames the first chain's residues."""
        coords = output.structure_coordinates
        if features.residue_atoms is not None:
            if sequence is None:
                raise ValueError(
                    "a design pack (binder_features) is written with its binder's "
                    "sequence: pass PSSM"
                )
            return writer(coords, sequence)
        st = writer(coords)
        if sequence is not None:
            for residue, k in zip(st[0][0], np.argmax(np.asarray(sequence), axis=-1)):
                residue.name = gemmi.expand_one_letter(TOKENS[k], gemmi.ResidueKind.AA)
        return st

    def set_sequence(
        self,
        features: ESMFold2Features,
        sequence: Float[Array, "L 20"],
        *,
        key: PRNGKeyArray,
    ) -> ESMFold2Features:
        """Set the binder's residue classes and MSA query row from `sequence`.

        For binder design features, also rebuild the binder's atoms and ESMC states from
        `argmax(sequence)`. No gradient flows through the rebuilt atoms or states."""
        del key
        f = set_binder_sequence(features.features, sequence)
        if features.residue_atoms is None:
            return dataclasses.replace(features, features=f)
        residues = jax.lax.stop_gradient(jnp.argmax(sequence, axis=-1))
        f, atom37_slot = refresh_binder_atoms(
            f, features.atom37_slot, features.residue_atoms, residues
        )
        states = (
            f["lm_hidden_states"].at[: len(residues)].set(self._binder_states(residues))
        )
        return dataclasses.replace(
            features, features=f | {"lm_hidden_states": states}, atom37_slot=atom37_slot
        )

    def _binder_states(self, residues: Int[Array, " N"]) -> Float[Array, "N layers D"]:
        """Return ESMC's hidden states for a binder with amino acids `residues`.

        Running ESMC on the binder alone gives the same states as running it on the whole
        system, because ESMC attends within chains, uses absolute positions, and the
        binder is the first chain."""
        n = residues.shape[0]
        ids = jnp.concatenate(
            [
                jnp.array([view.BOS]),
                jnp.asarray(ESMC_ID)[residues],
                jnp.array([view.EOS]),
            ]
        )
        states = esmc_states(
            self.model.trunk_init.esmc, ids, jnp.zeros(n + 2, int), jnp.arange(1, n + 1)
        )
        return jax.lax.stop_gradient(states)

    def jfold_features(self, features: ESMFold2Features) -> JFeatures:
        f = features.features
        # The language model is frozen.
        lm_hidden_states = jax.lax.stop_gradient(f["lm_hidden_states"])
        return jesm.features(f | {"lm_hidden_states": lm_hidden_states})

    def token_plddt(self, plddt, trunk: Trunk):
        """Average the pLDDT of each token's real atoms."""
        features = trunk.inputs.features
        atoms = features.atoms
        return mean_atoms_to_tokens(
            plddt[:, None],
            atoms.atom_to_token,
            atoms.mask,
            features.token_mask.shape[0],
        )[:, 0]

    def tokens(self, features: ESMFold2Features) -> TokenInfo:
        f = features.features
        res_type = probabilities(f["res_type"], NUM_CLASSES)
        return TokenInfo(
            sequence=res_type @ RES_TYPE.T,
            asym_id=f["asym_id"].astype(jnp.int32),
            residue_index=f["residue_index"].astype(jnp.int32),
            mask=f["token_attention_mask"].astype(bool),
        )

    def atom37(self, features: ESMFold2Features, coords: Float[Array, "A 3"]) -> Atom37:
        f = features.features
        coords37, mask = scatter_atom37(
            atom_coords=coords,
            atom_to_token=f["atom_to_token"].astype(jnp.int32),
            atom37_idx=features.atom37_slot,
            n_token=f["res_type"].shape[0],
        )
        return Atom37(coords=coords37, mask=mask)
