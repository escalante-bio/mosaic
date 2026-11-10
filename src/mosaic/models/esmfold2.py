"""ESMFold2 on jfold: featurized by `jfold.featurize` (torch-free), with jfold's converted
ESMFold2 and its ESMC (`jfold.pretrained`) for the forward pass.

Checkpoints by `jfold.pretrained` name: the release type's `esmfold2_fast` (no MSA
encoder; the default, for design) and `esmfold2`, and the experimental type's
`esmfold2_experimental[_fast][_cutoff2025]`.

Features are the view's dict with ESMC's states per token (`lm_hidden_states`), run once
at featurization with the designed positions read as X. A binder design pack also
carries each amino acid's atoms: `set_sequence` rebuilds the binder's atoms and ESMC
states from `argmax(sequence)`, so the geometry the trunk sees tracks the sequence.
Protein chains of the 20 amino acids and X only, without templates.
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
# `predict`: upstream `fold`'s defaults (20 recycles, 0.3 language-model dropout, 200
# sampling steps).
INFERENCE_RECYCLING_STEPS = 21
UPSTREAM_SAMPLING_STEPS = 200
LM_DROPOUT = 0.3
# Each a3m's first records, as upstream reads them (`MSA.from_a3m`'s `max_sequences`).
MSA_MAX_SEQUENCES = 8192
NUM_CLASSES = 33  # ESMFold2's residue classes
MAX_ATOMS = 14  # atoms per binder residue (Trp's), the room each one is allocated

# Each mosaic-20 residue's class, as a `[20, 33]` one-hot; and its ESMC input id.
RES_TYPE = np.eye(NUM_CLASSES, dtype=np.float32)[[view.TOKEN[c] for c in TOKENS]]
ESMC_ID = np.array([view.ESMC_TOKEN[c] for c in TOKENS])

# The view's entries the model does not read: ESMC's inputs (`lm_hidden_states` holds
# their states) and the training targets.
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
# A residue's atom entries, which a design pack's binder takes from its amino acids'.
_ATOM_FIELDS = ("ref_pos", "ref_element", "ref_charge", "ref_atom_name_chars")


class ESMFold2Features(eqx.Module):
    """ESMFold2's features: the view's dict with ESMC's states per token
    (`lm_hidden_states`, `[T, layers, D]`), each atom's atom37 slot (-1 for none), and
    for a design pack each amino acid's atoms (`residue_atoms`), from which `set_sequence`
    rebuilds the binder (the first chain)."""

    features: dict[str, Array]
    atom37_slot: Int[Array, " A"]
    residue_atoms: dict[str, Array] | None = None


def _designed_system(
    chains: list[TargetChain], design_char: str | None, design_geometry: str
) -> System:
    """`chains` as a jfold `System`, each residue equal to `design_char` as
    `design_geometry` (each chain's MSA fetched for its own sequence)."""
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
    """Each atom's atom37 slot in the view's dict `f`, -1 for none (and padding)."""
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
    """ESMC's states for each token: ESMC run on `ids` (its vocabulary; chains by
    `sequence_id`), each token's row (`positions`) taken."""
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
    """`chains`' features and their jfold `System`; residues equal to `design_char` are
    designed positions: a `design_geometry` residue, X to ESMC."""
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
    """Each amino acid's atoms (`TOKENS` order) as the view featurizes it alone, its real
    atoms first in `MAX_ATOMS` slots: the atom entries, the mask, atom37 slots and the
    distogram atom (`[20]`)."""
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
    """`f` with the binder's (the first `N` tokens') residue classes and MSA query row
    `sequence`, and its columns of the other MSA rows masked, so its MSA profile is
    `sequence` itself."""
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
    """`f` and its atom37 slots with the binder's atoms (its `N` tokens allocated
    `MAX_ATOMS` each) those of amino acids `residues` (`atoms`, `residue_atoms`): packed
    tightly at the front in order, the target's atoms right after them, padding at the
    end. Static shapes, so it runs under `jit`."""
    n = residues.shape[0]
    allocated = n * MAX_ATOMS
    num_atoms = f["ref_pos"].shape[0]
    count = atoms["atom_attention_mask"][residues].sum(-1)
    first = jnp.cumsum(count) - count
    local = jnp.arange(MAX_ATOMS)
    binder_slot = jnp.where(local < count[:, None], first[:, None] + local, num_atoms)
    binder_slot = binder_slot.reshape(-1)  # out of range: dropped
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
    """The distogram's bin centres: `num_bins` even bins from 2 Å to `max_distance`; 128
    bins (the experimental type's) between 1 and 57 Å, the inner edges even from 2 to
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
    """The language-model dropout (on at inference; every loop for the release type, in its
    trunk; once for the experimental, in its trunk init), MSA rows per loop, MSA column
    masking, and whether the token features carry the MSA profile (`None`: the
    checkpoint's own)."""
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
    """ESMFold2 on jfold, with upstream `fold`'s language-model dropout and sampler, and
    each trunk pass rematerialized."""

    remat_steps = True
    upstream_sampling_steps = UPSTREAM_SAMPLING_STEPS

    model: StructureModel  # with its ESMC (`model.trunk_init.esmc`)
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
        """`model`: a converted ESMFold2 with its ESMC. `ESMFold2.fast()`,
        `.experimental_fast()`, ... load the releases.

        `lm_dropout`: the rate of dropout on the language model's pair, on at inference
        as upstream's `fold` applies it (0 for none). `msa_max_depth`: MSA rows drawn per
        trunk pass (the query always kept; `None` keeps all). `msa_column_mask_rate`: share
        of MSA columns masked outside the query, once per forward pass (upstream's `fold`
        uses 0.1; 0 by default). `msa_profile`: whether the token features carry the MSA
        profile and deletion mean; `None` for the checkpoint's own (upstream's: zero for
        checkpoints with `disable_msa_features`)."""
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
        """The `jfold.pretrained` release `name`: downloaded from jfold's Hugging Face
        repository on first use, then read from mosaic's `jfold` cache."""
        # Read on the host, then moved over: loading ESMC-6B's large arrays straight onto
        # the accelerator fragments its memory pool enough that later multi-GB
        # allocations fail.
        with jax.default_device(jax.devices("cpu")[0]):
            model = pretrained.load(name, cache=cache_dir("jfold"))
        model = eqx.filter_shard(model, jax.sharding.SingleDeviceSharding(jax.devices()[0]))
        return cls(model, **settings)

    @classmethod
    def base(cls, **settings) -> "ESMFold2":
        """ESMFold2 (with an MSA encoder). `settings`: `__init__`'s."""
        return cls._release("esmfold2", **settings)

    @classmethod
    def fast(cls, **settings) -> "ESMFold2":
        """ESMFold2-Fast. `settings`: `__init__`'s."""
        return cls._release("esmfold2_fast", **settings)

    @classmethod
    def experimental(cls, **settings) -> "ESMFold2":
        """ESMFold2-Experimental. `settings`: `__init__`'s."""
        return cls._release("esmfold2_experimental", **settings)

    @classmethod
    def experimental_cutoff2025(cls, **settings) -> "ESMFold2":
        """ESMFold2-Experimental, 2025 data cutoff. `settings`: `__init__`'s."""
        return cls._release("esmfold2_experimental_cutoff2025", **settings)

    @classmethod
    def experimental_fast(cls, **settings) -> "ESMFold2":
        """ESMFold2-Experimental-Fast. `settings`: `__init__`'s."""
        return cls._release("esmfold2_experimental_fast", **settings)

    @classmethod
    def experimental_fast_cutoff2025(cls, **settings) -> "ESMFold2":
        """ESMFold2-Experimental-Fast, 2025 data cutoff. `settings`: `__init__`'s."""
        return cls._release("esmfold2_experimental_fast_cutoff2025", **settings)

    @classmethod
    def experimental_fast_base300m(cls, **settings) -> "ESMFold2":
        """ESMFold2-Experimental-Fast on ESMC-300M. `settings`: `__init__`'s."""
        return cls._release("esmfold2_experimental_fast_base300m", **settings)

    @classmethod
    def experimental_fast_base600m(cls, **settings) -> "ESMFold2":
        """ESMFold2-Experimental-Fast on ESMC-600M. `settings`: `__init__`'s."""
        return cls._release("esmfold2_experimental_fast_base600m", **settings)

    @classmethod
    def experimental_fast_base6b(cls, **settings) -> "ESMFold2":
        """ESMFold2-Experimental-Fast on ESMC-6B (1.5M steps). `settings`: `__init__`'s."""
        return cls._release("esmfold2_experimental_fast_base6b", **settings)

    @property
    def distogram_bins(self) -> np.ndarray:
        num_bins = self.model.distogram.linear.weight.shape[0]
        experimental = isinstance(self.model.trunk, RecyclingPairTrunk)
        return distogram_bin_centers(num_bins, 52.0 if experimental else 22.0)

    # -- host side ----------------------------------------------------------------------

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
        """`chains`' features and writer; residues equal to `design_char` are designed
        positions (a `design_geometry` residue, X to ESMC). Deterministic: `key` is unused."""
        del key
        features, system = self._featurize(chains, design_char, design_geometry)
        return features, AtomSitesWriter(view.atom_sites(system))

    def binder_features(
        self, binder_length: int, chains: list[TargetChain], *, key: PRNGKeyArray
    ) -> tuple[ESMFold2Features, BinderAtomSitesWriter]:
        """A design pack: a designed binder (chain 0, allocated as Trp, the largest
        residue) before `chains`, its atoms and ESMC states rebuilt from
        `argmax(sequence)` by `set_sequence`. Deterministic: `key` is unused."""
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
        """A design pack written with its binder `sequence`'s atoms; other features with
        their own atoms, the binder's residues named by `sequence` if given."""
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

    # -- traced ---------------------------------------------------------------------------

    def set_sequence(
        self,
        features: ESMFold2Features,
        sequence: Float[Array, "L 20"],
        *,
        key: PRNGKeyArray,
    ) -> ESMFold2Features:
        """The binder's residue classes and MSA query row; a design pack also rebuilds the
        binder's atoms and ESMC states from `argmax(sequence)` (no gradient flows through
        either)."""
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
        """ESMC's states for a binder of amino acids `residues`, run on the binder alone:
        the same as in the whole system, as ESMC attends within chains, with absolute
        positions, and the binder comes first."""
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
        # The language model is frozen: no gradient flows through it.
        lm_hidden_states = jax.lax.stop_gradient(f["lm_hidden_states"])
        return jesm.features(f | {"lm_hidden_states": lm_hidden_states})

    def token_plddt(self, plddt, trunk: Trunk):
        """The mean of each token's real atoms' pLDDT."""
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
