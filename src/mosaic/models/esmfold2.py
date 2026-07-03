"""ESMFold2 backend for mosaic structure prediction.

Factories come in two families, each in a `Fast` (no MSA encoder, default for
design) and `Full` (MSA-conditioned, for target prediction) variant:

- Release: `ESMFold2Fast()`, `ESMFold2Full()`.
- Experimental (AF3-style: LM added once to `z_init` outside the trunk loop,
  slimmer confidence head): `ESMFold2ExperimentalFast()`,
  `ESMFold2ExperimentalFull()`, plus `...2025()` variants on the 2025-cutoff
  checkpoints and the `base{300M,600M,6B}` training-step sweep via
  `ESMFold2ExperimentalFastBase(size, steps)` / `...Fast300M()` / `...Fast600M()`
  / `...Fast6B()` — lighter design models with non-functional confidence heads
  (validate with a full model).

All expose mosaic's standard `StructurePredictionModel` interface; recycling
(`recycling_steps`) and diffusion (`sampling_steps`) counts are per-call knobs
on `build_loss` / `predict`, defaulting to the module globals
`_DEFAULT_NUM_LOOPS` / `_DEFAULT_NUM_SAMPLING_STEPS`.

See `mosaic.losses.esmfold2` for the soft-sequence ESMC plumbing this
wrapper relies on.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import os
from pathlib import Path

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

import esmjfold2
from esmjfold2 import Features
from esmjfold2.esmc import ESMC
from esmjfold2.experimental import ESMFold2Experimental as JaxESMFold2Experimental
from esmjfold2.model import ESMFold2 as JaxESMFold2Release
from esmjfold2.structure_output import output_to_mmcif

from mosaic.losses.esmfold2 import (
    NUM_RES_TYPES,
    ESMFold2Loss,
    EsmFold2FeaturePack,
    _residue_atom_templates,
    build_atom37_scaffolding,
    distogram_bin_centers,
    forward_with_pssm,
    precompute_target_lm_hidden,
)
from mosaic.common import TOKENS
from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.structure_prediction import (
    PolymerType,
    StructurePrediction,
    StructurePredictionModel,
    TargetChain,
)

JaxESMFold2 = JaxESMFold2Release | JaxESMFold2Experimental


_DEFAULT_NUM_LOOPS = 3
_DEFAULT_NUM_SAMPLING_STEPS = 14
_DEFAULT_MSA_SERVER = "https://api.colabfold.com"
_DEFAULT_MSA_CACHE = Path(
    os.environ.get("MOSAIC_MSA_CACHE", "~/.cache/mosaic/msa")
).expanduser()


def _fetch_msa(sequence: str, max_sequences: int = 8192):
    """Query the ColabFold MSA server (via boltz's MMseqs2 client) for an a3m
    and return an `esm.utils.msa.MSA`. Cached by sequence hash under
    `MOSAIC_MSA_CACHE` so repeated featurizations skip the network.
    """
    from boltz.data.msa.mmseqs2 import run_mmseqs2
    from esm.utils.msa import MSA

    _DEFAULT_MSA_CACHE.mkdir(parents=True, exist_ok=True)
    seq_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]
    a3m_path = _DEFAULT_MSA_CACHE / f"{seq_hash}.a3m"

    if not a3m_path.exists():
        tmp_prefix = _DEFAULT_MSA_CACHE / seq_hash
        result = run_mmseqs2(
            [sequence],
            prefix=str(tmp_prefix),
            use_env=True,
            use_filter=True,
            use_pairing=False,
            host_url=_DEFAULT_MSA_SERVER,
        )
        # run_mmseqs2 returns either list[str] or tuple of lists depending on
        # version; unwrap defensively.
        a3m_strings = result[0] if isinstance(result, tuple) else result
        a3m_path.write_text(a3m_strings[0])

    return MSA.from_a3m(
        path=str(a3m_path), remove_insertions=True, max_sequences=max_sequences
    )


def _load_pretrained(
    checkpoint: str,
    *,
    experimental: bool = False,
    dtype=None,
) -> tuple[JaxESMFold2, ESMC]:
    """Load ESMFold2 + ESMC from a HuggingFace checkpoint and convert to JAX.

    ``experimental=True`` selects the ``ESMFold2ExperimentalModel`` torch class
    (its ``from_pretrained`` takes no ``esmc_precision``). ESMC is cast to fp32
    for conversion either way. Converts from the HF torch checkpoint via the
    Biohub ``transformers``/``esm`` forks (both are dependencies).
    """
    import torch

    if experimental:
        from transformers.models.esmfold2.modeling_esmfold2_experimental import (
            ESMFold2ExperimentalModel as TorchModelCls,
        )
        kwargs = dict(load_esmc=True, dtype=dtype or torch.float32)
    else:
        from transformers.models.esmfold2.modeling_esmfold2 import (
            ESMFold2Model as TorchModelCls,
        )
        kwargs = dict(
            load_esmc=True,
            dtype=dtype or torch.float32,
            esmc_precision="bf16",
        )

    torch_model = TorchModelCls.from_pretrained(checkpoint, **kwargs).eval()
    torch_model._esmc = torch_model._esmc.to(dtype=torch.float32)
    if hasattr(torch_model, "_esmc_fp8"):
        torch_model._esmc_fp8 = False

    eqx_esmc = esmjfold2.from_torch(torch_model._esmc)
    eqx_trunk = esmjfold2.from_torch(torch_model)

    del torch_model
    gc.collect()
    return eqx_trunk, eqx_esmc


def _featurize(
    chains: list[TargetChain],
    *,
    design_char: str | None = None,
    design_geometry: str = "G",
):
    """Build esmjfold2 features + chain_infos + design positions via the builder.

    Each chain is a real protein chain — except any residue equal to
    ``design_char`` marks a *designed* position. Designed residues are handed to
    the builder as ``design_geometry`` ("A" → N/CA/C/O/CB; "G" → N/CA/C/O) so the
    trunk gets a clean placeholder backbone there, rather than the degenerate
    atom-tokenized geometry of a literal "X". Every *non*-designed residue keeps
    its real identity, so it gets correct full-atom geometry and real res_type.

    Returns ``(raw_features, chain_infos, design_positions)`` where
    ``design_positions`` is the token-index array of the designed residues (used
    by the caller to blank them to UNK for the LM and to inject/score the design
    PSSM). ``design_char=None`` is the pure real-chain case → no design positions.
    Token indices assume one token per (standard/placeholder) residue, chains
    concatenated in order — the codebase-wide convention.
    """
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )

    proteins: list = []
    chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    design_positions: list[int] = []
    offset = 0
    for i, c in enumerate(chains):
        if c.polymer_type != PolymerType.PROTEIN:
            raise NotImplementedError(
                "ESMFold2 wrapper currently supports protein chains only; got "
                f"{c.polymer_type}"
            )
        if c.template_chain is not None:
            raise NotImplementedError(
                "ESMFold2 wrapper does not support template chains."
            )
        seq = c.sequence
        if design_char is not None and design_char in seq:
            design_positions.extend(
                offset + j for j, ch in enumerate(seq) if ch == design_char
            )
            # Hand the builder a clean placeholder backbone at designed
            # positions (the trunk reads geometry here; res_type is overwritten
            # by the PSSM and input_ids is blanked to UNK downstream).
            seq = seq.replace(design_char, design_geometry)
        # The Fast checkpoint has no MSA encoder but still uses query row 0 as
        # the `profile` aggregate, so passing the MSA through is correct in
        # both modes.
        msa = _fetch_msa(c.sequence) if c.use_msa else None
        proteins.append(
            ProteinInput(id=chain_letters[i], sequence=seq, msa=msa)
        )
        offset += len(c.sequence)

    spi = StructurePredictionInput(sequences=proteins)
    builder = ESMFold2InputBuilder()
    raw_features, chain_infos = builder.prepare_input(spi)

    # Note: we intentionally do NOT re-stamp the binder `token_bonds` block.
    # A standard-residue placeholder ("A"/"G") leaves it empty (chain
    # connectivity is implicit via residue_index, exactly as for a real chain
    # and as the target_only reprediction sees it).

    return raw_features, chain_infos, np.array(design_positions, dtype=np.int64)


def _build_pack(
    esmc: ESMC,
    raw: dict,
    *,
    design_positions: np.ndarray,
    unk_input_id: int,
) -> EsmFold2FeaturePack:
    features = Features.from_input_builder(raw)
    target_lm = precompute_target_lm_hidden(
        esmc,
        input_ids=np.asarray(raw["input_ids"]),
        asym_id=np.asarray(raw["asym_id"]),
        residue_index=np.asarray(raw["residue_index"]),
        mol_type=np.asarray(raw["mol_type"]),
        token_attention_mask=np.asarray(raw["token_attention_mask"]),
        design_positions=design_positions,
        unk_input_id=unk_input_id,
    )
    atom37_idx, bb_idx = build_atom37_scaffolding(
        ref_atom_name_chars=np.asarray(raw["ref_atom_name_chars"]),
        atom_to_token=np.asarray(raw["atom_to_token"]),
        atom_attention_mask=np.asarray(raw["atom_attention_mask"]),
        n_tokens=int(features.res_type.shape[1]),
    )
    return EsmFold2FeaturePack(
        features=features,
        target_lm_hidden=target_lm,
        atom37_idx=jnp.asarray(atom37_idx, dtype=jnp.int32),
        backbone_atom_idx=jnp.asarray(bb_idx, dtype=jnp.int32),
    )


class ESMFold2(StructurePredictionModel):
    """ESMFold2 wrapped behind mosaic's structure-prediction interface.

    Public construction: use the `ESMFold2Fast()` / `ESMFold2Full()` factories
    rather than `__init__` directly — those handle checkpoint download +
    alphabet probing.
    """

    esmf: JaxESMFold2
    esmc: ESMC
    res_type_perm: Float[Array, "20 33"]
    distogram_bins: Float[Array, "Bins"]
    # ESMC input-id for each mosaic-20 residue (LM re-encoding).
    esmc_token_of_aa: Int[Array, "20"]
    unk_input_id: int = eqx.field(static=True)

    def _check_msa_compat(self, chains: list[TargetChain]) -> None:
        # The Fast checkpoint has no MSA encoder; an MSA would put its
        # `profile` aggregate off-distribution, so refuse rather than
        # silently produce bad predictions.
        if self.esmf.msa_encoder is None and any(c.use_msa for c in chains):
            raise ValueError(
                "ESMFold2-Fast does not support MSAs (model.esmf.msa_encoder is "
                "None). Set use_msa=False on all chains, or use ESMFold2Full()."
            )

    def target_only_features(
        self,
        chains: list[TargetChain],
        *,
        design_char: str | None = None,
        design_geometry: str = "G",
    ):
        """Featurize a list of real protein chains.

        With ``design_char`` set, any residue equal to it (in any chain) is a
        *designed* position: the trunk gets a ``design_geometry`` ("G"/"A")
        backbone there and the frozen ESMC sees UNK; everything else is the real
        residue on every channel (full-atom geometry, real res_type, real LM
        token). This is the whole design contract — a binder is just a chain
        with designed positions in it.

        Returns ``(pack, chain_infos)``. The designed token positions (where the
        frozen ESMC sees UNK) are wherever ``design_char`` appears in the input
        sequences — the caller put them there, so it can recompute them.
        """
        self._check_msa_compat(chains)
        raw, chain_infos, design_positions = _featurize(
            chains, design_char=design_char, design_geometry=design_geometry,
        )
        pack = _build_pack(
            self.esmc, raw,
            design_positions=design_positions,
            unk_input_id=self.unk_input_id,
        )
        return pack, chain_infos

    def binder_features(
        self,
        binder_length: int,
        chains: list[TargetChain],
    ):
        """Design-time pack for a fully-designed binder (chain 0).

        Thin wrapper over ``target_only_features``: prepends an all-``X`` binder
        chain (UNK to the frozen ESMC — the design contract) at token positions
        ``[0, binder_length)``.

        The returned pack has ``refresh=True``: the forward refreshes the binder
        geometry + ESMC features in-loop from ``argmax(PSSM)``. That geometry
        refresh tight-packs each residue's atoms, so the binder is allocated at
        the 14-atom maximum (``"W"``); the placeholder geometry is overwritten
        before the trunk ever sees it, so its identity is immaterial and not a
        caller choice.

        For a partially-fixed binder (e.g. a VHH with a fixed framework) still
        use this function: make the whole binder a designed (all-``X``) chain and
        pin the framework with ``SetPositions`` during optimization. The in-loop
        geometry refresh means there's no need to bake the real framework into a
        ``target_only_features`` template, unlike the other backends.
        """
        pack, chain_infos = self.target_only_features(
            [TargetChain("X" * binder_length, use_msa=False), *chains],
            # "W" = the 14-atom (max) allocation the in-loop geom refresh needs.
            design_char="X", design_geometry="W",
        )
        # Tie geom + LM refresh on for the design pack.
        return dataclasses.replace(pack, refresh=True), chain_infos

    def build_loss(
        self,
        *,
        loss,
        features,
        recycling_steps: int = _DEFAULT_NUM_LOOPS,
        sampling_steps: int = _DEFAULT_NUM_SAMPLING_STEPS,
        msa_max_depth: int | None = 1024,
        lm_dropout: float = 0.0,
    ):
        """
        `lm_dropout`: per-step dropout on the LM→pair contribution. >0 sets the
        static `lm_dropout` field on the design trunk; the per-step optimizer
        key varies the mask. Always 0 at scoring time (the model's own trunk).

        The returned loss references `self.esmc` for the binder LM refresh. To
        share one ESMC across several losses + the PLL term, null it on the model
        first (`eqx.tree_at(lambda m: m.esmc, model, None)`) and splice a shared
        copy back in at call time — see `StackedEsmFold2PLL`.
        """
        esmf = self.esmf
        if lm_dropout:
            esmf = dataclasses.replace(esmf, lm_dropout=lm_dropout)
        return ESMFold2Loss(
            esmf=esmf,
            pack=features,
            loss=loss,
            res_type_perm=self.res_type_perm,
            distogram_bins=self.distogram_bins,
            num_loops=recycling_steps,
            num_sampling_steps=sampling_steps,
            msa_max_depth=msa_max_depth,
            geom_templates=_residue_atom_templates() if features.refresh else None,
            esmc=self.esmc,
            esmc_token_of_aa=self.esmc_token_of_aa,
            refresh_lm=features.refresh,
        )

    @eqx.filter_jit
    def model_output(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features,
        recycling_steps: int = _DEFAULT_NUM_LOOPS,
        sampling_steps: int = _DEFAULT_NUM_SAMPLING_STEPS,
        msa_max_depth: int | None = 1024,
        key,
    ):
        # Geometry + LM refresh are tied to the pack (`features.refresh`): on for
        # a `binder_features` design pack, off for a native pack.
        return forward_with_pssm(
            self.esmf, features, PSSM,
            self.res_type_perm,
            self.distogram_bins,
            key=key,
            num_loops=recycling_steps,
            num_sampling_steps=sampling_steps,
            msa_max_depth=msa_max_depth,
            geom_templates=_residue_atom_templates() if features.refresh else None,
            esmc=self.esmc,
            esmc_token_of_aa=self.esmc_token_of_aa,
            refresh_lm=features.refresh,
        )

    def predict(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features,
        writer=None,
        recycling_steps: int = _DEFAULT_NUM_LOOPS,
        sampling_steps: int = _DEFAULT_NUM_SAMPLING_STEPS,
        msa_max_depth: int | None = 1024,
        key,
    ) -> StructurePrediction:
        """Fold the (binder+)target complex and return scores + a structure.

        The forward follows the pack (`features.refresh`). A `binder_features`
        (design) pack runs the *design-time* forward — binder geometry + ESMC
        features refreshed in-loop from `argmax(PSSM)` — so the returned
        iPTM/pLDDT match what the design loss saw. A native `target_only_features`
        pack refreshes neither, for scoring a finished design from real
        sidechains / tight atom packing.

        Structure writing follows the same flag: a refresh pack's placeholder flat
        atom names/elements no longer describe the refreshed coordinates, so we
        write from the canonical atom37 view (`StructureModelOutput.to_structure`),
        which `refresh_binder_geometry` keeps layout-correct and whose identities
        come from `full_sequence` (the spliced PSSM). A native pack writes from its
        (tight) packing and needs `writer`.
        """
        out = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            msa_max_depth=msa_max_depth,
            key=key,
        )
        seq = PSSM if PSSM is not None else jnp.zeros((0, 20))
        iptm = -IPTMLoss()(seq, out, key=jax.random.key(0))[0]

        if features.refresh:
            if PSSM is None:
                # The refresh (geom + LM) is gated on `pssm is not None`, so with
                # no PSSM the forward leaves the W/atom14 placeholder geometry
                # (14-atom Trp, all-Ala identities) untouched — `to_structure`
                # would then write that garbage. A refresh pack is a design pack;
                # it always has a binder to splice, so require it explicitly.
                raise ValueError(
                    "predict() on a refresh (design) pack needs a PSSM; "
                    "PSSM=None leaves the placeholder binder geometry unrefreshed. "
                    "Use a native target_only_features pack for target-only folds."
                )
            # The in-loop geom refresh repacks the binder into the placeholder
            # (W/atom14) buffer, so the pack's flat atom names/elements no longer
            # match the coordinates. The atom37 view IS rebuilt to match (see
            # `refresh_binder_geometry`), and `to_structure` writes from it with
            # identities taken from `full_sequence` (the spliced PSSM) — so no
            # `chain_infos` relabeling is needed.
            st = out.to_structure()
        else:
            if writer is None:
                raise ValueError(
                    "predict() needs `writer` (the chain_infos from "
                    "target_only_features / binder_features) to write the native "
                    "structure; it's only optional for a refresh (design) pack."
                )
            pack: EsmFold2FeaturePack = features
            feats = pack.features
            mmcif_str = output_to_mmcif(
                coords=np.asarray(out.structure_coordinates),
                plddt=np.asarray(out.plddt),
                atom_mask=np.asarray(feats.atom_attention_mask[0]),
                ref_element=np.asarray(feats.ref_element[0]),
                ref_atom_name_chars=np.asarray(feats.ref_atom_name_chars[0]),
                chain_infos=writer,
            )
            st = gemmi.make_structure_from_block(
                gemmi.cif.read_string(mmcif_str).sole_block()
            )
            if PSSM is not None:
                # Residue names come from the placeholder `chain_infos` (e.g. all
                # ALA for an "A"*N binder), never from the PSSM — so relabel the
                # binder residues to the designed sequence (argmax of the PSSM).
                # Atom geometry stays at the placeholder (backbone + CB); only the
                # residue identities are corrected.
                binder_names = [
                    gemmi.expand_one_letter(TOKENS[int(k)], gemmi.ResidueKind.AA)
                    for k in np.argmax(np.asarray(PSSM), axis=-1)
                ]
                binder_chain = st[0][0]  # asym_id 0 → first chain → the binder
                for res, name in zip(binder_chain, binder_names):
                    res.name = name
        return StructurePrediction(
            st=st,
            plddt=out.plddt,
            pae=out.pae,
            iptm=iptm,
            model_output=out,
        )


def _probe_alphabets() -> tuple[np.ndarray, int, np.ndarray]:
    """Build the mosaic-20 → res_type-33 permutation matrix, UNK input id, and
    the ESMC input-id per mosaic-20 residue.

    The perm is read off `res_type` for a chain of the 20 canonical AAs in
    `TOKENS` order; the UNK id from a one-residue `X` chain; the ESMC ids from
    `input_ids` of that same 20-AA chain. The char→int mapping is
    checkpoint-specific, so probe it via the input builder.
    """
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )

    builder = ESMFold2InputBuilder()

    spi = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence=TOKENS)]
    )
    feats, _ = builder.prepare_input(spi)
    res_type = np.asarray(feats["res_type"])[0]
    assert res_type.shape == (len(TOKENS),), (
        f"expected res_type shape ({len(TOKENS)},), got {res_type.shape} "
        "— the input builder may have inserted extra tokens that break the "
        "1:1 mapping assumption"
    )
    res_type_perm = np.zeros((len(TOKENS), NUM_RES_TYPES), dtype=np.float32)
    for i in range(len(TOKENS)):
        res_type_perm[i, int(res_type[i])] = 1.0

    # ESMC input-id per residue (same 20-AA chain, in TOKENS order).
    esmc_token_of_aa = np.asarray(feats["input_ids"])[0].astype(np.int32)
    assert esmc_token_of_aa.shape == (len(TOKENS),)

    spi_unk = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence="X")]
    )
    feats_unk, _ = builder.prepare_input(spi_unk)
    unk_input_id = int(np.asarray(feats_unk["input_ids"])[0, 0])

    return res_type_perm, unk_input_id, esmc_token_of_aa


def _staticize_scalar_config(tree):
    """Coerce 0-d array leaves (jax or numpy) to python scalars.

    esmjfold2 / from_torch store configuration as 0-d arrays: TriangleMultiplication
    `flow` ("incoming"/"outgoing" — a string array), ESMC `n_heads` / `d_head`, the
    diffusion noise-schedule sigmas, layernorm eps, etc. `eqx.is_array` reports these
    as arrays, so `eqx.filter_jit` (the optimizer, `predict`) traces them as dynamic.
    That breaks string leaves outright and any scalar used as a *static* value —
    reshape dims (`n_heads`) and the numpy-built diffusion schedule both require
    concrete Python ints/floats. The model is frozen during design (gradients flow to
    the sequence, never the model), so making these scalars static is safe and is what
    the library's own code already assumes. Multi-element arrays (weights) are
    untouched.
    """
    def conv(x):
        if isinstance(x, (np.ndarray, jax.Array)) and getattr(x, "ndim", 1) == 0:
            return x.item()
        return x

    return jax.tree_util.tree_map(conv, tree)


def _make(checkpoint: str, *, experimental: bool = False) -> ESMFold2:
    esmf, esmc = _load_pretrained(checkpoint, experimental=experimental)
    esmf = _staticize_scalar_config(esmf)
    esmc = _staticize_scalar_config(esmc)
    res_type_perm, unk_input_id, esmc_token_of_aa = _probe_alphabets()
    # Distogram range (release 2.0–22.0 Å; experimental 2.0–52.0 Å, matching
    # its confidence head).
    min_dist, max_dist = (2.0, 52.0) if experimental else (2.0, 22.0)
    n_bins = int(esmf.distogram_head.weight.shape[0])
    return ESMFold2(
        esmf=esmf,
        esmc=esmc,
        res_type_perm=jnp.asarray(res_type_perm),
        distogram_bins=distogram_bin_centers(n_bins, min_dist, max_dist),
        esmc_token_of_aa=jnp.asarray(esmc_token_of_aa),
        unk_input_id=unk_input_id,
    )


def ESMFold2Fast() -> ESMFold2:
    """Single-sequence ESMFold2 (no MSA encoder). Default for binder design.

    Recycling (`recycling_steps`) and diffusion (`sampling_steps`) counts are
    per-call knobs on `build_loss` / `predict`, defaulting to the module globals
    `_DEFAULT_NUM_LOOPS` / `_DEFAULT_NUM_SAMPLING_STEPS`.
    """
    return _make("biohub/ESMFold2-Fast")


def ESMFold2Full() -> ESMFold2:
    """Full ESMFold2 with MSA encoder. Heavier; intended for target structure
    prediction with real MSAs. (Pass larger `recycling_steps` / `sampling_steps`
    to `predict` for full-quality inference.)
    """
    return _make("biohub/ESMFold2")


def ESMFold2ExperimentalFast() -> ESMFold2:
    """Single-sequence ESMFold2-Experimental-Fast.

    Closer to a standard AF3-style model than the release ``Fast``: no parcae
    refinement state, LM contribution added once to ``z_init`` outside the
    loop, slimmer confidence head (no PDE / no resolved). No MSA encoder.
    """
    return _make("biohub/ESMFold2-Experimental-Fast", experimental=True)

def ESMFold2ExperimentalFast2025() -> ESMFold2:
    """Single-sequence ESMFold2-Experimental-Fast.

    Closer to a standard AF3-style model than the release ``Fast``: no parcae
    refinement state, LM contribution added once to ``z_init`` outside the
    loop, slimmer confidence head (no PDE / no resolved). No MSA encoder.
    """
    return _make("biohub/ESMFold2-Experimental-Fast-Cutoff2025", experimental=True)


def ESMFold2ExperimentalFastBase(
    size: str = "600M", steps: str = "1500k"
) -> ESMFold2:
    """ESMFold2-Experimental-Fast ``base{size}`` training-step sweep.

    ``size`` ∈ {``"300M"``, ``"600M"``, ``"6B"``}; ``steps`` selects the
    training-step checkpoint tag ∈ {``"250k"``, ``"500k"``, ``"750k"``,
    ``"1000k"``, ``"1500k"``}. The 300M / 600M variants are lighter single-
    sequence design models — small enough to sit alongside a second structure
    model (e.g. ``ProtenixV2``) in one kernel without OOM.

    Note: these checkpoints' confidence heads are non-functional
    (``model_output`` returns pLDDT / PAE as 0), so re-fold designs with a full
    model such as ``ESMFold2ExperimentalFast2025`` for held-out iPTM validation.
    """
    return _make(
        f"biohub/ESMFold2-Experimental-Fast-base{size}-step{steps}",
        experimental=True,
    )


def ESMFold2ExperimentalFast300M(steps: str = "1500k") -> ESMFold2:
    """300M ESMFold2-Experimental-Fast design model. See ``ESMFold2ExperimentalFastBase``."""
    return ESMFold2ExperimentalFastBase("300M", steps)


def ESMFold2ExperimentalFast600M(steps: str = "1500k") -> ESMFold2:
    """600M ESMFold2-Experimental-Fast design model. See ``ESMFold2ExperimentalFastBase``."""
    return ESMFold2ExperimentalFastBase("600M", steps)


def ESMFold2ExperimentalFast6B(steps: str = "1500k") -> ESMFold2:
    """6B ESMFold2-Experimental-Fast design model (large). See ``ESMFold2ExperimentalFastBase``."""
    return ESMFold2ExperimentalFastBase("6B", steps)


def ESMFold2ExperimentalFull() -> ESMFold2:
    """ESMFold2-Experimental with MSA encoder. AF3-style architecture; intended
    for target structure prediction with real MSAs.
    """
    return _make("biohub/ESMFold2-Experimental", experimental=True)

def ESMFold2ExperimentalFull2025() -> ESMFold2:
    """ESMFold2-Experimental with MSA encoder. AF3-style architecture; intended
    for target structure prediction with real MSAs.
    """
    return _make("biohub/ESMFold2-Experimental-Cutoff2025", experimental=True)



