"""ESMFold2 backend for mosaic structure prediction.

Factories come in two families, each in a `Fast` (no MSA encoder, default for
design) and `Full` (MSA-conditioned, for target prediction) variant:

- Release: `ESMFold2Fast()`, `ESMFold2Full()`.
- Experimental (AF3-style: LM added once to `z_init` outside the trunk loop,
  slimmer confidence head): `ESMFold2ExperimentalFast()`,
  `ESMFold2ExperimentalFull()`, plus `...2025()` variants on the 2025-cutoff
  checkpoints.

All expose mosaic's standard `StructurePredictionModel` interface; recycling
(`recycling_steps`) and diffusion (`sampling_steps`) counts are per-call knobs
on `build_loss` / `predict`, defaulting to the module globals
`_DEFAULT_NUM_LOOPS` / `_DEFAULT_NUM_SAMPLING_STEPS`.

See `mosaic.losses.esmfold2` for the soft-sequence ESMC plumbing this
wrapper relies on.
"""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

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
    # and as the target_only reprediction sees it). Leaving it empty matched/
    # beat the previously re-stamped peptide-bond version in design tests.

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
        *,
        placeholder_char: str = "A",
    ):
        """Design-time pack for a fully-designed binder (chain 0).

        Thin wrapper over ``target_only_features``: prepends an all-``X`` binder
        chain, so every binder position gets ``placeholder_char`` backbone
        geometry + UNK to the frozen ESMC (the design contract), and the binder
        lands at token positions ``[0, binder_length)``. For a partially-fixed
        binder (e.g. a VHH with only CDR ``X`` positions) call
        ``target_only_features([TargetChain(template_with_X), *chains],
        design_char="X")`` directly to keep the real framework.
        """
        pack, chain_infos = self.target_only_features(
            [TargetChain("X" * binder_length, use_msa=False), *chains],
            design_char="X", design_geometry=placeholder_char,
        )
        return pack, chain_infos

    def build_loss(
        self,
        *,
        loss,
        features,
        recycling_steps: int = _DEFAULT_NUM_LOOPS,
        sampling_steps: int = _DEFAULT_NUM_SAMPLING_STEPS,
        msa_max_depth: int | None = 1024,
        stop_recycling_grad: bool = False,
    ):
        return ESMFold2Loss(
            esmf=self.esmf,
            pack=features,
            loss=loss,
            res_type_perm=self.res_type_perm,
            distogram_bins=self.distogram_bins,
            num_loops=recycling_steps,
            num_sampling_steps=sampling_steps,
            msa_max_depth=msa_max_depth,
            stop_recycling_grad=stop_recycling_grad,
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
        return forward_with_pssm(
            self.esmf, features, PSSM,
            self.res_type_perm,
            self.distogram_bins,
            key=key,
            num_loops=recycling_steps,
            num_sampling_steps=sampling_steps,
            msa_max_depth=msa_max_depth,
        )

    def predict(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features,
        writer,
        recycling_steps: int = _DEFAULT_NUM_LOOPS,
        sampling_steps: int = _DEFAULT_NUM_SAMPLING_STEPS,
        msa_max_depth: int | None = 1024,
        key,
    ) -> StructurePrediction:
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


def _probe_alphabets() -> tuple[np.ndarray, int]:
    """Build the mosaic-20 → res_type-33 permutation matrix and UNK input id.

    The perm is read off `res_type` for a chain of the 20 canonical AAs in
    `TOKENS` order; the UNK id from a one-residue `X` chain. The char→int
    mapping is checkpoint-specific, so probe it via the input builder.
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

    spi_unk = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence="X")]
    )
    feats_unk, _ = builder.prepare_input(spi_unk)
    unk_input_id = int(np.asarray(feats_unk["input_ids"])[0, 0])

    return res_type_perm, unk_input_id


def _make(checkpoint: str, *, experimental: bool = False) -> ESMFold2:
    esmf, esmc = _load_pretrained(checkpoint, experimental=experimental)
    res_type_perm, unk_input_id = _probe_alphabets()
    # Distogram range, verified empirically on 1UBQ (release 2.0–22.0 Å;
    # experimental 2.0–52.0 Å, matching its confidence head).
    min_dist, max_dist = (2.0, 52.0) if experimental else (2.0, 22.0)
    n_bins = int(esmf.distogram_head.weight.shape[0])
    return ESMFold2(
        esmf=esmf,
        esmc=esmc,
        res_type_perm=jnp.asarray(res_type_perm),
        distogram_bins=distogram_bin_centers(n_bins, min_dist, max_dist),
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



