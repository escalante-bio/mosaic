"""Mosaic StructurePredictionModel wrapper for OpenFold3.
- OF3(): mosaic StructurePredictionModel implementation
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array, Float
from jopenfold3._vendor.openfold3.core.data.framework.single_datasets.inference import (
    InferenceDataset,
)
from jopenfold3._vendor.openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessorSettings,
)
from jopenfold3._vendor.openfold3.core.data.resources.residues import (
    RESTYPE_INDEX_3,
    STANDARD_RESIDUES_WITH_GAP_3,
    MoleculeType,
)
from jopenfold3._vendor.openfold3.core.data.tools.colabfold_msa_server import (
    MsaComputationSettings,
    augment_main_msa_with_query_sequence,
    preprocess_colabfold_msas,
)
from jopenfold3._vendor.openfold3.projects.of3_all_atom.config.dataset_config_components import (
    MSASettings,
    TemplateSettings,
)
from jopenfold3._vendor.openfold3.projects.of3_all_atom.config.dataset_configs import (
    InferenceJobConfig,
)
from jopenfold3._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
    Chain,
    InferenceQuerySet,
    Query,
)
from jopenfold3.batch import Batch
from jopenfold3.model import OpenFold3
from mosaic.losses.of3 import (
    MultiSampleOF3Loss,
    OF3FromTrunkOutput,
    set_binder_sequence,
)

from mosaic.common import LinearCombination, LossTerm
from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.structure_prediction import (
    PolymerType,
    StructurePrediction,
    StructurePredictionModel,
    TargetChain,
)

# ---------------------------------------------------------------------------
# Template feature computation from gemmi.Chain
# ---------------------------------------------------------------------------


def _extract_residue_atoms(
    chain,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract backbone and pseudo-beta atom coords from a gemmi.Chain.

    Returns:
        frame_coords: [N_res, 3, 3] — N, CA, C coordinates (NaN if missing)
        pseudo_beta_coords: [N_res, 3] — CB (or CA for GLY) (NaN if missing)
        mask arrays are derived from NaN presence
        res_names: list of 3-letter residue names
    """
    residues = [r for r in chain if r.entity_type.name == "Polymer" or True]
    n_res = len(residues)

    frame_coords = np.full((n_res, 3, 3), np.nan, dtype=np.float32)
    pseudo_beta_coords = np.full((n_res, 3), np.nan, dtype=np.float32)
    res_names = []

    for i, res in enumerate(residues):
        res_names.append(res.name)
        atoms = {a.name: np.array([a.pos.x, a.pos.y, a.pos.z]) for a in res}
        if "N" in atoms:
            frame_coords[i, 0] = atoms["N"]
        if "CA" in atoms:
            frame_coords[i, 1] = atoms["CA"]
        if "C" in atoms:
            frame_coords[i, 2] = atoms["C"]
        # Pseudo-beta: CB for non-glycine, CA for glycine
        if res.name == "GLY":
            if "CA" in atoms:
                pseudo_beta_coords[i] = atoms["CA"]
        else:
            if "CB" in atoms:
                pseudo_beta_coords[i] = atoms["CB"]

    return frame_coords, pseudo_beta_coords, res_names


def _make_frames(frame_coords: np.ndarray) -> np.ndarray:
    """Build local coordinate frames from N, CA, C positions.

    Uses Gram-Schmidt: e0 = normalize(C - CA), e1 = GS(N - CA, e0), e2 = e0 x e1.
    Matches AF3's make_transform_from_reference(a=N, b=CA, c=C) which does
    Rot3Array.from_two_vectors(C - CA, N - CA).

    Args:
        frame_coords: [N, 3, 3] — (N, CA, C) per residue

    Returns:
        rotations: [N, 3, 3] — rotation matrices (columns are e0, e1, e2)
        translations: [N, 3] — CA positions
    """
    n_pos = frame_coords[:, 0]  # N
    ca_pos = frame_coords[:, 1]  # CA
    c_pos = frame_coords[:, 2]  # C

    # e0 = normalize(C - CA)
    e0 = c_pos - ca_pos
    e0 = e0 / (np.linalg.norm(e0, axis=-1, keepdims=True) + 1e-8)

    # e1 = GS orthogonalize (N - CA) against e0, then normalize
    e1 = n_pos - ca_pos
    e1 = e1 - np.sum(e1 * e0, axis=-1, keepdims=True) * e0
    e1 = e1 / (np.linalg.norm(e1, axis=-1, keepdims=True) + 1e-8)

    # e2 = e0 x e1
    e2 = np.cross(e0, e1)

    # Rotation matrix: rows are e0, e1, e2 (to match from_two_vectors convention)
    rotations = np.stack([e0, e1, e2], axis=-2)  # [N, 3, 3]
    return rotations, ca_pos


def _compute_template_unit_vectors(
    frame_coords: np.ndarray,
    backbone_frame_mask: np.ndarray,
) -> np.ndarray:
    """Compute unit vectors between residue pairs in local frames.

    Args:
        frame_coords: [N, 3, 3] — N, CA, C per residue
        backbone_frame_mask: [N] — 1.0 if all backbone atoms present

    Returns:
        unit_vectors: [N, N, 3]
    """
    # Replace NaN with 0 for computation
    frame_coords = np.nan_to_num(frame_coords, nan=0.0)
    rotations, translations = _make_frames(frame_coords)  # [N,3,3], [N,3]

    # For each pair (i, j): R_i^T @ (CA_j - CA_i), normalized
    # delta[i, j] = CA_j - CA_i
    delta = translations[None, :, :] - translations[:, None, :]  # [N, N, 3]

    # Rotate into local frame: R_i @ delta[i,j] (R has rows as basis vectors)
    local = np.einsum("ijk,ilk->ilj", rotations, delta)  # [N, N, 3]

    # Normalize
    norms = np.linalg.norm(local, axis=-1, keepdims=True) + 1e-8
    unit_vectors = local / norms

    # Apply 2D mask
    mask_2d = backbone_frame_mask[:, None] * backbone_frame_mask[None, :]
    return unit_vectors * mask_2d[..., None]


def _compute_template_distogram(
    pseudo_beta_coords: np.ndarray,
    pseudo_beta_mask: np.ndarray,
    min_bin: float = 3.25,
    max_bin: float = 50.75,
    n_bins: int = 39,
    inf_value: float = 1e8,
) -> np.ndarray:
    """Compute binned pairwise distances between pseudo-beta atoms.

    Args:
        pseudo_beta_coords: [N, 3]
        pseudo_beta_mask: [N]

    Returns:
        distogram: [N, N, 39]
    """
    coords = np.nan_to_num(pseudo_beta_coords, nan=0.0)
    # Squared pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]  # [N, N, 3]
    sq_dist = np.sum(diff**2, axis=-1, keepdims=True)  # [N, N, 1]

    # Bin edges (squared)
    lower = np.linspace(min_bin, max_bin, n_bins) ** 2
    upper = np.concatenate([lower[1:], np.array([inf_value])])

    distogram = ((sq_dist > lower) & (sq_dist < upper)).astype(np.float32)

    # Apply mask
    mask_2d = pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]
    return distogram * mask_2d[..., None]


def _compute_template_restype(res_names: list[str], n_tokens: int) -> np.ndarray:
    """One-hot encode residue types into OF3's 32-dim vocabulary.

    Returns: [n_tokens, 32] with GAP for positions beyond the template.
    """

    n_classes = len(STANDARD_RESIDUES_WITH_GAP_3)  # 32
    gap_idx = RESTYPE_INDEX_3["GAP"]
    unk_idx = RESTYPE_INDEX_3["UNK"]

    restype = np.zeros((n_tokens, n_classes), dtype=np.int32)
    for i in range(n_tokens):
        if i < len(res_names):
            idx = RESTYPE_INDEX_3.get(res_names[i], unk_idx)
        else:
            idx = gap_idx
        restype[i, idx] = 1

    return restype


def compute_template_features(
    template_chain,
    query_sequence: str,
    n_tokens: int,
    token_offset: int,
    n_templates: int = 4,
) -> dict[str, np.ndarray]:
    """Compute OF3 template features from a gemmi.Chain.

    Produces features for a single template, placed at template slot 0.
    Remaining slots are zero-padded. The template features are placed at
    token positions [token_offset : token_offset + len(query_sequence)].

    Args:
        template_chain: gemmi.Chain with the template structure
        query_sequence: sequence of the chain this template is for
        n_tokens: total number of tokens in the batch
        token_offset: first token position for this chain
        n_templates: number of template slots (default 4)

    Returns:
        Dict with template_distogram, template_unit_vector, template_restype,
        template_backbone_frame_mask, template_pseudo_beta_mask arrays,
        each with a leading [n_templates] dimension.
    """
    frame_coords, pseudo_beta_coords, res_names = _extract_residue_atoms(template_chain)
    n_res = len(res_names)

    # Masks: valid where not NaN
    backbone_frame_mask = (~np.isnan(frame_coords).any(axis=(-2, -1))).astype(
        np.float32
    )
    pseudo_beta_mask = (~np.isnan(pseudo_beta_coords).any(axis=-1)).astype(np.float32)

    # Compute per-chain features
    distogram = _compute_template_distogram(pseudo_beta_coords, pseudo_beta_mask)
    unit_vector = _compute_template_unit_vectors(frame_coords, backbone_frame_mask)
    restype = _compute_template_restype(res_names, n_res)

    # Place into full [n_tokens, ...] arrays at the right offset
    full_distogram = np.zeros((n_tokens, n_tokens, 39), dtype=np.float32)
    full_unit_vector = np.zeros((n_tokens, n_tokens, 3), dtype=np.float32)
    full_restype = np.zeros((n_tokens, 32), dtype=np.int32)
    full_backbone_mask = np.zeros(n_tokens, dtype=np.float32)
    full_pseudo_beta_mask = np.zeros(n_tokens, dtype=np.float32)

    # Fill GAP for all positions in restype

    gap_idx = RESTYPE_INDEX_3["GAP"]
    full_restype[:, gap_idx] = 1

    # Insert template features at the right positions
    end = min(token_offset + n_res, n_tokens)
    src_end = end - token_offset
    s = token_offset

    full_distogram[s:end, s:end, :] = distogram[:src_end, :src_end, :]
    full_unit_vector[s:end, s:end, :] = unit_vector[:src_end, :src_end, :]
    full_restype[s:end, :] = restype[:src_end, :]
    full_backbone_mask[s:end] = backbone_frame_mask[:src_end]
    full_pseudo_beta_mask[s:end] = pseudo_beta_mask[:src_end]

    # Expand to [n_templates, ...], only slot 0 is filled
    def pad_templates(arr):
        shape = (n_templates,) + arr.shape
        out = np.zeros(shape, dtype=arr.dtype)
        out[0] = arr
        return out

    return {
        "template_distogram": pad_templates(full_distogram),
        "template_unit_vector": pad_templates(full_unit_vector),
        "template_restype": pad_templates(full_restype),
        "template_backbone_frame_mask": pad_templates(full_backbone_mask),
        "template_pseudo_beta_mask": pad_templates(full_pseudo_beta_mask),
    }


# ---------------------------------------------------------------------------
# Featurization helpers
# ---------------------------------------------------------------------------


def _build_query_set(chains: list[TargetChain], seed: int = 0):
    """Build an InferenceQuerySet from mosaic TargetChain objects."""

    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    of3_chains = []
    for i, tc in enumerate(chains):
        cid = chain_ids[i] if i < len(chain_ids) else f"chain{i}"
        mol_type = {
            PolymerType.PROTEIN: MoleculeType.PROTEIN,
            PolymerType.RNA: MoleculeType.RNA,
            PolymerType.DNA: MoleculeType.DNA,
        }[tc.polymer_type]

        of3_chains.append(
            Chain(
                molecule_type=mol_type,
                chain_ids=[cid],
                description=f"chain_{i}",
                sequence=tc.sequence,
            )
        )

    query = Query(query_name="design", chains=of3_chains)
    return InferenceQuerySet(seeds=[seed], queries={"design": query})


def _compute_msas(query_set, chains: list[TargetChain], msa_dir: Path):
    """Compute MSAs: ColabFold for chains with use_msa=True, dummy for the rest.

    To avoid submitting binder/placeholder chains to ColabFold, we build a
    separate query set with only the MSA-needing chains, run ColabFold on
    that, then copy the resulting MSA paths back to the full query set.
    """

    msa_dir.mkdir(parents=True, exist_ok=True)
    settings = MsaComputationSettings(msa_output_directory=msa_dir)

    msa_chains = [tc for tc in chains if tc.use_msa]
    if not msa_chains:
        return augment_main_msa_with_query_sequence(query_set, settings)

    # Build a subset query set with only MSA-needing chains
    msa_query_set = _build_query_set(msa_chains)
    msa_query_set = preprocess_colabfold_msas(msa_query_set, settings)

    # Map ColabFold results back: match by sequence
    msa_by_seq = {}
    for q in msa_query_set.queries.values():
        for c in q.chains:
            if c.main_msa_file_paths is not None:
                msa_by_seq[c.sequence] = c.main_msa_file_paths

    for q in query_set.queries.values():
        for c in q.chains:
            if c.sequence in msa_by_seq:
                c.main_msa_file_paths = msa_by_seq[c.sequence]

    # Fill remaining chains with dummy MSAs
    return augment_main_msa_with_query_sequence(query_set, settings)


def _featurize(query_set, seed: int = 0) -> tuple[dict, object]:
    """Run PyTorch featurization pipeline, return (features_dict, atom_array)."""

    config = InferenceJobConfig(
        query_set=query_set,
        seeds=[seed],
        msa=MSASettings(subsample_main=False),
        template=TemplateSettings(take_top_k=True),
        template_preprocessor_settings=TemplatePreprocessorSettings(),
    )
    dataset = InferenceDataset(config)
    features = dataset[0]

    if not features.get("valid_sample", torch.tensor([False])).item():
        raise RuntimeError("OF3 featurization failed (valid_sample=False)")

    atom_array = features.pop("atom_array")
    for key in ["query_id", "seed", "repeated_sample", "valid_sample"]:
        features.pop(key, None)

    return features, atom_array


def _inject_templates(batch: Batch, chains: list[TargetChain]) -> Batch:
    """Inject template features computed from gemmi.Chain objects into a Batch.

    For each chain with a template_chain, computes template features and
    overwrites the corresponding fields in the batch.
    """
    n_tokens = batch.token_mask.shape[1]
    template_chains = [
        (i, tc) for i, tc in enumerate(chains) if tc.template_chain is not None
    ]
    if not template_chains:
        return batch

    # Figure out token offsets per chain from sequence lengths
    offsets = []
    offset = 0
    for tc in chains:
        offsets.append(offset)
        offset += len(tc.sequence)

    # Compute and merge template features
    # Start from the existing (zero) template features
    merged = {
        "template_distogram": np.array(batch.template_distogram[0]),
        "template_unit_vector": np.array(batch.template_unit_vector[0]),
        "template_restype": np.array(batch.template_restype[0]),
        "template_backbone_frame_mask": np.array(batch.template_backbone_frame_mask[0]),
        "template_pseudo_beta_mask": np.array(batch.template_pseudo_beta_mask[0]),
    }

    for chain_idx, tc in template_chains:
        feats = compute_template_features(
            template_chain=tc.template_chain,
            query_sequence=tc.sequence,
            n_tokens=n_tokens,
            token_offset=offsets[chain_idx],
            n_templates=merged["template_distogram"].shape[0],
        )
        # Add features (they don't overlap — each chain writes to different token positions)
        for key in feats:
            merged[key] = merged[key] + feats[key]

    # Inject into batch (add back batch dim)
    return eqx.tree_at(
        lambda b: (
            b.template_distogram,
            b.template_unit_vector,
            b.template_restype,
            b.template_backbone_frame_mask,
            b.template_pseudo_beta_mask,
        ),
        batch,
        (
            jnp.array(merged["template_distogram"])[None],
            jnp.array(merged["template_unit_vector"])[None],
            jnp.array(merged["template_restype"])[None],
            jnp.array(merged["template_backbone_frame_mask"])[None],
            jnp.array(merged["template_pseudo_beta_mask"])[None],
        ),
    )


# ---------------------------------------------------------------------------
# biotite → gemmi conversion
# ---------------------------------------------------------------------------


def _biotite_array_to_gemmi_struct(atom_array, pred_coord=None):
    """Convert a biotite AtomArray to a gemmi.Structure."""

    if pred_coord is not None:
        atom_array = copy.deepcopy(atom_array)
        atom_array.coord = pred_coord

    structure = gemmi.Structure()
    model = gemmi.Model("0")
    chains: dict = {}
    for atom in atom_array:
        chain = chains.setdefault(atom.chain_id, {})
        if int(atom.res_id) not in chain:
            r = gemmi.Residue()
            r.name = atom.res_name
            r.seqid = gemmi.SeqId(atom.res_id, " ")
            r.entity_type = gemmi.EntityType.Polymer
            chain[int(atom.res_id)] = r
        residue = chain[int(atom.res_id)]
        ga = gemmi.Atom()
        ga.pos = gemmi.Position(*atom.coord)
        ga.element = gemmi.Element(atom.element)
        ga.name = atom.atom_name
        residue.add_atom(ga)

    for k in chains:
        c = gemmi.Chain(k)
        c.append_residues(list(chains[k].values()))
        model.add_chain(c)
    structure.add_model(model)
    return structure


class OF3(StructurePredictionModel):
    """Mosaic-compatible wrapper around the OpenFold3 JAX model."""

    model: OpenFold3
    default_sampling_steps: int = 20
    default_num_samples: int = 1

    def __init__(self, default_sampling_steps: int = 20, default_num_samples: int = 1):
        # load the model...

        jax_model = OpenFold3.load()
        # Switch to vanilla ODE sampler
        jax_model = eqx.tree_at(
            lambda m: (
                m.sample_diffusion.gamma_0,
                m.sample_diffusion.step_scale,
                m.sample_diffusion.noise_scale,
            ),
            jax_model,
            (0.0, 1.0, 1.0),
        )

        self.model = jax_model
        self.default_sampling_steps = default_sampling_steps
        self.default_num_samples = default_num_samples

    def target_only_features(self, chains: list[TargetChain]) -> tuple[Batch, object]:
        """Featurize target chains via the PyTorch upstream pipeline.

        Returns ``(batch, atom_array)`` where ``atom_array`` is the writer
        object for producing CIF/PDB output.

        If any chain has a ``template_chain`` (gemmi.Chain), template
        features are computed directly from the structure and injected
        into the Batch — bypassing OF3's file-based template pipeline.
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix="of3_mosaic_"))
        msa_dir = tmp_dir / "msas"

        query_set = _build_query_set(chains)
        query_set = _compute_msas(query_set, chains=chains, msa_dir=msa_dir)
        features, atom_array = _featurize(query_set)
        batch = Batch.from_torch_dict(features)

        # Inject template features if any chain has a template_chain
        batch = _inject_templates(batch, chains)

        return batch, atom_array

    def binder_features(
        self, binder_length: int, chains: list[TargetChain]
    ) -> tuple[Batch, object]:
        """Featurize a binder + target chains.

        Prepends a poly-X (UNK) placeholder binder of the given length.
        UNK is uninformative (the model was trained to treat it as unknown),
        so the atom features don't contradict the designed restype during
        optimization.
        """
        binder = TargetChain(sequence="X" * binder_length, use_msa=False)
        return self.target_only_features([binder] + list(chains))

    def build_loss(
        self,
        *,
        loss: LossTerm | LinearCombination,
        features: Batch,
        recycling_steps: int = 3,
        sampling_steps: int | None = None,
    ) -> LossTerm:
        return self.build_multisample_loss(
            loss=loss,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            num_samples=1,
        )

    def build_multisample_loss(
        self,
        *,
        loss: LossTerm | LinearCombination,
        features: Batch,
        recycling_steps: int = 3,
        sampling_steps: int | None = None,
        num_samples: int = 4,
        reduction=jnp.mean,
    ) -> MultiSampleOF3Loss:
        if sampling_steps is None:
            sampling_steps = self.default_sampling_steps
        return MultiSampleOF3Loss(
            model=self.model,
            batch=features,
            loss=loss,
            num_cycles=recycling_steps + 1,
            sampling_steps=sampling_steps,
            num_samples=num_samples,
            reduction=reduction,
        )


    def model_output(self, *, PSSM = None, features, recycling_steps = 1, sampling_steps = None, key):
        if sampling_steps is None:
            sampling_steps = self.default_sampling_steps
        if PSSM is not None:
            batch = set_binder_sequence(PSSM, features)
        else:
            batch = features
        init_emb, trunk_emb = self.model.run_trunk(
            batch,
            recycling_steps + 1,
            key=key,
        )
        return OF3FromTrunkOutput(
            model=self.model,
            batch=batch,
            init_emb=init_emb,
            trunk_emb=trunk_emb,
            sampling_steps=sampling_steps,
            key=key,
        )

    @eqx.filter_jit
    def _coords_and_confidences(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: Batch,
        recycling_steps: int = 3,
        sampling_steps: int | None = None,
        key,
    ):
        
        output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            key=key,
        )
        coords = output.structure_coordinates[0, 0]
        seq = PSSM if PSSM is not None else jnp.zeros((0, 20))
        iptm = -IPTMLoss()(seq, output, key=jax.random.key(0))[0]
        return (coords, output.plddt, output.pae, iptm)

    def predict(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: Batch,
        writer,
        recycling_steps: int = 3,
        sampling_steps: int | None = None,
        key,
    ) -> StructurePrediction:

        coords, plddt, pae, iptm = jax.tree.map(
            np.array,
            self._coords_and_confidences(
                PSSM=PSSM,
                features=features,
                recycling_steps=recycling_steps,
                sampling_steps=sampling_steps,
                key=key,
            ),
        )

        return StructurePrediction(
            st=_biotite_array_to_gemmi_struct(writer, coords),
            plddt=plddt,
            pae=pae,
            iptm=iptm,
        )
