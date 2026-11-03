
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import boltz.data.const as const
import boltz.main as boltz_main
import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import joltz
import numpy as np
import torch
from boltz.data.const import ref_atoms
from boltz.model.models.boltz2 import Boltz2 as TorchBoltz2
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from joltz import InitialEmbedding
from joltz import TrunkState as JoltzTrunkState

from mosaic.cache import resolve_cache
from mosaic.losses.atom37 import ATOM37_INDEX, scatter_atom37
from mosaic.losses.structure_prediction import PAE_BINS, StructureModelOutput
from mosaic.sampling import Denoise, Sampler, schedules
from mosaic.sampling.core import mean_centering, no_alignment, rigid_align_stop_gradient
from mosaic.sampling.schedules import Schedule
from mosaic.structure_prediction import (
    Atom37,
    Confidence,
    DiffusionStructurePredictionModel,
    Distogram,
    TargetChain,
    TokenInfo,
    TrunkState,
)


def _denoise(structure_module: joltz.AtomDiffusion2, conditioning: PyTree, x, sigma):
    with jax.default_matmul_precision("float32"):
        # `key` is unused: the score model has no dropout.
        return structure_module.preconditioned_network_forward(
            x[None], sigma, conditioning, key=None
        )[0]


def boltz2_denoiser(
    model: joltz.Joltz2, features: PyTree, initial_embedding, trunk_state
) -> tuple[Denoise, Float[Array, " A"]]:
    """Bind the trunk outputs (`Boltz2.embed` and `Boltz2.step`).

    Returns the denoiser and the atom mask.
    """
    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
        model.diffusion_conditioning(
            trunk_state.s,
            trunk_state.z,
            initial_embedding.relative_position_encoding,
            features,
        )
    )
    conditioning = {
        "s_trunk": trunk_state.s,
        "s_inputs": initial_embedding.s_inputs,
        "feats": features,
        "multiplicity": 1,
        # Step-invariant, hoisted out of the loop.
        "diffusion_conditioning": {
            "q": q,
            "c": c,
            "to_keys": to_keys,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        },
    }
    denoise = eqx.Partial(_denoise, model.structure_module, conditioning)
    return denoise, features["atom_pad_mask"][0]


def boltz2_sampler(
    structure_module: joltz.AtomDiffusion2,
    num_steps: int,
    *,
    schedule: schedules.Schedule | None = None,
) -> Sampler:
    """The `Sampler` reproducing `AtomDiffusion2.sample`.

    `schedule` defaults to the model's own `Karras`."""
    m = structure_module
    if schedule is None:
        schedule = schedules.Karras(rho=m.rho)
    sigmas = schedules.append_zero(
        schedule(
            np.linspace(0.0, 1.0, num_steps),
            sigma_data=m.sigma_data,
            s_max=m.sigma_max,
            s_min=m.sigma_min,
        )
    )
    return Sampler(
        sigmas=sigmas,
        gamma_0=m.gamma_0,
        gamma_min=m.gamma_min,
        noise_scale=m.noise_scale,
        step_scale=m.step_scale,
        center=mean_centering,
        align=rigid_align_stop_gradient if m.alignment_reverse_diff else no_alignment,
    )


def _build_boltz_atom37_table() -> np.ndarray:
    """Boltz `res_type` slot → atom37 lookup `[N_slots, max_tokatom]`.

    Slots 2..21 of `res_type` are the 20 standard amino acids in mosaic
    order (`ref_atoms[3letter]` gives per-residue atom names). Returns an
    int table where -1 means "atom not in atom37 layout for this residue".
    """
    n_slots = len(const.tokens)
    aa_3letter = const.tokens[2:22]  # 'ALA' .. 'VAL'
    max_tokatom = max(len(ref_atoms[r]) for r in aa_3letter)
    table = -np.ones((n_slots, max_tokatom), dtype=np.int32)
    for slot, three_letter in enumerate(const.tokens):
        if three_letter not in ref_atoms:
            continue
        for tokatom_idx, atom_name in enumerate(ref_atoms[three_letter]):
            if atom_name in ATOM37_INDEX:
                table[slot, tokatom_idx] = ATOM37_INDEX[atom_name]
    return table


_BOLTZ_TOKATOM_TO_ATOM37 = _build_boltz_atom37_table()


def load_boltz2(checkpoint_path: Path | None = None):
    checkpoint_path = resolve_cache(checkpoint_path, "boltz", "boltz2_conf.ckpt")
    if not checkpoint_path.exists():
        print(f"Downloading Boltz checkpoint to {checkpoint_path}")
        cache = checkpoint_path.parent
        cache.mkdir(parents=True, exist_ok=True)
        boltz_main.download_boltz2(cache)

    torch_model = TorchBoltz2.load_from_checkpoint(
        checkpoint_path,
        strict=True,
        map_location="cpu",
        # Note: these args ARE NOT USED during prediction, but are needed to load the model
        predict_args={
            "recycling_steps": 0,
            "sampling_steps": 25,
            "diffusion_samples": 1,
        },
        diffusion_process_args=asdict(boltz_main.Boltz2DiffusionParams()),
        # ema=False,
        msa_args=asdict(
            boltz_main.MSAModuleArgs(
                subsample_msa=True,
                num_subsampled_msa=1024,
                use_paired_feature=True,
            )
        ),
        pairformer_args=asdict(boltz_main.PairformerArgsV2()),
    ).eval()

    model = joltz.from_torch(torch_model)
    _model_params, _model_static = eqx.partition(model, eqx.is_inexact_array)
    return eqx.combine(jax.device_put(_model_params), _model_static)


## Duplicated code
class StructureWriter:
    writer: boltz_main.BoltzWriter
    atom_pad_mask: torch.Tensor
    record: any
    out_dir: str
    temp_dir_handle: TemporaryDirectory

    def __init__(
        self,
        *,
        features_dict,
        target_dir: Path,
        output_dir: Path,
        temp_dir_handle: TemporaryDirectory,
    ):
        self.writer = boltz_main.BoltzWriter(
            data_dir=target_dir,
            output_dir=output_dir,
            output_format="mmcif",
            boltz2=True,
        )
        self.atom_pad_mask = features_dict["atom_pad_mask"].unsqueeze(0)
        self.record = features_dict["record"][0]
        self.out_dir = output_dir
        self.temp_dir_handle = temp_dir_handle

    def __call__(self, sample_atom_coords) -> gemmi.Structure:
        confidence = torch.ones(1)

        pred_dict = {
            "exception": False,
            # Accepts [A, 3] or [1, A, 3].
            "coords": torch.tensor(np.array(sample_atom_coords).reshape(1, -1, 3)).unsqueeze(0),
            "masks": self.atom_pad_mask,
            "confidence_score": confidence,
        }
        self.writer.write_on_batch_end(
            None,
            None,
            pred_dict,
            None,
            {"record": [self.record]},
            None,
            None,
        )
        return gemmi.read_structure(
            str((Path(self.out_dir) / self.record.id) / f"{self.record.id}_model_0.cif")
        )


def load_features_and_structure_writer(
    input_yaml_str: str,
    cache=None,
) -> PyTree:
    print("Loading data")
    cache = resolve_cache(cache, "boltz", create=True)
    out_dir_handle = (
        TemporaryDirectory()
    )  # this is sketchy -- we have to remember not to let this get garbage collected
    out_dir = Path(out_dir_handle.name)
    # dump the yaml to a file
    input_data_path = out_dir / "input.yaml"
    input_data_path.write_text(input_yaml_str)
    data = boltz_main.check_inputs(input_data_path)
    # Process inputs
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    manifest = boltz_main.process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=True,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        boltz2=True,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    if manifest is None:
        print("Something odd happened with manifest, trying to reload.")
        manifest = boltz_main.Manifest.load(processed_dir / "manifest.json")

    processed = boltz_main.BoltzProcessedInput(
        manifest=manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )

    # Create data module
    data_module = boltz_main.Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=0,
        mol_dir=mol_dir,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
        override_method=None,
    )

    # Load the features for the single example
    features_dict = list(data_module.predict_dataloader())[0]

    # convert features to numpy arrays
    features = {k: np.array(v) for k, v in features_dict.items() if k != "record"}

    ## one-hot the MSA

    features["msa"] = jax.nn.one_hot(features["msa"], const.num_tokens)
    # fix up some dtypes
    # features["method_feature"] = features["method_feature"].astype(np.int32)

    writer = StructureWriter(
        features_dict=features_dict,
        target_dir=processed.targets_dir,
        output_dir=out_dir / "output",
        temp_dir_handle=out_dir_handle,
    )

    return jax.tree.map(jnp.array, features), writer


def set_binder_sequence(
    new_sequence: Float[Array, "N 20"],
    features: PyTree,
):
    """Replace features related to first N tokens with `new_sequence.` Used for hallucination/binder design."""
    # features = jax.tree.map(lambda v: v.astype(jnp.float32), features)
    features["res_type"] = features["res_type"].astype(jnp.float32)
    features["msa"] = features["msa"].astype(jnp.float32)
    features["profile"] = features["profile"].astype(jnp.float32)
    assert len(new_sequence.shape) == 2
    assert new_sequence.shape[1] == 20
    binder_len = new_sequence.shape[0]

    # We only use the standard 20 amino acids, but boltz has 33 total tokens.
    # zero out non-standard AA types
    zero_padded_sequence = jnp.pad(new_sequence, ((0, 0), (2, 11)))
    n_msa = features["msa"].shape[1]
    print("n_msa", n_msa)

    # We assume there are no MSA hits for the binder sequence
    binder_profile = jnp.zeros_like(features["profile"][0, :binder_len])
    binder_profile = binder_profile.at[:binder_len].set(zero_padded_sequence) / n_msa
    binder_profile = binder_profile.at[:, 1].set((n_msa - 1) / n_msa)

    return features | {
        "res_type": features["res_type"]
        .at[0, :binder_len, :]
        .set(zero_padded_sequence),
        "msa": features["msa"].at[0, 0, :binder_len, :].set(zero_padded_sequence),
        "profile": features["profile"].at[0, :binder_len].set(binder_profile),
    }


BOLTZ2_DISTOGRAM_BINS = np.linspace(start=2.0, stop=22.0, num=64)


def boltz2_atom37(features: PyTree, coords: Float[Array, "A 3"]):
    """Atom37 view of Boltz-2 coordinates."""
    features_unbatched = jax.tree.map(lambda x: x[0], features)
    assert ref_atoms["UNK"][:4] == ["N", "CA", "C", "O"]
    first_atom_idx = jax.vmap(lambda atoms: jnp.nonzero(atoms, size=1)[0][0])(
        features_unbatched["atom_to_token"].T
    )
    n_tokens = features_unbatched["res_type"].shape[0]
    res_slot = features_unbatched["res_type"].argmax(-1)             # [N_token]
    atom_to_token = features_unbatched["atom_to_token"].argmax(-1)   # [N_atom]
    tokatom_idx = jnp.arange(atom_to_token.shape[0]) - first_atom_idx[atom_to_token]
    atom37_idx = jnp.asarray(_BOLTZ_TOKATOM_TO_ATOM37)[res_slot[atom_to_token], tokatom_idx]
    atom37_idx = jnp.where(
        features_unbatched["atom_pad_mask"] > 0.5, atom37_idx, jnp.int32(-1),
    )
    return scatter_atom37(coords, atom_to_token, atom37_idx, n_tokens)


def pad_atom_features(features: dict, pad_to: int):

    n_atoms = features["atom_pad_mask"].shape[-1]
    assert pad_to >= n_atoms

    def pad(v):
        pad_width = tuple((0, pad_to-n_atoms) if d == n_atoms else (0,0) for d in v.shape)
        return jnp.pad(v, pad_width)

    return jax.tree.map(pad, features)

def _prefix():
    return """version: 1
sequences:"""


def chain_yaml(chain_name: str, chain: TargetChain) -> str:
    raw = f"""  - {chain.polymer_type.lower()}:
        id: [{chain_name}]
        sequence: {chain.sequence}"""
    if not chain.use_msa:
        raw += """
        msa: empty"""

    return raw


def target_only_features(chains: list[TargetChain]):
    yaml = "\n".join(
        [_prefix()]
        + [
            chain_yaml(chain_id, c)
            for chain_id, c in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
        ]
    )

    tf, template_yaml = build_template_yaml("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
    if tf is not None:
        yaml += template_yaml

    features, writer = load_features_and_structure_writer(yaml)
    if tf is not None: # make sure we actually got a template
        assert np.sum(features["template_mask"]) > 0
    return (features, writer)



def build_template_yaml(chain_names: str, chains: list[TargetChain]):
    # boltz wants perfect .cifs :( 
    templates = {
        chain_id: c.template_chain
        for chain_id, c in zip(chain_names, chains)
        if c.template_chain != None
    }
    if len(templates) > 0:
        st = gemmi.Structure()
        model = gemmi.Model("0")
        entities = []

        for chain_id, chain in templates.items():
            chain.name = chain_id
            ent = gemmi.Entity(chain_id)
            ent.entity_type = gemmi.EntityType.Polymer
            ent.polymer_type = gemmi.PolymerType.PeptideL
            ent.subchains = [chain_id]
            ent.full_sequence = [r.name for r in chain]
            entities.append(ent)
            for r in chain:
                r.subchain = chain_id
            model.add_chain(chain)

        st.add_model(model)
        st.entities = gemmi.EntityList(entities)
        st.assign_subchains()
        st.setup_entities()
        st.ensure_entities()
        st.assign_label_seq_id()

        tf = NamedTemporaryFile(suffix=".cif")

        template_yaml = f"""
        
templates:
  - cif: {tf.name}
    chain_id: [{', '.join(k for k in templates)}]
    template_id: [{', '.join(k for k in templates)}]
"""
        
        st.setup_entities()
        doc = st.make_mmcif_document()
        doc.write_file(tf.name)
        return tf, template_yaml
    else:
        return None, None

def binder_features(binder_length, chains: list[TargetChain]):
    return target_only_features([TargetChain(sequence = "X"* binder_length, use_msa=False)] + chains)


# `model_output` and the loss builders.
_DESIGN_RECYCLING_STEPS = 1
# `predict`: upstream `boltz predict` (200 steps, `--recycling_steps 3`).
_UPSTREAM_SAMPLING_STEPS = 200
_INFERENCE_RECYCLING_STEPS = 4


class Boltz2(DiffusionStructurePredictionModel[dict, InitialEmbedding]):
    """Boltz-2. `recycling_steps` counts trunk passes, so it is one more than
    upstream's `--recycling_steps`."""

    model: eqx.Module
    design_recycling_steps: int = eqx.field(static=True)
    inference_recycling_steps: int = eqx.field(static=True)
    backprop_all_passes: bool = eqx.field(static=True)

    def __init__(
        self,
        cache_path: Path | None = None,
        *,
        design_recycling_steps: int = _DESIGN_RECYCLING_STEPS,
        inference_recycling_steps: int = _INFERENCE_RECYCLING_STEPS,
        backprop_all_passes: bool = False,
    ):
        self.model = load_boltz2(cache_path) if cache_path is not None else load_boltz2()
        self.design_recycling_steps = design_recycling_steps
        self.inference_recycling_steps = inference_recycling_steps
        self.backprop_all_passes = backprop_all_passes

    @staticmethod
    def target_only_features(chains: list[TargetChain]):
        return target_only_features(chains)

    @staticmethod
    def binder_features(binder_length, chains: list[TargetChain]):
        return binder_features(binder_length, chains)

    def upstream_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> Sampler:
        return boltz2_sampler(
            self.model.structure_module, num_steps or _UPSTREAM_SAMPLING_STEPS, schedule=schedule
        )

    # -- the structure-model contract -----------------------------------------

    def set_sequence(self, features: dict, sequence: Float[Array, "N 20"], *, key: PRNGKeyArray) -> dict:
        del key
        return set_binder_sequence(sequence, features)

    def embed(self, features: dict, *, key: PRNGKeyArray) -> InitialEmbedding:
        del key
        return self.model.embed_inputs(features)

    def init_state(self, features: dict, inputs: InitialEmbedding, *, key: PRNGKeyArray) -> TrunkState:
        del features, key
        return TrunkState(pair=jnp.zeros_like(inputs.z_init[0]), single=jnp.zeros_like(inputs.s_init[0]))

    def step(self, features: dict, inputs: InitialEmbedding, state: TrunkState, *, key: PRNGKeyArray) -> TrunkState:
        out, _ = self.model.trunk_iteration(
            _joltz_state(state), inputs, features, key=key, deterministic=True
        )
        return TrunkState(pair=out.z[0], single=out.s[0])

    def structure_denoiser(
        self, features: dict, inputs: InitialEmbedding, state: TrunkState
    ) -> tuple[Denoise, Float[Array, " A"]]:
        return boltz2_denoiser(self.model, features, inputs, _joltz_state(state))

    def distogram(self, features: dict, inputs: InitialEmbedding, state: TrunkState) -> Distogram:
        del features, inputs
        return Distogram(
            logits=self.model.distogram_module(state.pair[None])[0, :, :, 0, :],
            bins=jnp.array(BOLTZ2_DISTOGRAM_BINS),
        )

    def confidence(
        self, features: dict, inputs: InitialEmbedding, state: TrunkState, coords: Float[Array, "A 3"]
    ) -> Confidence:
        out = self.model.confidence_module(
            s_inputs=inputs.s_inputs,
            s=state.single[None],
            z=state.pair[None],
            x_pred=coords[None],
            feats=features,
            pred_distogram_logits=self.distogram(features, inputs, state).logits[None],
            key=jax.random.key(0),  # deterministic: no dropout
            deterministic=True,
        )
        return Confidence(
            plddt=out.plddt[0], pae_logits=out.pae_logits[0], pae_bins=jnp.array(PAE_BINS)
        )

    def tokens(self, features: dict) -> TokenInfo:
        return TokenInfo(
            sequence=features["res_type"][0][:, 2:22],
            asym_id=features["asym_id"][0],
            residue_index=features["residue_index"][0],
            mask=features["token_pad_mask"][0].astype(bool),
        )

    def atom37(self, features: dict, coords: Float[Array, "A 3"]) -> Atom37:
        atom37_coords, atom37_mask = boltz2_atom37(features, coords)
        return Atom37(coords=atom37_coords, mask=atom37_mask)

    def to_structure(self, features: dict, writer, output: StructureModelOutput, sequence) -> gemmi.Structure:
        del features, sequence
        return writer(output.structure_coordinates)


def _joltz_state(state: TrunkState) -> JoltzTrunkState:
    return JoltzTrunkState(s=state.single[None], z=state.pair[None])
