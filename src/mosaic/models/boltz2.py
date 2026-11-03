from pathlib import Path
from tempfile import NamedTemporaryFile

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from joltz import InitialEmbedding
from joltz import TrunkState as JoltzTrunkState

import mosaic.structure_model as sm
from mosaic.losses.boltz2 import (
    BOLTZ2_DISTOGRAM_BINS,
    boltz2_atom37,
    load_features_and_structure_writer,
    set_binder_sequence,
)
from mosaic.losses.boltz2 import (
    load_boltz2 as lb,
)
from mosaic.losses.structure_prediction import PAE_BINS, StructureModelOutput
from mosaic.sampling import Denoise, Sampler
from mosaic.sampling.boltz2 import boltz2_denoiser, boltz2_sampler
from mosaic.sampling.schedules import Schedule
from mosaic.structure_prediction import (
    Atom37,
    Confidence,
    DiffusionStructurePredictionModel,
    Distogram,
    StructurePrediction,
    TargetChain,
    TokenInfo,
    TrunkState,
)


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
_DESIGN_SAMPLING_STEPS = 25
_DESIGN_RECYCLING_STEPS = 1
# `predict`: upstream `boltz predict` (200 steps, `--recycling_steps 3`).
_INFERENCE_SAMPLING_STEPS = 200
_INFERENCE_RECYCLING_STEPS = 4


class Boltz2(DiffusionStructurePredictionModel[dict, InitialEmbedding]):
    """Boltz-2. `recycling_steps` counts trunk passes, so it is one more than
    upstream's `--recycling_steps`."""

    model: eqx.Module

    @property
    def design_passes(self) -> int:
        return _DESIGN_RECYCLING_STEPS

    @property
    def inference_passes(self) -> int:
        return _INFERENCE_RECYCLING_STEPS

    @property
    def backprop_all_passes(self) -> bool:
        return False

    def __init__(self, cache_path: Path | None = None):
        self.model = lb(cache_path) if cache_path is not None else lb()

    @staticmethod
    def target_only_features(chains: list[TargetChain]):
        return target_only_features(chains)

    @staticmethod
    def binder_features(binder_length, chains: list[TargetChain]):
        return binder_features(binder_length, chains)

    def design_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> Sampler:
        return boltz2_sampler(
            self.model.structure_module, num_steps or _DESIGN_SAMPLING_STEPS, schedule=schedule
        )

    def inference_sampler(
        self, num_steps: int | None = None, *, schedule: Schedule | None = None
    ) -> Sampler:
        return boltz2_sampler(
            self.model.structure_module, num_steps or _INFERENCE_SAMPLING_STEPS, schedule=schedule
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

    # -- public API (forwards to `mosaic.structure_model`) ----------------------

    def build_loss(
        self,
        *,
        loss,
        features,
        recycling_steps=_DESIGN_RECYCLING_STEPS,
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
        features,
        recycling_steps=_DESIGN_RECYCLING_STEPS,
        num_samples: int = 4,
        reduction=jnp.mean,
        sampler: Sampler | None = None,
        backprop_all_passes: bool | None = None,
    ) -> sm.MultiSampleLoss:
        """Run the trunk once and reduce the loss over `num_samples` samples."""
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
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        recycling_steps=_DESIGN_RECYCLING_STEPS,
        sampler: Sampler | None = None,
        key,
    ) -> StructureModelOutput:
        return sm.model_output(
            self, features, sequence=PSSM, num_passes=recycling_steps, sampler=sampler, key=key
        )

    def predict(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        writer: any,
        recycling_steps=_INFERENCE_RECYCLING_STEPS,
        sampler: Sampler | None = None,
        key,
    ) -> StructurePrediction:
        return sm.predict(
            self, features, writer, sequence=PSSM, num_passes=recycling_steps, sampler=sampler, key=key
        )


def _joltz_state(state: TrunkState) -> JoltzTrunkState:
    return JoltzTrunkState(s=state.single[None], z=state.pair[None])
