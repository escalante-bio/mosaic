# TODO: figure out how to NOT produce MSA for a target chain
# Note we use a vanilla ODE sampler for the structure module by default!
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import protenix
import protenix.inference
from pathlib import Path
from jaxtyping import Array, Float, PyTree
from ml_collections.config_dict import ConfigDict
from protenix.config import parse_configs
from protenix.configs.configs_base import configs as configs_base
from protenix.configs.configs_data import data_configs
from protenix.configs.configs_inference import inference_configs
from protenix.configs.configs_model_type import model_configs
from protenix.model.protenix import Protenix as TorchProtenix
from protenix.protenij import (
    from_torch,
)
import torch

from protenix.data.template import ChainInput, featurize

from mosaic.losses.protenix import (
    MultiSampleProtenixLoss,
    ProtenixFromTrunkOutput,
    biotite_array_to_gemmi_struct,
    get_trunk_state,
    set_binder_sequence,
)
from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.structure_prediction import (
    PolymerType,
    StructurePrediction,
    StructurePredictionModel,
    TargetChain,
)


def load_model(name="protenix_mini_default_v0.5.0", cache_path=Path("~/.protenix")):
    cache_path = cache_path.expanduser()
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        arg_str=f"--model_name {name}",
        fill_required_with_null=True,
    )
    configs.model_name = name
    configs.update({"load_checkpoint_dir": str(cache_path)})
    configs.update(ConfigDict(model_configs[name]))
    ###  Use vanilla ODE sampler.
    configs.sample_diffusion["gamma0"] = 0.0
    configs.sample_diffusion["step_scale_eta"] = 1.0
    configs.sample_diffusion["noise_scale_lambda"] = 1.0
    ###
    protenix.inference.download_infercence_cache(configs)
    checkpoint_path = f"{configs.load_checkpoint_dir}/{configs.model_name}.pt"
    checkpoint = torch.load(checkpoint_path)
    sample_key = [k for k in checkpoint["model"].keys()][0]
    print(f"Sampled key: {sample_key}")
    if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
        checkpoint["model"] = {
            k[len("module.") :]: v for k, v in checkpoint["model"].items()
        }
    model = TorchProtenix(configs)
    model.load_state_dict(state_dict=checkpoint["model"], strict=configs.load_strict)
    model.eval()
    jax_model = from_torch(model)
    print(
        f"protenix SM parameters: gamma0={jax_model.gamma0}, step_scale_eta={jax_model.step_scale_eta}, N_steps={jax_model.N_steps}"
    )
    return jax_model


class Protenix(StructurePredictionModel):
    protenix: eqx.Module
    default_sample_steps: int

    def target_only_features(self, chains: list[TargetChain]):
        for c in chains:
            if c.polymer_type != PolymerType.PROTEIN:
                assert False, (
                    "Protenix interface only supports Protein chains. Manually build features for more complex targets. "
                )

        features_dict, atom_array, _ = featurize(
            [
                ChainInput(
                    sequence=c.sequence,
                    compute_msa=c.use_msa,
                    template=c.template_chain,
                )
                for c in chains
            ]
        )

        return features_dict, atom_array

    def binder_features(self, binder_length, chains: list[TargetChain]):
        binder = TargetChain(sequence="X" * binder_length, use_msa=False)
        return self.target_only_features([binder] + chains)

    def build_loss(
        self,
        *,
        loss,
        features,
        recycling_steps=1,
        sampling_steps=None,
        initial_recycling_state=None,
    ):
        return self.build_multisample_loss(
            loss=loss,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            num_samples=1,
            initial_recycling_state=initial_recycling_state,
        )

    def build_multisample_loss(
        self,
        *,
        loss,
        features,
        recycling_steps=1,
        num_samples: int = 4,
        sampling_steps=None,
        reduction=jnp.mean,
        initial_recycling_state=None,
    ):
        if sampling_steps is None:
            sampling_steps = self.default_sample_steps
        return MultiSampleProtenixLoss(
            model=self.protenix,
            features=features,
            loss=loss,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            num_samples=num_samples,
            reduction=reduction,
            initial_recycling_state=initial_recycling_state,
        )

    def model_output(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        recycling_steps=1,
        sampling_steps=None,
        initial_recycling_state=None,
        key,
    ):
        if sampling_steps is None:
            sampling_steps = self.default_sample_steps
        features = set_binder_sequence(PSSM, features) if PSSM is not None else features

        initial_embedding, trunk_state = get_trunk_state(
            model=self.protenix,
            features=features,
            initial_recycling_state=initial_recycling_state,
            recycling_steps=recycling_steps,
            key=key,
        )

        return ProtenixFromTrunkOutput(
            model=self.protenix,
            features=features,
            sampling_steps=sampling_steps,
            initial_embedding=initial_embedding,
            trunk_state=trunk_state,
            key=key,
        )

    @eqx.filter_jit
    def _coords_and_confidences(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        recycling_steps=1,
        sampling_steps=None,
        initial_recycling_state=None,
        key,
    ):
        if sampling_steps is None:
            sampling_steps = self.default_sample_steps
        output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            initial_recycling_state=initial_recycling_state,
            key=key,
        )
        if PSSM is None:
            PSSM = jnp.zeros((0, 20))
        iptm = -IPTMLoss()(PSSM, output, key=jax.random.key(0))[0]
        return (output.structure_coordinates[0], output.pae, output.plddt, iptm)

    def predict(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        writer,
        recycling_steps=1,
        sampling_steps=None,
        initial_recycling_state=None,
        key,
    ):
        if sampling_steps is None:
            sampling_steps = self.default_sample_steps
        (coords, pae, plddt, iptm) = self._coords_and_confidences(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            initial_recycling_state=initial_recycling_state,
            key=key,
        )
        return StructurePrediction(
            st=biotite_array_to_gemmi_struct(writer, np.array(coords)),
            plddt=plddt,
            pae=pae,
            iptm=iptm,
        )


def ProtenixMini():
    return Protenix(load_model(name="protenix_mini_default_v0.5.0"), 2)


def ProtenixTiny():
    return Protenix(load_model(name="protenix_tiny_default_v0.5.0"), 2)


def ProtenixBase():
    return Protenix(load_model(name="protenix_base_default_v1.0.0"), 20)


def Protenix2025():
    return Protenix(load_model(name="protenix_base_20250630_v1.0.0"), 20)
