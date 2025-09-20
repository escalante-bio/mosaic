from mosaic.structure_prediction import StructurePredictionModel, TargetChain, StructurePrediction
from mosaic.af2.alphafold2 import AF2
from mosaic.losses.af2 import AlphaFoldLoss, AF2Output
from mosaic.losses.structure_prediction import IPTMLoss

from jaxtyping import Array, Float, PyTree
import equinox as eqx
import jax
import jax.numpy as jnp

class AlphaFold2(eqx.Module, StructurePredictionModel):
    af2_forward: callable
    stacked_parameters: PyTree

    def __init__(self, data_dir: str = "."):
        af2 = AF2(data_dir=data_dir)
        self.af2_forward = af2.alphafold_apply
        self.stacked_parameters = af2.stacked_model_params

    def target_only_features(self, chains: list[TargetChain]):
        for c in chains:
            assert c.polymer_type == "PROTEIN", "AF2 only supports protein chains"
            assert not c.use_msa, "AF2 interface does not support MSA yet"

        features, _ = AF2.build_features(
            [c.sequence for c in chains],
            {
                idx: c.template_chain
                for idx, c in enumerate(chains)
                if c.template_chain is not None
            },
        )
        return features, None

    def binder_features(self, binder_length, chains: list[TargetChain]):
        features, _ = self.target_only_features(
            [TargetChain(sequence="G" * binder_length, use_msa=False)] + chains
        )
        return features, None

    def build_loss(self, *, loss, features, recycling_steps=1, sampling_steps=None, name = "af2"):
        assert sampling_steps is None, "AF2 does not support sampling steps"
        return AlphaFoldLoss(
            forward=self.af2_forward,
            stacked_params=self.stacked_parameters,
            features=features,
            loss=loss,
            recycling_steps=recycling_steps,
            name = name
        )

    @eqx.filter_jit
    def _forward(self, PSSM, features, *, key, model_idx: int, recycling_steps: int, initial_guess=None):
        params = jax.tree.map(lambda v: v[model_idx], self.stacked_parameters)
        print("JIT compiling AF2...")
        # build full soft sequence
        full_sequence = jax.nn.one_hot(features.aatype, 21)
        # set binder sequence
        if PSSM is not None:
            full_sequence = full_sequence.at[: PSSM.shape[0], :20].set(
                PSSM
            )
         # run the model
        return self.af2_forward(
            params,
            jax.random.fold_in(key, 1),
            features=features,
            initial_guess=initial_guess,
            replace_target_feat=full_sequence,
            recycling_steps= recycling_steps,
        )

    def model_output(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        recycling_steps=1,
        sampling_steps=None,
        model_idx: int | None = None,
        key,
    ):
        if model_idx is None:
            model_idx = jax.random.randint(key=key, shape=(), minval=0, maxval=5)
            key = jax.random.fold_in(key, 0)
        else:
            model_idx = jax.device_put(model_idx)
        
        output = self._forward(
            PSSM,
            features,
            key=key,
            model_idx=model_idx,
            recycling_steps=recycling_steps,
            initial_guess=None,
        )

        return AF2Output(features=features, output=output)


    @eqx.filter_jit
    def _coords_and_confidences(self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        recycling_steps=1,
        sampling_steps=None,
        model_idx: int | None = None,
        key,
    ):
        output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            model_idx=model_idx,
            key=key,
        )

        # coords = output.structure_output.sample_atom_coords[0]
        pae = output.pae
        plddt = output.plddt
        if PSSM is None:
            PSSM = jnp.zeros((0, 20))
        iptm = -IPTMLoss()(PSSM, output, key=jax.random.key(0))[0]
        return output.output, pae, plddt, iptm

    def predict(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        writer: None = None,
        recycling_steps=1,
        sampling_steps=None,
        model_idx: int | None = None,
        key,
    ) -> StructurePrediction:
        if PSSM is not None:
            features = eqx.tree_at(lambda f: f.aatype, features, jnp.array(features.aatype).at[: PSSM.shape[0]].set(jnp.argmax(PSSM, axis=-1)))

        (afo, pae, plddt, iptm) = self._coords_and_confidences(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=None,
            model_idx=model_idx,
            key=key,
        )

        _, structure = AF2._postprocess_prediction(
            features, afo
        )

        return StructurePrediction(
            st=structure,
            plddt=plddt,
            pae=pae,
            iptm=iptm
        )