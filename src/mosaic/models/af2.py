from mosaic.structure_prediction import StructurePredictionModel, TargetChain
from mosaic.af2.alphafold2 import AF2
from mosaic.losses.af2 import AlphaFoldLoss, AF2Output

from jaxtyping import Array, Float, PyTree
import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp

class AlphaFold2(eqx.Module, StructurePredictionModel):
    # af2: AF2
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
        key,
    ):
        output = self._forward(
            PSSM,
            features,
            key=key,
            model_idx=jax.random.randint(key=key, shape=(), minval=0, maxval=5),
            recycling_steps=recycling_steps,
            initial_guess=None,
        )

        return AF2Output(features=features, output=output)

    def predict(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        writer: None,
        recycling_steps=1,
        sampling_steps=None,
        key,
    ):
        if PSSM is not None:
            features = eqx.tree_at(lambda f: f.aatype, features, jnp.array(features.aatype).at[: PSSM.shape[0]].set(jnp.argmax(PSSM, axis=-1)))
        model_output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=None,
            key=key,
        )

        _, structure = AF2._postprocess_prediction(
            features, model_output.output
        )

        return structure


af2 =AlphaFold2(data_dir = "/home/ubuntu/escalante/")

st = gemmi.read_structure("/home/ubuntu/escalante/PDL1.pdb")
st[0][0]
seq = gemmi.one_letter_code([r.name for r in st[0][0]])

features, _= af2.target_only_features([TargetChain(sequence=seq, use_msa=False, template_chain = st[0][0])])
features

features, _= af2.binder_features(80, [TargetChain(sequence=seq, use_msa=False, template_chain = st[0][0])])



features
features
features
''
import jax.numpy as jnp
# st_pred = af2.predict(PSSM = jax.nn.one_hot(jnp.ones(80, dtype=jnp.int32), 20), features=features, writer=None, recycling_steps=3, key = jax.random.PRNGKey(0))
# st_pred = af2.predict(PSSM = None, features=features, writer=None, recycling_steps=3, key = jax.random.PRNGKey(0))


# output = af2.model_output(features=features, recycling_steps=3, key = jax.random.PRNGKey(0))
# st_pred.write_minimal_pdb("test_af2.pdb")
# features, _ = af2.binder_features(5, [TargetChain(sequence="G" * 10, use_msa=False)])


from mosaic.losses.structure_prediction import PLDDTLoss

loss = af2.build_loss(loss=PLDDTLoss(), features=features, recycling_steps=3)


eqx.filter_jit(loss)(jax.nn.one_hot(jnp.ones(80, dtype=jnp.int32), 20), key = jax.random.PRNGKey(0))

st_pred[0]
# features

# features