# TODO: This is fairly rough, especially how we inject soft sequences into alphafold. Clean this up!
from dataclasses import dataclass
import jax
from jax import tree
from jaxtyping import Array, Float, PyTree

import gemmi

from .structure_prediction import AbstractStructureOutput
from ..common import LossTerm, LinearCombination
from ..af2.featurization import AFFeatures
from ..af2.alphafold2 import AFOutput, AF2

import numpy as np



@dataclass
class AF2Output(AbstractStructureOutput):
    features: AFFeatures
    output: AFOutput

    @property
    def full_sequence(self):
        return jax.nn.one_hot(self.features.aatype, 20)

    @property
    def asym_id(self):
        return self.features.asym_id

    @property
    def residue_idx(self):
        return self.features.residue_index

    @property
    def distogram_bins(self) -> Float[Array, "64"]:
        return np.linspace(
            start=2.3125, stop=21.6875, num=64
        )  # not quite right but whatever

    @property
    def distogram_logits(self) -> Float[Array, "N N 64"]:
        return self.output.distogram.logits

    @property
    def backbone_coordinates(self) -> Float[Array, "N 4 3"]:
        return self.output.structure_module.final_atom_positions[:, [0, 1, 2, 4], :]

    @property
    def plddt(self) -> Float[Array, "N"]:
        return self.output.plddt / 100

    @property
    def pae(self) -> Float[Array, "N N"]:
        return self.output.predicted_aligned_error

    @property
    def pae_logits(self) -> Float[Array, "N N 64"]:
        return self.output.pae_logits

    @property
    def pae_bins(self) -> Float[Array, "64"]:
        return self.output.pae_bin_centers

    @property
    def iptm(self) -> float:
        return self.output.iptm


class AlphaFoldLoss(LossTerm):
    forward: callable
    stacked_params: PyTree
    features: AFFeatures
    losses: LinearCombination
    name: str
    initial_guess: gemmi.Structure | None = None

    def predict(self, soft_sequence: Float[Array, "N 20"], *, key, model_idx: int):
        params = tree.map(lambda v: v[model_idx], self.stacked_params)
        # build full soft sequence
        full_sequence = jax.nn.one_hot(self.features.aatype, 21)
        # set binder sequence
        full_sequence = full_sequence.at[: soft_sequence.shape[0], :20].set(
            soft_sequence
        )
        # run the model
        output = self.forward(
            params,
            jax.random.fold_in(key, 1),
            features=self.features,
            initial_guess=None
            if self.initial_guess is None
            else AF2._initial_guess(self.initial_guess),
            replace_target_feat=full_sequence,
        )
        return output

    def __call__(self, soft_sequence: Float[Array, "N 20"], *, key):
        # pick a random model
        model_idx = jax.random.randint(key=key, shape=(), minval=0, maxval=5)

        output = self.predict(soft_sequence, key=key, model_idx=model_idx)

        v, aux = self.losses(
            soft_sequence,
            AF2Output(
                features=self.features,
                output=output,
            ),
            key=key,
        )

        return v, {f"{self.name}/{k}": v for k, v in aux.items()} | {
            f"{self.name}/model_idx": model_idx,
            f"{self.name}/loss": v,
        }