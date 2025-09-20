# TODO: This is fairly rough, especially how we inject soft sequences into alphafold. Clean this up!
from dataclasses import dataclass
from typing import Callable
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
        return np.linspace(start = 0.25, stop = 31.75, num=64)




class AlphaFoldLoss(LossTerm):
    forward: Callable
    stacked_params: PyTree
    features: AFFeatures
    losses: LinearCombination
    name: str
    initial_guess: gemmi.Structure | None = None
    recycling_steps: int = 1

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
            recycling_steps=self.recycling_steps,
        )
        return output

    def __call__(self, soft_sequence: Float[Array, "N 20"], *, key):
        # pick a random model
        model_idx = int(jax.random.randint(key=key, shape=(), minval=0, maxval=5).item())

        output = self.predict(soft_sequence, key=key, model_idx=model_idx)

        v, aux = self.losses(
            soft_sequence,
            AF2Output(
                features=self.features,
                output=output,
            ),
            key=key,
        )

        return v, {self.name: aux, f"{self.name}/model_idx": model_idx, f"{self.name}/loss": v}


class AF2SidechainRMSD(LossTerm):
    """Sidechain RMSD at specified positions using AF2 outputs and template atoms.

    - Uses predicted atom37 positions from AF2.
    - Uses template atom37 positions from features (template_all_atom_positions) as references.
    - Excludes backbone atoms (N, CA, C, O) from the RMSD calculation.
    - If positions is None, includes any residue with a nonzero template sidechain mask.
    """
    positions: tuple[int, ...] | None = None

    def __call__(self, sequence: Float[Array, "N 20"], output: AF2Output, *, key):
        from ..alphafold.common import residue_constants as rc

        import jax.numpy as jnp

        N = sequence.shape[0]
        pred_all37 = jnp.nan_to_num(output.output.structure_module.final_atom_positions[:N])

        # Template references: take first template (if any) and restrict to binder length
        tmpl_pos_all = jnp.nan_to_num(output.features.template_all_atom_positions)
        tmpl_mask_all = output.features.template_all_atom_mask
        if tmpl_pos_all.shape[0] == 0:
            z = jnp.asarray(0.0, dtype=jnp.float32)
            return z, {"af2_sc_rmsd": z}
        tmpl_pos = tmpl_pos_all[0, :N]
        tmpl_mask = tmpl_mask_all[0, :N]

        # Exclude backbone atoms
        backbone_idx = [rc.atom_order[k] for k in ["N","CA","C","O"]]
        sidechain_mask = jnp.ones_like(tmpl_mask).at[:, backbone_idx].set(0)
        sidechain_mask = sidechain_mask * tmpl_mask

        # Optionally restrict to provided positions
        if self.positions is not None and len(self.positions) > 0:
            keep = jnp.zeros((N,), dtype=tmpl_mask.dtype).at[jnp.asarray(self.positions, dtype=jnp.int32)].set(1)
            sidechain_mask = sidechain_mask * keep[:, None]

        # Weighted Kabsch alignment and RMSD
        def _np_kabsch(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            ab = a.T @ b
            u, s, vh = jnp.linalg.svd(ab, full_matrices=False)
            flip = jnp.linalg.det(u @ vh) < 0
            u_ = jnp.where(flip, -u[:, -1], u[:, -1])
            u = u.at[:, -1].set(u_)
            return u @ vh

        P_all = jnp.nan_to_num(pred_all37.reshape(-1, 3))
        Q_all = jnp.nan_to_num(tmpl_pos.reshape(-1, 3))
        w_mask = sidechain_mask.reshape(-1).astype(P_all.dtype)
        total = w_mask.sum()

        def _zero():
            z = jnp.asarray(0.0, dtype=jnp.float32)
            return z, {"af2_sc_rmsd": z}

        def _compute():
            w = w_mask / (total + 1e-8)
            P_mu = (P_all * w[:, None]).sum(0, keepdims=True)
            Q_mu = (Q_all * w[:, None]).sum(0, keepdims=True)
            R = _np_kabsch((P_all - P_mu) * w[:, None], (Q_all - Q_mu))
            align = lambda x: (x - P_mu) @ R + Q_mu
            sd = jnp.square(align(P_all) - Q_all).sum(-1)
            msd = (w * sd).sum()
            rmsd = jnp.sqrt(msd + 1e-8)
            return rmsd, {"af2_sc_rmsd": rmsd}

        import jax
        rmsd, aux = jax.lax.cond(total > 0, _compute, _zero)
        return rmsd, aux