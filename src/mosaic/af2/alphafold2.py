"""AF2 multimer model + initial guess"""

from dataclasses import asdict
from tempfile import NamedTemporaryFile
from pathlib import Path

import equinox as eqx
import gemmi
import haiku as hk
import jax
import numpy as np
from jax import tree
from jaxtyping import Array, Bool, Float
from tqdm import tqdm

from ..alphafold.common import protein, residue_constants
from ..alphafold.model import config, data, modules_multimer
from .confidence_metrics import confidence_metrics, _calculate_bin_centers
from .featurization import (
    AFFeatures,
    # aa_code,
    af2_get_atom_positions_gemmi,
    build_features,
    chain_template_features,
    empty_placeholder_template_features,
)


def from_string(s: str) -> gemmi.Structure:
    with NamedTemporaryFile(suffix=".pdb") as f:
        f.write(s.encode("utf-8"))
        f.flush()
        st = gemmi.read_pdb(f.name)

    st.setup_entities()
    return st


class Distogram(eqx.Module):
    bin_edges: Float[Array, "63"]
    logits: Float[Array, "N N 63"]


class StructureModuleOutputs(eqx.Module):
    final_atom_mask: Bool[Array, "N 37"]
    final_atom_positions: Float[Array, "N 37 3"]


class AFOutput(eqx.Module):
    distogram: Distogram
    iptm: float
    predicted_aligned_error: Float[Array, "N N"]
    pae_logits: Float[Array, "N N 64"]
    pae_bin_centers: Float[Array, "64"]
    predicted_lddt_logits: Float[Array, "N 50"]
    plddt: Float[Array, "N"]
    structure_module: StructureModuleOutputs


class AF2:
    def __init__(self, num_recycle=1, model_name="model_1_multimer_v3", data_dir="."):
        model_name = "model_1_multimer_v3"
        assert "multimer" in model_name, f"{model_name} is not a multimer model"

        if not (Path(data_dir)/"params").exists():
            print(f"Could not find AF2 parameters in {data_dir}/params. \n Running `download_params.sh .`")
            # run download_params.sh
            from subprocess import run
            run(["bash", "download_params.sh", data_dir], check=True)

        try: 
            model_params = [
                data.get_model_haiku_params(model_name=model_name, data_dir=data_dir)
                for model_name in tqdm(
                    [f"model_{i}_multimer_v3" for i in range(1, 6)],
                    desc="Loading AF2 params",
                )
            ]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not find AF2 parameters in {data_dir}/params. \n Run `download_params.sh .`. \n {e}"
            )
        cfg = config.model_config(model_name)
        cfg.num_recycle = num_recycle
        cfg.model.num_recycle = num_recycle
        cfg.max_msa_clusters = 1
        cfg.max_extra_msa = 1
        # cfg.common.max_extra_msa = 1
        cfg.masked_msa_replace_fraction = 0
        cfg.subbatch_size = None
        cfg.model.num_ensemble_eval = 1
        # cfg.model.recycle_early_stop_tolerance = 0.5
        cfg.model.global_config.subbatch_size = None
        cfg.model.global_config.eval_dropout = False
        cfg.model.global_config.deterministic = True
        cfg.model.global_config.use_remat = True
        self.cfg = cfg

        # haiku transform forward function
        def _forward_fn(
            features: AFFeatures, initial_guess=None, is_training=False, **kwargs
        ) -> AFOutput:
            print("JIT compiling AF2...")
            model = modules_multimer.AlphaFold(cfg.model)
            prediction_results = model(
                asdict(features),
                is_training=is_training,
                initial_guess=initial_guess,
                **kwargs,
            )
            # add confidences
            confidences = confidence_metrics(prediction_results)
            return AFOutput(
                distogram=Distogram(**prediction_results["distogram"]),
                iptm=confidences["iptm"],
                predicted_aligned_error=confidences["predicted_aligned_error"],
                pae_logits=prediction_results["predicted_aligned_error"]["logits"],
                pae_bin_centers=_calculate_bin_centers(prediction_results["predicted_aligned_error"]["breaks"]),
                predicted_lddt_logits=prediction_results["predicted_lddt"]["logits"],
                plddt=confidences["plddt"],
                structure_module=StructureModuleOutputs(
                    final_atom_mask=prediction_results["structure_module"][
                        "final_atom_mask"
                    ],
                    final_atom_positions=prediction_results["structure_module"][
                        "final_atom_positions"
                    ],
                ),
            )

        transformed = hk.transform(_forward_fn)
        self.alphafold_apply = transformed.apply
        self.jitted_apply = jax.jit(transformed.apply)

        self.stacked_model_params = tree.map(
            lambda *v: np.stack(v), *model_params
        )

    def _postprocess_prediction(self, features: AFFeatures, prediction: AFOutput):
        final_atom_mask = prediction.structure_module.final_atom_mask
        b_factors = prediction.plddt[:, None] * final_atom_mask
        # todo: this next step is blocking!
        # need to recursively turn prediction into a dictionary

        # prediction = asdict(prediction)
        unrelaxed_protein = protein.from_prediction(
            asdict(features),
            jax.tree.map(np.array, asdict(prediction)),
            b_factors,
            remove_leading_feature_dimension=False,
        )

        # prediction contains some very large values, let's select some to return
        selected_keys = ["predicted_aligned_error", "plddt", "iptm"]

        # return {k: np.array(prediction[k]) for k in selected_keys} | {
        #     "structure_module": prediction.structure_module, "prediction": prediction
        # }, from_string(protein.to_pdb(unrelaxed_protein))
        return prediction, from_string(protein.to_pdb(unrelaxed_protein))
    
    @staticmethod
    def _initial_guess(st: gemmi.Structure):
        ca_idx = residue_constants.atom_order["CA"]
        cb_idx = residue_constants.atom_order["CB"]
        initial_guess_all_atoms, mask = af2_get_atom_positions_gemmi(st)
        c_beta_missing = mask[:, cb_idx] == 0
        # if c_beta missing (e.g. for backbone-only structures) set position to ca
        initial_guess_all_atoms[c_beta_missing, cb_idx] = initial_guess_all_atoms[
            c_beta_missing, ca_idx
        ]
        return initial_guess_all_atoms


    def build_features(self, chains: list[str], template_chains: dict[int, gemmi.Chain] = {}, initial_guess: gemmi.Structure | None = None):
        assert isinstance(template_chains, dict)
        assert all(
            isinstance(k, int) and isinstance(v, gemmi.Chain)  # type(v) == gemmi.Chain
            for k, v in template_chains.items()
        )
        assert all(0 <= k < len(chains) for k in template_chains.keys())

        features = build_features(
            [
                (
                    chain_aa,
                    chain_template_features(template_chains[chain_idx])
                    if chain_idx in template_chains
                    else empty_placeholder_template_features(0, len(chain_aa)),
                )
                for (chain_idx, chain_aa) in enumerate(chains)
            ]
        )

        if initial_guess is not None:
            initial_guess = self._initial_guess(initial_guess)

        return features, initial_guess
    
    def predict(
        self,
        chains: list[str],
        template_chains: dict[int, gemmi.Chain] = {},
        initial_guess: gemmi.Structure | None = None,
        model_idx=0,
        *,
        key
    ):
        assert isinstance(template_chains, dict)
        assert all(
            isinstance(k, int) and isinstance(v, gemmi.Chain)  # type(v) == gemmi.Chain
            for k, v in template_chains.items()
        )
        assert 0 <= model_idx < 5
        assert all(0 <= k < len(chains) for k in template_chains.keys())

        features, initial_guess = self.build_features(chains, template_chains, initial_guess)

        results = self.jitted_apply(
            jax.tree.map(lambda v: v[model_idx], self.stacked_model_params),
            key, 
            features,
            initial_guess,
        )
        return self._postprocess_prediction(features, results)


    def __call__(
        self,
        st: gemmi.Structure,
        template_chains: dict[int, gemmi.Chain] = dict(),
        use_initial_guess: bool = False,
        model_idx=0,
    ):
        """Make a prediction possibly using templates and initial guess from `st.` If you don't have a structure and want to make an unconditional prediction use `predict` instead."""
        return self.batch_templated_predict(
            [st], template_chains, use_initial_guess, model_idx=model_idx
        )[0]
