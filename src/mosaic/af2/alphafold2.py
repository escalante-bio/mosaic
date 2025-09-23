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




def load_af2(data_dir: str = "."):
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
    cfg = config.model_config("model_1_multimer_v3")
    cfg.max_msa_clusters = 1
    cfg.max_extra_msa = 1
    cfg.masked_msa_replace_fraction = 0
    cfg.subbatch_size = None
    cfg.model.num_ensemble_eval = 1
    cfg.model.global_config.subbatch_size = None
    cfg.model.global_config.eval_dropout = True
    cfg.model.global_config.deterministic = False
    cfg.model.global_config.use_remat = True
    cfg.model.num_extra_msa = 1
    

        # haiku transform forward function
    def _forward_fn(
        features: dict, recycling_steps: int,  initial_guess=None, is_training=False, **kwargs
    ) -> AFOutput:
        print("JIT compiling AF2...")
        model = modules_multimer.AlphaFold(cfg.model)
        prediction_results = model(
            batch=features,
            num_recycling_iterations=recycling_steps,
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

    stacked_model_params = tree.map(
        lambda *v: np.stack(v), *model_params
    )

    return (transformed.apply, stacked_model_params)

def _postprocess_prediction(features, prediction: AFOutput):
    final_atom_mask = prediction.structure_module.final_atom_mask
    b_factors = prediction.plddt[:, None] * final_atom_mask
    # todo: this next step is blocking!
    # need to recursively turn prediction into a dictionary

    unrelaxed_protein = protein.from_prediction(
        features,
        jax.tree.map(np.array, asdict(prediction)),
        b_factors,
        remove_leading_feature_dimension=False,
    )

    # prediction contains some very large values, let's select some to return
    return prediction, from_string(protein.to_pdb(unrelaxed_protein))

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


