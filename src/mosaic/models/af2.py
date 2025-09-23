from mosaic.structure_prediction import (
    StructurePredictionModel,
    TargetChain,
    StructurePrediction,
)
from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.common import tokenize
from mosaic.alphafold.common import residue_constants,protein
from mosaic.alphafold.model import config, data, modules_multimer
from mosaic.losses.confidence_metrics import confidence_metrics, _calculate_bin_centers




from jaxtyping import Array, Float, PyTree, Bool
import equinox as eqx
import jax
import jax.numpy as jnp
import gemmi
import numpy as np

from dataclasses import dataclass
from jax import tree
from tempfile import NamedTemporaryFile
from pathlib import Path
from dataclasses import asdict

from tqdm import tqdm
import haiku as hk



from mosaic.structure_prediction import AbstractStructureOutput
from ..common import LossTerm, LinearCombination


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





def set_binder_sequence(PSSM, features: dict):
    if PSSM is None:
        PSSM = jnp.zeros((0, 20))
    assert PSSM.shape[-1] == 20
    binder_length = PSSM.shape[0]
    # full soft sequence
    soft_sequence = jnp.concatenate(
        (
            jnp.pad(PSSM, [[0, 0], [0, 1]]),
            jax.nn.one_hot(features["aatype"][binder_length:], 21),
        )
    )

    L = features["aatype"].shape[0]
    
    #Do not touch this. One-hot seems necessary for multimer models to work properly.
    hard_pssm = (
        jax.lax.stop_gradient(
            jax.nn.one_hot(soft_sequence.argmax(-1), 21) - soft_sequence
        )
        + soft_sequence
    )
    msa_feat = (
        jnp.zeros((1, L, 49))
        .at[..., 0:21]
        .set(soft_sequence)
        .at[..., 25:46]
        .set(hard_pssm)
    )

    return features | {
        "msa_feat": msa_feat,
        "target_feat": soft_sequence,
        "aatype": jnp.argmax(soft_sequence, axis=-1),
    }


@dataclass
class AF2Output(AbstractStructureOutput):
    features: dict
    output: AFOutput

    @property
    def full_sequence(self):
        return jax.nn.one_hot(self.features["aatype"], 20)

    @property
    def asym_id(self):
        return self.features["asym_id"]

    @property
    def residue_idx(self):
        return self.features["residue_index"]

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
        return np.linspace(start=0.25, stop=31.75, num=64)


class AlphaFoldLoss(LossTerm):
    forward: callable
    stacked_params: PyTree
    features: dict
    loss: LinearCombination
    name: str
    initial_guess: any = None
    recycling_steps: int = 1

    def __call__(self, soft_sequence: Float[Array, "N 20"], *, key):
        # pick a random model
        model_idx = jax.random.randint(key=key, shape=(), minval=0, maxval=5)

        params = tree.map(lambda v: v[model_idx], self.stacked_params)

        output = self.forward(
            params,
            jax.random.fold_in(key, 1),
            features=set_binder_sequence(soft_sequence, self.features),
            initial_guess=None if self.initial_guess is None else self.initial_guess,
            recycling_steps=self.recycling_steps,
        )

        v, aux = self.loss(
            soft_sequence,
            AF2Output(
                features=self.features,
                output=output,
            ),
            key=key,
        )

        return v, {
            self.name: aux,
            f"{self.name}/model_idx": model_idx,
            f"{self.name}/loss": v,
        }


def af2_get_atom_positions_gemmi(st) -> tuple[np.ndarray, np.ndarray]:
    return tree.map(
        lambda *v: np.concatenate(v), *[af2_atom_positions(chain) for chain in st[0]]
    )


def af2_atom_positions(chain: gemmi.Chain) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance(chain, gemmi.Chain)
    all_residues = list(chain)
    num_res = len(all_residues)
    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num])

    for res_idx, res in enumerate(all_residues):
        for atom in res:
            atom_name = atom.name
            x, y, z = atom.pos.x, atom.pos.y, atom.pos.z
            if atom_name in residue_constants.atom_order.keys():
                all_positions[res_idx, residue_constants.atom_order[atom_name]] = [
                    x,
                    y,
                    z,
                ]
                all_positions_mask[res_idx, residue_constants.atom_order[atom_name]] = (
                    1.0
                )
            elif atom_name.upper() == "SE" and res.name() == "MSE":
                # Put the coordinates of the selenium atom in the sulphur column.
                all_positions[res_idx, residue_constants.atom_order["SD"]] = [x, y, z]
                all_positions_mask[res_idx, residue_constants.atom_order["SD"]] = 1.0

    return all_positions[None], all_positions_mask[None]


def make_af_features(chains: list[TargetChain]) -> dict[str, jax.Array]:
    assert all(not c.use_msa for c in chains), "AF2 interface does not support MSAs"

    # check for missing residues in template chains
    for c in chains:
        if c.template_chain is not None:
            gemmi_seq = gemmi.one_letter_code([r.name for r in c.template_chain])
            if gemmi_seq != c.sequence:
                raise Exception(f"Template sequence does not match sequence for {c}")

    # TODO: handle homo-multimers better?
    L = sum(len(c.sequence) for c in chains)
    index_within_chain = np.concatenate(
        [np.arange(len(c.sequence), dtype=int) for c in chains]
    )
    chain_index = np.concatenate(
        [
            np.full(shape=len(c.sequence), fill_value=idx + 1)
            for (idx, c) in enumerate(chains)
        ]
    )

    raw_features = {
        "target_feat": np.zeros((L, 20)),
        "msa_feat": np.zeros((1, L, 49)),
        "aatype": np.concatenate([tokenize(c.sequence) for c in chains]),
        "all_atom_positions": None,  # np.zeros((L, 37, 3)),
        "seq_mask": np.ones(L),
        "msa_mask": np.ones((1, L)),
        "residue_index": index_within_chain,
        "extra_deletion_value": np.zeros((1, L)),
        "extra_has_deletion": np.zeros((1, L)),
        "extra_msa": np.zeros((1, L), int),
        "extra_msa_mask": np.zeros((1, L)),
        "extra_msa_row_mask": np.zeros(1),
        "asym_id": chain_index,
        "sym_id": chain_index,
        "entity_id": chain_index,
    }

    template_features = [
        af2_atom_positions(tc.template_chain)
        if tc.template_chain
        else (
            np.zeros((1, len(tc.sequence), 37, 3)),
            np.zeros((1, len(tc.sequence), 37)),
        )
        for tc in chains
    ]
    template_positions, template_mask = jax.tree.map(
        lambda *v: jnp.concatenate(v, 1), *template_features
    )

    template_aatype = np.concatenate(
        [
            np.zeros(len(c.sequence), dtype=int)
            if not c.template_chain
            else tokenize(c.sequence)
            for c in chains
        ]
    )

    return raw_features | {
        "template_aatype": template_aatype[None],
        "template_all_atom_mask": template_mask,
        "template_all_atom_positions": template_positions,
    }


class AlphaFold2(eqx.Module, StructurePredictionModel):
    af2_forward: callable
    stacked_parameters: PyTree

    def __init__(self, data_dir: str = "."):
        (forward_function, stacked_params) = load_af2(data_dir=data_dir)
        self.af2_forward = forward_function
        self.stacked_parameters = stacked_params

    def target_only_features(self, chains: list[TargetChain]):
        for c in chains:
            assert c.polymer_type == "PROTEIN", "AF2 only supports protein chains"
            assert not c.use_msa, "AF2 interface does not support MSA yet"

        return make_af_features(chains=chains), None

    def binder_features(self, binder_length, chains: list[TargetChain]):
        features, _ = self.target_only_features(
            [TargetChain(sequence="G" * binder_length, use_msa=False)] + chains
        )
        return features, None

    def build_loss(
        self, *, loss, features, recycling_steps=1, sampling_steps=None, name="af2"
    ):
        assert sampling_steps is None, "AF2 does not support sampling steps"
        return AlphaFoldLoss(
            forward=self.af2_forward,
            stacked_params=self.stacked_parameters,
            features=features,
            loss=loss,
            recycling_steps=recycling_steps,
            name=name,
        )

    @eqx.filter_jit
    def _forward(
        self,
        PSSM,
        features,
        *,
        key,
        model_idx: int,
        recycling_steps: int,
        initial_guess=None,
    ):
        params = jax.tree.map(lambda v: v[model_idx], self.stacked_parameters)
        print("JIT compiling AF2...")
        # set binder sequence
        if PSSM is None:
            PSSM = jnp.zeros((0, 20))

        features = set_binder_sequence(PSSM, features)
        # run the model
        return self.af2_forward(
            params,
            jax.random.fold_in(key, 1),
            features=features,
            initial_guess=initial_guess,
            recycling_steps=recycling_steps,
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
    def _coords_and_confidences(
        self,
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
        (afo, pae, plddt, iptm) = self._coords_and_confidences(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=None,
            model_idx=model_idx,
            key=key,
        )

        _, structure = _postprocess_prediction(set_binder_sequence(PSSM, features), afo)

        return StructurePrediction(st=structure, plddt=plddt, pae=pae, iptm=iptm)
