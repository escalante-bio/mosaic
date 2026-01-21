from mosaic.structure_prediction import (
    StructurePredictionModel,
    TargetChain,
    StructurePrediction,
)
from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.common import tokenize
from mosaic.alphafold.common import residue_constants, protein
from mosaic.alphafold.model import config, data, modules_multimer, modules, state
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
    recycling_state: state.AlphaFoldState


def load_af2(data_dir: str = "~/.alphafold", multimer=True):
    data_dir = Path(data_dir).expanduser()

    if not (data_dir / "params").exists():
        print(f"Downloading AF2 parameters to {data_dir}/params...")
        import httpx
        import tarfile

        params_dir = data_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
        tar_path = params_dir / "alphafold_params_2022-12-06.tar"

        with httpx.stream("GET", url, follow_redirects=True) as r:
            r.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)

        with tarfile.open(tar_path) as tar:
            tar.extractall(path=params_dir)
        tar_path.unlink()

    try:
        model_params = [
            data.get_model_haiku_params(model_name=model_name, data_dir=data_dir)
            for model_name in tqdm(
                [f"model_{i}_{'multimer_v3' if multimer else 'ptm'}" for i in range(1, 6 if multimer else 3)],
                desc="Loading AF2 params",
            )
        ]
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find AF2 parameters in {data_dir}/params. {e}"
        )
    stacked_model_params = tree.map(lambda *v: np.stack(v), *model_params)

    cfg = config.model_config("model_1_multimer_v3" if multimer else "model_1_ptm")
    cfg.max_msa_clusters = 1
    cfg.max_extra_msa = 1
    cfg.masked_msa_replace_fraction = 0
    cfg.subbatch_size = None
    cfg.model.num_ensemble_eval = 1
    cfg.model.global_config.subbatch_size = None
    cfg.model.global_config.eval_dropout = False
    cfg.model.global_config.deterministic = True
    cfg.model.global_config.use_remat = True
    cfg.model.num_extra_msa = 1
    cfg.model.resample_msa_in_recycling = False

    #haiku transform forward function
    init_model = modules_multimer.AlphaFold if multimer else modules.AlphaFold
    def _forward_fn(
        features: dict,
        previous_rep: state.AlphaFoldState,
        use_dropout=False,
        **kwargs,
    ) -> AFOutput:
        print("JIT compiling AF2...")
        model = init_model(cfg.model)
        prediction_results, state = model(
            batch=features,
            prev_rep=previous_rep,
            use_dropout=use_dropout,
            **kwargs,
        )
        confidences = confidence_metrics(prediction_results)
        return AFOutput(
            distogram=Distogram(**prediction_results["distogram"]),
            iptm=confidences["iptm"],
            predicted_aligned_error=confidences["predicted_aligned_error"],
            pae_logits=prediction_results["predicted_aligned_error"]["logits"],
            pae_bin_centers=_calculate_bin_centers(
                prediction_results["predicted_aligned_error"]["breaks"]
            ),
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
            recycling_state=state,
        )

    transformed = hk.transform(_forward_fn)
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

def multimer_to_monomer_features(features: dict):

    monomer_features = {}
    has_break = jnp.concatenate([jnp.array([0]), jnp.diff(features['asym_id'])])
#    between_segment_residues = jnp.where(has_break, 1, 0) #this doesnt seem to matter much
    between_segment_residues = np.zeros(features['asym_id'].shape, dtype=int)
    target_feat = jnp.concatenate([
        between_segment_residues[:, None],
        jax.nn.one_hot(features['aatype'], 21)
        ], axis=-1)
    monomer_features['target_feat'] = target_feat
    monomer_features['residue_index'] = jnp.cumsum(has_break)*50 + jnp.arange(features['asym_id'].size)
    monomer_features['template_all_atom_masks'] = features['template_all_atom_mask']
    monomer_features['template_mask'] = np.ones(1)

    monomer_features['template_pseudo_beta'], monomer_features['template_pseudo_beta_mask'] = jax.vmap(modules.pseudo_beta_fn)(
        features['template_aatype'], features['template_all_atom_positions'], features['template_all_atom_mask']
        )

    return features | monomer_features


def set_binder_sequence(PSSM, features: dict, multimer: bool=True):
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

    # Do not touch this. One-hot seems necessary for multimer models to work properly.
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

    out = features | {
        "msa_feat": msa_feat,
        "target_feat": soft_sequence,
        "aatype": jnp.argmax(soft_sequence, axis=-1),
    }
    
    if not multimer:
        return multimer_to_monomer_features(out)
    return out


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
        "all_atom_positions": np.zeros((L, 37, 3)),
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


class AlphaFold2(StructurePredictionModel):
    af2_forward: callable
    stacked_parameters: PyTree
    multimer: bool

    def __init__(self, data_dir: str = "~/.alphafold", multimer=True):
        (forward_function, stacked_params) = load_af2(data_dir=data_dir, multimer=multimer)
        self.af2_forward = forward_function
        self.stacked_parameters = stacked_params
        self.multimer = multimer

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
        self, *, loss, features, recycling_steps=1, sampling_steps=None, name="af2", use_dropout=False, initial_state=None
    ):
        assert sampling_steps is None, "AF2 does not support sampling steps"
        return AlphaFoldLoss(
            model=self,
            features=features,
            loss=loss,
            recycling_steps=recycling_steps,
            name=name,
            use_dropout=use_dropout,
            initial_state=initial_state,
        )

    def model_output(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        recycling_steps=1,
        sampling_steps=None,
        model_idx: int | None = None,
        use_dropout: bool = False,
        recycling_state: state.AlphaFoldState | None = None,
        key,
    ):
        features = set_binder_sequence(PSSM, features, self.multimer)
        N = features["aatype"].shape[0]

        if model_idx is None:
            model_idx = jax.random.randint(key=key, shape=(), minval=0, maxval=5 if self.multimer else 2)
            key = jax.random.fold_in(key, 0)
        else:
            model_idx = jax.device_put(model_idx)

        if recycling_state is None:
            recycling_state = state.AlphaFoldState(
                prev_pos=jnp.zeros((N, residue_constants.atom_type_num, 3)),
                prev_msa_first_row=jnp.zeros((N, 256)),
                prev_pair=jnp.zeros((N, N, 128)),
            )

        params = jax.tree.map(lambda v: v[model_idx], self.stacked_parameters)

        # recycling iterations
        def body_fn(state: state.AlphaFoldState, _):
            state = jax.tree.map(jax.lax.stop_gradient, state)
            output = self.af2_forward(
                params,
                jax.random.fold_in(key, 1),
                features=features,
                previous_rep=state,
                use_dropout=use_dropout,
            )
            return output.recycling_state, output

        _, outputs = jax.lax.scan(
            body_fn,
            recycling_state,
            length=recycling_steps,
        )

        return AF2Output(
            features=features, output=jax.tree.map(lambda v: v[-1], outputs)
        )

    @eqx.filter_jit
    def _coords_and_confidences(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        recycling_steps=1,
        model_idx: int | None = None,
        use_dropout: bool = False,
        recycling_state: state.AlphaFoldState | None = None,
        key,
    ):
        output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            model_idx=model_idx,
            key=key,
            use_dropout=use_dropout,
            recycling_state=recycling_state,
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
        use_dropout: bool = False,
        recycling_state: state.AlphaFoldState | None = None,
        key,
    ) -> StructurePrediction:
        (afo, pae, plddt, iptm) = self._coords_and_confidences(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            model_idx=model_idx,
            key=key,
            use_dropout=use_dropout,
            recycling_state=recycling_state,
        )

        _, structure = _postprocess_prediction(set_binder_sequence(PSSM, features, self.multimer), afo)

        return StructurePrediction(st=structure, plddt=plddt, pae=pae, iptm=iptm)



class AlphaFoldLoss(LossTerm):
    model: AlphaFold2
    features: dict
    loss: LinearCombination
    name: str
    initial_state: state.AlphaFoldState | None = None
    recycling_steps: int = 1
    use_dropout: bool = False

    def __call__(self, PSSM: Float[Array, "N 20"], *, key):
        # pick a random model
        
        model_idx = jax.random.randint(key=key, shape=(), minval=0, maxval=5 if self.model.multimer else 2)
        key = jax.random.fold_in(key, 0)
        output = self.model.model_output(
            PSSM=PSSM,
            features=self.features,
            recycling_steps=self.recycling_steps,
            sampling_steps=None,
            model_idx=model_idx,
            key=key,
            use_dropout=self.use_dropout,
            recycling_state=self.initial_state,
        )

        v, aux = self.loss(
            PSSM,
            output=output,
            key=key,
        )

        return v, {
            self.name: aux,
            f"{self.name}/model_idx": model_idx,
            f"{self.name}/loss": v,
        }
