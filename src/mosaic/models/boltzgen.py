#####
#
# Note: this is pretty rushed, will come back and clean up later
# data loading and structure writing is **terrible**
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import joltzgen
import numpy as np
import torch
from boltzgen.data import const
from boltzgen.data.data import (
    Structure,
    convert_ccd,
)
from boltzgen.data.feature.featurizer import (
    Featurizer,
    res_all_gly,
    res_from_atom14,
    res_from_atom37,
)
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.model.models.boltz import Boltz
from boltzgen.model.modules.masker import BoltzMasker
from boltzgen.task.predict.data_from_yaml import DataConfig, FromYamlDataModule
from boltzgen.task.predict.writer import DesignWriter
from jaxtyping import Array, Float, PyTree

from mosaic.losses.structure_prediction import AbstractStructureOutput
from ..util import pairwise_distance



def load_boltzgen(checkpoint_dir=Path("~/.boltz/").expanduser(), model_diverse=True):
    checkpoints = ["boltzgen1_adherence.ckpt", "boltzgen1_diverse.ckpt"]
    if not all((checkpoint_dir / ckpt).exists() for ckpt in checkpoints):
        print(f"Downloading Boltz folding checkpoints to {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for ckpt in checkpoints:
            subprocess.run(
                [
                    "wget",
                    "-O",
                    str(checkpoint_dir / ckpt),
                    f"https://huggingface.co/boltzgen/boltzgen-1/resolve/main/{ckpt}?download=true",
                ],
            )
            # ugh, torch is trash
            cpkt = torch.load(
                checkpoint_dir / ckpt, map_location="cpu", weights_only=False
            )
            del cpkt["hyper_parameters"]["validators"]  # these contain GPU tensors
            del cpkt["validators"]
            torch.save(cpkt, checkpoint_dir / ckpt)

        subprocess.run(
            [
                "wget",
                "-O",
                str(checkpoint_dir / "mols.zip"),
                "https://huggingface.co/datasets/boltzgen/inference-data/resolve/main/mols.zip?download=true",
            ]
        )

    torch_model = Boltz.load_from_checkpoint(
        checkpoint_dir / (checkpoints[0] if not model_diverse else checkpoints[1]),
        strict=True,
        map_location="cpu",
    ).eval()
    torch_model.structure_module.time_dilation = 2.667

    model = joltzgen.from_torch(torch_model)
    _model_params, _model_static = eqx.partition(model, eqx.is_inexact_array)
    return eqx.combine(jax.device_put(_model_params), _model_static)


def _generate_mmcif(
    self,
    prediction: any = None,
    batch: any = None,
    sample_id: str = None,
) -> None:
    if prediction["exception"]:
        self.failed += 1
        return
    n_samples, _, _ = prediction["coords"].shape

    # TODO: remove this which is only here for temporary backward compatibility
    masker = BoltzMasker(mask=True, mask_backbone=False, mask_disto=True)
    feat_masked = masker(batch)
    prediction["ref_element"] = feat_masked["ref_element"]
    prediction["ref_atom_name_chars"] = feat_masked["ref_atom_name_chars"]
    """Write the predictions to disk."""
    # Check for extra molecules
    if batch["extra_mols"] is not None:
        extra_mols = batch["extra_mols"][0]
        for k, v in extra_mols.items():
            with open(self.mol_dir / f"{k}.pkl", "wb") as f:
                pickle.dump(v, f)

    # write samples to disk
    for n in range(n_samples):
        # get structure for all generated coords
        sample, native = {}, {}

        for k in set(prediction.keys()) & set(batch.keys()):
            if k == "coords":
                native[k] = batch[k][0][0].unsqueeze(0)
                sample[k] = prediction[k][n]

            if k in const.token_features:
                sample[k] = prediction[k][0]
                native[k] = batch[k][0]
            elif k in const.atom_features:
                if k == "coords":
                    native[k] = batch[k][0][0].unsqueeze(0)
                    sample[k] = prediction[k][n]
                else:
                    native[k] = batch[k][0]
                    sample[k] = prediction[k][0]
            elif k == "exception":
                sample[k] = prediction[k]
                native[k] = batch[k]
            else:
                # print(k)
                # print(batch[k].shape)
                try:
                    if batch[k] is not None:
                        native[k] = batch[k][0]
                        sample[k] = prediction[k][0]
                        native[k] = batch[k][0]
                except Exception as e:
                    print(e)

        if self.atom14:
            sample = res_from_atom14(sample)
        elif self.atom37:
            sample = res_from_atom37(sample)
        elif self.backbone_only:
            sample = res_all_gly(sample)

        design_mask = batch["design_mask"][0].bool()
        assert design_mask.sum() == sample["design_mask"].sum()

        if self.inverse_fold:
            token_ids = torch.argmax(sample["res_type"], dim=-1)
            tokens = [const.tokens[i] for i in token_ids]
            ccds = [convert_ccd(token) for token in tokens]

            ccds = torch.tensor(ccds).to(sample["res_type"])
            sample["ccd"][design_mask] = ccds[design_mask]

        try:
            structure, _, _ = Structure.from_feat(sample)
            str_native, _, _ = Structure.from_feat(native)

            # write structure to cif

            # design mask bfactor
            design_mask = batch["design_mask"][0].float()
            atom_design_mask = (
                sample["atom_to_token"].float() @ design_mask.unsqueeze(-1).float()
            )
            design_mask = native["design_mask"].float()

            atom_design_mask = atom_design_mask.squeeze().bool()
            bfactor = atom_design_mask * 100

            # binding type bfactor
            binding_type = batch["binding_type"][0].float()
            atom_binding_type = (
                sample["atom_to_token"].float() @ binding_type.unsqueeze(-1).float()
            )

            atom_binding_type = atom_binding_type.squeeze().bool()
            binding_type = native["binding_type"].float()
            bfactor[atom_binding_type == const.binding_type_ids["BINDING"]] = 60

            bfactor = atom_design_mask[sample["atom_pad_mask"].bool()].float()
            str_native.atoms["bfactor"] = bfactor.cpu().numpy()
            structure.atoms["bfactor"] = bfactor.cpu().numpy()

            # Add dummy (0-coord) design side chains if inverse fold
            if self.inverse_fold:
                atom_design_mask_no_pad = atom_design_mask[
                    native["atom_pad_mask"].bool()
                ]
                res_design_mask = np.array(
                    [
                        all(
                            atom_design_mask_no_pad[
                                res["atom_idx"] : res["atom_idx"] + res["atom_num"]
                            ]
                        )
                        for res in structure.residues
                    ]
                )
                structure = Structure.add_side_chains(
                    structure, residue_mask=res_design_mask
                )

            pred_binding_mask = prediction["binding_type"][0].cpu().bool().numpy()
            if self.design:
                chain_design_mask = (
                    prediction["chain_design_mask"][0].cpu().bool().numpy()
                )
            pred_design_mask = prediction["design_mask"][0].cpu().bool().numpy()
            design_color_features = np.ones_like(pred_binding_mask) * 0.8
            design_color_features[pred_binding_mask] = 1.0
            if self.design:
                design_color_features[chain_design_mask] = 0.0
            design_color_features[pred_design_mask] = 0.6

            # Create a mask to identify unique token-to-res mappings.
            # This is for small molecules where multiple tokens can be mapped to the same residue.
            token_to_res = prediction["token_to_res"][0].cpu().numpy()
            unique_mask = np.ones_like(token_to_res, dtype=bool)
            unique_mask[1:] = token_to_res[1:] != token_to_res[:-1]
            design_color_features = design_color_features[unique_mask]
            return gemmi.make_structure_from_block(
                gemmi.cif.read_string(
                    to_mmcif(
                        structure,
                        design_coloring=True,
                        color_features=design_color_features,
                    )
                )[0]
            )

        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()  # noqa: T201
            msg = f"predict/writer.py: Validation structure writing failed on {batch['id'][0]} with error {e}. Skipping."
            print(msg)


@dataclass
class BoltzGenWriter:
    writer: any
    torch_features: dict

    def __call__(self, coords: Float[Array, "... 3"]):
        return _generate_mmcif(
            self.writer,
            prediction=self.torch_features
            | {
                "coords": torch.tensor(np.array(coords)),
                "exception": False,
                "masks": self.torch_features["atom_pad_mask"].unsqueeze(0),
                "extra_mols": None,
                "structure_bonds": [torch.zeros(0)],  # hope you don't need bonds!
            },
            batch=self.torch_features
            | {
                "extra_mols": None,
                "target_msa_mask": torch.zeros(1, 1, 1),
                "structure_bonds": [torch.zeros(0)],  # lol
            },
        )


def load_features_and_structure_writer(
    yaml_string: str,
    moldir: Path = Path("~/.boltz/").expanduser() / "mols.zip",
    files: dict[str, Path] = {},
):
    with TemporaryDirectory() as temp_dir:
        with open(f"{temp_dir}/yaml.yaml", "w") as yaml_file:
            yaml_file.write(yaml_string)
            yaml_file.flush()

        for filename, p in files.items():
            dest_file = Path(f"{temp_dir}/{filename}")
            with open(p, "rb") as src_file, open(dest_file, "wb") as dest_file:
                dest_file.write(src_file.read())

        dataset_config = DataConfig(
            yaml_path=yaml_file.name,
            multiplicity=1,  # Multiplicity isn't used in get_sample, 1 is safe
            tokenizer=Tokenizer(),
            featurizer=Featurizer(),
            moldir=moldir,
            atom14=True,
            backbone_only=False,
            atom37=False,
            disulfide_prob=1.0,
            disulfide_on=True,
        )
        datamodule = FromYamlDataModule(
            dataset_config, batch_size=1, num_workers=0, pin_memory=False
        )
        dl = datamodule.predict_dataloader()

        features = next(iter(dl))

    features["structure_bonds"] = []
    torch_masker = BoltzMasker(mask=True, mask_backbone=False, mask_disto=True)
    features = torch_masker(features)

    # convert features to jax
    j_features = jax.tree.map(
        lambda v: jnp.array(v) if isinstance(v, torch.Tensor) else v, features
    ) | {"cyclic_period": jnp.zeros((1, 1))}

    j_features["msa"] = jax.nn.one_hot(j_features["msa"], num_classes=const.num_tokens)

    output_dir = TemporaryDirectory(delete=False).name

    return (
        j_features,
        BoltzGenWriter(
            DesignWriter(
                output_dir=output_dir,
                res_atoms_only=False,
                atom14=True,
                atom37=False,
                write_native=False,
            ),
            features,
        ),
    )


class Sampler(eqx.Module):
    """Hold conditioner information for repeated sampling from stucture module. Can be vmapped, jitted, etc."""

    trunk_s: Float[Array, "N S"]
    s_inputs: Float[Array, "N S"]
    feats: dict[str, any]
    q: Float[Array, "..."]
    c: Float[Array, "..."]
    to_keys: any
    atom_enc_bias: Float[Array, "..."]
    atom_dec_bias: Float[Array, "..."]
    token_trans_bias: Float[Array, "..."]

    @eqx.filter_jit
    @staticmethod
    def from_features(
        *,
        model: joltzgen.JoltzGen,
        features: dict[str, any],
        recycling_steps: int,
        key,
        deterministic: bool = True,
    ):
        initial_embedding = model.embed_inputs(features)

        trunk_state, key = model.recycle(
            initial_embedding=initial_embedding,
            recycling_steps=recycling_steps,
            feats=features,
            key=key,
            deterministic=deterministic,
        )

        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
            model.diffusion_conditioning(
                trunk_state.s,
                trunk_state.z,
                initial_embedding.relative_position_encoding,
                features,
            )
        )

        return Sampler(
            trunk_s=trunk_state.s,
            s_inputs=initial_embedding.s_inputs,
            feats=features,
            q=q,
            c=c,
            to_keys=to_keys,
            atom_enc_bias=atom_enc_bias,
            atom_dec_bias=atom_dec_bias,
            token_trans_bias=token_trans_bias,
        )

    def __call__(
        self,
        *,
        structure_module: joltzgen.AtomDiffusion,
        num_sampling_steps: int,
        step_scale: float,
        noise_scale: float,
        key,
        sample_schedule="dilated",
    ):
        return structure_module.sample(
            s_trunk=self.trunk_s,
            s_inputs=self.s_inputs,
            feats=self.feats,
            num_sampling_steps=num_sampling_steps,
            atom_mask=self.feats["atom_pad_mask"],
            multiplicity=1,
            diffusion_conditioning={
                "q": self.q,
                "c": self.c,
                "to_keys": self.to_keys,
                "atom_enc_bias": self.atom_enc_bias,
                "atom_dec_bias": self.atom_dec_bias,
                "token_trans_bias": self.token_trans_bias,
            },
            key=jax.random.fold_in(key, 2),
            step_scale=step_scale,
            noise_scale=noise_scale,
            sample_schedule=sample_schedule,
        )

def _coords_to_restype(coords, *, des_idx, threshold: float = 0.5):
    design_coords = coords[des_idx]
    design_coords = design_coords.reshape(len(design_coords) // 14, 14, 3)

    # For each sidechain atom, compute closest backbone atom and count them
    # while excluding those side chain atoms whose distance is above a threshold
    distances = pairwise_distance(
        design_coords[:, :4], design_coords[:, 4:]
    )  # torch.cdist(design_coords[:, :4], design_coords[:, 4:])
    value, argmin = jnp.min(distances, axis=1), jnp.argmin(distances, axis=1)
    argmin = jnp.where(value > threshold, -1, argmin)
    arange = jnp.arange(len(const.ref_atoms["GLY"]))
    counts = (argmin[:, :, None] == arange[None, None, :]).sum(1)
    # counts is num_res x 4
    with jax.ensure_compile_time_eval():
        count_matrix = np.zeros((20, 4))
        for k, v in const.placement_count_to_token.items():
            if const.token_ids[v] != 22:
                count_matrix[const.token_ids[v] - 2] = k

    dists = ((counts[:, None, :] - count_matrix[None, :, :]) ** 2).sum(-1)

    return dists.argmin(-1)


class CoordsToToken(eqx.Module):
    """Convert sampled coordinates to mosaic token indices. Class to make precomputing some things outside of JIT easier..."""

    des_idx: np.ndarray

    def __init__(self, features: dict[str, any]):
        design_mask = np.array(features["design_mask"]).astype(bool)
        mol_type = features["mol_type"]
        atom_to_token = np.array(features["atom_to_token"])
        token_index = np.array(features["token_index"])
        atom_pad_mask = np.array(features["atom_pad_mask"])
        design_mask = np.logical_and(
            design_mask, mol_type == const.chain_type_ids["PROTEIN"]
        )
        # Get designed atom coordinates in shape N//14 x 14 x 3
        atom_to_token = np.argmax(atom_to_token, axis=-1)
        token_indices = token_index[design_mask.astype(bool)]
        atom_design_mask = np.isin(atom_to_token, token_indices)
        atom_design_mask = np.logical_and(atom_design_mask, atom_pad_mask)
        self.des_idx = np.nonzero(atom_design_mask[0])

    @eqx.filter_jit
    def __call__(self, coords: Float[Array, "... 3"]):
        return _coords_to_restype(coords, des_idx=self.des_idx)

@dataclass
class BoltzGenOutput(AbstractStructureOutput):
    sample: jax.Array
    features: PyTree
    coords2token: CoordsToToken

    @property
    def full_sequence(self):
        binder_sequence = self.coords2token(self.sample[0])
        binder_sequence = jax.nn.one_hot(binder_sequence, 20, dtype=jnp.int32)
        binder_len = binder_sequence.shape[0]
        return self.features["res_type"][0, :, 2:22].at[:binder_len].set(binder_sequence)

    @property
    def asym_id(self):
        return self.features["asym_id"][0]

    @property
    def residue_idx(self):
        return self.features["residue_index"][0]

    @property
    def backbone_coordinates(self):
        # could precompute the index in load_features to avoid slow operation alarm
        bb_atom_inds = jnp.argmax(self.features["token_to_bb4_atoms"][0], axis=-1)
        return self.sample[0][bb_atom_inds]

    @property
    def structure_coordinates(self):
        return self.sample

    @property
    def ptm(self):
        raise NotImplementedError


