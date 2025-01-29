from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import joltz
import numpy as np
import torch
from boltz.data.const import ref_atoms
from boltz.main import (
    BoltzDiffusionParams,
    BoltzInferenceDataModule,
    BoltzProcessedInput,
    BoltzWriter,
    Manifest,
    check_inputs,
    process_inputs,
)
from boltz.model.model import Boltz1
from jax import tree
from jaxtyping import Array, Float, PyTree
from joltz import StructureModuleOutputs, TrunkOutputs

from ..common import LinearCombination, LossTerm
from ..proteinmpnn.mpnn import ProteinMPNN
from .protein_mpnn import boltz_to_mpnn_matrix


def load_boltz_model(
    checkpoint_path: Path = Path("~/.boltz/boltz1_conf.ckpt").expanduser(),
):
    predict_args = {
        "recycling_steps": 0,
        "sampling_steps": 25,
        "diffusion_samples": 1,
    }

    _torch_model = Boltz1.load_from_checkpoint(
        checkpoint_path,
        strict=True,
        map_location="cpu",
        predict_args=predict_args,
        diffusion_process_args=asdict(BoltzDiffusionParams()),
        ema=False,
    )

    model = joltz.from_torch(_torch_model)
    _model_params, _model_static = eqx.partition(model, eqx.is_inexact_array)
    model = eqx.combine(jax.device_put(_model_params), _model_static)

    return model


class StructureWriter:
    """
    Hacky class to write predicted structures to disk using a BoltzWriter
    """

    writer: BoltzWriter
    atom_pad_mask: torch.Tensor
    record: any
    out_dir: str

    def __init__(
        self,
        *,
        features_dict,
        target_dir: Path,
        output_dir: Path,
        output_format: str = "pdb",
    ):
        self.writer = BoltzWriter(
            data_dir=target_dir,
            output_dir=output_dir,
            output_format=output_format,
        )
        self.atom_pad_mask = features_dict["atom_pad_mask"].unsqueeze(0)
        self.record = features_dict["record"][0]
        self.out_dir = output_dir

    def __call__(self, sample_atom_coords):
        confidence = torch.ones(1)

        pred_dict = {
            "exception": False,
            "coords": torch.tensor(np.array(sample_atom_coords)).unsqueeze(0),
            "masks": self.atom_pad_mask,
            "confidence_score": confidence,
        }
        self.writer.write_on_batch_end(
            None,
            None,
            pred_dict,
            None,
            {"record": [self.record]},
            None,
            None,
        )
        # TODO: return path to output structure
        return (Path(self.out_dir) / self.record.id) / f"{self.record.id}_model_0.pdb"


def binder_fasta_seq(binder_len: int) -> str:
    return f">A|protein|empty\n{'X' * binder_len}\n"


def target_fasta_seq(
    sequence: str, polymer_type: str = "protein", use_msa=True, chain="B"
):
    return f">{chain}|{polymer_type}{'|empty' if not use_msa else ''}\n{sequence}\n"


def make_binder_features(
    binder_len: int,
    target_sequence: str,
    target_polymer_type: str = "protein",
    use_msa=True,
    out_dir: Path | None = None,
):
    if out_dir is None:
        out_dir = Path(f"{binder_len}_{target_sequence}")

    out_dir.mkdir(exist_ok=True, parents=True)

    fasta_path = out_dir / "protein.fasta"
    fasta_path.write_text(
        binder_fasta_seq(binder_len)
        + target_fasta_seq(target_sequence, target_polymer_type, use_msa)
    )
    return load_features_and_structure_writer(fasta_path, out_dir)


def make_binder_monomer_features(monomer_len: int, out_dir: Path | None = None):
    return make_monomer_features(
        "X" * monomer_len, out_dir, use_msa=False, polymer_type="protein"
    )


def make_monomer_features(
    seq: str, out_dir: Path | None = None, use_msa=True, polymer_type: str = "protein"
):
    if out_dir is None:
        out_dir = Path(f"monomer_{seq}")

    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "protein.fasta").write_text(
        target_fasta_seq(seq, polymer_type=polymer_type, use_msa=use_msa, chain="A")
    )
    return load_features_and_structure_writer(out_dir / "protein.fasta", out_dir)


def load_features_and_structure_writer(
    input_fasta_path: Path,
    out_dir: Path,
    cache=Path("~/.boltz/").expanduser(),
) -> tuple[PyTree, StructureWriter]:
    print("Loading data")
    data = check_inputs(input_fasta_path, out_dir, override=True)
    # Process inputs
    ccd_path = cache / "ccd.pkl"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        use_msa_server=True,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
    )
    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
    )

    # Create data module
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=0,
    )

    # Load the features for the single example
    features_dict = list(data_module.predict_dataloader())[0]

    # convert features to numpy arrays
    features = {
        k: jnp.array(v).astype(jnp.float32)
        for k, v in features_dict.items()
        if k != "record"
    }
    # set up structure writer
    writer = StructureWriter(
        features_dict=features_dict,
        target_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
    )
    return features, writer


def set_binder_sequence(
    new_sequence: Float[Array, "N 20"],
    features: PyTree,
):
    """Replace features related to first N tokens with `new_sequence.` Used for hallucination/binder design."""
    features = tree.map(lambda v: v.astype(jnp.float32), features)
    assert len(new_sequence.shape) == 2
    assert new_sequence.shape[1] == 20
    binder_len = new_sequence.shape[0]

    # We only use the standard 20 amino acids, but boltz has 33 total tokens.
    # zero out non-standard AA types
    zero_padded_sequence = jnp.pad(new_sequence, ((0, 0), (2, 11)))
    n_msa = features["msa"].shape[1]
    print("n_msa", n_msa)

    # We assume there are no MSA hits for the binder sequence
    binder_profile = jnp.zeros_like(features["profile"][0, :binder_len])
    binder_profile = binder_profile.at[:binder_len].set(zero_padded_sequence) / n_msa
    binder_profile = binder_profile.at[:, 1].set((n_msa - 1) / n_msa)

    return features | {
        "res_type": features["res_type"]
        .at[0, :binder_len, :]
        .set(zero_padded_sequence),
        "msa": features["msa"].at[0, 0, :binder_len, :].set(zero_padded_sequence),
        "profile": features["profile"].at[0, :binder_len].set(binder_profile),
    }


def contact_log_probability(
    distogram_logits: Float[Array, "... N N 64"],
    contact_dist: float,
    min_dist=2.0,
    max_dist=22.0,
) -> Float[Array, "... N N"]:
    """Compute log probability (under distogram) that D_ij < contact_dist."""
    distogram_logits = jax.nn.log_softmax(distogram_logits)
    mask = (
        jnp.linspace(start=min_dist, stop=max_dist, num=distogram_logits.shape[-1])
        < contact_dist
    )
    return jax.nn.logsumexp(distogram_logits, where=mask, axis=-1)


def contact_cross_entropy(
    distogram_logits: Float[Array, "... N N 64"],
    contact_dist: float,
    min_dist=2.0,
    max_dist=22.0,
) -> Float[Array, "... N N"]:
    """Compute partial entropy (under distogram) that D_ij < contact_dist."""
    distogram_logits = jax.nn.log_softmax(distogram_logits)
    mask = (
        jnp.linspace(start=min_dist, stop=max_dist, num=distogram_logits.shape[-1])
        < contact_dist
    )
    px_ = jax.nn.softmax(distogram_logits, where=mask, axis=-1)
    return (px_ * distogram_logits).sum(-1)


class ConfidenceLoss(LossTerm):
    """Loss term that needs access to confidence model outputs."""

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        trunk_output: TrunkOutputs,
        structure_output: StructureModuleOutputs,
        confidence_output: PyTree,
        key=None,
    ):
        raise NotImplementedError


class StructureLoss(LossTerm):
    """Loss term that needs access to structure module outputs."""

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        trunk_output: TrunkOutputs,
        structure_output: StructureModuleOutputs,
        key=None,
    ):
        raise NotImplementedError


class TrunkLoss(LossTerm):
    """Loss term that only needs access to trunk outputs."""

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        trunk_output: TrunkOutputs,
        key=None,
    ):
        raise NotImplementedError


class StructurePrediction(LossTerm):
    model: joltz.Joltz1
    features: PyTree
    name: str
    loss: LinearCombination
    sampling_steps: int = 25
    recycling_steps: int = 0

    def __call__(self, binder_sequence, *, key):
        # this is fairly ugly and needs to be sorted out eventually
        # but what's going on here is we really have three classes of loss terms that rely on different outputs from boltz
        # 1. trunk loss terms that only need the trunk outputs
        # 2. structure loss terms that need the structure outputs
        # 3. confidence loss terms that need the confidence outputs
        # We only run the structure and confidence modules if necessary

        # Replace binder sequence
        features = set_binder_sequence(
            binder_sequence,
            self.features,
        )
        # run the trunk
        trunk_embedding = self.model.trunk(features, self.recycling_steps)
        sampled_structure = None
        confidence_output = None
        total_loss = 0.0
        aux = {}

        def strip_batch_dim(t: PyTree):
            return tree.map(lambda v: v[0], t)

        def _sample_structure(key):
            nonlocal sampled_structure
            if sampled_structure is None:
                sampled_structure = self.model.sample_structure(
                    features, trunk_embedding, self.sampling_steps, key
                )

        for w, loss in zip(self.loss.weights, self.loss.losses):
            match loss:
                case StructureLoss():
                    _sample_structure(key)
                    key = jax.random.fold_in(key, 1)
                    l, a = loss(
                        binder_sequence,
                        features,
                        *(strip_batch_dim((trunk_embedding, sampled_structure))),
                        key=key,
                    )
                    key = jax.random.fold_in(key, 1)
                    total_loss += w * l
                    aux = aux | {f"{self.name}/{k}": v for k, v in a.items()}
                case ConfidenceLoss():
                    _sample_structure(key)
                    key = jax.random.fold_in(key, 1)
                    if confidence_output is None:
                        confidence_output = self.model.predict_confidence(
                            features, trunk_embedding, sampled_structure
                        )

                    l, a = loss(
                        binder_sequence,
                        features,
                        *strip_batch_dim(
                            (trunk_embedding, sampled_structure, confidence_output)
                        ),
                        key=key,
                    )
                    key = jax.random.fold_in(key, 1)
                    total_loss += w * l
                    aux = aux | {f"{self.name}/{k}": v for k, v in a.items()}
                case TrunkLoss():
                    l, a = loss(
                        binder_sequence,
                        features,
                        strip_batch_dim(trunk_embedding),
                        key=key,
                    )
                    key = jax.random.fold_in(key, 1)
                    total_loss += w * l
                    aux = aux | {f"{self.name}/{k}": v for k, v in a.items()}
                case _:
                    raise ValueError(
                        f"Unsupported loss type: {loss.__class__.__name__}"
                    )

        return total_loss, {
            self.name: total_loss,
        } | aux


class HelixLoss(TrunkLoss):
    max_distance = 6.0
    target_value: float = -2.0

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        trunk_output: TrunkOutputs,
        key=None,
    ):
        binder_len = sequence.shape[0]
        log_contact = contact_log_probability(
            trunk_output.pdistogram[:binder_len, :binder_len],
            self.max_distance,
        )
        value = jnp.diagonal(log_contact, 3).mean()

        loss = jax.nn.elu(self.target_value - value)

        return loss, {"helix": loss}


class RadiusOfGyration(TrunkLoss):
    target_radius: float | None = None

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        trunk_output: TrunkOutputs,
        key=None,
    ):
        # TODO: Why RMSE instead of MAE?
        binder_len = sequence.shape[0]
        dgram_radius_of_gyration = jnp.sqrt(
            jnp.fill_diagonal(
                (
                    jax.nn.softmax(trunk_output.pdistogram)[:binder_len, :binder_len]
                    * (np.linspace(2, 22, 64)[None, None, :] ** 2)
                ).sum(-1),  # expected squared distance
                0,
                inplace=False,
            ).mean()
            + 1e-8
        )

        rg_th = (
            2.38 * binder_len**0.365
            if self.target_radius is None
            else self.target_radius
        )
        return jax.nn.elu(dgram_radius_of_gyration - rg_th), {
            "radius_of_gyration": dgram_radius_of_gyration
        }


class DistogramCE(TrunkLoss):
    f: Float[Array, "... 64"]
    name: str

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        trunk_output: TrunkOutputs,
        key=None,
    ):
        binder_len = sequence.shape[0]
        # expand dims so self.f is broadcastable to network_output["pdistogram"] of size (N, N, 64)
        f = jnp.expand_dims(
            self.f, [i for i in range(trunk_output.pdistogram.ndim - self.f.ndim)]
        )

        ce = -jnp.fill_diagonal(
            (
                jax.nn.log_softmax(trunk_output.pdistogram)[:binder_len, :binder_len]
                * f
            ).sum(-1),
            0,
            inplace=False,
        ).mean()

        return ce, {self.name: ce}


class DistogramExpectation(TrunkLoss):
    f: Float[Array, "... 64"]
    name: str

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        trunk_output: TrunkOutputs,
        key=None,
    ):
        binder_len = sequence.shape[0]
        # expand dims so self.f is broadcastable to network_output["pdistogram"] of size (N, N, 64)
        f = jnp.expand_dims(
            self.f, [i for i in range(trunk_output.pdistogram.ndim - self.f.ndim)]
        )

        expectation = jnp.fill_diagonal(
            (jax.nn.softmax(trunk_output.pdistogram)[:binder_len, :binder_len] * f).sum(
                -1
            ),
            0,
            inplace=False,
        ).mean()

        return expectation, {self.name: expectation}


class SquaredRadiusOfGyration(TrunkLoss):
    target_radius: float | None = None

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        trunk_output: TrunkOutputs,
        key=None,
    ):
        # TODO: Why RMSE instead of MAE?
        binder_len = sequence.shape[0]
        dgram_radius_of_gyration = jnp.fill_diagonal(
            (
                jax.nn.softmax(trunk_output.pdistogram)[:binder_len, :binder_len]
                * (np.linspace(2, 22, 64)[None, None, :] ** 2)
            ).sum(-1),  # expected squared distance
            0,
            inplace=False,
        ).mean()

        rg_th = (
            2.38 * binder_len**0.365
            if self.target_radius is None
            else self.target_radius
        )
        return jax.nn.elu(dgram_radius_of_gyration - rg_th**2), {
            "sq_radius_of_gyration": dgram_radius_of_gyration
        }


class MAERadiusOfGyration(TrunkLoss):
    target_radius: float | None = None

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        trunk_output: TrunkOutputs,
        key=None,
    ):
        # TODO: Why RMSE instead of MAE?
        binder_len = sequence.shape[0]

        dgram_radius_of_gyration = jnp.fill_diagonal(
            (
                jax.nn.softmax(trunk_output.pdistogram)[:binder_len, :binder_len]
                * (np.linspace(2, 22, 64)[None, None, :])
            ).sum(-1),  # expected squared distance
            0,
            inplace=False,
        ).mean()

        rg_th = (
            2.38 * binder_len**0.365
            if self.target_radius is None
            else self.target_radius
        )
        return jax.nn.elu(dgram_radius_of_gyration - rg_th), {
            "radius_of_gyration": dgram_radius_of_gyration
        }


class WithinBinderContact(TrunkLoss):
    """Encourages contacts between residues."""

    max_contact_distance: float = 14.0
    min_sequence_separation: int = 8
    num_contacts_per_residue: int = 25

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        trunk_output: TrunkOutputs,
        key=None,
    ):
        binder_len = sequence.shape[0]
        log_contact_intra = contact_cross_entropy(
            trunk_output.pdistogram[:binder_len, :binder_len], self.max_contact_distance
        )
        # only count binder-binder contacts with sequence sep > min_sequence_separation
        within_binder_mask = (
            jnp.abs(jnp.arange(binder_len)[:, None] - jnp.arange(binder_len)[None, :])
            > self.min_sequence_separation
        )
        # for each position in binder find positions most likely to make contact
        binder_binder_max_p, _ = jax.vmap(
            lambda lcp: jax.lax.top_k(lcp, self.num_contacts_per_residue)
        )(log_contact_intra + (1 - within_binder_mask) * -30)
        average_log_prob = binder_binder_max_p.mean()

        return -average_log_prob, {"intra_contact": average_log_prob}


class PLDDTLoss(ConfidenceLoss):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        trunk_output: TrunkOutputs,
        structures: StructureModuleOutputs,
        confidences: PyTree,
        key=None,
    ):
        plddt = confidences["plddt"].mean()
        return -plddt, {"plddt": plddt}


class BinderTargetContact(TrunkLoss):
    """Encourages contacts between binder and target."""

    paratope_idx: list[int] | None = None
    paratope_size: int | None = None
    contact_distance: float = 20.0

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        trunk_output: TrunkOutputs,
        key=None,
    ):
        binder_len = sequence.shape[0]
        log_contact_inter = contact_cross_entropy(
            trunk_output.pdistogram[:binder_len, binder_len:],
            self.contact_distance,
        )

        # binder_target_max_p = log_contact_inter[:binder_len, binder_len:].max(-1)
        binder_target_max_p = jax.vmap(lambda v: jax.lax.top_k(v, 3)[0])(
            log_contact_inter
        ).mean(-1)
        # log probability of contacting target for each position in binder

        if self.paratope_idx is not None:
            binder_target_max_p = binder_target_max_p[self.paratope_idx]
        if self.paratope_size is not None:
            binder_target_max_p = jax.lax.top_k(
                binder_target_max_p, self.paratope_size
            )[0]

        average_log_prob = binder_target_max_p.mean()
        return -average_log_prob, {"target_contact": average_log_prob}


class BinderPAE(ConfidenceLoss):
    def __call__(
        self, tokens, features, trunk_output, structures, confidences, key=None
    ):
        binder_len = tokens.shape[0]

        pae_within = jnp.fill_diagonal(
            confidences["pae"][:binder_len, :binder_len], 0.0, inplace=False
        ).mean()

        return pae_within, {"bb_pae": pae_within}


class BinderTargetPAE(ConfidenceLoss):
    def __call__(
        self, tokens, features, trunk_output, structures, confidences, key=None
    ):
        binder_len = tokens.shape[0]

        p = confidences["pae"][:binder_len, binder_len:].mean()

        return p, {"bt_pae": p}


class TargetBinderPAE(ConfidenceLoss):
    def __call__(
        self, tokens, features, trunk_output, structures, confidences, key=None
    ):
        binder_len = tokens.shape[0]

        p = confidences["pae"][binder_len:, :binder_len].mean()

        return p, {"tb_pae": p}


class ActualRadiusOfGyration(StructureLoss):
    target_radius: float

    def __call__(self, tokens, features, trunk_output, structure, key):
        features = jax.tree_map(lambda x: x[0], features)
        binder_len = tokens.shape[0]

        first_atom_idx = jax.vmap(lambda atoms: jnp.nonzero(atoms, size=1)[0][0])(
            features["atom_to_token"].T
        )
        first_atom_coords = structure.sample_atom_coords[first_atom_idx]
        first_atom_coords = first_atom_coords[:binder_len]
        rg = jnp.sqrt(
            ((first_atom_coords - first_atom_coords.mean(0)) ** 2).sum(-1).mean()
        )

        return jax.nn.elu(rg - self.target_radius), {"actual_rg": rg}


class BoltzProteinMPNNLoss(StructureLoss):
    """Average log-likelihood of binder sequence given Boltz-predicted complex structure

    Args:

        mpnn: ProteinMPNN
        num_samples: int
        stop_grad: bool = True : Whether to stop gradient through the structure module output

    """

    mpnn: ProteinMPNN
    num_samples: int
    stop_grad: bool = True

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        trunk_output: TrunkOutputs,
        structure_output: StructureModuleOutputs,
        *,
        key,
    ):
        if self.stop_grad:
            structure_output = jax.tree.map(jax.lax.stop_gradient, structure_output)

        features = jax.tree_map(lambda x: x[0], features)

        binder_length = sequence.shape[0]

        total_length = features["res_type"].shape[0]
        # Get the atoms required for proteinMPNN:
        # In order these are N, C-alpha, C, O
        assert ref_atoms["UNK"][:4] == ["N", "CA", "C", "O"]
        # first step, which is a bit cryptic is to get the first atom for each token
        first_atom_idx = jax.vmap(lambda atoms: jnp.nonzero(atoms, size=1)[0][0])(
            features["atom_to_token"].T
        )
        # NOTE: this is completely fail if any tokens are non-protein!
        all_atom_coords = structure_output.sample_atom_coords
        coords = jnp.stack([all_atom_coords[first_atom_idx + i] for i in range(4)], -2)

        full_sequence = jnp.concatenate(
            [sequence, features["res_type"][binder_length:, 2:22]], 0
        )
        sequence_mpnn = full_sequence @ boltz_to_mpnn_matrix()
        mpnn_mask = jnp.ones(total_length, dtype=jnp.int32)
        # adjust residue idx by chain
        asym_id = features["asym_id"]
        # hardcode max number of chains = 16
        chain_lengths = (asym_id[:, None] == np.arange(16)[None]).sum(-2)
        # vector of length 16 with length of each chain
        res_idx_adjustment = jnp.cumsum(chain_lengths, -1) - chain_lengths
        # now add res_idx_adjustment to each chain
        residue_idx = (
            features["residue_index"]
            + (asym_id[:, None] == np.arange(16)[None]) @ res_idx_adjustment
        )
        # this is why I dislike vectorized code
        # add 100 residue gap to match proteinmpnn
        residue_idx += 100 * asym_id

        # alright, we have all our features.
        # encode the fixed structure
        h_V, h_E, E_idx = self.mpnn.encode(
            X=coords,
            mask=mpnn_mask,
            residue_idx=residue_idx,
            chain_encoding_all=asym_id,
        )

        def decoder_LL(key):
            # MPNN is cheap, let's call the decoder a few times to average over random decoding order
            # generate a decoding order
            # this should be random but end with the binder
            decoding_order = (
                jax.random.uniform(key, shape=(total_length,))
                .at[:binder_length]
                .add(2.0)
            )

            logits = self.mpnn.decode(
                S=sequence_mpnn,
                h_V=h_V,
                h_E=h_E,
                E_idx=E_idx,
                mask=mpnn_mask,
                decoding_order=decoding_order,
            )[0]

            return (
                (logits[:binder_length] * (sequence @ boltz_to_mpnn_matrix()))
                .sum(-1)
                .mean()
            )

        binder_ll = (
            jax.vmap(decoder_LL)(jax.random.split(key, self.num_samples))
        ).mean()

        return -binder_ll, {"protein_mpnn_ll": binder_ll}
