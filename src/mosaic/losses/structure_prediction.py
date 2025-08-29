import jax
from jaxtyping import Float, Array, Int

from abc import abstractmethod
import jax.numpy as jnp
import numpy as np
from mosaic.af2.confidence_metrics import predicted_tm_score, interaction_prediction_score

from ..common import LossTerm


# Each structure prediction model (AF2, boltz, boltz2, etc.) implements this interface for loss functionals
class AbstractStructureOutput:
    @property
    @abstractmethod
    def distogram_bins(self) -> Float[Array, "Bins"]:
        raise NotImplementedError

    @property
    @abstractmethod
    def distogram_logits(self) -> Float[Array, "N N Bins"]:
        raise NotImplementedError

    @property
    @abstractmethod
    def plddt(self) -> Float[Array, "N"]:
        raise NotImplementedError

    @property
    @abstractmethod
    def pae(self) -> Float[Array, "N N"]:
        raise NotImplementedError

    @property
    @abstractmethod
    def pae_logits(self) -> Float[Array, "N N Bins"]:
        raise NotImplementedError

    @property
    @abstractmethod
    def pae_bins(self) -> Float[Array, "Bins"]:
        raise NotImplementedError

    @property
    @abstractmethod
    def backbone_coordinates(self) -> Float[Array, "N 4 3"]:
        """
        Backbone coordinates of predicted structure in the order "N, CA, C, O".
        """
        raise NotImplementedError

    @property
    def ptm(self) -> Float[Array, "1"]:
        return predicted_tm_score(
            logits=self.pae_logits,
            breaks=self.pae_bins[:-1],  # not quite right but whatever, interface=False
        )

    @property
    @abstractmethod
    def full_sequence(self) -> Float[Array, "N 20"]:
        """
        Full sequence of the structure, including binder and target(s).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def asym_id(self) -> Float[Array, "N"]:
        raise NotImplementedError

    @property
    @abstractmethod
    def residue_idx(self) -> Int[Array, "N"]:
        """Residue index in each chain!"""
        raise NotImplementedError


def contact_cross_entropy(
    distogram_logits: Float[Array, "N N Bins"],
    contact_dist: float,
    bins: Float[Array, "Bins"],
) -> Float[Array, "... N N"]:
    """Compute partial entropy (under distogram) that D_ij < contact_dist."""
    assert bins.shape[-1] == distogram_logits.shape[-1]
    assert distogram_logits.ndim == 3

    distogram_logits = jax.nn.log_softmax(distogram_logits)
    contact_idx = np.searchsorted(
        bins,
        contact_dist,
    )

    px_ = jax.nn.softmax(distogram_logits[..., :contact_idx], axis=-1)

    return (px_ * distogram_logits[..., :contact_idx]).sum(-1)


def contact_log_probability(
    distogram_logits: Float[Array, "... N N 64"],
    contact_dist: float,
    bins: Float[Array, "Bins"],
) -> Float[Array, "... N N"]:
    """Compute log probability (under distogram) that D_ij < contact_dist."""
    assert bins.shape[-1] == distogram_logits.shape[-1]
    assert distogram_logits.ndim == 3
    distogram_logits = jax.nn.log_softmax(distogram_logits)
    mask = bins < contact_dist
    return jax.nn.logsumexp(distogram_logits, where=mask, axis=-1)


class WithinBinderContact(LossTerm):
    max_contact_distance: float = 14.0
    min_sequence_separation: int = 8
    num_contacts_per_residue: int = 25

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        log_contact_intra = contact_cross_entropy(
            output.distogram_logits[:binder_len, :binder_len],
            self.max_contact_distance,
            bins=output.distogram_bins,
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


class BinderTargetContact(LossTerm):
    paratope_idx: list[int] | None = None
    paratope_size: int | None = None
    contact_distance: float = 20.0
    epitope_idx: list[int] | None = None

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        log_contact_inter = contact_cross_entropy(
            output.distogram_logits[:binder_len, binder_len:],
            self.contact_distance,
            bins=output.distogram_bins,
        )
        if self.epitope_idx is not None:
            log_contact_inter = log_contact_inter[:, self.epitope_idx]

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


class HelixLoss(LossTerm):
    max_distance: float = 6.0
    target_value: float = -2.0

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        log_contact = contact_log_probability(
            output.distogram_logits[:binder_len, :binder_len],
            self.max_distance,
            bins=output.distogram_bins,
        )
        value = jnp.diagonal(log_contact, 3).mean()

        loss = jax.nn.elu(self.target_value - value)

        return loss, {"helix": loss}


class DistogramRadiusOfGyration(LossTerm):
    target_radius: float | None = None

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        # TODO: Why RMSE instead of MAE?
        binder_len = sequence.shape[0]
        dgram_radius_of_gyration = jnp.sqrt(
            jnp.fill_diagonal(
                (
                    jax.nn.softmax(output.distogram_logits)[:binder_len, :binder_len]
                    * (output.distogram_bins[None, None, :] ** 2)
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


class MAERadiusOfGyration(LossTerm):
    target_radius: float | None = None

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        binder_len = sequence.shape[0]

        dgram_radius_of_gyration = jnp.fill_diagonal(
            (
                jax.nn.softmax(output.distogram_logits)[:binder_len, :binder_len]
                * (output.distogram_bins[None, None, :])
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


class DistogramCE(LossTerm):
    f: Float[Array, "... Bins"]
    name: str
    l: float = -np.inf
    u: float = np.inf

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        # expand dims so self.f is broadcastable to network_output["pdistogram"] of size (N, N, Bins)
        f = jnp.expand_dims(self.f, [i for i in range(3 - self.f.ndim)])

        ce = -jnp.fill_diagonal(
            (
                jax.nn.log_softmax(output.distogram_logits)[:binder_len, :binder_len]
                * f
            ).sum(-1),
            0,
            inplace=False,
        ).mean()

        return ce.clip(self.l, self.u), {self.name: ce}


class PLDDTLoss(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        plddt = output.plddt[:binder_len].mean()
        return -plddt, {"plddt": plddt}


class WithinBinderPAE(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        pae_within = jnp.fill_diagonal(
            output.pae[:binder_len, :binder_len], 0, inplace=False
        ).mean()
        return pae_within, {"bb_pae": pae_within}


class BinderTargetPAE(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        pae = output.pae[:binder_len, binder_len:].mean()
        return pae, {"bt_pae": pae}


class TargetBinderPAE(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        pae = output.pae[binder_len:, :binder_len].mean()
        return pae, {"tb_pae": pae}


class IPTMLoss(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        # binder - target iptm -- we override asym-id in the case of multi-chain targets
        N = output.full_sequence.shape[0]
        asym_id = jnp.concatenate(
            (jnp.zeros(sequence.shape[0]), jnp.ones(N - sequence.shape[0]))
        ).astype(jnp.int32)
        logits = output.pae_logits
        if len(logits.shape) == 3:
            logits = logits[None]
        scores = jax.vmap(
            lambda logits: predicted_tm_score(
                asym_id=asym_id,
                logits=logits,
                breaks=output.pae_bins[:-1],
                interface=True,
            )
        )(logits)
        iptm = scores.mean()
        return iptm, {"iptm": iptm}

class IPSAELoss(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        # binder - target iptm -- we override asym-id in the case of multi-chain targets
        N = output.full_sequence.shape[0]
        asym_id = jnp.concatenate(
            (jnp.zeros(sequence.shape[0]), jnp.ones(N - sequence.shape[0]))
        ).astype(jnp.int32)
        logits = output.pae_logits
        if len(logits.shape) == 3:
            logits = logits[None]
        scores = jax.vmap(
            lambda logits: interaction_prediction_score(
                asym_id=asym_id,
                logits=logits,
                bin_centers=output.pae_bins,
                interface=True,
                pae_cutoff=15.0,
            )
        )(logits)
        ipsae = scores.mean()
        return ipsae, {"ipsae": ipsae}

class ActualRadiusOfGyration(LossTerm):
    target_radius: float

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        first_atom_coords = output.backbone_coordinates[:binder_len, 0]
        rg = jnp.sqrt(
            ((first_atom_coords - first_atom_coords.mean(0)) ** 2).sum(-1).mean()
        )

        return jax.nn.elu(rg - self.target_radius), {"actual_rg": rg}


class pTMEnergy(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        len_binder = sequence.shape[0]
        logits = output.pae_logits
        num_res = logits.shape[0]
        # Clip num_res to avoid negative/undefined d0.
        clipped_num_res = max(num_res, 19)

        d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

        pae_bin_centers = output.pae_bins

        g_d_b = 1.0 / (1 + jnp.square(pae_bin_centers) / jnp.square(d0))
        energy = jax.scipy.special.logsumexp(a=logits, b=g_d_b, axis=-1)
        # return negative mean over cross-chain pairs
        binder_target = energy[:len_binder, len_binder:].mean()
        target_binder = energy[len_binder:, :len_binder].mean()
        E = -(binder_target + target_binder) / 2
        return E, {"pTMEnergy": E}
