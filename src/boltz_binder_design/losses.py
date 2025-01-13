import equinox as eqx
import jax
import jax.numpy as jnp
import joltz
import numpy as np
from jaxtyping import Array, Float, PyTree


def set_binder_sequence(
    new_sequence: Float[Array, "N 20"],
    features: PyTree,
):
    """Replace features related to first N tokens with `new_sequence.` Used for hallucination/binder design."""
    features = jax.tree.map(lambda v: v.astype(jnp.float32), features)
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
    mask = jnp.linspace(start=min_dist, stop=max_dist, num=64) < contact_dist
    return jax.nn.logsumexp(distogram_logits, where=mask, axis=-1)


def contact_cross_entropy(
    distogram_logits: Float[Array, "... N N 64"],
    contact_dist: float,
    min_dist=2.0,
    max_dist=22.0,
) -> Float[Array, "... N N"]:
    """Compute partial entropy (under distogram) that D_ij < contact_dist."""
    distogram_logits = jax.nn.log_softmax(distogram_logits)
    mask = jnp.linspace(start=min_dist, stop=max_dist, num=64) < contact_dist
    px_ = jax.nn.softmax(distogram_logits, where=mask, axis=-1)
    return (px_ * distogram_logits).sum(-1)


class LossTerm(eqx.Module):
    def __call__(self, *args, **kwds) -> tuple[float, dict]:
        raise NotImplementedError

    def __rmul__(self, scalar: float):
        if not (isinstance(scalar, float) or isinstance(scalar, int)):
            return NotImplemented

        return LinearCombination(losses=[self], weights=jnp.array([scalar]))

    def __add__(self, other):
        return 1.0 * self + 1.0 * other


class ConfidenceLoss(LossTerm):
    """Tag that indicates a loss term that needs access to confidence model outputs."""

    pass


class LinearCombination(eqx.Module):
    """Weighted linear combination of loss terms."""

    # losses: list[tuple[float, any]]
    losses: list[LossTerm]
    weights: jax.Array

    def __call__(self, *args, **kwargs) -> tuple[float, dict]:
        r = 0.0
        aux = {}
        for w, loss in zip(self.weights, self.losses):
            v, a = loss(*args, **kwargs)
            r += w * v
            aux.update({f"{k}": v for k, v in a.items()})
        return r, aux

    def __rmul__(self, scalar: float):
        if not (isinstance(scalar, float) or isinstance(scalar, int)):
            return NotImplemented

        return LinearCombination(
            losses=self.losses,
            weights=self.weights * scalar,
        )

    def __add__(self, other):
        if isinstance(other, LossTerm):
            other = 1.0 * other  # lift to LinearCombination

        if not isinstance(other, LinearCombination):
            return NotImplemented

        return LinearCombination(
            losses=self.losses + other.losses,
            weights=jnp.concatenate([self.weights, other.weights]),
        )


class StructurePrediction(LossTerm):
    model: joltz.Joltz1
    features: PyTree
    name: str
    loss: LossTerm | LinearCombination
    sampling_steps: int = 25
    recycling_steps: int = 0

    def __call__(self, binder_sequence, *, key):
        needs_confidence = (
            isinstance(self.loss, ConfidenceLoss)
            or isinstance(self.loss, LinearCombination)
            and any(isinstance(l, ConfidenceLoss) for l in self.loss.losses)
        )
        if needs_confidence:
            print(f"Using confidence model for {self.name} loss.")
        # TODO: This will fail for nested LinearCombinations
        features = set_binder_sequence(
            binder_sequence,
            self.features,
        )
        o = self.model(
            features,
            key=key,
            sample_structure=needs_confidence,
            confidence_prediction=needs_confidence,
            num_sampling_steps=self.sampling_steps,
            recycling_steps=self.recycling_steps,
        )
        # strip unneeded batch dimension
        o = jax.tree_map(lambda v: v[0], o)

        loss, aux = self.loss(binder_sequence, features, o)
        return loss, {
            self.name: loss,
        } | {self.name + "/" + k: v for k, v in aux.items()}


class HelixLoss(LossTerm):
    max_distance = 6.0

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        network_output: PyTree,
    ):
        binder_len = sequence.shape[0]
        log_contact = contact_log_probability(
            network_output["pdistogram"][:binder_len, :binder_len],
            self.max_distance,
        )
        print("helix: ", jnp.diagonal(log_contact, 3).shape)
        loss = jnp.diagonal(log_contact, 3).mean()

        return -loss, {"helix": loss}


class RadiusOfGyration(LossTerm):
    target_radius: float | None = None

    def __call__(
        self, sequence: Float[Array, "N 20"], features: PyTree, network_output: PyTree
    ):
        # TODO: Why RMSE instead of MAE?
        binder_len = sequence.shape[0]
        dgram_radius_of_gyration = jnp.sqrt(
            jnp.fill_diagonal(
                (
                    jax.nn.softmax(network_output["pdistogram"])[
                        :binder_len, :binder_len
                    ]
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


class DistogramCE(LossTerm):
    f: Float[Array, "... 64"]
    name: str

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        network_output: PyTree,
    ):
        binder_len = sequence.shape[0]
        # expand dims so self.f is broadcastable to network_output["pdistogram"] of size (N, N, 64)
        f = jnp.expand_dims(
            self.f, [i for i in range(network_output["pdistogram"].ndim - self.f.ndim)]
        )

        ce = jnp.fill_diagonal(
            (
                jax.nn.log_softmax(network_output["pdistogram"])[
                    :binder_len, :binder_len
                ]
                * f
            ).sum(-1),
            0,
            inplace=False,
        ).mean()

        return ce, {self.name: ce}


class DistogramExpectation(LossTerm):
    f: Float[Array, "... 64"]
    name: str

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        features: PyTree,
        network_output: PyTree,
    ):
        binder_len = sequence.shape[0]
        # expand dims so self.f is broadcastable to network_output["pdistogram"] of size (N, N, 64)
        f = jnp.expand_dims(
            self.f, [i for i in range(network_output["pdistogram"].ndim - self.f.ndim)]
        )

        expectation = jnp.fill_diagonal(
            (
                jax.nn.softmax(network_output["pdistogram"])[:binder_len, :binder_len]
                * f
            ).sum(-1),
            0,
            inplace=False,
        ).mean()

        return expectation, {self.name: expectation}


class SquaredRadiusOfGyration(LossTerm):
    target_radius: float | None = None

    def __call__(
        self, sequence: Float[Array, "N 20"], features: PyTree, network_output: PyTree
    ):
        # TODO: Why RMSE instead of MAE?
        binder_len = sequence.shape[0]
        dgram_radius_of_gyration = jnp.fill_diagonal(
            (
                jax.nn.softmax(network_output["pdistogram"])[:binder_len, :binder_len]
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


class MAERadiusOfGyration(LossTerm):
    target_radius: float | None = None

    def __call__(
        self, sequence: Float[Array, "N 20"], features: PyTree, network_output: PyTree
    ):
        # TODO: Why RMSE instead of MAE?
        binder_len = sequence.shape[0]

        dgram_radius_of_gyration = jnp.fill_diagonal(
            (
                jax.nn.softmax(network_output["pdistogram"])[:binder_len, :binder_len]
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


class WithinBinderContact(LossTerm):
    """Encourages contacts between residues."""

    min_sequence_separation: int = 8
    num_contacts_per_residue: int = 2

    def __call__(
        self, sequence: Float[Array, "N 20"], features: PyTree, network_output: PyTree
    ):
        binder_len = sequence.shape[0]
        log_contact_intra = contact_cross_entropy(network_output["pdistogram"], 14.0)
        # only count binder-binder contacts with sequence sep > 8
        within_binder_mask = (
            jnp.abs(jnp.arange(binder_len)[:, None] - jnp.arange(binder_len)[None, :])
            > self.min_sequence_separation
        )
        # for each position in binder find positions most likely to make contact
        binder_binder_max_p, _ = jax.vmap(
            lambda lcp: jax.lax.top_k(lcp, self.num_contacts_per_residue)
        )(log_contact_intra[:binder_len, :binder_len] + (1 - within_binder_mask) * -30)
        average_log_prob = binder_binder_max_p.mean()

        return -average_log_prob, {"intra_contact": average_log_prob}


class PLDDTLoss(ConfidenceLoss):
    def __call__(
        self, sequence: Float[Array, "N 20"], features: PyTree, network_output: PyTree
    ):
        plddt = network_output["plddt"].mean()
        return -plddt, {"plddt": plddt}


class BinderTargetContact(LossTerm):
    """Encourages contacts between binder and target."""

    paratope_idx: list[int] | None = None
    paratope_size: int | None = None
    contact_distance: float = 20.0

    def __call__(
        self, sequence: Float[Array, "N 20"], features: PyTree, network_output: PyTree
    ):
        binder_len = sequence.shape[0]
        log_contact_inter = contact_cross_entropy(
            network_output["pdistogram"][:binder_len, binder_len:],
            self.contact_distance,
        )
        # binder - target contact loss...
        # complex_length = network_output["pdistogram"].shape[0]
        # binder_target_mask = (
        #     jnp.zeros((complex_length,), dtype=bool).at[:binder_len].set(True)
        # )
        # binder_target_max_p = jnp.max(
        #     log_contact_inter,
        #     where=binder_target_mask[:, None] != binder_target_mask[None, :],
        #     initial=-1e8,
        #     axis=-1,
        # )[:binder_len]
        # binder_target_max_p = log_contact_inter[:binder_len, binder_len:].max(-1)
        binder_target_max_p = jax.vmap(lambda v: jax.lax.top_k(v, 3)[0])(
            log_contact_inter
        ).mean(-1)
        print("BTMP", binder_target_max_p.shape)
        # log probability of contacting target for each position in binder

        if self.paratope_idx is not None:
            binder_target_max_p = binder_target_max_p[self.paratope_idx]
            print("paratope", binder_target_max_p.shape)
        if self.paratope_size is not None:
            binder_target_max_p = jax.lax.top_k(
                binder_target_max_p, self.paratope_size
            )[0]

        average_log_prob = binder_target_max_p.mean()
        return -average_log_prob, {"target_contact": average_log_prob}


class BinderPAE(ConfidenceLoss):
    def __call__(self, tokens, features, output):
        binder_len = tokens.shape[0]
        pae_mat = output["pae"]

        pae_within = jnp.fill_diagonal(
            pae_mat[:binder_len, :binder_len], 0.0, inplace=False
        ).mean()
        return pae_within, {"bb_pae": pae_within}


class BinderTargetPAE(ConfidenceLoss):
    def __call__(self, tokens, features, output):
        binder_len = tokens.shape[0]
        pae_mat = output["pae"]

        off_diag = pae_mat[:binder_len, binder_len:]
        p = off_diag.mean()

        return p, {"bt_pae": p}


class TargetBinderPAE(ConfidenceLoss):
    def __call__(self, tokens, features, output):
        binder_len = tokens.shape[0]
        pae_mat = output["pae"]

        off_diag = pae_mat[binder_len:, :binder_len]
        p = off_diag.mean()

        return p, {"tb_pae": p}
