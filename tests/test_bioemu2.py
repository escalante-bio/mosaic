"""BioEmu2 (biojemu2) wrapper: sequence injection, sampler, losses, prediction and
ensemble sampling.

Loads the converted ``bioemu2-preview`` weights (downloaded and converted on first use).
"""

import biojemu2
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.common import TOKENS
from mosaic.losses.bioemu2 import set_binder_sequence
from mosaic.losses.structure_prediction import BinderTargetContact
from mosaic.models.bioemu2 import BioEmu2Preview
from mosaic.sampling import Noise, integrate, sample
from mosaic.sampling.bioemu2 import bioemu2_denoiser, bioemu2_sampler
from mosaic.structure_prediction import PolymerType, TargetChain

pytestmark = pytest.mark.slow

TARGET = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
BINDER = "GWYKCPSHDLIFEGMRAQNTVG"  # every residue type


@pytest.fixture(scope="module")
def model():
    return BioEmu2Preview()


@pytest.fixture(scope="module")
def chains():
    return [
        TargetChain(TARGET, use_msa=False),
        TargetChain("GACTCG", polymer_type=PolymerType.DNA, use_msa=False),
    ]


def one_hot(seq):
    return jax.nn.one_hot(jnp.array([TOKENS.index(c) for c in seq]), 20)


def test_binder_sequence_matches_featurization(model, chains):
    """Setting the design pack's sequence gives the sequence channels of featurizing it."""
    design, _ = model.binder_features(len(BINDER), chains)
    native, _ = model.target_only_features([TargetChain(BINDER, use_msa=False), *chains])
    f = set_binder_sequence(one_hot(BINDER), design.features)
    g = native.features
    np.testing.assert_array_equal(f.tokens.res_type, g.tokens.res_type)
    np.testing.assert_array_equal(f.tokens.profile, g.tokens.profile)
    np.testing.assert_array_equal(f.trunk.msa, g.trunk.msa)


@pytest.mark.parametrize("mode", ["structure", "equilibrium"])
def test_sampler_matches_biojemu2(model, mode):
    """`bioemu2_sampler` on mosaic's sampler reproduces upstream's loop from the same noise."""
    pack, _ = model.target_only_features([TargetChain(TARGET, use_msa=False)])
    f = pack.features
    with jax.default_matmul_precision("highest"):
        trunk = eqx.filter_jit(lambda m, f: m.embed(f, num_recycles=0))(model.bioemu2, f)
        denoise, atom_mask = bioemu2_denoiser(model.bioemu2, f, trunk)
        noise = Noise.draw(jax.random.key(0), num_steps=20, num_atoms=atom_mask.shape[0])
        ours = integrate(denoise, atom_mask, bioemu2_sampler(model.bioemu2, 20, mode), noise)
        preset = {"structure": biojemu2.EDMSampler.structure_prediction,
                  "equilibrium": biojemu2.EDMSampler.equilibrium}[mode]  # fmt: skip
        theirs = biojemu2.sample_from_noise(
            model.bioemu2,
            model.bioemu2.static_context(f, trunk),
            f.dynamics,
            preset(20),
            biojemu2.Noise(noise.init, noise.churn, noise.rotations, noise.translations),
        )
    err = np.abs(ours - theirs)[np.asarray(atom_mask) > 0].max()
    assert err < 2e-3, err


def test_loss_gradient(model, chains):
    design, _ = model.binder_features(len(BINDER), chains)
    loss = model.build_multisample_loss(
        loss=BinderTargetContact(), features=design, recycling_steps=2, num_samples=2
    )
    pssm = jax.nn.softmax(jax.random.normal(jax.random.key(0), (len(BINDER), 20)))

    @eqx.filter_jit
    def value_and_grad(loss, pssm, key):
        return jax.value_and_grad(lambda p: loss(p, key=key)[0])(pssm)

    value, grad = value_and_grad(loss, pssm, jax.random.key(1))
    assert np.isfinite(value) and np.isfinite(grad).all() and np.abs(grad).max() > 0


def test_predict(model, chains):
    design, _ = model.binder_features(len(BINDER), chains)
    sampler = model.inference_sampler(20)
    pred = model.predict(
        PSSM=one_hot(BINDER), features=design, recycling_steps=2, sampler=sampler,
        key=jax.random.key(0),
    )  # fmt: skip
    assert len(pred.st[0][0]) == len(BINDER)  # atom37 view: binder first
    assert np.isnan(pred.plddt).all() and np.isnan(pred.iptm)

    native, system = model.target_only_features(chains)
    pred = model.predict(
        features=native, writer=system, recycling_steps=2, sampler=sampler,
        key=jax.random.key(0),
    )  # fmt: skip
    assert [len(ch) for ch in pred.st[0]] == [len(TARGET), 6]


def test_ensemble(model):
    """Trunk once, `jax.vmap` the sampler over keys: distinct all-atom samples."""
    pack, system = model.target_only_features([TargetChain(TARGET, use_msa=False)])
    denoise, atom_mask = model.denoiser(features=pack, recycling_steps=2)
    sampler = model.inference_sampler(20)
    keys = jax.random.split(jax.random.key(0), 3)
    coords = jax.vmap(lambda k: sample(denoise, atom_mask, sampler, key=k))(keys)
    assert np.isfinite(coords).all()
    assert not np.allclose(coords[0], coords[1])
    structures = [model.structure(system, x) for x in coords]
    assert all(st[0].count_atom_sites() == int(atom_mask.sum()) for st in structures)
