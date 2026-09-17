"""Local Biohub checkpoints must support Mosaic inference and sequence gradients."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from esm.models.esmc import EsmcConfig, EsmcForMaskedLM
from esm.models.esmfold2 import EsmFold2Config, EsmFold2ExperimentalModel, EsmFold2Model
from esm.models.esmfold2.protein_utils import prepare_protein_features
from safetensors.torch import save_file

from mosaic.common import TOKENS, LossTerm
from mosaic.losses.esmc import ESMCPseudoLikelihood, load_esmc
from mosaic.models import esmfold2 as adapter


@pytest.fixture
def config():
    return EsmFold2Config(
        hidden_size=32,
        pairwise_hidden_size=16,
        num_loops=1,
        num_diffusion_samples=1,
        lm_d_model=32,
        lm_num_layers=2,
        folding_trunk_num_hidden_layers=1,
        folding_trunk_num_attention_heads=2,
        atom_encoder={
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
        },
        structure_head={
            "inference_num_steps": 2,
            "distogram_bins": 8,
            "diffusion_module": {
                "atom_encoder": {
                    "hidden_size": 64,
                    "num_attention_heads": 2,
                    "num_hidden_layers": 1,
                },
                "token_hidden_size": 32,
                "c_z": 16,
                "fourier_dim": 16,
                "token_num_blocks": 1,
                "token_num_heads": 2,
            },
        },
        confidence_head={
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "distogram_bins": 8,
            "num_plddt_bins": 8,
            "num_pae_bins": 64,
            "num_pde_bins": 8,
        },
        parcae_num_coda_layers=1,
        lm_encoder={"num_hidden_layers": 1},
        msa_encoder={
            "enabled": True,
            "hidden_size": 32,
            "outer_hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "head_width": 8,
        },
    )


def _save_checkpoint(model, path):
    model.config.save_pretrained(path)
    save_file(model.state_dict(), path / "model.safetensors")


def _assert_gradient(gradient):
    assert gradient.shape == (2, 20)
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 0


def test_native_esmc_loading_and_sequence_gradient(tmp_path, monkeypatch):
    monkeypatch.setenv("MOSAIC_CACHE_DIR", str(tmp_path / "cache"))
    torch.manual_seed(4)
    model = EsmcForMaskedLM(
        EsmcConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2)
    ).eval()
    _save_checkpoint(model, tmp_path)
    converted = load_esmc(str(tmp_path))
    ids = np.array([[0, 5, 6, 2]], dtype=np.int32)
    with torch.no_grad():
        expected = model(torch.from_numpy(ids)).logits.numpy()
    actual = eqx.filter_jit(converted)(jnp.asarray(ids))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    loss = ESMCPseudoLikelihood(converted, stop_grad=False)
    (value, _), grad = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))(
        jnp.full((2, 20), 0.05), key=jax.random.key(0)
    )
    assert np.isfinite(value)
    _assert_gradient(grad)


class _DistogramLoss(LossTerm):
    def __call__(self, sequence, output, *, key):
        value = jnp.mean(jnp.square(output.distogram_logits))
        return value, {"distogram": value}


@pytest.mark.parametrize("experimental", [False, True])
def test_native_esmfold_loading_prediction_and_design(
    config, experimental, tmp_path, monkeypatch
):
    monkeypatch.setenv("MOSAIC_CACHE_DIR", str(tmp_path / "cache"))
    torch.manual_seed(0)
    config.type = "experimental" if experimental else "release"
    # Experimental design is single-sequence; its MSA encoder requires integer IDs.
    config.msa_encoder.enabled = not experimental
    config.esmc_config = EsmcConfig(
        hidden_size=32, num_attention_heads=4, num_hidden_layers=2
    )
    cls = EsmFold2ExperimentalModel if experimental else EsmFold2Model
    _save_checkpoint(cls(config).eval(), tmp_path)

    # Use the native protein-only featurizer to keep this checkpoint test offline.
    raw_alphabet = prepare_protein_features(TOKENS)
    perm = jax.nn.one_hot(np.asarray(raw_alphabet["res_type"])[0], 33)
    unk = int(prepare_protein_features("X")["input_ids"][0, 0])
    ids = np.asarray(raw_alphabet["input_ids"])[0]
    monkeypatch.setattr(adapter, "_probe_alphabets", lambda: (perm, unk, ids))
    model = adapter._make(str(tmp_path), experimental=experimental)
    pack = adapter._build_pack(
        model.esmc,
        prepare_protein_features("WWG"),
        design_positions=np.array([0, 1]),
        unk_input_id=unk,
    )
    pssm = jnp.full((2, 20), 0.05)
    output = model.model_output(
        PSSM=pssm,
        features=pack,
        recycling_steps=1,
        sampling_steps=2,
        key=jax.random.key(0),
    )
    assert output.distogram_logits.shape == (3, 3, 8)
    assert output.pae.shape == (3, 3)
    for leaf in jax.tree.leaves(output):
        assert np.isfinite(np.asarray(leaf)).all()
    np.testing.assert_allclose(output.full_sequence[:2], pssm)

    loss = model.build_loss(
        loss=_DistogramLoss(),
        features=pack,
        recycling_steps=1,
        sampling_steps=2,
    )
    # Refresh the binder LM, as in a design pack, without needing CCD geometry.
    loss = replace(loss, refresh_lm=True)
    (value, _), grad = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))(
        pssm, key=jax.random.key(1)
    )
    assert np.isfinite(value)
    _assert_gradient(grad)
