"""Check that each backend's default path runs without importing torch."""

import os
import subprocess

import pytest

SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQ"
FEATURIZE = {
    "mol1": (
        "from mosaic.models.mol1 import feature_pack, featurize, mol1_system\n"
        "feature_pack(featurize(mol1_system([TargetChain(SEQUENCE, use_msa=False)]), key=jax.random.key(0)))\n"
    ),
    "promera": (
        "from mosaic.models.promera import view_features\n"
        "view_features([TargetChain(SEQUENCE, use_msa=False)], key=jax.random.key(0))\n"
    ),
    "boltz2": (
        "from mosaic.models.boltz2 import target_only_features\n"
        "target_only_features([TargetChain(SEQUENCE, use_msa=False)], key=jax.random.key(0))\n"
    ),
    "openfold3": (
        "from mosaic.models.openfold3 import target_only_features\n"
        "target_only_features([TargetChain(SEQUENCE, use_msa=False)], key=jax.random.key(0))\n"
    ),
    "bioemu2": (
        "from mosaic.models.bioemu2 import featurize\n"
        "featurize([TargetChain(SEQUENCE, use_msa=False)], mode='structure')\n"
    ),
}
PREDICT = {
    "mol1": (
        "from mosaic.models.mol1 import Mol1\n"
        "model = Mol1()\n"
        "features = model.target_only_features(\n"
        "    [TargetChain(SEQUENCE, use_msa=False)], key=jax.random.key(0)\n"
        ")\n"
        "model.predict(features=features, recycling_steps=1,\n"
        "              sampler=model.sampler(2), key=jax.random.key(1))\n"
    ),
    "promera": (
        "from mosaic.models.promera import Promera\n"
        "model = Promera()\n"
        "features = model.binder_features(\n"
        "    4, [TargetChain(SEQUENCE, use_msa=False)], key=jax.random.key(0)\n"
        ")\n"
        "model.predict(PSSM=jax.nn.one_hot(jax.numpy.arange(4), 20), features=features,\n"
        "              recycling_steps=1, sampler=model.upstream_sampler(2),\n"
        "              key=jax.random.key(1))\n"
    ),
}


@pytest.mark.parametrize("backend", FEATURIZE)
def test_featurization_is_torch_free(backend):
    run_without_torch(FEATURIZE[backend])


@pytest.mark.model_forward
@pytest.mark.parametrize("backend", PREDICT)
def test_prediction_is_torch_free(backend):
    run_without_torch(PREDICT[backend])


def run_without_torch(code):
    script = (
        "import sys\n"
        "import jax\n"
        "from mosaic.structure_prediction import TargetChain\n"
        f"SEQUENCE = {SEQUENCE!r}\n"
        + code
        + "imported = {m.split('.')[0] for m in sys.modules} & {'torch'}\n"
        "assert not imported, imported\n"
    )
    env = os.environ | {"JAX_PLATFORMS": "cpu"}
    subprocess.run(["uv", "run", "python", "-c", script], check=True, env=env)
