"""Each backend's default path (jfold's featurization, and for Mol-1 and ProMERA a
prediction too) imports neither torch nor tinyprot. Runs on the CPU, in a fresh process."""

import os
import subprocess
import sys

import pytest

SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQ"
FEATURIZE = {
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
        "features, writer = model.target_only_features(\n"
        "    [TargetChain(SEQUENCE, use_msa=False)], key=jax.random.key(0)\n"
        ")\n"
        "model.predict(features=features, writer=writer, recycling_steps=1,\n"
        "              sampler=model.sampler(2), key=jax.random.key(1))\n"
    ),
    "promera": (
        "from mosaic.models.promera import Promera\n"
        "model = Promera()\n"
        "features, writer = model.binder_features(\n"
        "    4, [TargetChain(SEQUENCE, use_msa=False)], key=jax.random.key(0)\n"
        ")\n"
        "model.predict(PSSM=jax.nn.one_hot(jax.numpy.arange(4), 20), features=features,\n"
        "              writer=writer, recycling_steps=1, sampler=model.upstream_sampler(2),\n"
        "              key=jax.random.key(1))\n"
    ),
}


@pytest.mark.parametrize(
    "backend",
    [
        *FEATURIZE,
        *(pytest.param(name, marks=pytest.mark.model_forward) for name in PREDICT),
    ],
)
def test_default_path_is_torch_free(backend):
    script = (
        "import sys\n"
        "import jax\n"
        "from mosaic.structure_prediction import TargetChain\n"
        f"SEQUENCE = {SEQUENCE!r}\n"
        + (FEATURIZE | PREDICT)[backend]
        + "imported = {m.split('.')[0] for m in sys.modules} & {'torch', 'tinyprot'}\n"
        "assert not imported, imported\n"
    )
    env = os.environ | {"JAX_PLATFORMS": "cpu"}
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
