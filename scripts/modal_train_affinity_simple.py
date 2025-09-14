import os
import sys
from pathlib import Path

import modal


LOCAL_SRC = "/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/src"

image = (
    modal.Image.debian_slim(python_version="3.12.0")
    .apt_install("git")
    .env({
        "BOLTZ_CACHE": "/root/.boltz",
        "JAX_PLATFORMS": "cuda",
        "XLA_FLAGS": "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found --xla_gpu_cuda_data_dir=/usr/local/lib/python3.12/site-packages/nvidia",
        "LD_LIBRARY_PATH": ":".join([
            "/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
            "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib",
            "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib",
            "/usr/local/lib/python3.12/site-packages/nvidia/cusolver/lib",
            "/usr/local/lib/python3.12/site-packages/nvidia/cusparse/lib",
            "/usr/local/lib/python3.12/site-packages/nvidia/nvjitlink/lib",
        ]),
    })
    .pip_install("pip>=25.0", "setuptools>=80.0", "wheel>=0.45.0")
    .run_commands("python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.1+cpu")
    .run_commands(
        "python -m pip install jax==0.7.1 && "
        "python -m pip install jax-cuda12-plugin==0.7.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && "
        "python -m pip install nvidia-cuda-nvcc-cu12==12.8.93 nvidia-nvjitlink-cu12==12.8.93"
    )
    .pip_install(
        "nvidia-cuda-runtime-cu12==12.8.90",
        "nvidia-cuda-nvrtc-cu12==12.8.93",
        "nvidia-cuda-cupti-cu12==12.8.90",
        "nvidia-cublas-cu12==12.8.4.1",
        "nvidia-cufft-cu12==11.3.3.83",
        "nvidia-curand-cu12==10.3.9.90",
        "nvidia-cusolver-cu12==11.7.3.90",
        "nvidia-cusparse-cu12==12.5.8.93",
        "nvidia-nccl-cu12==2.27.3",
        "nvidia-cudnn-cu12==9.10.2.21",
    )
    .pip_install("optax==0.2.4", "equinox==0.11.7", "pandas>=2.2.0", "numpy>=1.26.0", "loguru>=0.7.2")
    .run_commands(
        "python -m pip install git+https://github.com/nboyd/joltz.git && "
        "python -m pip install boltz==2.2.1"
    )
    .add_local_dir(LOCAL_SRC, "/app/src")
)

app = modal.App("affinity-simple", image=image)

boltz_cache = modal.Volume.from_name("boltz-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("results-kcat", create_if_missing=True)
data_vol = modal.Volume.from_name("kcat-data", create_if_missing=True)


def _add_src_paths():
    # embedded source added at image build time
    src = Path("/app/src")
    if src.exists():
        sys.path.insert(0, str(src))


@app.function(
    gpu="H100",
    timeout=2 * 60 * 60,
    volumes={"/root/.boltz": boltz_cache, "/results": results_vol, "/data": data_vol},
)
def run_simple_train(
    *,
    data_csv: str = "/data/kcat.csv",
    sequence_col: str = "Sequence",
    smiles_col: str = "Smiles",
    target_col: str = "log_kcat",
    fold_col: str = "fold",
    fold: int = 0,
    epochs: int = 5,
    n_train: int = 256,
    n_val: int = 256,
    matmul_precision: str = "bfloat16",
    quantize_bf16: bool = True,
    trunk_lora_rank: int = 0,
    trunk_lora_alpha: float = 1.0,
    no_msa: bool = False,
):
    from loguru import logger
    import pandas as pd
    import numpy as np

    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["JOLTZ_BF16"] = "1" if bool(quantize_bf16) else "0"
    os.environ["JOLTZ_SKIP_ONEHOT"] = "1"
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_memory_fraction=0.85")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    if int(trunk_lora_rank) > 0:
        os.environ["JOLTZ_LORA_RANK"] = str(int(trunk_lora_rank))
        os.environ["JOLTZ_LORA_ALPHA"] = str(float(trunk_lora_alpha))

    _add_src_paths()

    from mosaic_workflows.affinity_simple import AffinitySimpleReadout
    from mosaic.losses.boltz2 import load_boltz2, load_features_and_structure_writer, Boltz2Output
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax

    try:
        from jax import config as jconfig
        jconfig.update("jax_default_matmul_precision", str(matmul_precision))
    except Exception:
        pass

    df = pd.read_csv(data_csv)
    if target_col not in df.columns:
        candidates = [c for c in df.columns if ("kcat" in c.lower()) and ("/" not in c)]
        if candidates:
            y_raw = df[candidates[0]].astype("float32").values
            df[target_col] = np.log10(np.clip(y_raw, 1e-8, None)).astype("float32")
        else:
            raise RuntimeError("No target column found")
    if sequence_col not in df.columns:
        seq_candidates = [c for c in df.columns if any(k in c.lower() for k in ("sequence","seq"))]
        sequence_col = seq_candidates[0]
    if smiles_col not in df.columns:
        smi_candidates = [c for c in df.columns if any(k in c.lower() for k in ("smiles","smile"))]
        smiles_col = smi_candidates[0]
    assert fold_col in df.columns

    train_df = df[df[fold_col] != fold].reset_index(drop=True).head(int(n_train))
    val_df = df[df[fold_col] == fold].reset_index(drop=True).head(int(n_val))

    boltz2 = load_boltz2()

    def yaml_for(seq: str, smi: str) -> str:
        return (
            "sequences:\n"
            "  - protein:\n"
            f"      id: A\n      sequence: {seq.strip()}\n      msa: empty\n"
            "  - ligand:\n"
            f"      id: B\n      smiles: '{smi.strip()}'\n"
            "properties:\n"
            "  - affinity:\n"
            "      binder: B\n"
        )

    # Probe dims
    probe_yaml = yaml_for(str(train_df.iloc[0][sequence_col]), str(train_df.iloc[0][smiles_col]))
    probe_feats, _ = load_features_and_structure_writer(probe_yaml)
    probe_out = Boltz2Output(joltz2=boltz2, features=probe_feats, deterministic=True, key=jax.random.key(0), recycling_steps=0)
    z_dim = int(probe_out.trunk_state.z.shape[-1])
    s_dim = int(probe_out.initial_embedding.s_inputs.shape[-1])
    readout = AffinitySimpleReadout(token_z=z_dim, token_s=s_dim, key=jax.random.key(1))

    def embed(yml: str):
        feats, _ = load_features_and_structure_writer(yml)
        if bool(no_msa) and "msa" in feats:
            feats = feats | {"msa": jnp.zeros_like(feats["msa"]) }
        out = Boltz2Output(joltz2=boltz2, features=feats, deterministic=True, key=jax.random.key(7), recycling_steps=0)
        f = out.features
        return {
            "z": out.trunk_state.z.astype(jnp.float32),
            "s_inputs": out.initial_embedding.s_inputs.astype(jnp.float32),
            "feats": {
                "token_pad_mask": f["token_pad_mask"],
                "mol_type": f["mol_type"],
                "affinity_token_mask": f["affinity_token_mask"],
            },
        }

    train_embeds, train_labels = [], []
    for _, row in train_df.iterrows():
        yml = yaml_for(str(row[sequence_col]), str(row[smiles_col]))
        train_embeds.append(embed(yml))
        train_labels.append(float(row[target_col]))

    val_embeds, val_labels = [], []
    for _, row in val_df.iterrows():
        yml = yaml_for(str(row[sequence_col]), str(row[smiles_col]))
        val_embeds.append(embed(yml))
        val_labels.append(float(row[target_col]))

    @eqx.filter_jit
    def predict(readout, embed):
        return readout(z=embed["z"], s_inputs=embed["s_inputs"], feats=embed["feats"])  # scalar

    def loss_fn(model, embed, y_true):
        y_pred = predict(model, embed)
        y_pred = jnp.asarray(y_pred, dtype=jnp.float32)
        return (y_pred - y_true) ** 2

    loss_and_grad = eqx.filter_value_and_grad(loss_fn)
    optim = optax.adamw(learning_rate=2e-4, weight_decay=1e-2)
    opt_state = optim.init(eqx.filter(readout, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(model, embed, y_true, opt_state):
        loss_val, grads = loss_and_grad(model, embed, y_true)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state2 = optim.update(grads, opt_state, params)
        model2 = eqx.apply_updates(model, updates)
        return model2, opt_state2, loss_val

    for e in range(int(epochs)):
        sq = []
        for emb, y in zip(train_embeds, train_labels):
            if not (y == y):
                continue
            readout, opt_state, l = train_step(readout, emb, jnp.array(y, dtype=jnp.float32), opt_state)
            sq.append(float(l))
        logger.info({"epoch": int(e), "train_mse": float(np.mean(np.array(sq))) if len(sq) else float("nan")})

    preds, labels = [], []
    for emb, y in zip(val_embeds, val_labels):
        if not (y == y):
            continue
        yp = predict(readout, emb)
        preds.append(float(yp)); labels.append(y)
    import numpy as _np
    preds = _np.array(preds); labels = _np.array(labels)
    rmse = float(_np.sqrt(_np.mean((preds - labels) ** 2))) if len(labels) else float("nan")
    logger.info({"val_rmse": rmse})


@app.local_entrypoint()
def main(
    data_csv: str,
    sequence_col: str = "Sequence",
    smiles_col: str = "Smiles",
    target_col: str = "log_kcat",
    fold_col: str = "fold",
    fold: int = 0,
    epochs: int = 5,
    n_train: int = 256,
    n_val: int = 256,
    matmul_precision: str = "bfloat16",
    quantize_bf16: bool = True,
    trunk_lora_rank: int = 0,
    trunk_lora_alpha: float = 1.0,
    no_msa: bool = False,
):
    run_simple_train.remote(
        data_csv=data_csv,
        sequence_col=sequence_col,
        smiles_col=smiles_col,
        target_col=target_col,
        fold_col=fold_col,
        fold=fold,
        epochs=epochs,
        n_train=n_train,
        n_val=n_val,
        matmul_precision=matmul_precision,
        quantize_bf16=quantize_bf16,
        trunk_lora_rank=trunk_lora_rank,
        trunk_lora_alpha=trunk_lora_alpha,
        no_msa=no_msa,
    )
