import os
import sys
from pathlib import Path
import argparse


def _add_src_paths():
    here = Path(__file__).resolve().parents[1]
    src = here / "src"
    if src.exists():
        sys.path.insert(0, str(src))
    ext_boltz = here / "_external" / "boltz" / "src"
    if ext_boltz.exists():
        sys.path.insert(0, str(ext_boltz))


def _yaml_for_seq_smiles(seq: str, smiles: str) -> str:
    seq = seq.strip(); smiles = smiles.strip()
    return (
        "sequences:\n"
        "  - protein:\n"
        f"      id: A\n      sequence: {seq}\n      msa: empty\n"
        "  - ligand:\n"
        f"      id: B\n      smiles: '{smiles}'\n"
        "properties:\n"
        "  - affinity:\n"
        "      binder: B\n"
    )


def main():
    parser = argparse.ArgumentParser("Simple affinity trainer leveraging Joltz/Boltz2")
    parser.add_argument("--data-csv", type=str, required=True)
    parser.add_argument("--sequence-col", type=str, default="sequence")
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--target-col", type=str, default="log_kcat")
    parser.add_argument("--fold-col", type=str, default="fold")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-val", type=int, default=128)
    args = parser.parse_args()

    _add_src_paths()
    os.environ.setdefault("BOLTZ_CACHE", str(Path.home() / ".boltz"))
    os.environ.setdefault("JAX_PLATFORMS", "cuda")

    import pandas as pd
    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax

    from mosaic_workflows.affinity_simple import AffinitySimpleReadout
    from mosaic.losses.boltz2 import load_boltz2, load_features_and_structure_writer, Boltz2Output

    df = pd.read_csv(args.data_csv)
    if args.target_col not in df.columns:
        candidates = [c for c in df.columns if ("kcat" in c.lower()) and ("/" not in c)]
        if candidates:
            y_raw = df[candidates[0]].astype("float32").values
            df[args.target_col] = np.log10(np.clip(y_raw, 1e-8, None)).astype("float32")
        else:
            raise RuntimeError("No target column found")
    if args.sequence_col not in df.columns:
        seq_candidates = [c for c in df.columns if any(k in c.lower() for k in ("sequence","seq"))]
        args.sequence_col = seq_candidates[0]
    if args.smiles_col not in df.columns:
        smi_candidates = [c for c in df.columns if any(k in c.lower() for k in ("smiles","smile"))]
        args.smiles_col = smi_candidates[0]
    assert args.fold_col in df.columns

    train_df = df[df[args.fold_col] != args.fold].reset_index(drop=True).head(args.n_train)
    val_df = df[df[args.fold_col] == args.fold].reset_index(drop=True).head(args.n_val)

    boltz2 = load_boltz2()

    # Probe dims from the first training example
    assert len(train_df) > 0, "Empty training split"
    probe_yaml = _yaml_for_seq_smiles(str(train_df.iloc[0][args.sequence_col]), str(train_df.iloc[0][args.smiles_col]))
    probe_features, _ = load_features_and_structure_writer(probe_yaml)
    probe_out = Boltz2Output(joltz2=boltz2, features=probe_features, deterministic=True, key=jax.random.key(123), recycling_steps=0)
    z_dim = int(probe_out.trunk_state.z.shape[-1])
    s_dim = int(probe_out.initial_embedding.s_inputs.shape[-1])

    readout = AffinitySimpleReadout(token_z=z_dim, token_s=s_dim, key=jax.random.key(0))

    # Precompute embeddings (z, s_inputs, feats) once per example
    def _embed_yaml(yaml_str: str):
        features, _ = load_features_and_structure_writer(yaml_str)
        out = Boltz2Output(joltz2=boltz2, features=features, deterministic=True, key=jax.random.key(7), recycling_steps=0)
        feats = out.features
        return {
            "z": out.trunk_state.z.astype(jnp.float32),
            "s_inputs": out.initial_embedding.s_inputs.astype(jnp.float32),
            "feats": {
                "token_pad_mask": feats["token_pad_mask"],
                "mol_type": feats["mol_type"],
                "affinity_token_mask": feats["affinity_token_mask"],
            },
        }

    train_embeds, train_labels = [], []
    for _, row in train_df.iterrows():
        yaml_str = _yaml_for_seq_smiles(str(row[args.sequence_col]), str(row[args.smiles_col]))
        train_embeds.append(_embed_yaml(yaml_str))
        train_labels.append(float(row[args.target_col]))

    val_embeds, val_labels = [], []
    for _, row in val_df.iterrows():
        yaml_str = _yaml_for_seq_smiles(str(row[args.sequence_col]), str(row[args.smiles_col]))
        val_embeds.append(_embed_yaml(yaml_str))
        val_labels.append(float(row[args.target_col]))

    @eqx.filter_jit
    def predict_from_embed(readout, embed):
        return readout(z=embed["z"], s_inputs=embed["s_inputs"], feats=embed["feats"])  # scalar

    def _loss_fn(readout, embed, y_true):
        y_pred = predict_from_embed(readout, embed)
        y_pred = jnp.asarray(y_pred, dtype=jnp.float32)
        return (y_pred - y_true) ** 2

    loss_and_grad = eqx.filter_value_and_grad(_loss_fn)
    optim = optax.adamw(learning_rate=2e-4, weight_decay=1e-2)
    opt_state = optim.init(eqx.filter(readout, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(readout, embed, y_true, opt_state):
        loss_val, grads = loss_and_grad(readout, embed, y_true)
        params = eqx.filter(readout, eqx.is_inexact_array)
        updates, opt_state2 = optim.update(grads, opt_state, params)
        readout2 = eqx.apply_updates(readout, updates)
        return readout2, opt_state2, loss_val

    for epoch in range(int(args.epochs)):
        sq = []
        for embed, y_true in zip(train_embeds, train_labels):
            if not (y_true == y_true):
                continue
            readout, opt_state, lval = train_step(readout, embed, jnp.array(y_true, dtype=jnp.float32), opt_state)
            sq.append(float(lval))
        train_mse = float(np.mean(np.array(sq))) if len(sq) else float("nan")
        print({"epoch": epoch, "train_mse": train_mse})

    # Validation
    preds, labels = [], []
    for embed, y_true in zip(val_embeds, val_labels):
        if not (y_true == y_true):
            continue
        y_pred = predict_from_embed(readout, embed)
        preds.append(float(y_pred)); labels.append(y_true)
    preds = np.array(preds); labels = np.array(labels)
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2))) if len(labels) else float("nan")
    print({"val_rmse": rmse})


if __name__ == "__main__":
    main()


