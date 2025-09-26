from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast


# ---------------- ESM2 embedder ----------------


class ESM2Embedder:
    """Tiny wrapper over fair-esm for mean-pooled token embeddings.

    Keeps API minimal for reuse: call with list[str] -> ndarray [N, D].
    """

    def __init__(self, model_id: str = "facebook/esm2_t6_8M_UR50D", device: Optional[str] = None):
        import torch
        import esm

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_id.endswith("_UR50D"):
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        else:
            model, alphabet = esm.pretrained.load_model_and_alphabet(model_id)
        self._model = model.eval().to(self._device)
        self._batch_converter = alphabet.get_batch_converter()

    def __call__(self, seqs: List[str]):
        import numpy as np

        if len(seqs) == 0:
            return np.zeros((0, 320), dtype=np.float32)
        data = [(str(i), s) for i, s in enumerate(seqs)]
        batch_labels, batch_strs, batch_tokens = self._batch_converter(data)
        batch_tokens = batch_tokens.to(self._device)
        with self._torch.no_grad():
            out = self._model(batch_tokens, repr_layers=[6])
            reps = out["representations"][6]  # [B, L+2, D]
            # drop cls/eos; build masks per sequence length
            means: List[np.ndarray] = []
            for i, s in enumerate(seqs):
                toks = reps[i]
                toks = toks[1 : len(s) + 1]
                means.append(toks.mean(dim=0).detach().cpu().numpy())
        return np.stack(means, axis=0).astype("float32")


# ---------------- Ridge predictors for CSV metrics ----------------


@dataclass
class RidgePredictors:
    W: Any  # [D, K]
    b: Any  # [K]
    mu: Any  # [D]
    sd: Any  # [D]
    cols: Tuple[str, ...]


def train_ridge_predictors(
    embedder: ESM2Embedder,
    df,
    target_cols: Iterable[str],
    seq_col: str = "sequence",
    lam: float = 1e-2,
) -> RidgePredictors:
    import numpy as np

    cols = tuple([c for c in target_cols if c in df.columns])
    rows = [str(s) for s in df.get(seq_col, []) if isinstance(s, str) and len(s) > 0]
    if len(cols) == 0 or len(rows) == 0:
        raise ValueError("No target columns or sequences available in CSV for training predictors.")
    X = embedder(rows)  # [N, D]
    Y = np.stack([df.loc[df[seq_col] == s, list(cols)].iloc[0].to_numpy(dtype=np.float64) for s in rows], axis=0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    Xn = (X - mu) / sd
    I = np.eye(Xn.shape[1], dtype=np.float64)
    W = np.linalg.solve(Xn.T @ Xn + lam * I, Xn.T @ Y)  # [D, K]
    b = Y.mean(axis=0)
    return RidgePredictors(W=W.astype("float32"), b=b.astype("float32"), mu=mu.astype("float32"), sd=sd.astype("float32"), cols=cols)


def predict_ridge(predictors: RidgePredictors, embedder: ESM2Embedder, seqs: List[str]) -> List[Dict[str, float]]:
    import numpy as np

    if len(seqs) == 0:
        return []
    X = embedder(seqs)
    Xn = (X - predictors.mu) / predictors.sd
    Y = Xn @ predictors.W + predictors.b  # [N, K]
    Y = Y.astype("float32")
    out: List[Dict[str, float]] = []
    for i in range(Y.shape[0]):
        d = {col: float(Y[i, j]) for j, col in enumerate(predictors.cols)}
        out.append(d)
    return out


# ---------------- CLEAN scorer (ESM2 + Torch head + EC centroid) ----------------


class CleanHead:
    """Loads a Torch linear head and applies it to ESM2 embeddings.

    Expects a state_dict with 'weight' and optional 'bias'. No fallbacks.
    """

    def __init__(self, weight, bias=None):
        import numpy as np

        self.W = weight.astype("float32")  # [D, K] or [K, D] depending on storage; we enforce [D, K]
        self.b = None if bias is None else bias.astype("float32")

    def __call__(self, x):
        import numpy as np

        y = x @ self.W
        if self.b is not None:
            y = y + self.b
        return y


def load_clean_head_from_torch(path: str, embed_dim: int) -> CleanHead:
    import torch
    import numpy as np

    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # find linear weight/bias
    weight = None
    bias = None
    for k, v in sd.items():
        if k.endswith("weight") and getattr(v, "ndim", 0) == 2:
            w = v.detach().cpu().numpy()
            weight = w
        elif k.endswith("bias") and getattr(v, "ndim", 0) == 1:
            b = v.detach().cpu().numpy()
            bias = b
    if weight is None:
        raise ValueError(f"No linear 'weight' found in {path}")
    # ensure shape [D, K]
    if weight.shape[0] != embed_dim and weight.shape[1] == embed_dim:
        weight = weight.T
    if weight.shape[0] != embed_dim:
        raise ValueError(f"CLEAN head weight shape {weight.shape} incompatible with embed_dim={embed_dim}")
    if bias is not None and bias.shape[0] != weight.shape[1]:
        raise ValueError(f"CLEAN head bias shape {bias.shape} incompatible with weight shape {weight.shape}")
    return CleanHead(weight=weight, bias=bias)


def build_clean_scorer(
    *,
    embedder: ESM2Embedder,
    clean_head_path: str,
    embedding_path: str,
    labels_path: str,
    ec_label: str,
) -> Callable[[str], float]:
    import torch
    import numpy as np

    # Load head
    # Infer embed_dim by a probe on one sequence
    probe = embedder(["M"])  # [1, D]
    embed_dim = probe.shape[1]
    head = load_clean_head_from_torch(clean_head_path, embed_dim=embed_dim)

    # Load EC label embeddings (precomputed)
    data = torch.load(str(embedding_path), map_location="cpu")
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, dict):
        tensor = next(iter(data.values()))
    elif isinstance(data, (list, tuple)):
        tensor = data[0]
    else:
        raise TypeError(f"Unsupported CLEAN embedding container: {type(data)!r}")
    with open(labels_path, "r") as fh:
        raw = fh.read().strip()
    ec_labels = [entry.strip() for entry in raw.split(",") if entry.strip()]
    idxs = [i for i, lbl in enumerate(ec_labels) if lbl == str(ec_label)]
    if not idxs:
        raise ValueError(f"EC label '{ec_label}' not found in {labels_path}")
    centroid = tensor[idxs].mean(0).detach().cpu().numpy().astype("float32")
    # Project centroid through the same head for consistent space
    # If centroid is not the same dimension, try to infer by projecting embedder output then matching dims
    # Here we assume centroid already in embedding space; map both via head
    centroid_proj = head(centroid)
    centroid_proj = centroid_proj / (np.linalg.norm(centroid_proj) + 1e-8)

    def score(seq: str) -> float:
        z = embedder([seq])[0]  # [D]
        z = head(z)
        z = z / (np.linalg.norm(z) + 1e-8)
        return float((z * centroid_proj).sum())

    return score


# ---------------- Boltz2 scorer interface (single-GPU) ----------------


def build_boltz2_scorer(ligand_smiles: Optional[str] = None, cache: Optional[str] = None, xla_cuda_dir: Optional[str] = None) -> Callable[[str], Dict[str, float]]:
    # Environment for XLA
    if xla_cuda_dir:
        os.environ["XLA_FLAGS"] = f"--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found --xla_gpu_cuda_data_dir={xla_cuda_dir}"
    os.environ.setdefault("JAX_PLATFORMS", "cuda")

    import jax
    import jax.numpy as jnp
    from mosaic_rl.utils import AA_VOCAB
    from mosaic.losses.boltz2 import load_boltz2, load_features_and_structure_writer, set_binder_sequence, Boltz2Output

    def build_yaml(binder_len: int) -> str:
        lines = ["version: 1", "sequences:"]
        lines.append(f"  - protein:\n      id: A\n      sequence: {'X'*binder_len}\n      msa: empty")
        if ligand_smiles is not None and len(str(ligand_smiles)) > 0:
            lines.append(f"  - ligand:\n      id: L\n      smiles: '{ligand_smiles}'")
        return "\n".join(lines)

    model = load_boltz2()

    def score(seq: str) -> Dict[str, float]:
        binder_len = len(seq)
        vocab = AA_VOCAB
        onehot = jnp.zeros((binder_len, 20), dtype=jnp.float32)
        idx_map = {aa: i for i, aa in enumerate(vocab)}
        for i, ch in enumerate(seq):
            if ch in idx_map:
                onehot = onehot.at[i, idx_map[ch]].set(1.0)
        yaml_str = build_yaml(binder_len)
        features, _ = load_features_and_structure_writer(
            yaml_str,
            cache=Path(str(cache or os.environ.get("BOLTZ_CACHE", "/root/.boltz"))).expanduser(),
        )
        features = set_binder_sequence(onehot, features)
        out = Boltz2Output(joltz2=model, features=features, deterministic=True, key=jax.random.PRNGKey(0))
        from mosaic.losses.structure_prediction import predicted_tm_score
        logits = out.pae_logits
        bins = out.pae_bins
        asym = out.asym_id.astype(int)
        iptm_val = predicted_tm_score(logits=logits, bin_centers=bins, asym_id=asym, interface=True).max()
        return {"boltz2_plddt": float(out.plddt[:binder_len].mean()), "boltz2_iptm": float(iptm_val)}

    return score


# ---------------- Compose GRPO reward ----------------


# Back-compat: accept plain dicts for reward options; avoid dataclasses for configs
RewardOpts = dict  # type: ignore


def compose_reward(opts: Dict[str, Any]) -> Callable[[List[str], List[str]], List[float]]:
    import numpy as np
    from .utils import AA_VOCAB

    def parse_weight_spec(spec: str) -> Dict[str, float]:  # backward-compat helper if needed elsewhere
        out: Dict[str, float] = {}
        for item in str(spec).split(','):
            if not item.strip():
                continue
            if '=' not in item:
                continue
            k, v = item.split('=', 1)
            out[k.strip()] = float(v.strip())
        return out

    def extract_sequence(text: str, eos_token: Optional[str]) -> str:
        s = str(text)
        if eos_token and eos_token in s:
            s = s.split(eos_token, 1)[0]
        allowed = set(AA_VOCAB)
        return "".join([c for c in s.upper() if c in allowed])

    weights: Dict[str, float] = cast(Dict[str, float], opts.get("weights", {}))
    csv_path: Optional[str] = cast(Optional[str], opts.get("csv_path"))
    csv_seq_col: str = cast(str, opts.get("csv_seq_col", "sequence"))
    csv_pred_cols: Tuple[str, ...] = cast(Tuple[str, ...], opts.get("csv_pred_cols", ("total_score", "efield_score", "ncaa_interface_score")))
    use_csv_predictors: bool = bool(opts.get("use_csv_predictors", False))
    use_clean: bool = bool(opts.get("use_clean", False))
    clean_ec_label: Optional[str] = cast(Optional[str], opts.get("clean_ec_label"))
    clean_head_path: Optional[str] = cast(Optional[str], opts.get("clean_head_path"))
    clean_embedding_path: Optional[str] = cast(Optional[str], opts.get("clean_embedding_path"))
    clean_labels_path: Optional[str] = cast(Optional[str], opts.get("clean_labels_path"))
    esm_model_id: str = cast(str, opts.get("esm_model_id", "facebook/esm2_t6_8M_UR50D"))
    boltz_ligand_smiles: Optional[str] = cast(Optional[str], opts.get("boltz_ligand_smiles"))
    xla_cuda_dir: Optional[str] = cast(Optional[str], opts.get("xla_cuda_dir"))
    eos_token: Optional[str] = cast(Optional[str], opts.get("eos_token"))
    motif_positions: Optional[Tuple[int, ...]] = cast(Optional[Tuple[int, ...]], opts.get("motif_positions"))
    motif_identities: Optional[Tuple[str, ...]] = cast(Optional[Tuple[str, ...]], opts.get("motif_identities"))

    embedder: Optional[ESM2Embedder] = None
    predictors: Optional[RidgePredictors] = None
    clean_score_fn: Optional[Callable[[str], float]] = None
    csv_df = None
    if use_csv_predictors or use_clean:
        import pandas as pd
        if not csv_path:
            raise ValueError("csv_path is required when using CSV predictors or CLEAN.")
        csv_df = pd.read_csv(str(csv_path))
        embedder = ESM2Embedder(model_id=esm_model_id)
    if use_csv_predictors and csv_df is not None:
        assert embedder is not None
        embedder = cast(ESM2Embedder, embedder)
        predictors = train_ridge_predictors(embedder, csv_df, csv_pred_cols, seq_col=csv_seq_col)
    if use_clean:
        if not (clean_ec_label and clean_head_path and clean_embedding_path and clean_labels_path):
            raise ValueError("CLEAN requires clean_ec_label, clean_head_path, clean_embedding_path, clean_labels_path")
        assert embedder is not None
        embedder = cast(ESM2Embedder, embedder)
        clean_score_fn = build_clean_scorer(
            embedder=embedder,
            clean_head_path=str(clean_head_path),
            embedding_path=str(clean_embedding_path),
            labels_path=str(clean_labels_path),
            ec_label=str(clean_ec_label),
        )

    boltz_score_fn: Optional[Callable[[str], Dict[str, float]]] = None
    if ("boltz2_plddt" in weights) or ("boltz2_iptm" in weights):
        boltz_score_fn = build_boltz2_scorer(ligand_smiles=boltz_ligand_smiles, xla_cuda_dir=xla_cuda_dir)

    def f(_prompts: List[str], completions: List[str]) -> List[float]:
        results: List[float] = []
        csv_map: Dict[str, Dict[str, float]] = {}
        if csv_df is not None:
            for s, row in zip(csv_df.get(csv_seq_col, []), csv_df.to_dict(orient="records")):
                if isinstance(s, str):
                    csv_map[s] = {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
        # Batch ESM predictors once
        pred_rows: List[Dict[str, float]] = []
        if predictors is not None:
            pred_rows = predict_ridge(predictors, embedder, completions)  # type: ignore[arg-type]
        for i, raw in enumerate(completions):
            seq = extract_sequence(raw, eos_token)
            total = 0.0
            # Length term
            if "length100" in weights:
                total += weights["length100"] * (-abs(len(seq) - 100) / 100.0)
            # Motif position hits
            if (
                (motif_positions is not None)
                and (motif_identities is not None)
                and ("motif_pos" in weights)
            ):
                hits = 0.0
                positions = cast(Tuple[int, ...], motif_positions)
                pos_list: List[int] = [int(p) for p in positions]
                identities = tuple(s.upper() for s in cast(Tuple[str, ...], motif_identities))
                for p, aa in zip(pos_list, identities):
                    idx = int(p) - 1
                    if 0 <= idx < len(seq) and seq[idx] == aa:
                        hits += 1.0
                frac = hits / max(1, len(pos_list))
                total += weights["motif_pos"] * float(frac)
            # CSV exact-match
            if len(csv_map) > 0:
                cm = csv_map.get(seq)
                if cm is not None:
                    for col in csv_pred_cols:
                        if col in weights and col in cm:
                            total += weights[col] * float(cm[col])
            # Predictor heads
            if predictors is not None:
                for col in predictors.cols:
                    if col in weights:
                        total += weights[col] * float(pred_rows[i].get(col, 0.0))
            # CLEAN
            if clean_score_fn is not None and ("clean_cosine" in weights):
                total += weights["clean_cosine"] * float(clean_score_fn(seq))
            # Boltz2
            if boltz_score_fn is not None:
                b = boltz_score_fn(seq)
                if "boltz2_plddt" in weights:
                    total += weights["boltz2_plddt"] * float(b.get("boltz2_plddt", 0.0))
                if "boltz2_iptm" in weights:
                    total += weights["boltz2_iptm"] * float(b.get("boltz2_iptm", 0.0))
            results.append(float(total))
        return results

    return f


def build_reward_scorers(opts: Dict[str, Any]) -> tuple[Callable[[List[str], List[str]], List[float]], Callable[[List[str]], List[Dict[str, float]]]]:
    """Return (score_total, score_components) callables built from the same options.

    - score_total(prompts, completions) -> List[float]
    - score_components(completions) -> List[Dict[str, float]] with per-term metrics
    """
    # Reuse compose_reward setup by partially rebuilding context here
    from .utils import AA_VOCAB
    weights: Dict[str, float] = cast(Dict[str, float], opts.get("weights", {}))
    csv_path: Optional[str] = cast(Optional[str], opts.get("csv_path"))
    csv_seq_col: str = cast(str, opts.get("csv_seq_col", "sequence"))
    csv_pred_cols: Tuple[str, ...] = cast(Tuple[str, ...], opts.get("csv_pred_cols", ("total_score", "efield_score", "ncaa_interface_score")))
    use_csv_predictors: bool = bool(opts.get("use_csv_predictors", False))
    use_clean: bool = bool(opts.get("use_clean", False))
    clean_ec_label: Optional[str] = cast(Optional[str], opts.get("clean_ec_label"))
    clean_head_path: Optional[str] = cast(Optional[str], opts.get("clean_head_path"))
    clean_embedding_path: Optional[str] = cast(Optional[str], opts.get("clean_embedding_path"))
    clean_labels_path: Optional[str] = cast(Optional[str], opts.get("clean_labels_path"))
    esm_model_id: str = cast(str, opts.get("esm_model_id", "facebook/esm2_t6_8M_UR50D"))
    boltz_ligand_smiles: Optional[str] = cast(Optional[str], opts.get("boltz_ligand_smiles"))
    xla_cuda_dir: Optional[str] = cast(Optional[str], opts.get("xla_cuda_dir"))
    eos_token: Optional[str] = cast(Optional[str], opts.get("eos_token"))
    motif_positions: Optional[Tuple[int, ...]] = cast(Optional[Tuple[int, ...]], opts.get("motif_positions"))
    motif_identities: Optional[Tuple[str, ...]] = cast(Optional[Tuple[str, ...]], opts.get("motif_identities"))

    embedder: Optional[ESM2Embedder] = None
    predictors: Optional[RidgePredictors] = None
    clean_score_fn: Optional[Callable[[str], float]] = None
    csv_df = None
    if use_csv_predictors or use_clean:
        import pandas as pd
        if not csv_path:
            raise ValueError("csv_path is required when using CSV predictors or CLEAN.")
        csv_df = pd.read_csv(str(csv_path))
        embedder = ESM2Embedder(model_id=esm_model_id)
    if use_csv_predictors and csv_df is not None:
        assert embedder is not None
        embedder = cast(ESM2Embedder, embedder)
        predictors = train_ridge_predictors(embedder, csv_df, csv_pred_cols, seq_col=csv_seq_col)
    if use_clean:
        if not (clean_ec_label and clean_head_path and clean_embedding_path and clean_labels_path):
            raise ValueError("CLEAN requires clean_ec_label, clean_head_path, clean_embedding_path, clean_labels_path")
        assert embedder is not None
        embedder = cast(ESM2Embedder, embedder)
        clean_score_fn = build_clean_scorer(
            embedder=embedder,
            clean_head_path=str(clean_head_path),
            embedding_path=str(clean_embedding_path),
            labels_path=str(clean_labels_path),
            ec_label=str(clean_ec_label),
        )

    boltz_score_fn: Optional[Callable[[str], Dict[str, float]]] = None
    if ("boltz2_plddt" in weights) or ("boltz2_iptm" in weights):
        boltz_score_fn = build_boltz2_scorer(ligand_smiles=boltz_ligand_smiles, xla_cuda_dir=xla_cuda_dir)

    def extract_sequence(text: str) -> str:
        s = str(text)
        if eos_token and eos_token in s:
            s = s.split(eos_token, 1)[0]
        allowed = set(AA_VOCAB)
        return "".join([c for c in s.upper() if c in allowed])

    # total scorer via compose_reward for consistency
    total_fn = compose_reward(opts)

    def components_fn(completions: List[str]) -> List[Dict[str, float]]:
        outs: List[Dict[str, float]] = []
        csv_map: Dict[str, Dict[str, float]] = {}
        if csv_df is not None:
            for s, row in zip(csv_df.get(csv_seq_col, []), csv_df.to_dict(orient="records")):
                if isinstance(s, str):
                    csv_map[s] = {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
        pred_rows: List[Dict[str, float]] = []
        if predictors is not None:
            pred_rows = predict_ridge(predictors, cast(ESM2Embedder, embedder), completions)
        for i, raw in enumerate(completions):
            seq = extract_sequence(raw)
            rec: Dict[str, float] = {}
            # length term
            rec["length_term"] = float(-abs(len(seq) - 100) / 100.0)
            # motif positions
            if motif_positions is not None and motif_identities is not None:
                hits = 0.0
                pos_list: List[int] = [int(p) for p in cast(Tuple[int, ...], motif_positions)]
                ids = tuple(s.upper() for s in cast(Tuple[str, ...], motif_identities))
                for p, aa in zip(pos_list, ids):
                    idx = int(p) - 1
                    if 0 <= idx < len(seq) and seq[idx] == aa:
                        hits += 1.0
                rec["motif_pos_score"] = float(hits / max(1, len(pos_list)))
            # CSV exact
            cm = csv_map.get(seq)
            if cm is not None:
                for col in csv_pred_cols:
                    if col in cm:
                        rec[col] = float(cm[col])
            # Predictors
            if predictors is not None:
                for col in predictors.cols:
                    rec[col] = float(pred_rows[i].get(col, 0.0))
            # CLEAN
            if clean_score_fn is not None:
                rec["clean_cosine"] = float(clean_score_fn(seq))
            # Boltz2
            if boltz_score_fn is not None:
                b = boltz_score_fn(seq)
                if "boltz2_plddt" in b:
                    rec["boltz2_plddt"] = float(b["boltz2_plddt"]) 
                if "boltz2_iptm" in b:
                    rec["boltz2_iptm"] = float(b["boltz2_iptm"]) 
            outs.append(rec)
        return outs

    return total_fn, components_fn


