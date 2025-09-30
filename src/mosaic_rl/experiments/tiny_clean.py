"""Reproduction of the tiny-clean-test GRPO loop using Mosaic RL primitives."""

from __future__ import annotations

import importlib
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import torch
from datasets import Dataset

from mosaic_rl.rewards import CleanHead, load_clean_head_from_torch
from mosaic_rl.hf import build_hf_dataset_phase, build_hf_grpo_phase
from mosaic_rl.utils import AA_VOCAB, sanitize_sequence
from mosaic_workflows import run_workflow


@dataclass
class TinyCleanConfig:
    workspace: Path
    label: str
    binder_len: int
    model: str
    reference_model: str
    clean_head_path: Path
    embedding_path: Path
    results_dir: Path
    device: str = "cuda"
    num_generations: int = 20
    max_new_tokens: int = 150
    top_k: int = 9
    schedule: Sequence[float] = field(default_factory=lambda: (2e-5,))
    tokenizer_id: str | None = None
    use_esm: bool = False
    ec_labels_path: Path | None = None
    esm_model: str = "esm1b_t33_650M_UR50S"

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace)
        self.clean_head_path = Path(self.clean_head_path)
        self.embedding_path = Path(self.embedding_path)
        self.results_dir = Path(self.results_dir)
        self.model = str(self.model)
        self.reference_model = str(self.reference_model)
        if self.tokenizer_id is not None:
            self.tokenizer_id = str(self.tokenizer_id)
        if self.ec_labels_path is not None:
            self.ec_labels_path = Path(self.ec_labels_path)
        if isinstance(self.schedule, float):
            self.schedule = (float(self.schedule),)
        else:
            self.schedule = tuple(float(x) for x in self.schedule)

    @property
    def data_dir(self) -> Path:
        return self.workspace / "data"

    @property
    def inputs_dir(self) -> Path:
        return self.data_dir / "inputs"

    @property
    def esm_dir(self) -> Path:
        return self.data_dir / "esm_data"

    @property
    def dataset_dir(self) -> Path:
        return self.workspace / "dataset"

    @property
    def resolved_ec_labels_path(self) -> Path:
        if self.ec_labels_path is not None:
            return self.ec_labels_path
        search_root = self.clean_head_path.resolve()
        for _ in range(6):
            candidate = search_root.parent / "ec_lables_clean_list.txt"
            if candidate.exists():
                return candidate
            search_root = search_root.parent
        raise FileNotFoundError(
            "Unable to locate 'ec_lables_clean_list.txt'; provide `ec_labels_path` in TinyCleanConfig."
        )


def _calculate_perplexity(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, _ = outputs[:2]
    return math.exp(loss)


def _calculate_log_likelihood(model, ref_model, input_ids):
    with torch.no_grad():
        outputs_model = model(input_ids, labels=input_ids)
        outputs_ref = ref_model(input_ids, labels=input_ids)
    loss, _ = outputs_model[:2]
    ref_loss, _ = outputs_ref[:2]
    intrinsic_reward = -(loss - ref_loss)
    return intrinsic_reward


def generate_sequences(cfg: TinyCleanConfig, checkpoint: str, iteration: int) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(cfg.inputs_dir, exist_ok=True)
    tokenizer_id = cfg.tokenizer_id or checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(cfg.device)
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.reference_model).to(cfg.device)

    sequences = []
    fasta_lines: List[str] = []

    encoded_prompt = tokenizer(cfg.label, return_tensors="pt").to(cfg.device)
    outputs = model.generate(
        **encoded_prompt,
        max_new_tokens=cfg.max_new_tokens,
        num_return_sequences=cfg.num_generations,
        do_sample=True,
        temperature=1.0,
        top_k=cfg.top_k,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    for seq_idx, output in enumerate(outputs):
        completion_ids = output[encoded_prompt["input_ids"].shape[1]:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        if not completion:
            continue
        full_input = torch.cat([encoded_prompt["input_ids"], completion_ids.unsqueeze(0)], dim=1)
        ppl = _calculate_perplexity(model, full_input)
        intr = _calculate_log_likelihood(model, ref_model, full_input)
        header_id = f"{cfg.label}_iteration{iteration}_{seq_idx}"
        header = f">{header_id}\t{ppl:.4f}\t{float(intr):.4f}"
        fasta_lines.append(f"{header}\n{completion}\n")
        sequences.append({
            "prompt": cfg.label,
            "sequence": completion,
            "perplexity": float(ppl),
            "intrinsic_reward": float(intr),
        })

    fasta_file = cfg.inputs_dir / f"seq_gen_iteration{iteration}.fasta"
    with open(fasta_file, "w") as f:
        f.writelines(fasta_lines)

    return sequences


def _dummy_embed_fn(binder_len: int) -> Callable[[str], np.ndarray]:
    rng = np.random.default_rng(0)
    vocab_vecs = rng.standard_normal((len(AA_VOCAB), 1280)).astype(np.float32)

    def embed(seq: str) -> np.ndarray:
        sanitized = sanitize_sequence(seq, binder_len)
        vec = np.zeros(1280, dtype=np.float32)
        for ch in sanitized:
            vec += vocab_vecs[AA_VOCAB.index(ch)]
        return vec / max(len(sanitized), 1)

    return embed


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    current_id: str | None = None
    buffer: list[str] = []
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    entries.append((current_id, "".join(buffer)))
                current_id = line[1:]
                buffer = []
            else:
                buffer.append(line)
        if current_id is not None:
            entries.append((current_id, "".join(buffer)))
    return entries


def _load_reference_embedding(cfg: TinyCleanConfig) -> np.ndarray:
    data = torch.load(cfg.embedding_path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, dict):
        tensor = next(iter(data.values()))
    elif isinstance(data, (list, tuple)):
        tensor = data[0]
    else:
        raise TypeError(f"Unsupported CLEAN embedding container: {type(data)!r}")
    if not isinstance(tensor, torch.Tensor):  # pragma: no cover - safety guard
        raise TypeError(f"Expected torch.Tensor for CLEAN embeddings, got {type(tensor)!r}")

    labels_file = cfg.resolved_ec_labels_path
    with open(labels_file, "r") as handle:
        raw = handle.read().strip()
        ec_labels = [entry.strip() for entry in raw.split(",") if entry.strip()]

    matches = [idx for idx, lbl in enumerate(ec_labels) if lbl == cfg.label]
    if not matches:
        raise ValueError(f"EC label '{cfg.label}' not found in {labels_file}")

    stacked = tensor[matches]
    reference = stacked.mean(0)
    return reference.detach().cpu().numpy().astype(np.float32)


def _invoke_esm_extract(cfg: TinyCleanConfig, fasta_path: Path) -> None:
    esm_module = importlib.import_module("esm")
    script = Path(esm_module.__file__).parent / "scripts" / "extract.py"
    if not script.exists():
        raise FileNotFoundError("esm extract.py script not found; ensure fair-esm is installed correctly.")
    subprocess.run(
        [
            sys.executable,
            str(script),
            cfg.esm_model,
            str(fasta_path),
            str(cfg.esm_dir),
            "--include",
            "mean",
        ],
        check=True,
    )


def _ensure_esm_embeddings(cfg: TinyCleanConfig, iteration: int, sequence_ids: Iterable[str]) -> Dict[str, np.ndarray]:
    cfg.esm_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = cfg.inputs_dir / f"seq_gen_iteration{iteration}.fasta"
    ids = list(sequence_ids)
    missing = [seq_id for seq_id in ids if not (cfg.esm_dir / f"{seq_id}.pt").exists()]
    if missing:
        _invoke_esm_extract(cfg, fasta_path)

    embeddings: Dict[str, np.ndarray] = {}
    for seq_id in ids:
        path = cfg.esm_dir / f"{seq_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing ESM embedding for {seq_id} at {path}")
        record = torch.load(path, map_location="cpu")
        rep = record["mean_representations"][33]
        embeddings[seq_id] = rep.detach().cpu().numpy().astype(np.float32)
    return embeddings


def _compute_clean_reward(head: CleanHead, target: jnp.ndarray, embedding: np.ndarray) -> float:
    vector = jnp.asarray(embedding, dtype=jnp.float32)
    logits = head(vector)
    numerator = jnp.vdot(logits, target)
    denom = jnp.linalg.norm(logits) * jnp.linalg.norm(target) + 1e-8
    return float(numerator / denom)


def score_sequences(cfg: TinyCleanConfig, iteration: int, *, use_esm: bool | None = None) -> tuple[Dataset, Dataset]:

    use_esm = cfg.use_esm if use_esm is None else use_esm
    fasta_path = cfg.inputs_dir / f"seq_gen_iteration{iteration}.fasta"
    records = _read_fasta(fasta_path)

    cleaned_records: list[tuple[str, str, str]] = []
    for header, sequence in records:
        seq = sequence.strip()
        if not seq:
            continue
        seq_id = header.split("\t", 1)[0]
        cleaned_records.append((header, seq, seq_id))

    if not cleaned_records:
        raise RuntimeError(f"No sequences found in {fasta_path}")

    if use_esm:
        embedding_map = _ensure_esm_embeddings(cfg, iteration, (seq_id for _, _, seq_id in cleaned_records))

        def get_embedding(seq_id: str, seq: str) -> np.ndarray:
            return embedding_map[seq_id]

    else:
        dummy_embed = _dummy_embed_fn(cfg.binder_len)

        def get_embedding(seq_id: str, seq: str) -> np.ndarray:
            return dummy_embed(seq)

    clean_head = load_clean_head_from_torch(str(cfg.clean_head_path), embed_dim=1280)
    target = jnp.asarray(_load_reference_embedding(cfg), dtype=jnp.float32)

    entries = []
    for header, seq, seq_id in cleaned_records:
        embedding = get_embedding(seq_id, seq)
        reward = _compute_clean_reward(clean_head, target, embedding)
        completion = f"{seq}<|eos|>"
        entries.append({"prompt": cfg.label, "completion": completion, "reward": float(reward)})

    dataset = Dataset.from_list(entries)
    split = dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)

    train_path = cfg.dataset_dir / f"iteration_{iteration}" / "train"
    eval_path = cfg.dataset_dir / f"iteration_{iteration}" / "eval"
    train_path.mkdir(parents=True, exist_ok=True)
    eval_path.mkdir(parents=True, exist_ok=True)
    split["train"].save_to_disk(train_path)
    split["test"].save_to_disk(eval_path)

    return split["train"], split["test"]


def train_model(cfg: TinyCleanConfig, train_dataset: Dataset, eval_dataset: Dataset, checkpoint: str, step_index: int) -> str:
    # Online GRPO using CLEAN reward; avoids TRL dependency differences.
    import importlib
    import torch

    lr = float(cfg.schedule[min(step_index, len(cfg.schedule) - 1)])

    # Prepare CLEAN components
    clean_head = load_clean_head_from_torch(str(cfg.clean_head_path), embed_dim=1280)
    target = jnp.asarray(_load_reference_embedding(cfg), dtype=jnp.float32)

    if cfg.use_esm:
        esm_lib = importlib.import_module("esm")
        model_loader = getattr(esm_lib.pretrained, cfg.esm_model)
        esm_model, alphabet = model_loader()
        esm_model.eval()
        batch_converter = alphabet.get_batch_converter()

        def embed_batch(seqs: list[str]) -> np.ndarray:
            batch = [(str(i), s) for i, s in enumerate(seqs)]
            with torch.no_grad():
                _, _, tokens = batch_converter(batch)
                out = esm_model(tokens, repr_layers=[33], return_contacts=False)
                reps = out["representations"][33][:, 1:-1].mean(1).cpu().numpy().astype(np.float32)
            return reps
    else:
        dummy = _dummy_embed_fn(cfg.binder_len)

        def embed_batch(seqs: list[str]) -> np.ndarray:
            return np.stack([dummy(s) for s in seqs], axis=0)

    def scorer(prompts: Sequence[str], completions: Sequence[str]) -> Sequence[float]:
        seqs = [sanitize_sequence(c, cfg.binder_len) for c in completions]
        embeddings = embed_batch(seqs)
        # Compute CLEAN cosine reward per sequence
        vector = jnp.asarray(embeddings, dtype=jnp.float32)  # [B,1280]
        z = jax.vmap(clean_head)(vector)
        denom = jnp.linalg.norm(z, axis=-1) * jnp.linalg.norm(target) + 1e-8
        cos = jnp.vdot(z, jnp.repeat(target[None, :], z.shape[0], axis=0)) / denom  # type: ignore[arg-type]
        return [float(c) for c in cos]

    phase = build_hf_grpo_phase(
        name=f"tiny_clean_train_{step_index}",
        model=checkpoint,
        tokenizer=cfg.tokenizer_id or checkpoint,
        prompts=[cfg.label],
        scorer=scorer,
        steps=1,
        generations=cfg.num_generations,
        max_new_tokens=cfg.max_new_tokens,
        results_dir=cfg.results_dir,
        schedule=lambda g, p: {"lr": lr, "device": cfg.device, "top_k": cfg.top_k},
    )
    workflow = {
        "binder_len": cfg.binder_len,
        "seed": 0,
        "phases": [phase],
        "initial_x": {"checkpoint": checkpoint},
    }
    result = run_workflow(workflow)
    return result["x"]["checkpoint"]


def run_pipeline(cfg: TinyCleanConfig, iterations: int) -> None:
    cfg.inputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.dataset_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    current_checkpoint = cfg.model
    prev_train = None
    prev_eval = None

    for iteration in range(iterations):
        if iteration > 0 and prev_train is not None and prev_eval is not None:
            current_checkpoint = train_model(cfg, prev_train, prev_eval, current_checkpoint, iteration)
        generate_sequences(cfg, current_checkpoint, iteration)
        prev_train, prev_eval = score_sequences(cfg, iteration)


__all__ = ["TinyCleanConfig", "run_pipeline", "generate_sequences", "score_sequences", "train_model"]
