"""Mosaic-style reproduction of the ProtRL ranked-DPO loop."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from datasets import Dataset

from mosaic_rl.hf import build_hf_dataset_phase
from mosaic_workflows import run_workflow


@dataclass
class ProtRLConfig:
    workspace: Path
    label: str
    model: str
    reference_model: str
    results_dir: Path
    tokenizer_id: str
    max_new_tokens: int = 1024
    top_k: int = 9
    num_generations: int = 20
    schedule: Sequence[float] = field(default_factory=lambda: (2e-5,))
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace)
        self.results_dir = Path(self.results_dir)
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    @property
    def inputs_dir(self) -> Path:
        return self.workspace / "data" / "inputs"

    @property
    def dataset_dir(self) -> Path:
        return self.workspace / "dataset"


def _calculate_perplexity(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, _ = outputs[:2]
    return math.exp(loss)


def _intrinsic_reward(model, ref_model, input_ids):
    with torch.no_grad():
        outputs_model = model(input_ids, labels=input_ids)
        outputs_ref = ref_model(input_ids, labels=input_ids)
    loss, _ = outputs_model[:2]
    ref_loss, _ = outputs_ref[:2]
    return -(loss - ref_loss)


def generate_sequences(cfg: ProtRLConfig, checkpoint: str, iteration: int) -> list[Dict[str, float]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(cfg.device)
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.reference_model).to(cfg.device)

    encoded_prompt = tokenizer(cfg.label, return_tensors="pt").to(cfg.device)
    outputs = model.generate(
        **encoded_prompt,
        max_new_tokens=cfg.max_new_tokens,
        num_return_sequences=cfg.num_generations,
        do_sample=True,
        temperature=1.0,
        top_k=cfg.top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    rows = []
    lines: list[str] = []
    prompt_len = encoded_prompt["input_ids"].shape[1]
    for idx, output in enumerate(outputs):
        completion_ids = output[prompt_len:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        # Only remove spaces to keep original content; do not trim other characters
        if completion:
            completion = completion.replace(" ", "")
        if not completion:
            continue
        packed = torch.cat([encoded_prompt["input_ids"], completion_ids.unsqueeze(0)], dim=1)
        perplexity = _calculate_perplexity(model, packed)
        intrinsic = _intrinsic_reward(model, ref_model, packed)
        header_id = f"{cfg.label}_iteration{iteration}_{idx}"
        header = f">{header_id}\t{perplexity:.4f}\t{float(intrinsic):.4f}"
        lines.append(f"{header}\n{completion}\n")
        rows.append(
            {
                "perplexity": float(perplexity),
                "intrinsic_reward": float(intrinsic),
                "completion": completion,
                "header": header_id,
            }
        )

    fasta_path = cfg.inputs_dir / f"seq_gen_{cfg.label}_iteration{iteration}.fasta"
    with open(fasta_path, "w") as handle:
        handle.writelines(lines)

    return rows


def _read_fasta(path: Path) -> Iterable[tuple[str, float, float, str]]:
    current = None
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]
                parts = header.split("\t")
                seq_id = parts[0]
                ppl = float(parts[1]) if len(parts) > 1 else 0.0
                intrinsic = float(parts[2]) if len(parts) > 2 else 0.0
                current = (seq_id, ppl, intrinsic)
            elif current is not None:
                yield current[0], current[1], current[2], line
                current = None


def score_sequences(cfg: ProtRLConfig, iteration: int) -> tuple[Dataset, Dataset]:

    fasta_path = cfg.inputs_dir / f"seq_gen_{cfg.label}_iteration{iteration}.fasta"
    rows = []
    for _, ppl, intrinsic, sequence in _read_fasta(fasta_path):
        # Mirror external heuristic: prefer low perplexity and positive intrinsic reward.
        reward = float(intrinsic) - math.log(ppl + 1e-6)
        completion = f"{sequence}<|endoftext|>"
        rows.append({"prompt": cfg.label, "completion": completion, "reward": reward})

    dataset = Dataset.from_list(rows)
    split = dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)

    train_dir = cfg.dataset_dir / f"iteration_{iteration}" / "train"
    eval_dir = cfg.dataset_dir / f"iteration_{iteration}" / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    split["train"].save_to_disk(train_dir)
    split["test"].save_to_disk(eval_dir)

    return split["train"], split["test"]


def train_model(cfg: ProtRLConfig, train_dataset: Dataset, eval_dataset: Dataset, checkpoint: str, iteration: int) -> str:
    from trl import GRPOConfig

    lr = float(cfg.schedule[min(iteration, len(cfg.schedule) - 1)])
    training_args = GRPOConfig(
        output_dir=str(cfg.results_dir),
        logging_steps=100,
        num_train_epochs=1,
        learning_rate=lr,
        beta=0.01,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="no",
        num_generations=cfg.num_generations,
    )

    def build_optimizer(model):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.1,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        return optimizer, scheduler

    phase = build_hf_dataset_phase(
        name=f"protrl_train_{iteration}",
        model=checkpoint,
        tokenizer=cfg.tokenizer_id,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
        results_dir=cfg.results_dir,
        step_index=iteration,
        optimizer_builder=build_optimizer,
    )

    workflow = {
        "binder_len": 0,
        "seed": 0,
        "phases": [phase],
        "initial_x": {"checkpoint": checkpoint},
    }
    result = run_workflow(workflow)
    return result["x"]["checkpoint"]


def run_pipeline(cfg: ProtRLConfig, iterations: int) -> None:
    current_checkpoint = cfg.model
    train_dataset = None
    eval_dataset = None

    for iteration in range(iterations):
        if iteration > 0 and train_dataset is not None and eval_dataset is not None:
            current_checkpoint = train_model(cfg, train_dataset, eval_dataset, current_checkpoint, iteration)
        generate_sequences(cfg, current_checkpoint, iteration)
        train_dataset, eval_dataset = score_sequences(cfg, iteration)


__all__ = ["ProtRLConfig", "run_pipeline", "generate_sequences", "score_sequences", "train_model"]
