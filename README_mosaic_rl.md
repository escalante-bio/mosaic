# mosaic_rl

`mosaic_rl` extends the Mosaic and Mosaic-Workflows ecosystem with loss-first RL tooling. You can either optimise workflow logits directly (great for prototyping) or fine-tune a HuggingFace causal LM, all while keeping the exact same phase/optimizer mental model: you still write closures, you still wire plain dictionaries, and you still call `mosaic_workflows.run_workflow`.

## Principles

The package sticks to the core Mosaic principles:

1. **Loss-first composition** – All RL logic consumes a loss closure returning `(value, aux)` and internally interprets the value as `-reward`. There are no trainer objects, annoted dataclasses, or opaque managers.
2. **Plain dict orchestration** – Phases remain dictionaries with `name`, `build_loss`, `optimizer`, `steps`, `schedule`, `transforms`, and `analyzers`. New knobs live inside the existing schedule dict, so you never learn new phase keys.
3. **Composable optimisers** – RL updates are just new optimiser functions (`grpo_logits`, etc.) that obey the Mosaic-Workflows signature. They can be mixed with gradient optimisers and plugged into Binder-Games without adapter layers.
4. **Explicit randomness & state** – Random keys are threaded explicitly. Policy state is represented by the logits array (`x`) and the usual `best_x` return slot; no implicit global state is stored.
5. **Tiny modules, no inheritance** – Helpers live in small modules (`optimizers`, `regularizers`, `utils`). Everything is a pure function or an Equinox-compatible PyTree so that JAX transformations remain straightforward.

## Package layout

```
src/mosaic_rl/
  __init__.py            # public exports (optimisers, helpers)
  optimizers.py          # Logit-level GRPO + HuggingFace RL optimisers
  regularizers.py        # pluggable KL-style regularisers
  utils.py               # sequence sanitation, decoding helpers
  sampling.py            # categorical sampling utilities
  hf.py                  # Mosaic-style phase builders for HuggingFace models
```

## Core optimiser API

`mosaic_rl.optimizers.grpo_logits` matches the signature of existing optimisers:

```python
def grpo_logits(*,
                loss_function,
                x,
                n_steps,
                key=None,
                schedule=None,
                transforms=None,
                trajectory_fn=None,
                aux_context=None,
                update_loss_state=False,
                **kwargs):
    ...
```

Behaviour:

- `loss_function` is invoked on one-hot samples; its scalar value is treated as `-reward`.
- `schedule` can be a dict or a callable `schedule(global_step, phase_step) -> dict`.
- Recognised schedule keys include `lr`, `temperature`, `num_samples`, and `regularizers`.
- Policy state lives in `x` (current logits); `best_x` carries the best policy by average reward.
- `aux_context` may contain `reference_logits` for KL regularisation.
- `trajectory_fn` receives Aux metrics (`reward/mean`, `reward/std`, etc.) and the sampled probabilities at each step.

### HuggingFace optimiser API

`mosaic_rl.optimizers.hf_grpo_optimizer` powers the higher-level `build_hf_grpo_phase` helper. The phase returns a resource dictionary instead of a numeric loss; the optimiser loads the specified HF model/tokenizer, generates completions, scores them, and performs a REINFORCE-style gradient step using PyTorch/Transformers. State is passed via the usual `x`/`best_x` slots (containing checkpoint paths), so phases chain exactly like the logit-level variant.

## Regularisers

Regularisers are pluggable functions stored in `mosaic_rl.regularizers`. They accept `(probs, reference_probs, weight)` and return `(penalty, grad, diagnostics)`.

Example schedule entry:

```python
schedule=lambda g, p: {
    "lr": 5e-3,
    "temperature": 1.0,
    "regularizers": [
        ("reverse_kl", 0.5),
        ("forward_kl", 0.1),
    ],
}
```

`grpo_logits` looks up the registered regularisers and adds their gradients/metrics.

## Usage examples

### Single-player phase

```python
import mosaic.losses.trigram as trigram
from mosaic_workflows import run_workflow
from mosaic_rl.optimizers import grpo_logits
from mosaic_workflows.transforms import temperature_on_logits

binder_len = 30

def build_loss():
    return trigram.TrigramLL.from_pkl("trigram_seg.pkl")

phase = {
    "name": "trigram_grpo",
    "build_loss": build_loss,
    "optimizer": grpo_logits,
    "steps": 200,
    "schedule": lambda g, p: {"lr": 5e-3, "temperature": 1.2, "regularizers": [("reverse_kl", 0.05)]},
    "transforms": {"pre_logits": [temperature_on_logits(scale_key="temperature")]},
}

workflow = {"binder_len": binder_len, "seed": 0, "phases": [phase]}
run_workflow(workflow)
```

### Hybrid workflow (gradient warm-up + RL fine-tune)

```python
from mosaic_workflows import adamw_logits

phases = [
    {"name": "grad", "build_loss": build_loss, "optimizer": adamw_logits, "steps": 50, "schedule": lambda g, p: {"lr": 0.01}},
    {"name": "grpo", "build_loss": build_loss, "optimizer": grpo_logits, "steps": 150, "schedule": lambda g, p: {"lr": 2e-3}},
]
```

### Binder-Games integration

```python
from binder_games import build_minmax_phase, make_minmax_loss
from mosaic_rl.optimizers import grpo_logits

phase = build_minmax_phase(
    name="minmax_rl",
    build_loss=lambda: make_minmax_loss(build_loss(), build_loss()),
    steps=120,
    optimizers={"x": grpo_logits, "y": grpo_logits},
    schedule=lambda g, p: {
        "x": {"lr": 3e-3, "temperature": 1.0, "regularizers": [("reverse_kl", 0.2)]},
        "y": {"lr": 3e-3, "temperature": 1.0, "regularizers": [("reverse_kl", 0.2)]},
    },
)
```

### HuggingFace model fine-tuning

```python
from mosaic_workflows import run_workflow
from mosaic_rl import build_hf_grpo_phase

def reward_len(prompts, completions):
    return [float(len(c)) for c in completions]

phase = build_hf_grpo_phase(
    name="tiny_rl",
    model="sshleifer/tiny-gpt2",
    tokenizer="sshleifer/tiny-gpt2",
    prompts=["Hello"],
    scorer=reward_len,
    steps=1,
    generations=1,
    max_new_tokens=8,
    results_dir="out_smoke/tiny_rl",
)

workflow = {
    "binder_len": 1,
    "seed": 0,
    "phases": [phase],
    "initial_x": {"checkpoint": "sshleifer/tiny-gpt2"},
}

run_workflow(workflow)
```

This produces an updated checkpoint under `out_smoke/tiny_rl/` while logging reward metrics in the workflow trajectory.

## Why this design sticks to Mosaic

- **Predictable mental model**: users only swap the optimiser function. All other wiring (loss builders, schedules, transforms, analyzers) remains identical.
- **No new phase schema**: real-world workflows can incrementally experiment with RL phases without refactoring.
- **Composable with Binder Games**: multi-player self-play is just a different optimiser per player.
- **JAX friendly**: policy state is a pure PyTree (logits array). There are no hidden Python objects or side-channel state.
- **Testable & inspectable**: because everything is functional, trajectory metrics and analyser hooks behave identically to gradient runs.

## Roadmap

- Additional optimisers (DPO-style updates, actor-critic baselines) following the same signature.
- Extra regularisers (χ², entropy bonuses) registered through `regularizers.register()`.
- Higher-level convenience builders (e.g. `build_protrl_phase`) that still output standard phase dicts.

For now, `grpo_logits` plus the helper utilities are enough to unlock REINFORCE/GRPO-style experimentation without ever leaving the Mosaic comfort zone.
