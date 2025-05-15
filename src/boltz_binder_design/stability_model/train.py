### Simple model (mostly a demonstration for how to incorporate a custom model as a loss)
# Basic *absolute* (delta G, ~not~ delta-delta G) stability model trained on top of frozen ESMC on the Megascale dataset
# Specifically this is trained to minimize MSE on the split described here: https://github.com/SimonKitSangChu/EsmTherm?tab=readme-ov-file
# Could *definitely* be improved.
# If you want to run this yourself you'll need to download the dataset and install a few additional dependencies.

import equinox as eqx
import jax
import numpy as onp
import optax
from datasets import load_from_disk
from jax import tree
from jaxtyping import Array, Float, Int
import tqdm
from esmj import from_torch, ESMC

# load torch model, convert to JAX
from esm.models.esmc import ESMC as TORCH_ESMC

esm = from_torch(TORCH_ESMC.from_pretrained("esmc_300m").to("cpu"))

dataset = load_from_disk("../EsmTherm/datasets/dataset")


class Datum(eqx.Module):
    tokens: Int[Array, "N"]
    deltaG: Float


def to_datum(d):
    tokens = esm.tokenize(d["sequence"]).astype("uint8")
    return Datum(tokens=tokens, deltaG=d["deltaG"])


def group_by_len_and_transform(dataset):
    groups = {}
    for d in tqdm.tqdm(dataset):
        l = len(d["sequence"])
        if l not in groups:
            groups[l] = []
        groups[l].append(to_datum(d))
    return groups


grouped_dataset = group_by_len_and_transform(dataset["train"])
grouped_val = group_by_len_and_transform(dataset["val"])


def sample_batch(grouped_dataset: dict[int], batch_size, replace=True):
    # sample length
    p = onp.array([len(v) for v in grouped_dataset.values()])
    p = p / p.sum()
    length = onp.random.choice(list(grouped_dataset.keys()), p=p)
    dataset = grouped_dataset[length]

    indices = onp.random.choice(len(dataset), batch_size, replace=replace)
    batch = [dataset[int(i)] for i in indices]

    return tree.map(lambda *v: onp.stack(v), *batch)


# subsample the validation set
val_batches = [sample_batch(grouped_val, 256) for _ in range(50)]


def apply_head(esm, tokens, head):
    esm_output = jax.tree.map(lambda v: v[0], esm(tokens[None]))
    # ESMC uses a batch dimension
    return head(esm_output.embedding.mean(axis=0))


class TrainState(eqx.Module):
    head: eqx.Module
    esm: ESMC
    opt_state: optax.OptState


@eqx.filter_jit
def loss_batch(head, esm, batch):
    return (
        jax.vmap(lambda d: (apply_head(esm, d.tokens, head) - d.deltaG) ** 2)(batch)
    ).mean()


@eqx.filter_jit(donate="all")
def opt_step(state, batch, optim):
    loss, grad = eqx.filter_value_and_grad(loss_batch)(state.head, state.esm, batch)
    updates, opt_state = optim.update(grad, state.opt_state, state.head)
    head = eqx.apply_updates(state.head, updates)
    return TrainState(head, state.esm, opt_state), loss


head = eqx.nn.MLP(
        in_size=960,
        out_size="scalar",
        width_size=2 * 960,
        activation=jax.nn.elu,
        depth=1,
        key=jax.random.key(0),
    )
optim = optax.adam(1e-3)
state = TrainState(
    head = head, 
    esm = esm,
    opt_state = optim.init(eqx.filter(head, eqx.is_inexact_array))
)

for _ in tqdm.tqdm(range(1250)):
    batch = sample_batch(grouped_dataset, 256)
    state, loss = opt_step(state, batch, optim)
    print(
        f"{loss: 0.2f}, {onp.var(batch.deltaG): 0.2f} {loss / onp.var(batch.deltaG): 0.2f}"
    )

optim = optax.adam(1e-4)
state = TrainState(
    head = state.head, 
    esm = state.esm,
    opt_state = optim.init(eqx.filter(state.head, eqx.is_inexact_array))
)
for _ in tqdm.tqdm(range(750)):
    batch = sample_batch(grouped_dataset, 256)
    state, loss = opt_step(state, batch, optim)
    print(
        f"{loss: 0.2f}, {onp.var(batch.deltaG): 0.2f} {loss / onp.var(batch.deltaG): 0.2f}"
    )


val_stats = []
for batch in val_batches:
    loss = loss_batch(state.head, state.esm, batch)
    # print(f"val {loss: 0.2f}, {onp.var(batch.deltaG): 0.2f} {loss / onp.var(batch.deltaG): 0.2f}")
    val_stats.append((loss, onp.var(batch.deltaG), loss / onp.var(batch.deltaG)))


jax.tree.map(lambda *x: onp.array(x).mean(), *val_stats)


eqx.tree_serialise_leaves(
    "stability.eqx", state.head
)