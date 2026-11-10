"""ProteinMPNN's torch checkpoints (`weights/*.pt`, in the repository; the package leaves
them out) converted to `mpnn.py`'s modules and saved in jfold's format beside them
(`weights/<name>/`), which is what mosaic loads; needs torch.

    uv run python -m mosaic.proteinmpnn.convert

Each torch module becomes its JAX counterpart (the same class name in `mpnn.py`), its
submodules, parameters and attributes filling the counterpart's fields of the same names."""

import dataclasses
import datetime
import hashlib
from functools import singledispatch

import numpy as np
import torch
from jfold import io

from . import mpnn, torch_mpnn

CHECKPOINTS = ("v_48_020", "soluble_v_48_020", "abmpnn")
HIDDEN_DIM, NUM_LAYERS = 128, 3


@singledispatch
def from_torch(x):
    raise NotImplementedError(f"no conversion for {type(x).__name__}")


for plain in (int, float, bool, type(None)):
    from_torch.register(plain, lambda x: x)
from_torch.register(torch.Tensor, lambda x: np.asarray(x.detach().float()))
from_torch.register(torch.nn.ModuleList, lambda x: [from_torch(m) for m in x])
from_torch.register(
    torch.nn.Linear, lambda m: mpnn.Linear(weight=from_torch(m.weight), bias=from_torch(m.bias))
)
from_torch.register(
    torch.nn.LayerNorm,
    lambda m: mpnn.LayerNorm(weight=from_torch(m.weight), bias=from_torch(m.bias), eps=m.eps),
)
from_torch.register(torch.nn.Embedding, lambda m: mpnn.Embedding(weight=from_torch(m.weight)))


def _module(cls):
    """Convert a torch module into `cls`: each field from the torch module's submodule,
    parameter or attribute of that name; a field it lacks must be optional (None)."""

    def convert(m: torch.nn.Module):
        values = dict(m.named_children()) | dict(m.named_parameters(recurse=False))
        kwargs = {}
        for field in dataclasses.fields(cls):
            if field.name in values:
                kwargs[field.name] = from_torch(values[field.name])
            elif hasattr(m, field.name):
                kwargs[field.name] = from_torch(getattr(m, field.name))
            else:
                kwargs[field.name] = None
        unread = set(values) - set(kwargs)
        if unread:
            raise ValueError(f"{type(m).__name__}: no {cls.__name__} field for {sorted(unread)}")
        return cls(**kwargs)

    return convert


for torch_cls, cls in [
    (torch_mpnn.PositionalEncodings, mpnn.PositionalEncodings),
    (torch_mpnn.PositionWiseFeedForward, mpnn.PositionWiseFeedForward),
    (torch_mpnn.EncLayer, mpnn.EncLayer),
    (torch_mpnn.DecLayer, mpnn.DecLayer),
    (torch_mpnn.ProteinFeatures, mpnn.ProteinFeatures),
    (torch_mpnn.ProteinMPNN, mpnn.ProteinMPNN),
    (torch.nn.GELU, mpnn.GELU),
    (torch.nn.Dropout, mpnn.Dropout),
]:
    from_torch.register(torch_cls, _module(cls))


def torch_model(checkpoint_path) -> torch_mpnn.ProteinMPNN:
    """The torch ProteinMPNN of a checkpoint, in evaluation mode."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = torch_mpnn.ProteinMPNN(
        num_letters=21,
        node_features=HIDDEN_DIM,
        edge_features=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def convert(checkpoint_path) -> mpnn.ProteinMPNN:
    return from_torch(torch_model(checkpoint_path))


def main():
    for name in CHECKPOINTS:
        source = mpnn.WEIGHTS / f"{name}.pt"
        model = convert(source)
        metadata = {
            "model": f"proteinmpnn-{name}",
            "checkpoint": {
                "source": f"mosaic/proteinmpnn/weights/{name}.pt",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
            "converted_at": datetime.date.today().isoformat(),
        }
        io.save(model, mpnn.WEIGHTS / name, metadata=metadata, namespaces=("mosaic.",))
        print(f"{source} -> {mpnn.WEIGHTS / name}")


if __name__ == "__main__":
    main()
