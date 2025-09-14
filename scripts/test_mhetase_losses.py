import argparse
from pathlib import Path
import sys
import os
import numpy as np
import jax
import jax.numpy as jnp

# Ensure local src is importable
_WS = "/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/src"
os.environ.setdefault("PYTHONPATH", _WS + os.pathsep + os.environ.get("PYTHONPATH", ""))
if _WS not in sys.path:
    sys.path.insert(0, _WS)

from importlib.machinery import SourceFileLoader
import types
def _stub_module(name: str, attrs=None):
    m = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    sys.modules[name] = m

# Stub external deps and heavy modules not needed for these unit tests
_stub_module("gemmi")
_stub_module("mosaic.af2.alphafold2", {"AF2": type("AF2", (), {})})
class _AlphaFoldLoss:
    def __init__(self, *a, **k):
        pass
    def __call__(self, *a, **k):
        return jnp.asarray(0.0, dtype=jnp.float32), {}
_stub_module("mosaic.losses.af2", {"AlphaFoldLoss": _AlphaFoldLoss})
class _ClippedLoss:
    def __init__(self, *a, **k):
        self.loss = k.get("loss", None)
        self.l = k.get("l", 0.0)
        self.u = k.get("u", 1.0)
        self.name = k.get("name", "clipped")
    def __call__(self, *a, **k):
        if self.loss is None:
            return jnp.asarray(0.0, dtype=jnp.float32), {self.name: jnp.asarray(0.0)}
        v, aux = self.loss(*a, **k)
        v2 = jnp.clip(v, self.l, self.u)
        return v2, aux
class _ClippedGradient:
    def __init__(self, *a, **k):
        self.loss = k.get("loss")
    def __call__(self, seq, *a, **k):
        if self.loss is None:
            return jnp.asarray(0.0, dtype=jnp.float32), {}
        return self.loss(seq, *a, **k)
_stub_module("mosaic.losses.transformations", {"ClippedLoss": _ClippedLoss, "ClippedGradient": _ClippedGradient})
class _Boltz2Loss:
    def __init__(self, *a, **k):
        pass
    def __call__(self, *a, **k):
        return jnp.asarray(0.0, dtype=jnp.float32), {}
def _load_boltz2():
    return None
def _load_feats(*a, **k):
    return None, None
_stub_module("mosaic.losses.boltz2", {"Boltz2Loss": _Boltz2Loss, "load_boltz2": _load_boltz2, "load_features_and_structure_writer": _load_feats})
class _ProteinMPNN:
    @staticmethod
    def from_pretrained():
        return _ProteinMPNN()
_stub_module("mosaic.proteinmpnn.mpnn", {"ProteinMPNN": _ProteinMPNN})
class _InverseFoldingSequenceRecovery:
    def __init__(self, *a, **k):
        pass
    def __call__(self, *a, **k):
        return jnp.asarray(0.0, dtype=jnp.float32), {"sequence_recovery": jnp.asarray(0.0)}
_stub_module("mosaic.losses.protein_mpnn", {"InverseFoldingSequenceRecovery": _InverseFoldingSequenceRecovery})
_stub_module("mosaic_workflows.optimizers", {"sgd_logits_adapter": lambda *a, **k: None, "simplex_APGM_adapter": lambda *a, **k: None})
_stub_module("mosaic_workflows.transforms", {"temperature_on_logits": lambda *a, **k: (lambda x: x), "e_soft_on_logits": lambda *a, **k: (lambda x: x), "gradient_normalizer": lambda *a, **k: (lambda x: x), "position_mask": lambda *a, **k: (lambda x: x), "per_position_allowed_tokens": lambda *a, **k: (lambda x: x)})
_stub_module("mosaic_workflows.design", {"run_workflow": lambda wf: {"trajectory": []}})
_MS_FILE = os.path.join(_WS, "mosaic_workflows", "mhetase_scaffold.py")
ms = SourceFileLoader("mhetase_scaffold", _MS_FILE).load_module()


def _parse_pdb_backbone(pdb_path: Path, residue_numbers: list[int]) -> np.ndarray:
    res_set = set(int(x) for x in residue_numbers)
    # Collect N, CA, C per residue number (first occurrence)
    coords: dict[int, dict[str, np.ndarray | None]] = {rn: {"N": None, "CA": None, "C": None} for rn in res_set}
    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            try:
                atom = line[12:16].strip()
                resi = int(line[22:26])
                if resi in res_set and atom in ("N", "CA", "C"):
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    if coords[resi][atom] is None:
                        coords[resi][atom] = np.array([x, y, z], dtype=np.float32)
            except Exception:
                continue
    bb: list[np.ndarray] = []
    for rn in residue_numbers:
        c = coords[int(rn)]
        assert c["N"] is not None and c["CA"] is not None and c["C"] is not None, f"Missing backbone for residue {rn}"
        bb.append(np.stack([c["N"], c["CA"], c["C"]], axis=0))
    return np.stack(bb, axis=0)


def _mock_predict_fn(binder_len: int, num_bins: int = 64):
    bins = jnp.append(0.0, jnp.linspace(2.3125, 21.6875, num_bins - 1))
    key0 = jax.random.key(0)
    W = jax.random.normal(key0, (20, 3)) * 0.1
    def predict(probs, *, key, state):
        L = probs.shape[0]
        dir_vecs = probs @ W
        dir_vecs = dir_vecs / (jnp.linalg.norm(dir_vecs, axis=-1, keepdims=True) + 1e-6)
        steps = 3.8 * dir_vecs
        ca = jnp.cumsum(steps, axis=0)
        n = ca - 1.33 * dir_vecs
        c = ca + 1.33 * dir_vecs
        backbone = jnp.stack([n, ca, c, c], axis=1)
        d = jnp.sqrt(jnp.maximum(1e-6, jnp.sum((ca[:, None] - ca[None, :]) ** 2, axis=-1)))
        gamma = 1.5
        d_exp = d[..., None]
        logits = -gamma * (d_exp - bins[None, None, :]) ** 2
        class _Out:
            distogram_bins = bins
            distogram_logits = logits
            backbone_coordinates = backbone
            plddt = jnp.ones((L,)) * 0.9
        return _Out()
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-path", required=True)
    ap.add_argument("--pdb-residues", required=True, help="comma-separated residue indices, e.g. 225,448,484")
    ap.add_argument("--binder-len", type=int, required=True)
    ap.add_argument("--ser", type=int, required=True)
    ap.add_argument("--his", type=int, required=True)
    ap.add_argument("--asp", type=int, required=True)
    ap.add_argument("--recycling-steps", type=int, default=1)
    ap.add_argument("--optimize-steps", type=int, default=0)
    args = ap.parse_args()

    pdb_path = Path(args.pdb_path)
    residues = [int(x) for x in args.pdb_residues.split(",")]
    motif_bb = _parse_pdb_backbone(pdb_path, residues)

    motif_positions = {"ser": int(args.ser), "his": int(args.his), "asp": int(args.asp)}
    predict = _mock_predict_fn(args.binder_len)

    # No workflow needed for unit tests; exercise losses directly

    vocab = "ARNDCQEGHILKMFPSTWYV"
    allowed = np.ones((args.binder_len, 20), dtype=np.float32)
    allowed[motif_positions["ser"], :] = 0.0; allowed[motif_positions["ser"], vocab.index("S")] = 1.0
    allowed[motif_positions["his"], :] = 0.0; allowed[motif_positions["his"], vocab.index("H")] = 1.0
    allowed[motif_positions["asp"], :] = 0.0; allowed[motif_positions["asp"], vocab.index("D")] = 1.0
    probs0 = jnp.asarray(allowed)
    probs0 = probs0 / (probs0.sum(-1, keepdims=True) + 1e-8)

    # Build a single mock output for isolated losses
    # predict already assigned
    mock_out = predict(probs0, key=jax.random.key(0), state={})

    # Inspect motif backbone extraction
    ca = motif_bb[:, 1, :]
    dmat = np.sqrt(np.sum((ca[:, None, :] - ca[None, :, :]) ** 2, axis=-1))
    print({"motif_K": motif_bb.shape[0], "motif_backbone_shape": tuple(motif_bb.shape), "motif_ca_dists": dmat.tolist()})

    tests = []
    # Motif RMSD (CA)
    tests.append(("MotifRMSDCA", ms.MotifRMSDCA(motif_positions=(args.ser, args.his, args.asp), motif_template_ca=ca.astype(np.float32))))
    # Motif distogram CCE
    tests.append(("MotifDistogramCCE", ms.MotifDistogramCCE(motif_positions=(args.ser, args.his, args.asp), motif_template_ca=ca.astype(np.float32), max_pair_distance=20.0)))
    # Contact loss (ColabDesign-style)
    tests.append(("ContactLoss", ms.ContactLoss(cutoff=14.0, binary=True, num=2, num_pos=1, seqsep=9, exclude_positions=(args.ser, args.his, args.asp))))
    # PLDDT loss
    tests.append(("PLDDTLoss", ms.PLDDTLoss(exclude_positions=(args.ser, args.his, args.asp))))
    # Composition / priors
    tests.append(("NoCysteine", ms.NoCysteine()))
    tests.append(("SeqEntropyLoss", ms.SeqEntropyLoss()))

    def run_test(name, loss_obj, needs_output=True):
        # Losses take (sequence, output, key); some ignore output
        if needs_output:
            def f(p):
                out = predict(p, key=jax.random.key(0), state={})
                v, _ = loss_obj(p, out, jax.random.key(0))
                return v
            v, aux = loss_obj(probs0, predict(probs0, key=jax.random.key(0), state={}), jax.random.key(0))
        else:
            def f(p):
                out = predict(p, key=jax.random.key(0), state={})
                v, _ = loss_obj(p, out, jax.random.key(0))
                return v
            v, aux = loss_obj(probs0, predict(probs0, key=jax.random.key(0), state={}), jax.random.key(0))
        g = jax.grad(f)(probs0)
        gnorm = jnp.linalg.norm(g)
        ok = bool(jnp.isfinite(v) & jnp.isfinite(gnorm) & (gnorm > 0))
        print({"loss": name, "value": float(v), "grad_norm": float(gnorm), "finite": ok, "aux_keys": list(aux.keys()) if isinstance(aux, dict) else type(aux).__name__})
        return ok

    # Run individual tests; specify which need output
    results = []
    for (nm, lo) in tests:
        needs_out = True
        results.append((nm, run_test(nm, lo, needs_output=needs_out)))
    # Summary
    print({"summary": results})

    # Extra: Direct RMSD smoke test (no predictor), to validate implementation numerically
    # Build a minimal differentiable backbone where motif residues are scaled by a small delta
    K = motif_bb.shape[0]
    L = int(args.binder_len)
    sel = (int(args.ser), int(args.his), int(args.asp))
    rmsd_loss = ms.MotifRMSDCA(motif_positions=sel, motif_template_ca=ca.astype(np.float32))

    def rmsd_value(delta: float):
        # Construct backbone with zeros then insert motif_backbone + delta shift
        bb = jnp.zeros((L, 4, 3), dtype=jnp.float32)
        Q = jnp.asarray(motif_bb.astype(np.float32))  # [K,3,3]
        scale = jnp.asarray(1.0 + delta, dtype=jnp.float32)
        P_ncac = Q * scale  # isotropic scaling to avoid being removed by Kabsch
        # scatter into bb at selected indices (N,CA,C -> channels 0,1,2)
        idx = jnp.array(sel, dtype=jnp.int32)
        bb = bb.at[idx, 0, :].set(P_ncac[:, 0, :])
        bb = bb.at[idx, 1, :].set(P_ncac[:, 1, :])
        bb = bb.at[idx, 2, :].set(P_ncac[:, 2, :])
        class _Out:
            backbone_coordinates = bb
        v, _ = rmsd_loss(probs0, _Out(), jax.random.key(0))
        return v

    val0 = float(rmsd_value(0.0))
    val1 = float(rmsd_value(0.5))
    dval = float(jax.grad(rmsd_value)(0.5))
    print({"rmsd_smoke": {"rmsd_delta0": val0, "rmsd_delta0.5": val1, "grad_at_0.5": dval}})

    # Optional: simple optimization trajectory with combined losses and mock predictor
    if int(args.optimize_steps) > 0:
        ser = int(args.ser); his = int(args.his); asp = int(args.asp)
        excl = (ser, his, asp)
        comb = (
            1.0 * ms.ContactLoss(cutoff=14.0, binary=True, num=2, num_pos=1, seqsep=9, exclude_positions=excl)
            + 0.1 * ms.PLDDTLoss(exclude_positions=excl)
            + 1.0 * ms.MotifDistogramCCE(motif_positions=(ser, his, asp), motif_template_ca=ca.astype(np.float32), max_pair_distance=20.0)
            + 0.5 * ms.MotifRMSDCA(motif_positions=(ser, his, asp), motif_template_ca=ca.astype(np.float32))
        )

        key = jax.random.key(0)
        logits = jnp.zeros((args.binder_len, 20), dtype=jnp.float32)
        # initialize harder at motif positions
        logits = logits.at[ser, :].set(-5.0).at[ser, vocab.index("S")].set(5.0)
        logits = logits.at[his, :].set(-5.0).at[his, vocab.index("H")].set(5.0)
        logits = logits.at[asp, :].set(-5.0).at[asp, vocab.index("D")].set(5.0)

        def _flatten(aux_any):
            if isinstance(aux_any, dict):
                return aux_any
            if isinstance(aux_any, list):
                flat = {}
                for it in aux_any:
                    if isinstance(it, dict):
                        flat.update({k: (float(v) if hasattr(v, "__float__") else v) for k, v in it.items()})
                return flat
            return {}

        def loss_from_logits(x):
            p = jax.nn.softmax(x, axis=-1)
            out = predict(p, key=key, state={})
            v, aux = comb(p, out, key=key)
            return v, _flatten(aux)

        def value(x):
            p = jax.nn.softmax(x, axis=-1)
            out = predict(p, key=key, state={})
            v, _ = comb(p, out, key=key)
            return v

        lr = 0.5
        traj = []
        for step in range(int(args.optimize_steps)):
            v, aux = loss_from_logits(logits)
            traj.append({
                "step": step,
                "loss": float(v),
                "motif_cce": float(aux.get("motif_cce", 0.0)),
                "motif_rmsd": float(aux.get("motif_rmsd", 0.0)),
                "con": float(aux.get("con", 0.0)),
                "plddt": float(aux.get("plddt", 0.0)),
            })
            g = jax.grad(value)(logits)
            logits = logits - lr * g
        # Print first 20 steps of trajectory
        for row in traj[:20]:
            print({"trajectory": row})


if __name__ == "__main__":
    main()


