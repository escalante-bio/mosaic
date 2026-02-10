import io
import warnings
import hashlib
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
from jaxtyping import Float, Array

import gemmi

from collections import defaultdict

ref_charge = defaultdict(int)
ref_charge[('ARG', 'NH2')] = 1
ref_charge[('LYZ', 'NZ')] = 1
ref_charge[('GLU', 'OE2')] = -1
ref_charge[('ASP', 'OD2')] = -1

def pairwise_distance(
    a: Float[Array, "... N D"], b: Float[Array, "... M D"]
) -> Float[Array, "... N M"]:
    r = a[..., :, None, :] - b[..., None, :, :]
    return jnp.sqrt(jnp.sum(r * r, axis=-1) + 1e-8)

def zero_nan_pullback(og_fn: callable):
    fn = jax.custom_vjp(og_fn)

    def _forward(*args, **kwargs):
        return jax.vjp(og_fn, *args, **kwargs)

    def _backward(pullback, g):
        return jax.tree.map(jnp.nan_to_num, pullback(g))

    fn.defvjp(_forward, _backward)
    return fn

_safe_SVD = zero_nan_pullback(partial(jnp.linalg.svd, full_matrices=False))

def project_to_SO3(M: Float[Array, "3 3"]) -> Float[Array, "3 3"]:
    U, _, Vt = _safe_SVD(M)
    d = jnp.sign(jnp.linalg.det(Vt @ U.T))
    R = U @ jnp.diag(jnp.array([1, 1, d])) @ Vt
    return R

def kabsch(
    P: Float[Array, "N 3"], Q: Float[Array, "M 3"]
):
    """
    Solve the optimization problem

        min_{T in SE(3)} || vmap(T)(P) - Q ||^2

    """

    assert P.shape == Q.shape, "Point sets must have same shape"
    assert P.shape[-1] == 3, "Points must be 3D"

    centroid_P = jnp.mean(P, axis=0)
    centroid_Q = jnp.mean(Q, axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    R = project_to_SO3(P_centered.T @ Q_centered)
    t = centroid_Q - centroid_P @ R

    return R, t

def unaligned_rmsd(
    P: Float[Array, "N 3"], Q: Float[Array, "M 3"]
):
    return jnp.sqrt(jnp.sum((P-Q)**2) / P.shape[0])

def calculate_rmsd(
    P: Float[Array, "N 3"], Q: Float[Array, "M 3"]
):
    """
    Aligned RMSD between two 3D point clouds.
    """
    R, t = kabsch(P, Q)
    return unaligned_rmsd(P@R + t, Q)

def fold_in(key: jax.dtypes.prng_key, name: str) -> jax.dtypes.prng_key:
    # hash name to int
    h = hashlib.sha256(name.encode("utf-8")).digest()
    h = int.from_bytes(h[-4:], "big")
    return jax.random.fold_in(key, h)


@dataclass(frozen=True, slots=True)
class _At:
    path: list[object]
    pytree: object
    cur_val: object  # we track this only so we can distinguish between DictKeys and SequenceKeys, seems a bit silly

    def _accessor(self, node):
        for key in self.path:
            match key:
                case jtu.DictKey(key):
                    node = node[key]
                case jtu.GetAttrKey(key):
                    node = getattr(node, key)
                case jtu.SequenceKey(key):
                    node = node[key]

        return node

    def __getattr__(self, key) -> "_At":
        return _At(
            self.path + [jtu.GetAttrKey(key)], self.pytree, getattr(self.cur_val, key)
        )

    def __getitem__(self, key) -> "_At":
        if isinstance(self.cur_val, dict):
            return _At(self.path + [jtu.DictKey(key)], self.pytree, self.cur_val[key])
        else:
            return _At(
                self.path + [jtu.SequenceKey(key)], self.pytree, self.cur_val[key]
            )

    def __call__(self, new_value):
        return eqx.tree_at(self._accessor, self.pytree, new_value)

    def replace(self, replace_fn: callable):
        return eqx.tree_at(self._accessor, self.pytree, replace_fn=replace_fn)


def At(pytree: object) -> _At:
    return _At([], pytree, pytree)


def add_chem_comp(doc: gemmi.cif.Document):
    block = doc[0]
    loop = block.init_loop("_chem_comp.", ["id", "type", "mon_nstd_flag", "name", "pdbx_synonyms", "formula", "formula_weight"])
    loop.add_row(["ALA", gemmi.cif.quote("L-peptide linking"), "y", "ALANINE", "?", gemmi.cif.quote("C3 H7 N O2"), "89.093"])
    loop.add_row(["ARG", gemmi.cif.quote("L-peptide linking"), "y", "ARGININE", "?", gemmi.cif.quote("C6 H15 N4 O2 1"), "175.209"])
    loop.add_row(["ASN", gemmi.cif.quote("L-peptide linking"), "y", "ASPARAGINE", "?", gemmi.cif.quote("C4 H8 N2 O3"), "132.118"])
    loop.add_row(["ASP", gemmi.cif.quote("L-peptide linking"), "y", gemmi.cif.quote("ASPARTIC ACID"), "?", gemmi.cif.quote("C4 H7 N O4"), "133.103"])
    loop.add_row(["CYS", gemmi.cif.quote("L-peptide linking"), "y", "CYSTEINE", "?", gemmi.cif.quote("C3 H7 N O2 S"), "121.158"])
    loop.add_row(["GLN", gemmi.cif.quote("L-peptide linking"), "y", "GLUTAMINE", "?", gemmi.cif.quote("C5 H10 N2 O3"), "146.144"])
    loop.add_row(["GLU", gemmi.cif.quote("L-peptide linking"), "y", gemmi.cif.quote("GLUTAMIC ACID"), "?", gemmi.cif.quote("C5 H9 N O4"), "147.129"])
    loop.add_row(["GLY", gemmi.cif.quote("peptide linking"), "y", "GLYCINE", "?", gemmi.cif.quote("C2 H5 N O2"), "75.067"])
    loop.add_row(["HIS", gemmi.cif.quote("L-peptide linking"), "y", "HISTIDINE", "?", gemmi.cif.quote("C6 H10 N3 O2 1"), "156.162"])
    loop.add_row(["ILE", gemmi.cif.quote("L-peptide linking"), "y", "ISOLEUCINE", "?", gemmi.cif.quote("C6 H13 N O2"), "131.173"])
    loop.add_row(["LEU", gemmi.cif.quote("L-peptide linking"), "y", "LEUCINE", "?", gemmi.cif.quote("C6 H13 N O2"), "131.173"])
    loop.add_row(["LYS", gemmi.cif.quote("L-peptide linking"), "y", "LYSINE", "?", gemmi.cif.quote("C6 H15 N2 O2 1"), "147.195"])
    loop.add_row(["MET", gemmi.cif.quote("L-peptide linking"), "y", "METHIONINE", "?", gemmi.cif.quote("C5 H11 N O2 S"), "149.211"])
    loop.add_row(["PHE", gemmi.cif.quote("L-peptide linking"), "y", "PHENYLALANINE", "?", gemmi.cif.quote("C9 H11 N O2"), "165.189"])
    loop.add_row(["PRO", gemmi.cif.quote("L-peptide linking"), "y", "PROLINE", "?", gemmi.cif.quote("C5 H9 N O2"), "115.130"])
    loop.add_row(["SER", gemmi.cif.quote("L-peptide linking"), "y", "SERINE", "?", gemmi.cif.quote("C3 H7 N O3"), "105.093"])
    loop.add_row(["THR", gemmi.cif.quote("L-peptide linking"), "y", "THREONINE", "?", gemmi.cif.quote("C4 H9 N O3"), "119.119"])
    loop.add_row(["TRP", gemmi.cif.quote("L-peptide linking"), "y", "TRYPTOPHAN", "?", gemmi.cif.quote("C11 H12 N2 O2"), "204.225"])
    loop.add_row(["TYR", gemmi.cif.quote("L-peptide linking"), "y", "TYROSINE", "?", gemmi.cif.quote("C9 H11 N O3"), "181.189"])
    loop.add_row(["VAL", gemmi.cif.quote("L-peptide linking"), "y", "VALINE", "?", gemmi.cif.quote("C5 H11 N O2"), "117.146"])
    return doc
