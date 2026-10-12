"""Residue-selection (binder-/target-relative) variants of structure losses.

Each grouped value is checked against an independent explicit-loop reference
computed in NumPy on absolute (full-complex) indices.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mosaic.losses.structure_prediction as sp

N, L = 30, 12
KEY = jax.random.key(0)


def _make(seed=0):
    k = jax.random.split(jax.random.key(seed), 5)
    return sp.StructureModelOutput(
        distogram_logits=jax.random.normal(k[4], (N, N, 64)),
        distogram_bins=jnp.linspace(2, 22, 64),
        plddt=jax.random.uniform(k[0], (N,)),
        pae=jax.random.uniform(k[1], (N, N)) * 30,
        pae_logits=jax.random.normal(k[2], (N, N, 64)) * 2,
        pae_bins=jnp.asarray(sp.PAE_BINS),
        structure_coordinates=jnp.zeros((N, 3)),
        backbone_coordinates=jax.random.normal(k[3], (N, 4, 3)) * 5,
        full_sequence=jnp.zeros((N, 20)),
        asym_id=jnp.zeros(N),
        residue_idx=jnp.arange(N),
        atom37_coords=jnp.zeros((N, 37, 3)),
        atom37_mask=jnp.ones((N, 37)),
    )


OUT, SEQ = _make(), jnp.zeros((L, 20))
BR = [1, 4, 5, 9]  # binder-relative (== absolute)
TR = [2, 8, 9, 17]  # target-relative
TA = [L + i for i in TR]  # absolute
PAE = np.asarray(OUT.pae)


def _val(term, out=OUT):
    return float(term(SEQ, out, key=KEY)[0])


def test_plddt_selection():
    assert _val(sp.PLDDTLoss(binder_residues=BR)) == pytest.approx(
        -np.mean(np.asarray(OUT.plddt)[BR]), rel=1e-5
    )
    both = np.asarray(OUT.plddt)[BR + TA]
    assert _val(sp.PLDDTLoss(binder_residues=BR, target_residues=TR)) == pytest.approx(
        -both.mean(), rel=1e-5
    )


def test_within_pae_matches_loop():
    ref = sum(PAE[i, j] for i in BR for j in BR if i != j) / len(BR) ** 2
    assert _val(sp.WithinBinderPAE(binder_residues=BR)) == pytest.approx(ref, rel=1e-5)


def test_cross_pae_orientation_and_relative_indexing():
    assert _val(sp.BinderTargetPAE(BR, TR)) == pytest.approx(
        PAE[np.ix_(BR, TA)].mean(), rel=1e-5
    )
    assert _val(sp.TargetBinderPAE(BR, TR)) == pytest.approx(
        PAE[np.ix_(TA, BR)].mean(), rel=1e-5
    )
    # only the target selection given -> all binder residues
    assert _val(sp.BinderTargetPAE(target_residues=TR)) == pytest.approx(
        PAE[np.ix_(range(L), TA)].mean(), rel=1e-5
    )
    mask = np.zeros(N - L, bool)
    mask[TR] = True
    assert _val(sp.BinderTargetPAE(BR, mask)) == _val(sp.BinderTargetPAE(BR, TR))


def _ref_ipsae(out, rows, cols, cutoff=10.0):
    probs = np.asarray(jax.nn.softmax(out.pae_logits, -1))
    bins = np.asarray(out.pae_bins)
    pae = (probs * bins).sum(-1)
    best = -np.inf
    for i in rows:
        js = [j for j in cols if j != i and pae[i, j] < cutoff]
        if not js:
            val = 0.0
        else:
            d0 = 1.24 * (max(len(js), 27) - 15) ** (1 / 3) - 1.8
            tm = (probs[i] / (1 + bins**2 / d0**2)).sum(-1)
            val = float(np.mean([tm[j] for j in js]))
        best = max(best, val)
    return best


def test_ipsae_matches_loop():
    assert -_val(sp.BinderTargetIPSAE(BR, TR)) == pytest.approx(
        _ref_ipsae(OUT, BR, TA), rel=1e-4
    )
    assert -_val(sp.TargetBinderIPSAE(BR, TR)) == pytest.approx(
        _ref_ipsae(OUT, TA, BR), rel=1e-4
    )
    both = min(_ref_ipsae(OUT, BR, TA), _ref_ipsae(OUT, TA, BR))
    assert -_val(sp.IPSAE_min(BR, TR)) == pytest.approx(both, rel=1e-4)
    # target-only selection (the "which target residues does the binder sit on" case)
    assert -_val(sp.BinderTargetIPSAE(target_residues=TR)) == pytest.approx(
        _ref_ipsae(OUT, range(L), TA), rel=1e-4
    )


def test_iptm_and_ptm_match_loop():
    probs = np.asarray(jax.nn.softmax(OUT.pae_logits, -1))
    bins = np.asarray(OUT.pae_bins)
    d0 = 1.24 * (max(N, 19) - 15) ** (1 / 3) - 1.8
    tm = (probs / (1 + bins**2 / d0**2)).sum(-1)

    def best(rows, cols):
        return max(np.mean([tm[i, j] for j in cols if j != i]) for i in rows)

    assert -_val(sp.IPTMLoss(BR, TR)) == pytest.approx(
        max(best(BR, TA), best(TA, BR)), rel=1e-4
    )
    assert -_val(sp.BinderTargetIPTM(BR, TR)) == pytest.approx(best(BR, TA), rel=1e-4)
    ptm = max(np.mean([tm[i, j] for j in BR]) for i in BR)  # self pairs included
    assert -_val(sp.BinderPTMLoss(BR)) == pytest.approx(ptm, rel=1e-4)


def test_ptm_energy_matches_loop():
    logits = np.asarray(OUT.pae_logits, dtype=np.float64)
    bins = np.asarray(OUT.pae_bins)
    d0 = 1.24 * (N - 15) ** (1 / 3) - 1.8
    e = np.log((np.exp(logits) * (1 / (1 + bins**2 / d0**2))).sum(-1))
    ref = -(e[np.ix_(BR, TA)].mean() + e[np.ix_(TA, BR)].mean()) / 2
    assert _val(sp.pTMEnergy(BR, TR)) == pytest.approx(ref, rel=1e-4)


# ---- distogram terms -------------------------------------------------------
DG = np.asarray(OUT.distogram_logits, dtype=np.float64)
BINS = np.asarray(OUT.distogram_bins)


def _logsm(x):
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def _p_contact_log(i, j, cutoff):
    return np.log(np.exp(_logsm(DG[i, j]))[BINS < cutoff].sum())


def test_within_binder_contact_matches_loop():
    sel = [0, 2, 3, 6, 7, 11]
    term = sp.WithinBinderContact(
        max_contact_distance=14.0,
        min_sequence_separation=2,
        num_contacts_per_residue=2,
        binder_residues=sel,
    )

    def partial_entropy(i, j):  # contact_cross_entropy
        lp = _logsm(DG[i, j])
        m = BINS < 14.0
        q = np.exp(lp[m]) / np.exp(lp[m]).sum()
        return (q * lp[m]).sum()

    per_row = []
    for i in sel:
        vals = sorted(
            [partial_entropy(i, j) for j in sel if abs(i - j) > 2], reverse=True
        )
        vals += [-30.0 + partial_entropy(i, j) for j in sel if abs(i - j) <= 2]
        per_row.append(np.mean(sorted(vals, reverse=True)[:2]))
    assert -_val(term) == pytest.approx(np.mean(per_row), rel=1e-4)


def test_inter_contact_matches_loop():
    term = sp.ESMFoldInterContact(
        contact_distance=10.0, num_contacts=2, binder_residues=BR, target_residues=TR
    )

    def ce(i, j):
        lp = _logsm(DG[i, j])
        m = BINS < 10.0
        q = np.exp(lp[m]) / np.exp(lp[m]).sum()
        return -(q * lp[m]).sum()

    ref = np.mean([np.mean(sorted(ce(t, b) for b in BR)[:2]) for t in TA])
    assert _val(term) == pytest.approx(ref, rel=1e-4)


def test_helix_selection_uses_i_i3_pairs():
    sel = [
        0,
        3,
        4,
        7,
        10,
    ]  # pairs (0,3), (4,7), (7,10); 3 and 4 have no (i,i+3) partner
    lp = [_p_contact_log(i, i + 3, 6.0) for i in (0, 4, 7)]
    ref = float(jax.nn.elu(-2.0 - np.mean(lp)))
    assert _val(sp.HelixLoss(binder_residues=sel)) == pytest.approx(ref, rel=1e-4)
    with pytest.raises(ValueError):
        _val(sp.HelixLoss(binder_residues=[0, 5]))


def test_radius_of_gyration_selection():
    sel = [0, 3, 4, 7, 10]
    n = len(sel)
    esq = np.zeros((n, n))
    mae = np.zeros((n, n))
    for a, i in enumerate(sel):
        for b, j in enumerate(sel):
            p = np.exp(_logsm(DG[i, j]))
            esq[a, b], mae[a, b] = (p * BINS**2).sum(), (p * BINS).sum()
    np.fill_diagonal(esq, 0), np.fill_diagonal(mae, 0)
    th = 2.38 * n**0.365
    rg = np.sqrt(esq.mean() + 1e-8)
    assert _val(sp.DistogramRadiusOfGyration(binder_residues=sel)) == pytest.approx(
        float(jax.nn.elu(rg - th)), rel=1e-4
    )
    assert _val(sp.MAERadiusOfGyration(binder_residues=sel)) == pytest.approx(
        float(jax.nn.elu(mae.mean() - th)), rel=1e-4
    )
    xyz = np.asarray(OUT.backbone_coordinates)[sel, 0]
    arg = np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(-1).mean())
    assert _val(
        sp.ActualRadiusOfGyration(target_radius=3.0, binder_residues=sel)
    ) == pytest.approx(float(jax.nn.elu(arg - 3.0)), rel=1e-4)


def test_distogram_ce_and_iptm_proxy_run_on_selection():
    f = jnp.linspace(0, 1, 64)
    sel = [0, 3, 4, 7]
    ce = _val(sp.DistogramCE(f=f, name="x", binder_residues=sel))
    lsm = _logsm(DG) * np.asarray(f)
    sub = lsm[np.ix_(sel, sel)].sum(-1)
    np.fill_diagonal(sub, 0)
    assert ce == pytest.approx(-sub.mean(), rel=1e-4)
    v = _val(sp.DistogramIPTMProxy(binder_residues=sel, target_residues=TR))
    assert -1.0 <= v <= 0.0


# ---- general ---------------------------------------------------------------
def test_explicit_full_selection_equals_default():
    binder, target = list(range(L)), list(range(N - L))
    for cls in (
        sp.BinderTargetPAE,
        sp.TargetBinderPAE,
        sp.IPTMLoss,
        sp.IPSAE_min,
        sp.pTMEnergy,
        sp.BinderTargetIPTM,
        sp.BinderTargetIPSAE,
        sp.TargetBinderIPSAE,
        sp.DistogramIPTMProxy,
        sp.ESMFoldInterContact,
    ):
        assert _val(cls()) == pytest.approx(
            _val(cls(binder_residues=binder, target_residues=target)), rel=1e-6
        ), cls
    for cls in (
        sp.PLDDTLoss,
        sp.WithinBinderPAE,
        sp.BinderPTMLoss,
        sp.HelixLoss,
        sp.WithinBinderContact,
        sp.ESMFoldGlobularity,
        sp.DistogramRadiusOfGyration,
        sp.MAERadiusOfGyration,
    ):
        assert _val(cls()) == pytest.approx(
            _val(cls(binder_residues=binder)), rel=1e-6
        ), cls


def test_validation():
    assert _val(sp.PLDDTLoss(binder_residues=[3, 3, 1])) == _val(
        sp.PLDDTLoss(binder_residues=[1, 3])
    )
    with pytest.raises(ValueError, match="binder_residues"):
        _val(sp.PLDDTLoss(binder_residues=[L]))  # target residue 0 is NOT binder idx L
    with pytest.raises(ValueError, match="target_residues"):
        _val(sp.BinderTargetPAE(target_residues=[N - L]))
    with pytest.raises(ValueError):
        _val(sp.PLDDTLoss(binder_residues=[-1]))
    with pytest.raises(ValueError):
        sp.PLDDTLoss(binder_residues=[])
    with pytest.raises(TypeError):
        sp.PLDDTLoss(binder_residues=[0.5])


def test_jit_grad_composition_and_gradient_locality():
    loss = (
        0.5 * sp.BinderTargetPAE(BR, TR)
        + 1.0 * sp.PLDDTLoss(binder_residues=BR)
        + 1.0 * sp.BinderTargetIPSAE(BR, TR)
        + 0.1 * sp.WithinBinderContact(binder_residues=[0, 3, 7, 11])
    )

    @eqx.filter_jit
    def f(pae_logits, dg):
        out = eqx.tree_at(
            lambda o: (o.pae_logits, o.distogram_logits), OUT, (pae_logits, dg)
        )
        return loss(SEQ, out, key=KEY)[0]

    v, g = jax.value_and_grad(f, argnums=(0, 1))(OUT.pae_logits, OUT.distogram_logits)
    assert np.isfinite(v) and all(np.isfinite(np.asarray(x)).all() for x in g)

    gi = jax.grad(
        lambda lg: sp.BinderTargetIPSAE(BR, TR)(
            SEQ, eqx.tree_at(lambda o: o.pae_logits, OUT, lg), key=KEY
        )[0]
    )(OUT.pae_logits)
    nz = np.abs(np.asarray(gi)).sum(-1) > 0
    expected = np.zeros((N, N), bool)
    expected[np.ix_(BR, TA)] = True
    assert not (nz & ~expected).any()
