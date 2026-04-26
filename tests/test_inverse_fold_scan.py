"""Regression test for the scan-based inverse_fold.

Builds a tiny synthetic StructureModelOutput, runs inverse_fold, and
compares against an inline for-loop reimplementation of the same body
to confirm the scan change is semantically equivalent.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.proteinmpnn.mpnn import load_mpnn_sol
from mosaic.losses.protein_mpnn import inverse_fold, boltz_to_mpnn_matrix
from mosaic.losses.structure_prediction import StructureModelOutput


def _make_fake_output(N: int, key) -> StructureModelOutput:
    coord_key, seq_key = jax.random.split(key)
    backbone = jax.random.normal(coord_key, (N, 4, 3))
    full_seq = jax.nn.one_hot(
        jax.random.randint(seq_key, (N,), 0, 20), 20, dtype=jnp.int32
    )
    asym_id = jnp.zeros(N, dtype=jnp.int32)
    residue_idx = jnp.arange(N, dtype=jnp.int32)
    return StructureModelOutput(
        distogram_logits=jnp.zeros((N, N, 1)),
        distogram_bins=jnp.zeros((1,)),
        plddt=jnp.zeros(N),
        pae=jnp.zeros((N, N)),
        pae_logits=jnp.zeros((N, N, 1)),
        pae_bins=jnp.zeros((1,)),
        structure_coordinates=backbone,
        backbone_coordinates=backbone,
        full_sequence=full_seq,
        asym_id=asym_id,
        residue_idx=residue_idx,
    )


def _reference_for_loop(mpnn, binder_length, output, temp, key, jacobi_iterations, bias):
    """Inline reimplementation matching the pre-scan body exactly."""
    coords = output.backbone_coordinates
    total_length = output.full_sequence.shape[0]
    mpnn_mask = jnp.ones(total_length, dtype=jnp.int32)
    asym_id = output.asym_id
    chain_lengths = (asym_id[:, None] == np.arange(16)[None]).sum(-2)
    res_idx_adjustment = jnp.cumsum(chain_lengths, -1) - chain_lengths
    residue_idx = (
        output.residue_idx
        + (asym_id[:, None] == np.arange(16)[None]) @ res_idx_adjustment
    )
    residue_idx += 100 * asym_id

    h_V, h_E, E_idx = mpnn.encode(
        X=coords, mask=mpnn_mask,
        residue_idx=residue_idx, chain_encoding_all=asym_id, key=key,
    )
    decoding_order = (
        jax.random.uniform(key, shape=(total_length,)).at[:binder_length].add(2.0)
    )
    gumbel = jax.random.gumbel(key, (binder_length, 20))

    def seq_to_logits(sequence):
        full_sequence = output.full_sequence.at[:binder_length].set(
            jax.nn.one_hot(sequence, 20, dtype=jnp.int32)
        )
        sequence_mpnn = full_sequence @ boltz_to_mpnn_matrix()
        logits = mpnn.decode(
            S=sequence_mpnn, h_V=h_V, h_E=h_E, E_idx=E_idx,
            mask=mpnn_mask, decoding_order=decoding_order,
        )[0]
        return logits[:binder_length] @ boltz_to_mpnn_matrix().T

    sequence = jax.random.randint(key=key, minval=0, maxval=20, shape=binder_length)
    for _ in range(jacobi_iterations):
        logits = seq_to_logits(sequence)
        if bias is not None:
            logits += bias
        sequence = (logits + temp * gumbel).argmax(-1)
    return sequence


def test_scan_matches_for_loop():
    mpnn = load_mpnn_sol(0.0)
    key = jax.random.key(42)
    out = _make_fake_output(N=24, key=key)
    binder_length = 12
    temp = 0.01
    iters = 5

    scan_seq = inverse_fold(
        mpnn=mpnn, binder_length=binder_length, output=out,
        temp=temp, key=key, jacobi_iterations=iters,
    )
    ref_seq = _reference_for_loop(
        mpnn=mpnn, binder_length=binder_length, output=out,
        temp=temp, key=key, jacobi_iterations=iters, bias=None,
    )

    assert scan_seq.shape == (binder_length,)
    assert jnp.array_equal(scan_seq, ref_seq), (
        f"scan output {scan_seq} != for-loop output {ref_seq}"
    )


def test_scan_matches_for_loop_with_bias():
    mpnn = load_mpnn_sol(0.0)
    key = jax.random.key(7)
    out = _make_fake_output(N=20, key=key)
    binder_length = 10
    temp = 0.05
    iters = 4
    bias = jax.random.normal(jax.random.key(99), (binder_length, 20))

    scan_seq = inverse_fold(
        mpnn=mpnn, binder_length=binder_length, output=out,
        temp=temp, key=key, jacobi_iterations=iters, bias=bias,
    )
    ref_seq = _reference_for_loop(
        mpnn=mpnn, binder_length=binder_length, output=out,
        temp=temp, key=key, jacobi_iterations=iters, bias=bias,
    )

    assert jnp.array_equal(scan_seq, ref_seq)


if __name__ == "__main__":
    test_scan_matches_for_loop()
    print("test_scan_matches_for_loop: PASS")
    test_scan_matches_for_loop_with_bias()
    print("test_scan_matches_for_loop_with_bias: PASS")
