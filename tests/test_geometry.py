import numpy as np
import pytest

from mosaic.geometry import cyclic_offset_matrix


def test_invalid_offset_type_raises():
    with pytest.raises(ValueError):
        cyclic_offset_matrix(4, offset_type=0)


def test_offset_type_1_wraparound_distance_is_antisymmetric_in_magnitude():
    m = cyclic_offset_matrix(4, offset_type=1)
    # signed like the linear offset, but magnitude is the wraparound distance:
    # |m| is symmetric (a proper distance), even though m itself is antisymmetric.
    assert np.array_equal(np.abs(m), np.abs(m).T)
    # residue 0 and residue 3 are adjacent on the ring (distance 1), not 3.
    assert abs(m[0, 3]) == 1
    assert abs(m[0, 2]) == 2
    # interior pairs match the ordinary linear distance.
    assert abs(m[1, 2]) == 1
    # for length=4, the maximum possible unsigned cyclic distance is 2 (L/2).
    assert m.max() == 2


def test_offset_type_2_matches_linear_offset_away_from_wraparound():
    length = 6
    m = cyclic_offset_matrix(length, offset_type=2)
    i = np.arange(length)
    linear_offset = i[:, None] - i[None, :]
    # for interior pairs the cyclic path isn't shorter, so type 2 == linear offset.
    assert m[1, 2] == linear_offset[1, 2]
    assert m[2, 3] == linear_offset[2, 3]


def test_offset_type_2_wraparound_pair_is_close():
    length = 6
    m = cyclic_offset_matrix(length, offset_type=2)
    # residue 0 and residue length-1 are wraparound-adjacent: cyclic distance 1,
    # far shorter than the linear distance (length-1), so it should be re-signed
    # to a small magnitude rather than the large linear offset.
    assert abs(m[0, length - 1]) == 1


def test_offset_type_3_saturates_long_range_pairs():
    length = 8
    m = cyclic_offset_matrix(length, offset_type=3)
    # a pair whose cyclic distance exceeds 2 should be saturated to +/-32.
    far_pairs_mask = np.abs(cyclic_offset_matrix(length, offset_type=1)) > 2
    assert np.all(np.abs(m[far_pairs_mask]) == 32)


def test_offset_type_3_saturation_value_is_configurable():
    # a caller using AF2 with a non-default max_relative_idx must be able to
    # keep offset_type=3's saturation consistent with it.
    length = 8
    m = cyclic_offset_matrix(length, offset_type=3, saturation_value=64)
    far_pairs_mask = np.abs(cyclic_offset_matrix(length, offset_type=1)) > 2
    assert np.all(np.abs(m[far_pairs_mask]) == 64)
