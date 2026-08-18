# cyclic_offset_matrix below is adapted from ColabDesign
# (https://github.com/sokrypton/ColabDesign), Copyright (c) 2021 Sergey
# Ovchinnikov, MIT License. See ./NOTICE.

import numpy as np


def cyclic_offset_matrix(
    length: int, offset_type: int = 2, saturation_value: int = 32
) -> np.ndarray:
    """Wraparound sequence-separation matrix for a head-to-tail cyclic chain.

    Port of ColabDesign's ``cyclic_offset`` (af_cyc_design.ipynb), as productionized
    in grasp's ``utils/cyclic_offset.py``. For each residue pair (i, j), computes the
    shortest signed separation around the ring instead of the linear ``i - j``.

    offset_type:
      1: unsigned cyclic distance (min(|i-j|, length-|i-j|)).
      2: cyclic distance, re-signed to match the linear offset's sign, but only
         overriding the linear distance where the cyclic path is shorter.
      3: like 2, but pairs with cyclic distance > 2 are saturated to
         ``saturation_value`` (biases losses away from long-range pairs).

    saturation_value: magnitude used by offset_type=3 for long-range pairs. Defaults
      to 32 to match AF2-multimer's own relpos clipping (``max_relative_idx``), which
      is what ``offset`` ultimately feeds into via
      ``EmbeddingsAndEvoformer._relative_encoding``'s ``batch["offset"]`` override —
      keep this in sync with ``max_relative_idx`` if that's ever configured
      differently. Unused for offset_type 1/2.
    """
    if offset_type not in (1, 2, 3):
        raise ValueError(f"Invalid offset_type: {offset_type}. Must be 1, 2, or 3.")

    i = np.arange(length)
    ij = np.stack([i, i + length], -1)
    offset = i[:, None] - i[None, :]
    c_offset = np.abs(ij[:, None, :, None] - ij[None, :, None, :]).min((2, 3))

    if offset_type >= 2:
        a = c_offset < np.abs(offset)
        c_offset[a] = -c_offset[a]
    if offset_type == 3:
        idx = np.abs(c_offset) > 2
        c_offset[idx] = saturation_value * np.sign(c_offset[idx])

    return c_offset * np.sign(offset)
