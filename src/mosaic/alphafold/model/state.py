import equinox as eqx
from jaxtyping import Array, Float
class AlphaFoldState(eqx.Module):
    prev_pos: Float[Array, "num_residue atom_type_num 3"]
    prev_msa_first_row: Float[Array, "num_residue msa_channel"]
    prev_pair: Float[Array, "num_residue num_residue pair_channel"]
