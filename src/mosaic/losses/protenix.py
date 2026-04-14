import copy

# set "PROTENIX_DATA_ROOT_DIR" env variable
import os
from pathlib import Path

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree
from protenix.data.constants import PRO_STD_RESIDUES
from protenix.protenij import (
    InitialEmbedding,
    TrunkEmbedding,
)
from protenix.protenij import Protenix as Protenij

from mosaic.common import TOKENS, LinearCombination, LossTerm
from mosaic.losses.structure_prediction import PAE_BINS, StructureModelOutput

os.environ["PROTENIX_DATA_ROOT_DIR"] = str(Path("~/.protenix").expanduser())


def biotite_atom_to_gemmi_atom(atom):
    ga = gemmi.Atom()
    ga.pos = gemmi.Position(*atom.coord)
    ga.element = gemmi.Element(atom.element)
    ga.name = atom.atom_name
    return ga


def new_gemmi_residue(atom):
    r = gemmi.Residue()
    r.name = atom.res_name
    r.seqid = gemmi.SeqId(atom.res_id, " ")
    r.entity_type = gemmi.EntityType.Polymer
    return r


def biotite_array_to_gemmi_struct(atom_array, pred_coord=None, per_atom_plddt=None):
    if pred_coord is not None:
        atom_array = copy.deepcopy(atom_array)
        atom_array.coord = pred_coord
    structure = gemmi.Structure()
    model = gemmi.Model("0")
    chains = {}
    for atom_idx, atom in enumerate(atom_array):
        chain = chains.setdefault(atom.chain_id, {})
        residue = chain.setdefault(int(atom.res_id), new_gemmi_residue(atom))
        gemmi_atom = biotite_atom_to_gemmi_atom(atom)
        if per_atom_plddt is not None:
            gemmi_atom.b_iso = per_atom_plddt[atom_idx]
        residue.add_atom(gemmi_atom)
    for k in chains:
        chain = gemmi.Chain(k)
        chain.append_residues(list(chains[k].values()))
        model.add_chain(chain)
    structure.add_model(model)
    return structure


def _build_boltz_to_protenix_matrix():
    T = np.zeros((len(TOKENS), 32))
    for i, tok in enumerate(TOKENS):
        protenix_idx = PRO_STD_RESIDUES[
            gemmi.expand_one_letter(tok, gemmi.ResidueKind.AA)
        ]
        T[i, protenix_idx] = 1
    return T

BOLTZ_TO_PROTENIX = _build_boltz_to_protenix_matrix()


def set_binder_sequence(new_sequence: Float[Array, "N 20"], features: PyTree):
    binder_len = new_sequence.shape[0]
    protenix_sequence = new_sequence @ BOLTZ_TO_PROTENIX
    n_msa = features["msa"].shape[0]
    print("n_msa", n_msa)

    zero_msa_idx = 20  # GAP #31#20
    n_fake_seq = 1

    # TODO: we may need to be more aggressive here and upweight the profile
    # We assume there are no MSA hits for the binder sequence
    binder_profile = jnp.zeros_like(features["profile"][:binder_len])
    binder_profile = (
        binder_profile.at[:binder_len].set(protenix_sequence) * n_fake_seq / n_msa
    )
    binder_profile = binder_profile.at[:, zero_msa_idx].set(
        (n_msa - n_fake_seq) / n_msa
    )
    # binder_profile = protenix_sequence
    return features | {
        "restype": features["restype"].at[:binder_len, :].set(protenix_sequence),
        # "msa": features["msa"].at[:, :binder_len].set(protenix_sequence.argmax(-1)),
        "profile": features["profile"].at[:binder_len].set(binder_profile),
    }


def get_trunk_state(
    *,
    model: Protenij,
    features: PyTree,
    initial_recycling_state: TrunkEmbedding | None,
    recycling_steps: int,
    key: jax.Array,
) -> tuple[InitialEmbedding, TrunkEmbedding]:
    """ Compute trunk embedding."""
    # manual recycling
    state = initial_recycling_state
    initial_embedding = model.embed_inputs(
        input_feature_dict=features
    )  
    if state is None:
        state = TrunkEmbedding(
            s=jnp.zeros_like(initial_embedding.s_init),
            z=jnp.zeros_like(initial_embedding.z_init),
        )

    def body_fn(carry):
        iter, state, key = carry
        state = jax.tree.map(jax.lax.stop_gradient, state)
        s, z = state.s, state.z
        z = initial_embedding.z_init + model.linear_no_bias_z_cycle(
            model.layernorm_z_cycle(z)
        )
        if model.template_embedder.n_blocks > 0:
            z = z + model.template_embedder(features, z, pair_mask=None, key=key)
        z = model.msa_module(
            features,
            z,
            initial_embedding.s_inputs,
            pair_mask=None,
            key=key,
        )
        s = initial_embedding.s_init + model.linear_no_bias_s(model.layernorm_s(s))
        s, z = model.pairformer_stack(
            s, z, pair_mask=None, key=jax.random.fold_in(key, 1)
        )
        return (iter + 1, TrunkEmbedding(s=s, z=z), jax.random.fold_in(key, 1))

    # while loop first because jax doesn't respect the stop_gradient in body_fn
    _, state, key = jax.lax.while_loop(
        lambda carry: carry[0] < recycling_steps - 1,
        body_fn,
        (0, state, key),
    )
    state = jax.tree.map(jax.lax.stop_gradient, state)
    return initial_embedding, body_fn((0, state, key))[1]


PROTENIX_DISTOGRAM_BINS = np.linspace(start=2.3125, stop=21.6875, num=64)


def protenix_forward_from_trunk(
    model: Protenij,
    features: PyTree,
    initial_embedding: InitialEmbedding,
    trunk_state: TrunkEmbedding,
    sampling_steps: int,
    key: jax.Array,
) -> StructureModelOutput:
    """Run distogram, structure, and confidence from pre-computed trunk state."""
    distogram_logits = model.distogram_head(trunk_state.z)

    structure_coordinates = model.sample_structures(
        initial_embedding=initial_embedding,
        trunk_embedding=trunk_state,
        input_feature_dict=features,
        N_samples=1,
        N_steps=sampling_steps,
        key=key,
    )

    confidence = model.confidence_metrics(
        initial_embedding=initial_embedding,
        trunk_embedding=trunk_state,
        input_feature_dict=features,
        coordinates=structure_coordinates,
        key=key,
    )

    # pLDDT normalized to [0, 1]
    plddt = (
        jax.nn.softmax(confidence.plddt_logits[0][features["atom_rep_atom_idx"]])
        * jnp.linspace(0, 1, 50)[None, :]
    ).sum(-1)

    # PAE
    pae_logits = confidence.pae_logits[0]
    pae = (
        jax.nn.softmax(pae_logits) * PAE_BINS[None, None, :]
    ).sum(-1)

    # Backbone coordinates (N, CA, C, O)
    n_tokens = features["restype"].shape[0]
    first_atom_idx = jax.vmap(lambda atoms: jnp.nonzero(atoms, size=1)[0][0])(
        (features["atom_to_token_idx"][:, None] == jnp.arange(n_tokens)[None, :]).T
    )
    all_atom_coords = structure_coordinates[0]
    backbone_coordinates = jnp.stack(
        [all_atom_coords[first_atom_idx + i] for i in range(4)], -2
    )

    return StructureModelOutput(
        distogram_logits=distogram_logits,
        distogram_bins=PROTENIX_DISTOGRAM_BINS,
        plddt=plddt,
        pae=pae,
        pae_logits=pae_logits,
        pae_bins=PAE_BINS,
        structure_coordinates=structure_coordinates,
        backbone_coordinates=backbone_coordinates,
        full_sequence=features["restype"] @ BOLTZ_TO_PROTENIX.T,
        asym_id=features["asym_id"],
        residue_idx=features["residue_index"],
    )


class MultiSampleProtenixLoss(LossTerm):
    model: Protenij
    features: PyTree
    loss: LossTerm | LinearCombination
    recycling_steps: int = 1
    sampling_steps: int = 20
    num_samples: int = 4
    name: str = "protenix"
    initial_recycling_state: TrunkEmbedding | None = None
    reduction: any = jnp.mean

    """
        Run the structure and confidence modules multiple times from the same trunk output.
        When `reduction` is jnp.mean this is equivalent to the expected loss over multiple samples *assuming a deterministic trunk*, but faster.
        This will consume quite a bit of memory -- if you'd like to sacrifice some speed for memory, replace the vmap below with a jax.lax.map.
    """

    def __call__(self, sequence: Float[Array, "N 20"], key):
        """Compute the loss for a given sequence."""
        # Set the binder sequence in the features
        features = set_binder_sequence(sequence, self.features)

        # run trunk once
        initial_embedding, trunk_state = get_trunk_state(
            model=self.model,
            features=features,
            initial_recycling_state=self.initial_recycling_state,
            recycling_steps=self.recycling_steps,
            key=key,
        )

        # initialize from trunk outputs using vmap
        def apply_loss_to_single_sample(key):
            output = protenix_forward_from_trunk(
                model=self.model,
                features=features,
                initial_embedding=initial_embedding,
                trunk_state=trunk_state,
                sampling_steps=self.sampling_steps,
                key=key,
            )
            v, aux = self.loss(
                sequence=sequence,
                output=output,
                key=key,
            )

            return v, aux

        vs, auxs = jax.vmap(apply_loss_to_single_sample)(
            jax.random.split(key, self.num_samples)
        )
        sortperm = jnp.argsort(vs)
        return self.reduction(vs), jax.tree.map(lambda v: list(v[sortperm]), auxs)
