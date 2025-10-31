import marimo

__generated_with = "0.17.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch 
    import numpy as np
    return np, torch


@app.cell
def _(gemmi):
    framework = gemmi.read_structure("3eak-assembly1.cif")
    framework.remove_alternative_conformations()
    framework.remove_ligands_and_waters()
    return (framework,)


@app.cell
def _(framework):
    framework.write_minimal_pdb("framework.pdb")
    return


@app.cell
def _(framework, pdb_viewer):
    pdb_viewer(framework)
    return


@app.cell
def _(framework, gemmi):
    gemmi.one_letter_code([r.name for r in framework[0][0]])
    return


@app.cell
def _():
    framework_masked = "QVQLVESGGGLVQPGGSLRLSCAASXXXXXXXXXXXLGWFRQAPGQGLEAVAAXXXXXXXXYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCXXXXXXXXXXXXXXXXXXWGQGTLVTVS"
    return (framework_masked,)


@app.cell
def _():
    import jax.numpy as jnp
    return (jnp,)


@app.cell
def _():
    from mosaic.structure_prediction import TargetChain
    return (TargetChain,)


@app.cell
def _():
    import jax
    return (jax,)


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(framework_masked):
    L_BINDER = len(framework_masked)
    return (L_BINDER,)


@app.cell
def _():
    from mosaic.models.boltzgen import load_boltzgen, load_features_from_yaml, generate_mmcif, Sampler
    return Sampler, generate_mmcif, load_boltzgen, load_features_from_yaml


@app.cell
def _(load_boltzgen):
    boltzgen = load_boltzgen()
    return (boltzgen,)


@app.cell
def _(Path):
    scaffold_yaml = r"""
    path: framework.pdb
    include: 
      - chain: 
          id: A

    design:
      - chain:
          id: A
          res_index: 26..36,54..61,100..117

    structure_groups:
      - group:
          id: A
          visibility: 2
      - group:
          id: A
          visibility: 0
          res_index: 26..36,54..61,100..117
    """
    Path("framework.yaml").write_text(scaffold_yaml)
    return (scaffold_yaml,)


@app.cell
def _(L_BINDER):
    yaml_binder = r"""
    entities:
      - file: 
          path: framework.yaml


      - file:
          # file references are relative to the location of the .yaml file
          path: 1PEB.cif # .pdb files also work

          include: 
            - chain:
                id: A
    """.format(N = L_BINDER)
    return (yaml_binder,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    from pathlib import Path
    return (Path,)


@app.cell
def _(Path, load_features_from_yaml, scaffold_yaml, yaml_binder):
    scaffold_yaml

    features_binder, torch_features_binder, writer_binder = load_features_from_yaml(yaml_string=yaml_binder, files = {"1PEB.cif" : Path("1PEB.cif"), "framework.yaml" : Path("framework.yaml"), "framework.pdb" : Path("framework.pdb")})
    return features_binder, torch_features_binder, writer_binder


@app.cell
def _(boltzgen, features_binder, jax, jnp):
    disto, struct = boltzgen(
        features_binder,
        1,
        key=jax.random.key(0),  # np.random.randint(10000)),
        deterministic=True,
        num_sampling_steps=300,
        step_scale=jnp.array(1.8),
        noise_scale=jnp.array(0.95),
    )
    return


@app.cell
def _():
    return


@app.cell
def _(Sampler, boltzgen, features_binder, jax):
    sampler = Sampler.from_features(
        model = boltzgen,
        features=features_binder,
        key = jax.random.key(0),  # np.random.randint(10000)),
        deterministic=True,
        recycling_steps=3
    )
    return (sampler,)


@app.cell
def _():
    import equinox as eqx
    return (eqx,)


@app.cell
def _(eqx):
    @eqx.filter_jit
    def apply(m, *a, **kw):
        return m(*a, **kw)
    return (apply,)


@app.cell
def _(prediction):
    prediction.iptm
    return


@app.cell
def _(apply, boltzgen, jax, jnp, np, sampler):
    coords = apply(
        sampler,
        model=boltzgen,
        num_sampling_steps=500,
        step_scale=jnp.array(1.8),
        noise_scale=jnp.array(0.88),
        key=jax.random.key(np.random.randint(10000)),
    )
    return (coords,)


@app.cell
def _(sample):
    samples = [sample() for _ in range(10)]
    return (samples,)


@app.cell
def _(samples):
    best =sorted(samples)[-1]
    return (best,)


@app.cell
def _(
    TOKENS,
    apply,
    boltzgen,
    gemmi,
    generate_mmcif,
    jax,
    jnp,
    np,
    p_features,
    p_writer,
    protenix,
    sampler,
    torch,
    torch_features_binder,
    writer_binder,
):
    def sample():
        coords = apply(
            sampler,
            model=boltzgen,
            num_sampling_steps=500,
            step_scale=jnp.array(1.8),
            noise_scale=jnp.array(0.88),
            key=jax.random.key(np.random.randint(10000)),
        )
        st = generate_mmcif(
            writer_binder,
            prediction=torch_features_binder
            | {
                "coords": torch.tensor(
                    np.array(coords)
                ),  
                "exception": False,
                "masks": torch_features_binder["atom_pad_mask"].unsqueeze(0),
                "extra_mols": None,
                "structure_bonds": [torch.zeros(0)],
            },
            batch=torch_features_binder
            | {
                "extra_mols": None,
                "target_msa_mask": torch.zeros(1, 1, 1),
                "structure_bonds": [torch.zeros(0)],
            },
        )
        binder_seq = gemmi.one_letter_code([r.name for r in st[0][0]])
        prediction = protenix.predict(
            PSSM=jax.nn.one_hot([TOKENS.index(c) for c in binder_seq], 20),
            features=p_features,
            writer=p_writer,
            key=jax.random.key(1),
            recycling_steps=3
        )
        return (prediction.iptm, binder_seq)

    return (sample,)


@app.cell
def _(gemmi):
    from ipymolstar import PDBeMolstar
    def pdb_viewer(st: gemmi.Structure):
        """Display a PDB file using Molstar"""
        custom_data = {
            "data": st.make_pdb_string(),
            "format": "pdb",
            "binary": False,
        }
        return PDBeMolstar(custom_data=custom_data, visual_style="cartoon")
    return (pdb_viewer,)


@app.cell
def _(pdb_viewer, st):
    pdb_viewer(st)
    return


@app.cell
def _(pdb_viewer, prediction):
    pdb_viewer(prediction.st)
    return


@app.cell
def _():
    import gemmi
    return (gemmi,)


@app.cell
def _(gemmi, st):
    binder_seq = gemmi.one_letter_code([r.name for r in st[0][0]])
    binder_seq
    return (binder_seq,)


@app.cell
def _(framework_masked):
    framework_masked
    return


@app.cell
def _(TOKENS, best, framework_masked, jax):
    partial_pssm = jax.nn.one_hot([TOKENS.index(c) for (c,b) in zip(best[1], framework_masked) if b=="X"], 20)
    return (partial_pssm,)


@app.cell
def _(coords, generate_mmcif, np, torch, torch_features_binder, writer_binder):
    st = generate_mmcif(
        writer_binder,
        prediction=torch_features_binder
        | {
            "coords": torch.tensor(
                np.array(coords)
            ),  
            "exception": False,
            "masks": torch_features_binder["atom_pad_mask"].unsqueeze(0),
            "extra_mols": None,
            "structure_bonds": [torch.zeros(0)],
        },
        batch=torch_features_binder
        | {
            "extra_mols": None,
            "target_msa_mask": torch.zeros(1, 1, 1),
            "structure_bonds": [torch.zeros(0)],
        },
    )
    return (st,)


@app.cell
def _():
    # class Wrapper(eqx.Module):
    #     s: Sampler
    #     model: joltzgen.JoltzGen

    #     def __call__(self, atom_coords_noisy, t_hat, *, key):
    #         return self.model.structure_module.preconditioned_network_forward(
    #             atom_coords_noisy,
    #             t_hat,
    #             network_condition_kwargs={
    #                 "diffusion_conditioning": {
    #                     "q": self.s.q,
    #                     "c": self.s.c,
    #                     "to_keys": self.s.to_keys,
    #                     "atom_enc_bias": self.s.atom_enc_bias,
    #                     "atom_dec_bias": self.s.atom_dec_bias,
    #                     "token_trans_bias": self.s.token_trans_bias,
    #                 },
    #                 "multiplicity": 1,
    #                 "s_trunk": self.s.trunk_s,
    #                 "s_inputs": self.s.s_inputs,
    #                 "feats": self.s.feats,
    #             },
    #             key=key,
    #         )
    return


@app.cell
def _():
    # from joltzgen import compute_random_augmentation, center,weighted_rigid_align
    return


@app.cell
def _():
    # @eqx.filter_jit
    # def sample(
    #     self,
    #     atom_mask,
    #     num_sampling_steps,
    #     *,
    #     key,
    #     sample_schedule: Literal["dilated", "af3"] = "dilated",
    #     step_scale: float,
    #     noise_scale: float,
    #     model,
    #     model_b,
    #     alpha = 0.5
    # ):
    #     shape = (*atom_mask.shape, 3)

    #     sigmas = (
    #         self.sample_schedule(num_sampling_steps)
    #         if sample_schedule == "af3"
    #         else self.sample_schedule_dilated(num_sampling_steps)
    #     )

    #     gammas = jnp.where(sigmas > self.gamma_min, self.gamma_0, 0.0)

    #     # atom position is noise at the beginning

    #     @jax.checkpoint
    #     def sample_body_function(carry, input):
    #         (sigma_tm, sigma_t, gamma) = input
    #         atom_coords, key = carry
    #         random_R, random_tr = compute_random_augmentation(key=key)
    #         key = jax.random.fold_in(key, 1)
    #         atom_coords = center(atom_coords, atom_mask)
    #         atom_coords = (
    #             jnp.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
    #         )

    #         t_hat = sigma_tm * (1 + gamma)
    #         noise_var = noise_scale**2 * (t_hat**2 - sigma_tm**2)
    #         eps = (
    #             noise_scale
    #             * jnp.sqrt(noise_var)
    #             * jax.random.normal(shape=shape, key=key)
    #         )
    #         key = jax.random.fold_in(key, 1)
    #         atom_coords_noisy = atom_coords + eps

    #         def get_step(model, atom_coords_noisy):
    #             atom_coords_denoised = model(atom_coords_noisy, t_hat, key=key)

    #             # if self.alignment_reverse_diff:
    #             #     atom_coords_D = weighted_rigid_align(
    #             #         atom_coords_denoised,
    #             #         atom_coords_noisy,
    #             #         atom_mask[:, :atom_coords_noisy.shape[1]],
    #             #         atom_mask[:, :atom_coords_noisy.shape[1]],
    #             #     )

    #             return (atom_coords_noisy - atom_coords_denoised) / t_hat


    #         dir = get_step(model, atom_coords_noisy)
    #         dir_b = get_step(model_b, atom_coords_noisy[:, :1088])
    #         binder_dir = alpha * dir[0:1, :dir_b.shape[1]] + (1-alpha)* dir_b
    #         weighted_directions = dir.at[0:1, :dir_b.shape[1], :].set(binder_dir)

    #         atom_coords_next = (
    #             atom_coords_noisy
    #             + step_scale * (sigma_t - t_hat) * weighted_directions
    #         )

    #         return (atom_coords_next, jax.random.fold_in(key, 0)), None

    #     (atom_coords, _), _ = jax.lax.scan(
    #         sample_body_function,
    #         (
    #             sigmas[0] * jax.random.normal(shape=shape, key=key),
    #             jax.random.fold_in(key, 1),
    #         ),
    #         (sigmas[:-1], sigmas[1:], gammas[1:]),
    #     )

    #     return atom_coords
    return


@app.cell
def _():
    from mosaic.models.boltz2 import Boltz2
    return (Boltz2,)


@app.cell
def _(Boltz2):
    protenix = Boltz2()#ProtenixMini()
    return (protenix,)


@app.cell
def _(gemmi):
    gemmi.one_letter_code(
                    [r.name for r in gemmi.read_structure("1PEB.cif")[0][0]]
                )
    return


@app.cell
def _(TargetChain, framework_masked, gemmi, protenix):

    p_features, p_writer = protenix.target_only_features(
        chains=[
            TargetChain(
                sequence=framework_masked,
                use_msa=True,
            ),
            TargetChain(
                sequence=gemmi.one_letter_code(
                    [r.name for r in gemmi.read_structure("1PEB.cif")[0][0]]
                ),
                use_msa=True
            ),
        ]
    )
    return p_features, p_writer


@app.cell
def _(binder_seq):
    binder_seq
    return


@app.cell
def _(TOKENS, binder_seq, jax, p_features, p_writer, protenix):
    prediction = protenix.predict(
        PSSM=jax.nn.one_hot([TOKENS.index(c) for c in binder_seq], 20),
        features=p_features,
        writer=p_writer,
        key=jax.random.key(1),
        recycling_steps=3
    )
    return (prediction,)


@app.cell
def _(TOKENS, binder_seq, jax, plt):
    plt.imshow(jax.nn.one_hot([TOKENS.index(c) for c in binder_seq], 20))
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _(prediction):
    prediction.iptm
    return


@app.cell
def _(pdb_viewer, prediction):
    pdb_viewer(prediction.st)
    return


@app.cell
def _(mo, prediction):
    mo.download(data = prediction.st.make_pdb_string(), filename = "a.pdb")
    return


@app.cell
def _():
    from mosaic.common import TOKENS
    return (TOKENS,)


@app.cell
def _():
    import joltzgen
    return


@app.cell
def _():
    from typing import Literal
    return


@app.cell
def _():
    from mosaic.models.protenix import ProtenixMini
    return


@app.cell
def _(
    AbLangPseudoLikelihood,
    ESMCPseudoLikelihood,
    InverseFoldingSequenceRecovery,
    SetPositions,
    TrigramLL,
    jax,
    load_ablang,
    load_esmc,
    masked_framework_sequence,
    mpnn,
    np,
    p_features,
    protenix,
    sp,
):
    ablang, ablang_tokenizer = load_ablang("heavy")
    ablang_pll = AbLangPseudoLikelihood(
        model=ablang,
        tokenizer=ablang_tokenizer,
        stop_grad=True,
    )
    # and ESMC PLL
    ESMCPLL = ESMCPseudoLikelihood(load_esmc("esmc_300m"), stop_grad=True)

    structure_loss = (
        0.1 * sp.PLDDTLoss()
        + 1
        * sp.BinderTargetContact(
            paratope_idx=np.array(
                [i for (i, c) in enumerate(masked_framework_sequence) if c == "X"]
            ),
        )
        + 0.50 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.25 * sp.IPTMLoss()
        + 0.2 * sp.WithinBinderPAE()
        + 0.5 * sp.WithinBinderContact()
        + 0.0
        * InverseFoldingSequenceRecovery(
            mpnn,
            temp=jax.device_put(0.0001),
            bias=50.0
            * jax.nn.one_hot(
                SetPositions.from_sequence(
                    wildtype=masked_framework_sequence, loss=None
                ).wildtype,
                20,
            ),
        )
        # + 0.1 * ProteinMPNNLoss(mpnn, num_samples=4)
    )

    model_loss = protenix.build_loss(
        loss=structure_loss,
        features=p_features,
        recycling_steps=1,
        # return_coords=False,
        # return_state=False,
    )
    # set initial recycling state from precycling
    # model_loss = eqx.tree_at(
    #     lambda l: l.initial_recycling_state,
    #     model_loss,
    #     padded_embedding,
    #     is_leaf=lambda x: x is None,
    # )

    # add a small trigram LL term (mostly to avoid homopolymer runs)
    trigram_ll = TrigramLL.from_pkl()


    loss = SetPositions.from_sequence(
        wildtype=masked_framework_sequence,
        loss=0.1 * ESMCPLL + 2 * model_loss + 0.1 * ablang_pll + trigram_ll,
    )
    return (loss,)


@app.cell
def _():
    from mosaic.losses.ablang import AbLangPseudoLikelihood, load_ablang
    from mosaic.losses.esmc import ESMCPseudoLikelihood, load_esmc
    from mosaic.losses.protein_mpnn import (
        InverseFoldingSequenceRecovery,
    )
    from mosaic.losses.transformations import SetPositions
    return (
        AbLangPseudoLikelihood,
        ESMCPseudoLikelihood,
        InverseFoldingSequenceRecovery,
        SetPositions,
        load_ablang,
        load_esmc,
    )


@app.cell
def _(load_abmpnn):
    mpnn = load_abmpnn(backbone_noise=0.05)
    return (mpnn,)


@app.cell
def _():
    from mosaic.optimizers import gradient_MCMC, simplex_APGM
    from mosaic.proteinmpnn.mpnn import load_abmpnn
    from mosaic.losses.trigram import TrigramLL
    return TrigramLL, gradient_MCMC, load_abmpnn


@app.cell
def _():
    import mosaic.losses.structure_prediction as sp
    return (sp,)


@app.cell
def _(gemmi):
    target_structure = gemmi.read_structure("1PEB.cif")
    return (target_structure,)


@app.cell
def _(TargetChain, gemmi, protenix, target_structure):
    target_features, _ = protenix.target_only_features(
            chains=[
                TargetChain(
                    sequence=gemmi.one_letter_code([r.name for r in target_structure[0][0]]),
                    use_msa=True,
                ),
            ]
        )
    return (target_features,)


@app.cell
def _(jax, np, protenix, target_features):
    model_output = protenix.model_output(
            features=target_features,
            recycling_steps=5,
            key=jax.random.key(np.random.randint(10000)),
        )
    return


@app.cell
def _():
    # _target_embedding = model_output.trunk_state
    # # add zeros for binder to target_embedding

    # M = len(target_structure[0][0])

    # padded_embedding = eqx.tree_at(
    #     lambda e: (e.s, e.z),
    #     _target_embedding,
    #     (
    #         jnp.zeros((N + M, 384)).at[:,N:].set(_target_embedding.s),
    #         jnp.zeros((N + M, N + M, 128)).at[:,N:, N:].set(_target_embedding.z),
    #     ),
    # )
    return


@app.cell
def _(framework_masked):
    masked_framework_sequence = framework_masked
    return (masked_framework_sequence,)


@app.cell
def _(masked_framework_sequence):
    N = len(masked_framework_sequence)
    return


@app.cell
def _(gradient_MCMC, jax, loss, partial_pssm):
    s_mcmc = gradient_MCMC(
                loss=loss,
                sequence=jax.device_put(partial_pssm.argmax(-1)),
                steps=60,
                fix_loss_key=False,
                proposal_temp=1e-5,
                max_path_length=1,
            )
    return


@app.cell
def _():
    12
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
