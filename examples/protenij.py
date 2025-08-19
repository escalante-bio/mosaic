import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell
def _():
    from boltz_binder_design.af2.alphafold2 import AF2
    from boltz_binder_design.losses.af2 import AlphaFoldLoss
    import boltz_binder_design.losses.af2 as aflosses
    return AF2, AlphaFoldLoss


@app.cell
def _():
    from boltz_binder_design.common import LossTerm, LinearCombination
    return


@app.cell
def _():
    import boltz_binder_design.losses.structure_prediction as sp
    return (sp,)


@app.cell
def _():
    from protenix.protenij import Protenix as Protenij
    from protenix.protenij import InitialEmbedding, TrunkEmbedding, ConfidenceMetrics
    return (TrunkEmbedding,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _():
    import jax
    return (jax,)


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _():
    from boltz_binder_design.notebook_utils import pdb_viewer
    return (pdb_viewer,)


@app.cell
def _():
    from boltz_binder_design.losses.protein_mpnn import (
        InverseFoldingSequenceRecovery,
    )
    from boltz_binder_design.proteinmpnn.mpnn import ProteinMPNN
    import importlib
    import boltz_binder_design
    from boltz_binder_design.af2.confidence_metrics import predicted_tm_score
    import equinox as eqx
    from boltz_binder_design.losses.protenix import (
        load_protenix_mini,
        load_features_from_json,
        biotite_array_to_gemmi_struct,
        boltz_to_protenix_matrix,
        ProtenixLoss,
        set_binder_sequence,
    )

    from jaxtyping import Float, Array,PyTree
    import jax.numpy as jnp
    from boltz_binder_design.common import TOKENS
    return (
        InverseFoldingSequenceRecovery,
        ProteinMPNN,
        ProtenixLoss,
        TOKENS,
        biotite_array_to_gemmi_struct,
        boltz_binder_design,
        eqx,
        importlib,
        jnp,
        load_features_from_json,
        load_protenix_mini,
        set_binder_sequence,
    )


@app.cell
def _():
    binder_length = 100
    return (binder_length,)


@app.cell
def _():
    target_sequence = "DYSFSCYSQLEVNGSQHSLTCAFEDPDVNTTNLEFEICGALVEVKCLNFRKLQEIYFIETKKFLLIGKSNICVKVGEKSLTCKKIDLTTIVKPEAPFDLSVVYREGANDFVVTFNTSHLQKKYVKVLMHDVAYRQEKDENKWTHVNLSSTKLTLLQRKLQPAAMYEIKVRSIPDHYFKGFWSEWSPSYYFRT"
    target_name = "il7ra"
    return target_name, target_sequence


@app.cell
def _(binder_length, target_name, target_sequence):
    design_json = {
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "X" * binder_length,
                    "count": 1
                }
            },
            {
            "proteinChain": {
                    "sequence": target_sequence,
                    "count": 1
                }
            }
        ],
        "name": target_name 
    }
    return (design_json,)


@app.cell
def _(target_name, target_sequence):
    target_json = {
        "sequences": [
            {
            "proteinChain": {
                    "sequence": target_sequence,
                    "count": 1
                }
            }
        ],
        "name": target_name 
    }
    return (target_json,)


@app.cell
def _(load_features_from_json, target_json):
    target_only_features, target_only_structure = load_features_from_json(target_json)
    return target_only_features, target_only_structure


@app.cell
def _(design_json, load_features_from_json):
    design_features, design_structure = load_features_from_json(design_json)
    return design_features, design_structure


@app.cell
def _(ProteinMPNN, boltz_binder_design, importlib):
    mpnn = ProteinMPNN.from_pretrained(
            importlib.resources.files(boltz_binder_design)
            / "proteinmpnn/weights/soluble_v_48_020.pt"
        )
    return (mpnn,)


@app.cell
def _(
    biotite_array_to_gemmi_struct,
    jax,
    np,
    target_only_features,
    target_only_structure,
    te_target,
    unjitted_model,
):
    def _():
        feats = target_only_features
        initial_embedding = unjitted_model.embed_inputs(input_feature_dict = feats)
        coordinates = unjitted_model.sample_structures(initial_embedding=initial_embedding,
                trunk_embedding=te_target,
                input_feature_dict=feats,
                N_samples=1,
                N_steps=3,
                key=jax.random.key(0))
        return biotite_array_to_gemmi_struct(target_only_structure, np.array(coordinates[0]))


    st_target_only = _()

    return (st_target_only,)


@app.cell
def _(pdb_viewer, st_target_only):
    pdb_viewer(st_target_only)
    return


@app.cell
def _(jax, target_only_features, unjitted_model):

    def _():
        feats = target_only_features
        initial_embedding = unjitted_model.embed_inputs(input_feature_dict = feats)
        return unjitted_model.recycle(initial_embedding=initial_embedding, input_feature_dict = feats, recycling_steps = 10, key = jax.random.key(0))


    te_target = _()
    return (te_target,)


@app.cell
def _(eqx, load_protenix_mini):
    unjitted_model = load_protenix_mini()
    jax_model = eqx.filter_jit(unjitted_model)
    return jax_model, unjitted_model


@app.cell
def _(InverseFoldingSequenceRecovery, binder_length, jax, mpnn, sp):
    structure_loss = (
        sp.BinderTargetContact()
        + sp.WithinBinderContact()
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.05 * sp.IPTMLoss()
        + 0.4 * sp.WithinBinderPAE()
        + 0.00 * sp.pTMEnergy()
        + 0.1 * sp.PLDDTLoss()
        + 5.0 * InverseFoldingSequenceRecovery(mpnn, temp=jax.numpy.array(0.001))
        + 0.05*sp.ActualRadiusOfGyration(target_radius = 2.38 * binder_length**0.365)
        -0.5*sp.HelixLoss()
    )
    return (structure_loss,)


@app.cell
def _(bl2, boltz2, boltz_features, structure_loss):
    boltz_loss = bl2.Boltz2Loss(
            joltz2=boltz2,
            features=boltz_features,
            loss=structure_loss,
            deterministic=True,
            recycling_steps=0,
        )
    return (boltz_loss,)


@app.cell
def _():
    import boltz_binder_design.losses.boltz2 as bl2
    return (bl2,)


@app.cell
def _(bl2):
    boltz2 = bl2.load_boltz2()
    return (boltz2,)


@app.cell
def _(binder_length, bl2, make_yaml):
    boltz_features, boltz_writer = bl2.load_features_and_structure_writer(
            make_yaml("X" * binder_length),
        )
    return boltz_features, boltz_writer


@app.cell
def _(target_sequence):
    def make_yaml(binder_sequence: str):
        return """
    version: 1
    sequences:
      - protein:
          id: [A]
          sequence: {seq}
          msa: empty
      - protein:
          id: [B]
          sequence: {t}
    """.format(seq=binder_sequence,t = target_sequence)
    return (make_yaml,)


@app.cell
def _():
    from boltz_binder_design.optimizers import simplex_APGM
    return (simplex_APGM,)


@app.cell
def _():
    import optax
    return


@app.cell
def _(eqx):
    j_model = eqx.filter_jit(lambda model, *args, **kwargs: model(*args, **kwargs))

    return (j_model,)


@app.cell
def _():
    import boltz_binder_design.losses.boltz2 as bl
    return (bl,)


@app.cell
def _(bl, boltz2, j_model, jax, pdb_viewer):
    def predict(sequence, features, writer):
        o = j_model(
            boltz2,
            bl.set_binder_sequence(sequence, features),
            key=jax.random.key(5),
            num_sampling_steps = 25,
            deterministic =True
        )
        print(o[1])
        out_st = writer(o[1])
        viewer = pdb_viewer(out_st)
        print("plddt", o[2].plddt[: sequence.shape[0]].mean())
        # print("ipae", o[2].complex_ipae.item())
        return o, viewer
    return (predict,)


@app.cell
def _():
    import gemmi
    return


@app.cell
def _(PSSM_sharper, boltz_features, boltz_writer, predict):
    predict(PSSM_sharper, boltz_features, boltz_writer)
    return


@app.cell(hide_code=True)
def _(PSSM_sharper, plt):
    plt.imshow(PSSM_sharper)
    return


@app.cell
def _(PSSM_sharper, design_features, jax, jax_model, set_binder_sequence):
    out_jax = jax_model(
            input_feature_dict = set_binder_sequence(PSSM_sharper,jax.device_put(design_features)),
            N_cycle=3,
            N_sample=5,
            key=jax.random.key(2),
            N_steps=5

        ) 
    return (out_jax,)


@app.cell
def _(jax, jnp, out_jax, plt):
    plt.plot((jax.nn.softmax(out_jax.confidence_metrics.plddt_logits) * jnp.arange(50)/50).sum(-1).T)
    return


@app.cell
def _(boltz_loss, eqx):
    ebl = eqx.filter_jit(boltz_loss)
    return (ebl,)


@app.cell
def _(PSSM_sharper, ebl, jax):
    ebl(PSSM_sharper, key = jax.random.key(2))
    return


@app.cell
def _(PSSM_sharper, TOKENS):
    "".join(TOKENS[c] for c in PSSM_sharper.argmax(-1))
    return


@app.cell
def _(PSSM_sharper, eqx, jax, loss):
    eqx.filter_jit(loss)(PSSM_sharper, key = jax.random.key(0))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(biotite_array_to_gemmi_struct, design_structure, np, out_jax):
    st_pred = biotite_array_to_gemmi_struct(design_structure, pred_coord = np.array(out_jax.coordinates[3]))
    return (st_pred,)


@app.cell(hide_code=True)
def _(pdb_viewer, st_pred):
    pdb_viewer(st_pred)
    return


@app.cell
def _(PSSM_sharper, TOKENS, af2, jax, st_target_only, target_sequence):
    o_pred, st_pred_af = max([af2.predict(
        [
            "".join([TOKENS[i] for i in PSSM_sharper.argmax(-1)]),
            target_sequence,
        ],
        template_chains={1: st_target_only[0][0]},
        key=jax.random.key(1),
        model_idx=idx,
    ) for idx in range(1)], key = lambda T: T[0].iptm)
    return o_pred, st_pred_af


@app.cell
def _(o_pred):
    o_pred.iptm
    return


@app.cell
def _(pdb_viewer, st_pred_af):
    pdb_viewer(st_pred_af)
    return


@app.cell
def _():
    import esm as esm_m
    return


@app.cell
def _():
    from esm.pretrained import load_local_model
    return


@app.cell
def _():
    import esmj
    from esm.models.esmc import ESMC as TORCH_ESMC
    from boltz_binder_design.losses.esmc import ESMCPseudoLikelihood

    esm = esmj.from_torch(TORCH_ESMC.from_pretrained("esmc_300m").to("cpu"))
    ESMCPLL = ESMCPseudoLikelihood(esm)
    return


@app.cell
def _():
    from boltz_binder_design.optimizers import projection_simplex

    return


@app.cell
def _(af2, binder_length, st_target_only, target_sequence):
    af_features, initial_guess = af2.build_features(
        chains=["G" * binder_length, target_sequence],
        template_chains={1: st_target_only[0][0]},
    )
    return (af_features,)


@app.cell
def _(AlphaFoldLoss, af2, af_features, jax, structure_loss):
    loss_af =  AlphaFoldLoss(
                name="af",
                forward=af2.alphafold_apply,
                stacked_params=jax.device_put(af2.stacked_model_params),
                features=af_features,
                losses=structure_loss,
            ) #+ 0.05*ESMCPLL
    return


@app.cell
def _(TrunkEmbedding, binder_length, eqx, jnp, target_sequence, te_target):
    N = len(target_sequence) + binder_length

    _te = TrunkEmbedding(s=jnp.zeros((N, 384)), z=jnp.zeros((N, N, 128)))
    te = eqx.tree_at(
        lambda s: (s.s, s.z),
        _te,
        (
            _te.s.at[binder_length:].set(te_target.s),
            _te.z.at[binder_length:, binder_length:].set(te_target.z),
        ),
    )
    return (te,)


@app.cell
def _(ProtenixLoss, design_features, structure_loss, te, unjitted_model):
    loss = ProtenixLoss(
        unjitted_model,
        design_features,
        structure_loss,
        recycling_steps=1,
        sampling_steps=2,
        n_structures=1,
        initial_recycling_state=te,  
        return_coords = True
    )
    return (loss,)


@app.cell
def _(binder_length, boltz_binder_design, jax, loss, np):
    x = jax.nn.softmax(
                    0.50
                    * jax.random.gumbel(
                        key=jax.random.key(np.random.randint(100000)),
                        shape=(binder_length, 20),
                    )
                )

    (_, aux), _ =boltz_binder_design.optimizers._eval_loss_and_grad(
        x=x, loss_function=loss, key=jax.random.key(0)
    )
    return


@app.cell
def _(AF2):
    af2 = AF2(num_recycle=3)
    return (af2,)


@app.cell
def _(binder_length, jax, loss, np, simplex_APGM):

    PSSM = jax.nn.softmax(
                    0.5
                    * jax.random.gumbel(
                        key=jax.random.key(np.random.randint(100000)),
                        shape=(binder_length, 20),
                    )
                )

    for _outer in range(20):
        print(_outer)
        _,PSSM = simplex_APGM(
                loss_function=loss,
                x=PSSM,
                n_steps=2,
                stepsize=0.15 * np.sqrt(binder_length),
                momentum=0.5,
                scale=1.0,
                update_loss_state=True
            )

    return (PSSM,)


@app.cell
def _(PSSM, binder_length, loss, np, simplex_APGM):
    PSSM_sharper = PSSM
    for _ in range(5*2):
        _,PSSM_sharper = simplex_APGM(
                loss_function=loss,#loss,
                x=PSSM_sharper,
                n_steps=2,
                stepsize=0.1 * np.sqrt(binder_length),
                momentum=0.0,
                scale = 1.5,
                update_loss_state=True,
                logspace=False
            )

    return (PSSM_sharper,)


@app.cell
def _(o_pred):
    o_pred.iptm
    return


@app.cell
def _(PSSM_sharper, plt):
    plt.imshow(PSSM_sharper)
    return


@app.cell
def _(PSSM, plt):
    plt.imshow(PSSM)
    return


@app.cell
def _(mo, st_pred):
    mo.download(data = st_pred.make_minimal_pdb(), filename = "tnf.pdb")
    return


@app.cell
def _(mo, st_pred_af):
    mo.download(data = st_pred_af.make_minimal_pdb(), filename = "af_tnf.pdb")
    return


if __name__ == "__main__":
    app.run()
