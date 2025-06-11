import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _():
    import boltz_binder_design.losses.boltz2 as bl2
    from boltz_binder_design.optimizers import design_bregman_optax, simplex_APGM
    from boltz_binder_design.common import TOKENS
    import gemmi
    from boltz_binder_design.af2.alphafold2 import AF2
    import numpy as np
    import optax
    import jax
    import equinox as eqx
    from boltz_binder_design.notebook_utils import pdb_viewer
    return (
        AF2,
        TOKENS,
        bl2,
        design_bregman_optax,
        eqx,
        jax,
        np,
        optax,
        pdb_viewer,
        simplex_APGM,
    )


@app.cell
def _():

    target_sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRL" 
    return (target_sequence,)


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
          sequence: {target}
    """.format(seq=binder_sequence, target=target_sequence)
    return (make_yaml,)


@app.cell
def _():
    binder_length = 75
    return (binder_length,)


@app.cell
def _(binder_length, bl2, make_yaml):
    features, boltz_writer = bl2.load_features_and_structure_writer(
        make_yaml("X" * binder_length),
        templates={},
    )
    return boltz_writer, features


@app.cell
def _(bl2):
    boltz2 = bl2.load_boltz2()
    return (boltz2,)


@app.cell
def _(bl2, boltz2, features):
    loss = bl2.Boltz2Loss(
        joltz2=boltz2,
        features=features,
        loss=2 * bl2.BinderTargetContact() + bl2.WithinBinderContact(),
        deterministic=True,
        recycling_steps=0,
    )
    return (loss,)


@app.cell
def _(binder_length, jax, loss, np, simplex_APGM):
    _, PSSM = simplex_APGM(
        loss_function=loss,
        n_steps=50,
        x=jax.nn.softmax(
            0.5*jax.random.gumbel(
                key=jax.random.key(np.random.randint(100000)),
                shape=(binder_length, 20),
            )
        ),
        stepsize=0.1 * np.sqrt(binder_length),
        momentum=0.90,
    )

    return (PSSM,)


@app.cell
def _(PSSM, binder_length, design_bregman_optax, jax, loss, np, optax):
    logits = jax.numpy.log(PSSM + 1e-5)
    logits_sharper, _ = design_bregman_optax(
        loss_function=loss,
        n_steps=50,
        x=logits,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.025),
            optax.sgd(0.5 * np.sqrt(binder_length), momentum=0.0),
        ),
    )

    logits_sharper, _ = design_bregman_optax(
        loss_function=loss,
        n_steps=25,
        x=logits_sharper,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.1),
            optax.sgd(1.5 * np.sqrt(binder_length), momentum=0.0),
        ),
    )
    return (logits_sharper,)


@app.cell
def _(eqx):
    j_model = eqx.filter_jit(lambda model, *args, **kwargs: model(*args, **kwargs))
    return (j_model,)


@app.cell
def _(bl2, boltz2, j_model, jax, pdb_viewer):
    def predict(sequence, features, writer):
        o = j_model(
            boltz2,
            bl2.set_binder_sequence(sequence, features),
            key=jax.random.key(5),
            deterministic=True,
            num_sampling_steps=25,
        )
        st = writer(o[1])
        viewer = pdb_viewer(st)
        # print("plddt", o["plddt"][: sequence.shape[0]].mean())
        # print("ipae", o["complex_ipae"].item())
        return o, st, viewer
    return (predict,)


@app.cell
def _(boltz_writer, features, jax, logits_sharper, predict):
    soft_output, soft_pred_st, _viewer = predict(
        jax.nn.softmax(1000*logits_sharper), features, boltz_writer
    )
    _viewer
    return


@app.cell
def _(binder_seq, bl2, jax, logits_sharper, make_yaml, predict):
    hard_output, pred_st, _viewer = predict(
        jax.nn.softmax(1000*logits_sharper), *bl2.load_features_and_structure_writer(input_yaml_str=make_yaml(binder_seq))
    )
    _viewer
    return hard_output, pred_st


@app.cell
def _(mo, pred_st):
    mo.download(data=pred_st.make_pdb_string(), filename="a.pdb", label = "Boltz-2 predicted complex")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(hard_output, plt):
    _f = plt.figure()
    plt.imshow(hard_output[2].pae[0])
    plt.colorbar()
    _f
    return


@app.cell
def _(hard_output, plt):
    plt.plot(hard_output[2].plddt[0])
    return


@app.cell
def _(AF2):
    af = AF2(num_recycle=2)
    return (af,)


@app.cell
def _(TOKENS, logits_sharper):
    binder_seq = "".join(TOKENS[i] for i in logits_sharper.argmax(-1))
    binder_seq
    return (binder_seq,)


@app.cell
def _(mo):
    mo.md("""Make a template structure of the target alone we can use with AF2 multimer""")
    return


@app.cell
def _(bl2, target_sequence):
    template_features, template_writer = bl2.load_features_and_structure_writer(
    """version: 1
    sequences:
      - protein:
          id: [A]
          sequence: {seq}
    """.format(seq = target_sequence)
    )
    return template_features, template_writer


@app.cell
def _(
    TOKENS,
    jax,
    predict,
    target_sequence,
    template_features,
    template_writer,
):
    _, template_st, template_viewer = predict(
        jax.nn.one_hot([TOKENS.index(c) for c in target_sequence], 20),
        template_features,
        template_writer,
    )
    template_viewer
    return (template_st,)


@app.cell
def _(af, binder_seq, jax, plt, target_sequence, template_st):
    iptms = [
        af.predict(
            chains=[binder_seq, target_sequence],
            key=jax.random.key(2),
            template_chains={1: template_st[0][0]},
            model_idx=idx,
        )[0].iptm
        for idx in range(5)
    ]

    plt.plot(iptms)
    plt.xlabel("AF2 model idx")
    plt.ylabel("IPTM")
    return


@app.cell
def _(af, binder_seq, jax, target_sequence, template_st):
    af_o, af_st = af.predict(
        chains=[binder_seq, target_sequence],
        key=jax.random.key(2),
        template_chains={1: template_st[0][0]},
        model_idx=0,
    )
    return af_o, af_st


@app.cell
def _(af_o):
    af_o.iptm
    return


@app.cell
def _(af_st, pdb_viewer):
    pdb_viewer(af_st)
    return


@app.cell
def _(af_o, plt):
    _f = plt.figure()
    plt.imshow(af_o.predicted_aligned_error)
    plt.colorbar()
    plt.title("AF2 PAE")
    _f
    return


@app.cell
def _(jax, logits_sharper, plt):
    plt.imshow(jax.nn.softmax(logits_sharper))
    return


@app.cell(hide_code=True)
def _(PSSM, plt):
    plt.imshow(PSSM)
    return


if __name__ == "__main__":
    app.run()
