import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

with app.setup:
    import jax

    import matplotlib.pyplot as plt
    import boltz_binder_design.losses.boltz2 as bl2
    from boltz_binder_design.optimizers import design_bregman_optax, simplex_APGM
    from boltz_binder_design.common import TOKENS
    from boltz_binder_design.af2.alphafold2 import AF2
    import numpy as np
    import optax

    import equinox as eqx
    from boltz_binder_design.notebook_utils import pdb_viewer
    import boltz_binder_design.losses.structure_prediction as sp

    boltz2 = bl2.load_boltz2()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    ---
    **Warning**

    1. You'll almost certainly need a GPU or TPU to run this
    2. Because JAX uses JIT compilation the first execution of a cell may take quite a while
    3. You might have to run these optimization methods multiple times before you get a reasonable binder
    4. If you change targets you'll likely have to fiddle with hyperparameters!
    5. This is pretty experimental, I highly recommend you stick with BindCraft if you're designing a minibinder against a protein target
    ---
    """
    )
    return


@app.cell
def _():
    target_sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLR" 
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
          sequence: {t}
    """.format(seq=binder_sequence,t = target_sequence)
    return (make_yaml,)


@app.cell
def _():
    binder_length = 75
    return (binder_length,)


@app.cell
def _(binder_length, make_yaml):
    features, boltz_writer = bl2.load_features_and_structure_writer(
        make_yaml("X" * binder_length),
        templates={},
    )
    return boltz_writer, features


@app.cell
def _(ProteinMPNN, ProteinMPNNLoss, features):
    loss = bl2.Boltz2Loss(
        joltz2=boltz2,
        features=features,
        loss=2 * sp.BinderTargetContact() + sp.WithinBinderContact()  + 0.25*ProteinMPNNLoss(mpnn=ProteinMPNN.from_pretrained(), num_samples=8, stop_grad=True),
        deterministic=True,
        recycling_steps=0,
    )
    return (loss,)


@app.cell
def _(mo):
    mo.md("""Adding the ProteinMPNN log likelihood term to the loss above tends to generate sequences that AF2-multimer also likes, but is slower because we have to run the Boltz-2 structure module. Try removing it for faster generation!""")
    return


@app.cell
def _(binder_length, loss):
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
        momentum=0.9,
    )

    return (PSSM,)


@app.cell
def _(PSSM, binder_length, loss):
    logits = jax.numpy.log(PSSM + 1e-5)
    logits_sharper, _ = design_bregman_optax(
        loss_function=loss,
        n_steps=25,
        x=logits,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.025),
            optax.sgd(0.5 * np.sqrt(binder_length), momentum=0.0),
        ),
    )

    logits_sharper, _ = design_bregman_optax(
        loss_function=loss,
        n_steps=15,
        x=logits_sharper,
        optim=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.add_decayed_weights(-0.1),
            optax.sgd(1.5 * np.sqrt(binder_length), momentum=0.0),
        ),
    )
    return (logits_sharper,)


@app.cell
def _():
    j_model = eqx.filter_jit(lambda model, *args, **kwargs: model(*args, **kwargs))
    return (j_model,)


@app.cell
def _(j_model):
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
def _(boltz_writer, features, logits_sharper, predict):
    soft_output, soft_pred_st, _viewer = predict(
        jax.nn.softmax(1000*logits_sharper), features, boltz_writer
    )
    _viewer
    return soft_output, soft_pred_st


@app.cell
def _(soft_output):
    _f = plt.figure()
    plt.imshow(soft_output[2].pae[0])
    plt.colorbar()
    _f
    return


@app.cell
def _(binder_seq, logits_sharper, make_yaml, predict):
    hard_output, pred_st, _viewer = predict(
        jax.nn.softmax(logits_sharper*1000), *bl2.load_features_and_structure_writer(input_yaml_str=make_yaml(binder_seq))
    )
    _viewer
    return hard_output, pred_st


@app.cell
def _(mo, pred_st):
    mo.download(data=pred_st.make_pdb_string(), filename="a.pdb", label = "Boltz-2 predicted complex")
    return


@app.cell
def _(hard_output):
    _f = plt.figure()
    plt.imshow(hard_output[2].pae[0])
    plt.colorbar()
    _f
    return


@app.cell
def _(hard_output, soft_output):
    plt.plot(hard_output[2].plddt[0])
    plt.plot(soft_output[2].plddt[0])
    return


@app.cell
def _():
    af = AF2(num_recycle=4)
    return (af,)


@app.cell
def _(logits_sharper):
    binder_seq = "".join(TOKENS[i] for i in logits_sharper.argmax(-1))
    binder_seq
    return (binder_seq,)


@app.cell
def _(mo):
    mo.md("""Make a template structure of the target alone we can use with AF2 multimer""")
    return


@app.cell
def _(target_sequence):
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
def _(predict, target_sequence, template_features, template_writer):
    _, template_st, template_viewer = predict(
        jax.nn.one_hot([TOKENS.index(c) for c in target_sequence], 20),
        template_features,
        template_writer,
    )
    template_viewer
    return (template_st,)


@app.cell
def _(af, binder_seq, target_sequence, template_st):
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
def _(af, binder_seq, target_sequence, template_st):
    af_o, af_st = af.predict(
        chains=[binder_seq, target_sequence],
        key=jax.random.key(2),
        template_chains={1: template_st[0][0]},
        model_idx=0,
    )
    return af_o, af_st


@app.cell
def _(af_st):
    pdb_viewer(af_st)
    return


@app.cell
def _(af_o):
    _f = plt.figure()
    plt.imshow(af_o.predicted_aligned_error)
    plt.colorbar()
    plt.title("AF2 PAE")
    _f
    return


@app.cell
def _(logits_sharper):
    plt.imshow(jax.nn.softmax(logits_sharper))
    return


@app.cell(hide_code=True)
def _(PSSM):
    plt.imshow(PSSM)
    return


@app.cell
def _():
    from boltz_binder_design.proteinmpnn.mpnn import ProteinMPNN
    from boltz_binder_design.losses.protein_mpnn import FixedStructureInverseFoldingLL, ProteinMPNNLoss
    return FixedStructureInverseFoldingLL, ProteinMPNN, ProteinMPNNLoss


@app.cell
def _(mo):
    mo.md("""Let's do it live! We can inverse fold the predicted complex using MPNN and the jacobi iteration in a few lines of code.""")
    return


@app.cell
def _():
    from boltz_binder_design.common import LossTerm
    return (LossTerm,)


@app.cell
def _(LossTerm):
    class GumbelPerturbation(LossTerm):
        key: any

        def __call__(self, sequence, key):
            v = (jax.random.gumbel(self.key, sequence.shape)*sequence).sum()
            return v, {"gumbel": v}
    return (GumbelPerturbation,)


@app.cell
def _(FixedStructureInverseFoldingLL, ProteinMPNN, soft_pred_st):
    if_ll = FixedStructureInverseFoldingLL.from_structure(
            soft_pred_st,
            ProteinMPNN.from_pretrained(),
            stop_grad=True
        )
    return (if_ll,)


@app.cell
def _():
    from boltz_binder_design.optimizers import _eval_loss_and_grad

    def jacobi(loss, iters, sequence, key):
        for _ in range(iters):
            (v, aux), g = _eval_loss_and_grad(loss, jax.nn.one_hot(sequence, 20), key = key)
            sequence = g.argmin(-1)
            print(v)

        return sequence
    return (jacobi,)


@app.cell
def _(GumbelPerturbation, binder_length, if_ll, jacobi):
    seq_mpnn = jacobi(
        if_ll + 0.0005 * GumbelPerturbation(jax.random.key(np.random.randint(1000000))),
        10,
        np.random.randint(low=0, high=20, size=(binder_length)),
        key=jax.random.key(np.random.randint(1000000)),
    )
    return (seq_mpnn,)


@app.cell
def _(af, seq_mpnn, target_sequence, template_st):
    [af.predict(
            chains=["".join(TOKENS[i] for i in seq_mpnn), target_sequence],
            key=jax.random.key(2),
            template_chains={1: template_st[0][0]},
            model_idx=idx,
        )[0].iptm
        for idx in range(5)
    ]
    return


@app.cell
def _(af, seq_mpnn, target_sequence, template_st):
    _, _af_st = af.predict(
        chains=["".join(TOKENS[i] for i in seq_mpnn), target_sequence],
        key=jax.random.key(2),
        template_chains={1: template_st[0][0]},
        model_idx=1,
    )
    pdb_viewer(_af_st)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
