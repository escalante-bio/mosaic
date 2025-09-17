import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import marimo as mo

    import matplotlib.pyplot as plt
    from mosaic.optimizers import simplex_APGM
    from mosaic.common import TOKENS
    from mosaic.af2.alphafold2 import AF2
    import numpy as np

    import equinox as eqx
    from mosaic.notebook_utils import pdb_viewer
    import mosaic.losses.structure_prediction as sp

    from mosaic.models.boltz2 import Boltz2
    from mosaic.structure_prediction import TargetChain


@app.cell
def _():
    from mosaic.models.af2 import AlphaFold2
    return (AlphaFold2,)


@app.cell
def _(AlphaFold2):
    model_af = AlphaFold2()
    return (model_af,)


@app.cell
def _():
    return


@app.cell
def _():
    model = Boltz2()
    return (model,)


@app.cell
def _():
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
    target_sequence = "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA" 
    return (target_sequence,)


@app.cell
def _(binder_length, model_af, target_sequence, template_st):
    af_f, _ = model_af.binder_features(binder_length=binder_length, chains = [TargetChain(target_sequence, use_msa=False, template_chain=template_st[0][0])])
    return (af_f,)


@app.cell
def _(PSSM_sharper, af_f, model_af):
    af_pred = model_af.predict(features=af_f, PSSM = PSSM_sharper, writer = None, key = jax.random.key(0))
    return (af_pred,)


@app.cell
def _(af_pred):
    pdb_viewer(af_pred)
    return


@app.cell
def _(binder_length, model, target_sequence):
    features, structure_writer = model.binder_features(binder_length=binder_length, chains = [TargetChain(target_sequence)])
    return features, structure_writer


@app.cell
def _():
    binder_length = 75
    return (binder_length,)


@app.cell
def _(ProteinMPNN):
    mpnn = ProteinMPNN.from_pretrained()
    return (mpnn,)


@app.cell
def _(InverseFoldingSequenceRecovery, features, model, mpnn):
    loss = model.build_loss(
        loss=2 * sp.BinderTargetContact()
        + sp.WithinBinderContact()
        + 5.0 * InverseFoldingSequenceRecovery(mpnn, temp=jax.numpy.array(0.05)),
        features=features,
    )
    return (loss,)


@app.cell
def _():
    mo.md("""Adding the ProteinMPNN log likelihood term to the loss above tends to generate sequences that AF2-multimer also likes, but is slower because we have to run the Boltz-2 structure module. Try removing it for faster generation!""")
    return


@app.cell
def _(binder_length, loss):
    _, PSSM = simplex_APGM(
        loss_function=loss,
        n_steps=75,
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

    PSSM_sharper, _ = simplex_APGM(
        loss_function=loss,
        n_steps=50,
        x=PSSM,
        stepsize = 0.5 * np.sqrt(binder_length),
        scale = 1.5,
        momentum=0.0
    )
    return (PSSM_sharper,)


@app.cell
def _():
    j_model = eqx.filter_jit(lambda model, *args, **kwargs: model(*args, **kwargs))
    return


@app.cell
def _(model):
    def predict(sequence, features, writer):
        return model.predict(PSSM = sequence, features = features, writer = writer, key = jax.random.key(0))
    return (predict,)


@app.cell
def _(PSSM_sharper, features, predict, structure_writer):
    soft_pred_st = predict(
        PSSM_sharper, features, structure_writer
    )
    pdb_viewer(soft_pred_st)
    return (soft_pred_st,)


@app.cell
def _(soft_output):
    _f = plt.figure()
    plt.imshow(soft_output[2].pae[0])
    plt.colorbar()
    _f
    return


@app.cell
def _(PSSM_sharper, features, predict, structure_writer):
    pred_st = predict(
        PSSM_sharper, features, structure_writer
    )
    pdb_viewer(pred_st)
    return (pred_st,)


@app.cell
def _(pred_st):
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
    af = AF2()
    return (af,)


@app.cell
def _(PSSM_sharper):
    binder_seq = "".join(TOKENS[i] for i in PSSM_sharper.argmax(-1))
    binder_seq
    return (binder_seq,)


@app.cell
def _():
    mo.md("""Make a template structure of the target alone we can use with AF2 multimer""")
    return


@app.cell
def _(model, target_sequence):
    template_features, template_writer = model.target_only_features(chains=[TargetChain(sequence=target_sequence)])
    return template_features, template_writer


@app.cell
def _(predict, target_sequence, template_features, template_writer):
    template_st = predict(
        jax.nn.one_hot([TOKENS.index(c) for c in target_sequence], 20),
        template_features,
        template_writer,
    )
    pdb_viewer(template_st)
    return (template_st,)


@app.cell
def _(template_st):
    template_st
    return


@app.cell
def _():
    12
    return


@app.cell
def _(af, binder_seq, target_sequence, template_st):
    iptms = [
        af.predict(
            chains=[binder_seq, target_sequence],
            key=jax.random.key(2),
            template_chains={1: template_st[0][0]},
            model_idx=idx,
            recycling_steps=3
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
def _(PSSM_sharper):
    plt.imshow(PSSM_sharper)
    return


@app.cell(hide_code=True)
def _(PSSM):
    plt.imshow(PSSM)
    return


@app.cell
def _():
    from mosaic.proteinmpnn.mpnn import ProteinMPNN
    from mosaic.losses.protein_mpnn import FixedStructureInverseFoldingLL, InverseFoldingSequenceRecovery
    return (
        FixedStructureInverseFoldingLL,
        InverseFoldingSequenceRecovery,
        ProteinMPNN,
    )


@app.cell
def _():
    mo.md("""Let's do it live! We can inverse fold the predicted complex using MPNN and the jacobi iteration in a few lines of code.""")
    return


@app.cell
def _():
    from mosaic.common import LossTerm
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
def _(FixedStructureInverseFoldingLL, mpnn, soft_pred_st):
    if_ll = FixedStructureInverseFoldingLL.from_structure(
            soft_pred_st,
            mpnn,
            stop_grad=True
        )
    return (if_ll,)


@app.cell
def _():
    from mosaic.optimizers import _eval_loss_and_grad

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
    mo.md("""For fun let's design 10 complexes""")
    return


@app.cell
def _(binder_length, boltz_writer, features, loss, predict):
    def design():
        _, PSSM = simplex_APGM(
            loss_function=loss,
            n_steps=75,
            x=jax.nn.softmax(
                0.5*jax.random.gumbel(
                    key=jax.random.key(np.random.randint(100000)),
                    shape=(binder_length, 20),
                )
            ),
            stepsize=0.1 * np.sqrt(binder_length),
            momentum=0.9,
        )
        _, soft_pred_st, _ = predict(
            PSSM, features, boltz_writer
        )
        return soft_pred_st
    return


@app.cell
def _():
    # designs = [design() for _ in mo.status.progress_bar(range(1))]
    return


@app.cell
def _():
    from mosaic.notebook_utils import gemmi_structure_from_models
    return (gemmi_structure_from_models,)


@app.cell
def _(designs, gemmi_structure_from_models):
    complexes = gemmi_structure_from_models("designs", [st[0] for st in designs])
    return (complexes,)


@app.cell
def _(complexes):
    pdb_viewer(complexes)
    return


if __name__ == "__main__":
    app.run()
