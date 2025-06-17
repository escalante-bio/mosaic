import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    return


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import gemmi
    import jax
    from boltz_binder_design.proteinmpnn.mpnn import ProteinMPNN
    return ProteinMPNN, gemmi, jax


@app.cell
def _():
    from boltz_binder_design.notebook_utils import pdb_viewer
    return (pdb_viewer,)


@app.cell
def _():
    from boltz_binder_design.losses.transformations import (
        FixedPositionsPenalty,
        SetPositions,
    )
    from boltz_binder_design.optimizers import gradient_MCMC
    from boltz_binder_design.common import TOKENS
    import jax.numpy as jnp
    return SetPositions, TOKENS, gradient_MCMC, jnp


@app.cell
def _():
    import boltz_binder_design.losses.structure_prediction as sp
    return (sp,)


@app.cell
def _():
    from boltz_binder_design.af2.alphafold2 import AF2
    from boltz_binder_design.losses.af2 import AlphaFoldLoss

    from boltz_binder_design.losses.protein_mpnn import (
        FixedStructureInverseFoldingLL,
    )
    return AF2, AlphaFoldLoss, FixedStructureInverseFoldingLL


@app.cell
def _(AF2):
    af2 = AF2()
    return (af2,)


@app.cell
def _(gemmi, pdb_viewer):
    wildtype_complex = gemmi.read_pdb("7opb.pdb")
    wildtype_complex.remove_ligands_and_waters()
    wildtype_complex.remove_empty_chains()
    pdb_viewer(gemmi.read_structure("7opb.pdb"))
    return (wildtype_complex,)


@app.cell
def _(gemmi, wildtype_complex):
    wt_binder_sequence = gemmi.one_letter_code(
        [r.name for r in wildtype_complex[0][1]]
    )
    return (wt_binder_sequence,)


@app.cell
def _(gemmi, wildtype_complex):
    target_sequence = gemmi.one_letter_code(
        [r.name for r in wildtype_complex[0][0]]
    )
    return (target_sequence,)


@app.cell
def _(af2, jax, target_sequence, wildtype_complex, wt_binder_sequence):
    # Make an AF2 prediction of the WT structure using templates and an initial guess -- this will be used in the loss function
    wt_pred_o, wt_pred = af2.predict(
        [wt_binder_sequence, target_sequence],
        template_chains={0: wildtype_complex[0][1], 1: wildtype_complex[0][0]},
        key=jax.random.key(0),
        model_idx=0,
        initial_guess=wildtype_complex,
    )
    return


@app.cell
def _(af2, target_sequence, wildtype_complex, wt_binder_sequence):
    # Make complex features for AF 0 - use a template for the target structrue
    ft_features, _ = af2.build_features(
        chains=[
            "G" * len(wt_binder_sequence),
            target_sequence,
        ],
        template_chains={1: wildtype_complex[0][0]},
    )
    return (ft_features,)


@app.cell
def _(FixedStructureInverseFoldingLL, ProteinMPNN, wildtype_complex):
    wt_inverse_folding_LL = FixedStructureInverseFoldingLL.from_structure(
        wildtype_complex,
        ProteinMPNN.from_pretrained(),
    )
    return (wt_inverse_folding_LL,)


@app.cell
def _(AlphaFoldLoss, af2, ft_features, jax, sp, wt_inverse_folding_LL):
    loss = (
        AlphaFoldLoss(
            name="af",
            forward=af2.alphafold_apply,
            stacked_params=jax.device_put(af2.stacked_model_params),
            features=ft_features,
            # initial_guess=st_wt,
            losses=0.01 * sp.PLDDTLoss()
            + 1 * sp.BinderTargetContact()
            + 0.1 * sp.TargetBinderPAE()
            + 0.1 * sp.BinderTargetPAE()
            + 0.0 * sp.IPTMLoss()
        )
        + 0.0 * wt_inverse_folding_LL
    )
    return (loss,)


@app.cell
def _():
    # Let's try redesigning the whole binder using MCMC
    return


@app.cell
def _(wt_binder_sequence):
    len(wt_binder_sequence)
    return


@app.cell
def _(TOKENS, gradient_MCMC, jax, jnp, loss, wt_binder_sequence):
    gradient_MCMC(
        loss=loss,
        sequence=jnp.array([TOKENS.index(AA) for AA in wt_binder_sequence]),
        steps=100,
        key=jax.random.key(1),
    )
    return


@app.cell
def _(wt_binder_sequence):
    wt_binder_sequence
    return


@app.cell
def _():
    interface_idx = [
        17,
        18,
        19,
        20,
        22,
        23,
        26,
        27,
        30,
        35,
        36,
        40,
        44,
        47,
        48,
        50,
        51,
        54,
    ]
    return (interface_idx,)


@app.cell
def _(SetPositions, TOKENS, interface_idx, jnp, loss, wt_binder_sequence):
    interface_only_loss = SetPositions(
        loss=loss,
        variable_positions=jnp.array(interface_idx),
        wildtype=[TOKENS.index(AA) for AA in wt_binder_sequence],
    )
    return (interface_only_loss,)


@app.cell
def _(redesigned_interface):
    redesigned_interface
    return


@app.cell
def _():
    return


@app.cell
def _(
    TOKENS,
    gradient_MCMC,
    interface_idx,
    interface_only_loss,
    jax,
    jnp,
    wt_binder_sequence,
):
    redesigned_interface = gradient_MCMC(
        loss=interface_only_loss,
        sequence=jnp.array(
            [TOKENS.index(wt_binder_sequence[i]) for i in interface_idx]
        ),
        steps=100,
        key=jax.random.key(1),
    )
    return (redesigned_interface,)


@app.cell
def _(TOKENS, interface_only_loss, jax, redesigned_interface):
    new_sequence = "".join(
        TOKENS[c]
        for c in interface_only_loss.sequence(
            jax.nn.one_hot(redesigned_interface, 20)
        ).argmax(-1)
    )
    return (new_sequence,)


@app.cell
def _(new_sequence):
    new_sequence
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
