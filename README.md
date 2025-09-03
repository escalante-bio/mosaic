

## Functional, multi-objective protein design using continuous relaxation.


 Protein design tasks almost always involve multiple constraints or properies that must be satisfied or optimized. For instance, in binder design one may want to simultaneously ensure:
- the chance of binding the intended target is high  
- the chance of binding to a similar off-target protein is low
- the binder expresses well in bacteria
- the binder is highly soluble. 

There has been a recent explosion in the application of machine learning to protein property prediction, resulting in fairly accurate predictors for each of these properties. What is currently lacking is an efficient and flexible method for combining these different predictors into one design framework. 

---

### Installation
We recommend using `uv`, e.g. run `uv sync --group jax-cuda` after cloning the repo to install dependencies.

To run the example notebook try `source .venv/bin/activate`, `marimo edit examples/example_notebook.py`.

> You may need to add various `uv` overrides for specific packages and your machine, take a look at [pyproject.toml](pyproject.toml)

> You'll need a GPU or TPU-compatible version of JAX for structure prediction. You might need to install this manually, i.e. ` uv add jax[cuda12].`

To automatically download the AF2 weights you'll need to install `aria2`: `apt-get install aria2`.

### Introduction

This project combines two simple components to make a powerful protein design framework:

- Gradient-based optimization over a continuous, relaxed sequence space (as in [ColabDesign](https://github.com/sokrypton/ColabDesign), RSO, BindCraft, etc)
- A functional, modular interface to easily combine multiple learned or hand-crafted loss terms and optimization algorithms (as in [A high-level programming language for generative protein design](https://www.biorxiv.org/content/10.1101/2022.12.21.521526v1.full.pdf) etc)

The key observation is that it's possible to use this continuous relaxation simultaneously with multiple learned objective terms [^1]. 

This allows us to easily construct objective functions that are combinations of multiple learned potentials and optimize them efficiently, like so:

```python
combined_loss = (
    Boltz1Loss(
        model=model,
        name="ART2B",
        loss=4 * sp.BinderTargetContact()
        + sp.RadiusOfGyration(target_radius=15.0)
        + sp.WithinBinderContact()
        + 0.3 * sp.HelixLoss()
        + ProteinMPNNLoss(mpnn, num_samples = 8),
        features=boltz_features,
        recycling_steps=0,
    )
    + 0.5 * esm_loss
    + trigram_ll
    + 0.1 * StabilityModel.from_pretrained(esm)
    + 0.5
    * Boltz1Loss(
        model=model,
        name="mono",
        loss=0.2 * sp.PLDDTLoss()
        + sp.RadiusOfGyration(target_radius=15.0)
        + 0.3 * sp.HelixLoss(),
        features=monomer_features,
        recycling_steps=0,
    )
)

_, logits_combined_objective = design_bregman_optax(
    loss_function=combined_loss,
    n_steps=150,
    x=np.random.randn(binder_length, 20) * 0.1,
    optim=optax.chain(
        optax.clip_by_global_norm(1.0), optax.sgd(np.sqrt(binder_length))
    ),
)

```

Here we're using ~5 different models to construct a loss function: the [Boltz-1](https://github.com/jwohlwend/boltz) structure prediction model (which is used _twice_: once to predict the binder-target complex and once to predict the binder as a monomer), ESM2, ProteinMPNN, an n-gram model, and a stability model trained on the [mega-scale](https://www.nature.com/articles/s41586-023-06328-6) dataset. 

It's super easy to define additional loss terms, which are JIT-compatible callable pytrees, e.g. 

```
class LogPCysteine(LossTerm):
    def __call__(self, soft_sequence: Float[Array, "N 20"], key = None):
        mean_log_p = jnp.log(soft_sequence[:, IDX_CYS] + 1E-8).mean()
        return mean_log_p, {"log_p_cys": mean_log_p}

```

There's no reason custom loss terms can't involve more expensive (differentiable) operations, e.g. running ProteinX, or an [EVOLVEpro-style fitness predictor](https://www.science.org/doi/10.1126/science.adr6006).

The [marimo notebook](examples/example_notebook.py) gives a few examples of how this can work.


> **WARNING**: ColabDesign, BindCraft, etc are well-tested and well-tuned methods for very specific problems. `mosaic` may require substantial hand-holding to work (tuning learning rates, etc), often produces proteins that fail simple in-silico tests, must be combined with standard filtering methods, hasn't been tested in any wetlab, etc. This is not for the faint of heart: the intent is to provide a framework in which to implement custom objective functions and optimization algorithms for your application.

It's very easy to swap in different optimizers. For instance, let's say we really wanted to try projected gradient descent on the hypercube $[0,1]^N$. We can implement that in a few lines of code:

```python
def RSO_box(
    *,
    loss_function,
    x: Float[Array, "N 20"],
    n_steps: int,
    optim=optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(1e-1)),
    key=None,
):
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 10000))

    opt_state = optim.init(x)
    
    for _iter in range(n_steps):
        (v, aux), g = _eval_loss_and_grad(
            x=x,
            loss_function=loss_function,
            key=key
        )
        updates, opt_state = optim.update(g, opt_state, x)
        x = optax.apply_updates(x, updates).clip(0,1)
        key = jax.random.fold_in(key, 0)
        _print_iter(_iter, aux, v)

    return x
```

Take a look at [optimizers.py](src/mosaic/optimizers.py) for a few examples of different optimizers.


### Models and losses

| Included models |
| :--- |
| [Boltz-1](#boltz1) |
| [Boltz-2](#boltz2) |
| [AlphaFold2](#alphafold2) |
| [Protenix (mini+tiny)](#protenix) |
| [ProteinMPNN](#proteinmpnn) |
| [ESM](#esm) |
| [stability](#stability) |
| [AbLang](#ablang) |
| [trigram](#trigram) |


---

#### Structure Prediction
---

We define a collection of (model agnostic!) structure prediction related losses [here](src/mosaic/losses/structure_prediction.py). It's also super easy to define your own using the provided interface.


#### Boltz1
---

First load the model using `load_boltz`. 
Next, we need to construct input features and a structure writer (which will produce `.cif` files). 
There are two methods for building inputs features. There are a few convenience functions provided in [boltz.py](src/mosaic/losses/boltz.py) for ease-of-use, e.g. 
`
    features, writer = make_binder_features(binder_len = 50, target_sequence = "GGGG")
`.
We also support the [boltz-1 yaml input specification](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md) for more complex targets (e.g. small molecules, PTMs, ...). For example:
```python
def ptm_yaml(binder_sequence: str):
    return (
        """
version: 1
sequences:
  - protein:
      id: [A]
      sequence: {seq}
      msa: empty
  - protein:
      id: [B]
      sequence: MFEARLVQGSILKKVLEALKDLINEACWDISSSGVNLQSMDSSHVSLVQLTLRSEGFDTYRCDRNLAMGVNLTSMSKILKCAGNEDIITLRAEDNADTLALVFEAPNQEKVSDYEMKLMDLDVEQLGIPEQEYSCVVKMPSGEFARICRDLSHIGDAVVISCAKDGVKFSASGELGNGNIKLSQTSNVDKEEEAVTIEMNEPVQLTFALRYLNFFTKATPLSSTVTLSMSADVPLVVEYKIADMGHLKYYLAPKIEDEEGS
      modifications:
          - position: 211   # index of residue, starting from 1
            ccd: PTR            # CCD code of the modified residue

""".format(seq = binder_sequence)
    )

features, writer = load_features_and_structure_writer(ptm_yaml("X" * binder_length))
```

Note that the binder comes first (by default `Boltz1Loss` optimizes the first `N` tokens).

Once we have our input features and structure writer we can construct a loss function, for example:

```python

import mosaic.losses.structure_prediction as sp
loss = Boltz1Loss(
        model=model,
        name="target",
        loss=2 * sp.BinderTargetContact(epitope_idx=list(range(205, 216)))
        + sp.WithinBinderContact(),
        features=features,
        recycling_steps=0,
        deterministic=False,
    )
```

> Internally we distinguish between three classes of losses: those that rely only on the trunk, structure module, or confidence module. For computational efficiency we only run the structure module or confidence module if required!

After you've designed your protein you can make a prediction and save a `.cif` using the same formula:
```python

final_features, final_writer = load_features_and_structure_writer(ptm_yaml(final_sequence))

j_model = eqx.filter_jit(lambda model, *args, **kwargs: model(*args, **kwargs))


def predict(features, writer):
    o = j_model(
        model,
        features,
        key=jax.random.key(5),
        sample_structure=True,
        confidence_prediction=True,
        deterministic=True,
    )
    st = writer(o["sample_atom_coords"])
    print("plddt", o["plddt"][: sequence.shape[0]].mean())
    print("ipae", o["complex_ipae"].item())
    return o, st

predict(final_features, final_writer)

```

---


#### Boltz2
---

Very similar to Boltz1, see [examples/boltz_notebook.py](examples/boltz_notebook.py).

#### Alphafold2
---

The first step is load the model:
```python

from mosaic.af2.alphafold2 import AF2
from mosaic.losses.af2 import AlphaFold
import mosaic.losses.af2 as aflosses


af2 = AF2(num_recycle = 1)
```

Then we load a target structure (if we want to use a template) and construct features.
```

target_st = gemmi.read_pdb(str(target_path)) # note this could be a prediction (e.g. from boltz-1)

# We use a template for the target chain!
af_features, initial_guess = af2.build_features(
    chains=["G" * binder_length, target_sequence],
    template_chains={1: target_st[0][0]},
)
```

Finally we can construct a loss. For example:
```python
import mosaic.losses.structure_prediction as sp
af_loss = (
        AlphaFold(
            name="af",
            forward=af2.alphafold_apply,
            stacked_params=jax.device_put(af2.stacked_model_params),
            features=af_features,
            losses=0.01 * sp.PLDDTLoss()
            + 1 * sp.BinderTargetContact()
            + 0.1 * sp.TargetBinderPAE()
            + 0.1 * sp.BinderTargetPAE()
        )
```

The `af2` object has a nice interface for prediction:
```python
output, structure = af2.predict(
        [
            binder_sequence,
            target_sequence,
        ],
        template_chains={1: target_st[0][0]},
        key=jax.random.key(3),
        model_idx=0,
    )
```


#### Protenix
---

See [protenij.py](examples/protenij.py) for an example of how to use this family of models. This loss function supports some advanced features to speed up hallucination, namely "pre-cycling" (running multiple recycling iterations on the target alone _before_ design) and "co-cycling" (running recycling and optimization steps in parallel), but can also be used analogously to Boltz or AF2. 


Example use as a prediction model: 
```python
model = load_protenix_mini()

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

design_features, design_structure = load_features_from_json(design_json)

### generate PSSM matrix using any design method

out_jax = model(
    input_feature_dict = set_binder_sequence(PSSM,jax.device_put(design_features)),
    N_cycle=3,
    N_sample=5,
    key=jax.random.key(2),
    N_steps=5
) 

st_pred = biotite_array_to_gemmi_struct(design_structure, pred_coord = np.array(out_jax.coordinates[0]))

```

See the notebook for more complex examples.


#### ProteinMPNN
---

Load your prefered ProteinMPNN (soluble or vanilla) model using 

```python
from mosaic.proteinmpnn.mpnn import ProteinMPNN

mpnn = ProteinMPNN.from_pretrained()
```

In the simplest case we have a single-chain structure or complex where the protein we're designing occurs as the first chain (note this can be a prediction). We can then construct the (negative) log-likelihood of the designed sequence under ProteinMPNN as a loss term:
```python
inverse_folding_LL = FixedStructureInverseFoldingLL.from_structure( gemmi.read_structure("scaffold.pdb"), mpnn)
```
This can then be added to whatever overall loss function you're constructing. 

Note that it is often helpful to clip the loss using, e.g.,  `ClippedLoss(inverse_folding_LL, 2, 100)`: over-optimizing ProteinMPNN likelihoods typically results in homopolymers. 

#### ProteinMPNN + structure prediction
---
ProteinMPNN can also be combined with live structure predictions. Mathematically this is 
$-\log P_\theta(s | AF2(s)),$ the log-likelihood of the sequence $s$ under inverse folding _of the predicted structure for that sequence_. 
This loss term is `ProteinMPNNLoss.`

Another very useful loss term is `InverseFoldingSequenceRecovery`: a continuous relaxation of sequence recovery after sampling with ProteinMPNN (roughly $\langle s, -E_{z \sim p_\theta(\cdot | AF2(s))} [z] \rangle$). We've found this term often speeds up design and increases filter pass rates.


#### ESM
---

Another useful loss term is the pseudolikelihood of the ESM2 protein language model (via [esm2quinox](https://github.com/patrick-kidger/esm2quinox/tree/main)); which is correlated with all kinds of useful properties (solubility, expressibility, etc).

This term can be constructed as follows:
```python
import esm
import esm2quinox
torch_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
ESM2PLL = ESM2PseudoLikelihood(esm2quinox.from_torch(torch_model))
```

In typical practice this loss should be clipped or squashed to avoid over-optimization (e.g. `ClippedLoss(ESM2PLL, 2, 100)`).

We also implement the corresponding loss for ESMC (via [esmj](https://github.com/escalante-bio/esmj)).
```python
from esmj import from_torch
from esm.models.esmc import ESMC as TORCH_ESMC

esm = from_torch(TORCH_ESMC.from_pretrained("esmc_300m").to("cpu"))
ESMCPLL = ESMCPseudoLikelihood(esm)
```

#### Stability
---

A simple delta G predictor trained on the megascale dataset. Might be a nice example of how to train and add a simple regression head on a small amount of data: [train.py](src/mosaic/stability_model/train.py).

```
stability_loss = StabilityModel.from_pretrained(esm)

```

#### AbLang
---
[AbLang](https://github.com/oxpig/AbLang), a family of antibody-specific language models.

```python
import ablang
import jablang

heavy_ablang = ablang.pretrained("heavy")
heavy_ablang.freeze()

abpll = AbLangPseudoLikelihood(
    model=jablang.from_torch(heavy_ablang.AbLang),
    tokenizer=heavy_ablang.tokenizer,
    stop_grad=True,
)


```


#### Trigram
---

A trigram language model as in [A high-level programming language for generative protein design](https://www.biorxiv.org/content/10.1101/2022.12.21.521526v1.full.pdf).

```python
trigram_ll = TrigramLL.from_pkl()
```

### Optimizers and loss transformations
---

We include some standard [optimizers] in (src/mosaic/optimizers.py).


First, `simplex_APGM,` which is an accelerated proximal gradient algorithm on the probability simplex. One critical hyperparameter is the stepsize, a reasonable first guess is `0.1*np.sqrt(binder_length)`. Another useful keyword argument is `scale`, which corresponds to $\ell_2$ regularization. Values larger than `1.0` encorage sparse solutions; a typical binder design run might start with `scale=1.0` to get an initial, soft solution and then ramp up to something higher to get a discrete solution. 

`simplex_APGM` also accepts a keyword argument, `logspace,` if this is set to true we run the algorithm in logspace, which corresponds to an accelerated proximal bregman method. In this case `scale` corresponds to entropic regularization.

We also include a discrete optimization algorithm, `gradient_MCMC`, which is a variant of MCMC with a proposal distribution defined using a taylor approximation to the objective function (see [Plug & Play Directed Evolution of Proteins with Gradient-based Discrete MCMC](https://arxiv.org/abs/2212.09925).) This algorithm is especially useful for finetuning either existing designs or the result of continuous optimization.


#### Loss transformations

We also provide a few [common transformations of loss functions](src/mosaic/losses/transformations.py). Of note are `ClippedLoss`, which ... wraps and clips another loss term. 

`SetPositions` and  `FixedPositionsPenalty` are useful for fixing certain positions of an existing design. 

`ClippedGradient` and `NormedGradient` respectively clip and normalize the gradients of individual loss terms, this can be useful when combining predictors with very different gradient norms, for example:
```python
loss = ClippedGradient(inverse_folding_LL, 1.0)  
    + ClippedGradient(ablang_pll, 1.0)
    + 0.25 * ClippedGradient(ESMCPLL, 1.0)
```

### Extensive theoretical discussion

Hallucination-based protein design workflows attempt to solve the following optimization problem:

$$\underset{s \in A^n}{\textrm{minimize}}~\ell(s).$$

Here $A$ is the set of amino acids, so the decision variable $s$ ranges over all protein sequences of length $n$. $~\ell: A^n \rightarrow \mathbf{R}$ is a loss functional that evaluates the quality of the protein $s$. In typical practice $\ell$ is some function of the output of a neural network; i.e. in [ColabDesign](https://github.com/sokrypton/ColabDesign) $\ell$ might be (negative) average pLDDT from AlphaFold. 

One challenge with naive approaches is that $A^n$ is extremely large and discrete optimization is difficult; while MCMC and other discrete algorithms have been used (see, e.g., [Rives et al](https://www.biorxiv.org/content/10.1101/2022.12.21.521526v1.full.pdf)) they are often *very* slow. 

ColabDesign, RSO, and BindCraft, among others, use the fact that $\ell$ has a particular structure that allows for a continuous relaxation of the original problem: almost every neural network first encodes the sequence $s$ into a one-hot matrix $P \in \mathbf{R}^{(n, c)}$. If we consider $\ell$ as a functional on $\mathbf{R}^{(n, c)}$ we can use automatic differentiation to do continuous optimization on either $\mathbf{R}^{(n, c)}$ or $\Delta_c^n$ ($n$ products of the probability simplex). 

> This is related to the classic optimization trick of optimizing over distributions rather than single points. First, $\underset{x}{\textrm{minimize }}f(x)$ is relaxed to $\underset{p \in \Delta}{\textrm{minimize }}E_p f(x)$. Next, if it makes sense to take the expectation of $x$ (as in the one-hot sequence case), we can interchange $f$ and $E$ to get the final relaxation: $$\underset{p \in \Delta}{\textrm{minimize }} f( E_p x) = \underset{p \in \Delta}{\textrm{minimize }} f(p).$$


Solutions to this relaxed optimization problem must then be translated into sequences; many different methods work here: RSO uses inverse folding of the predicted structure, BindCraft/ColabDesign uses a softmax with ramping temperature to encourage one-hot solutions, etc. 

By default we use a generalized proximal gradient method (mirror descent with entropic regularization) to do optimization over the simplex and to encourage sparse solutions, though it is very easy to swap in other optimization algorithms (e.g. projected gradient descent or composition with a softmax as in ColabDesign). 

Typically $\ell$ is formed by a single neural network (or an ensemble of the same architecture), but in practice we're interested in simultaneously optimizing different properties predicted by different neural networks. This has the added benefit of reducing the chance of finding so-called adversarial sequences. 

This kind of modular implementation of loss terms is also useful with modern RL-based alignment of generative models approaches: these forms of alignment can often be seen as _amortized optimization_. Typically, they train a generative model to minimize some combination of KL divergence minus a loss function, which can be a combination of in-silico predictors. Another use case is to provide guidance to discrete diffusion or flow models. 

[^1]: This requires us to treat neural networks as _simple parametric functions_ that can be combined programatically; **not** as complicated software packages that require large libraries (e.g. PyTorch lightning), bash scripts, or containers as is common practice in BioML. 

