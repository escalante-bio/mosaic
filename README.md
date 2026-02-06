## Functional, multi-objective protein design using continuous relaxation.


> **WARNING**: Unlike BindCraft (which is a well-tested and well-tuned method for generic binder design), `mosaic` may require substantial hand-holding (tuning learning rates, etc), often produces proteins that fail simple in-silico tests, should be combined with standard filtering methods, etc. This is not for the faint of heart: the intent is to provide a framework in which to implement custom objective functions and optimization algorithms for your application.

> **WARNING**: We rely heavily on just-in-time compilation via JAX; the first call to a JAX-compiled function will be slow. After that things should be pretty fast. If you're tuning loss weights or optimizer parameters, you should use an interactive session or a notebook!

 Protein design tasks almost always involve multiple constraints or properties that must be satisfied or optimized. For instance, in binder design one may want to simultaneously ensure:
- the chance of binding the intended target is high  
- the chance of binding to a similar off-target protein is low
- the binder expresses well in bacteria
- the binder is highly soluble. 

There has been a recent explosion in the application of machine learning to protein property prediction, resulting in fairly accurate predictors for each of these properties. What is currently lacking is an efficient and flexible method for combining these different predictors into one design/filtering/ranking framework. 

---
### Models and losses

| Included models |
| :--- |
| Boltz-1 |
| Boltz-2 |
| BoltzGen (design) |
| AlphaFold2 |
| [Protenix (miny, tiny, base, v1.0, 20250630_v1.0.0)](#protenix) |
| [ProteinMPNN (standard, soluble, AbMPNN)](#proteinmpnn) |
| [ESM (2 *or* C)](#esm) |
| [stability](#stability) |
| [AbLang](#ablang) |
| [trigram](#trigram) |



### Installation
We recommend using `uv`, e.g. run `uv sync --group jax-cuda` after cloning the repo to install dependencies.

To run the example notebook try `source .venv/bin/activate`, `marimo edit examples/example_notebook.py`.

> You may need to add various `uv` overrides for specific packages and your machine, take a look at [pyproject.toml](pyproject.toml)

> You'll need a GPU or TPU-compatible version of JAX for structure prediction. You might need to install this manually, i.e. ` uv add jax[cuda12].`


### Introduction

This project combines two simple components to make a powerful protein design framework:

- Gradient-based optimization over a continuous, relaxed sequence space (as in [ColabDesign](https://github.com/sokrypton/ColabDesign), RSO, BindCraft, etc)
- A functional, modular interface to easily combine multiple learned or hand-crafted loss terms and optimization algorithms (as in [A high-level programming language for generative protein design](https://www.biorxiv.org/content/10.1101/2022.12.21.521526v1.full.pdf) etc)

The key observation is that it's possible to use this continuous relaxation simultaneously with multiple learned objective terms [^1]. 

This allows us to easily construct objective functions that are combinations of multiple learned potentials and optimize them efficiently, like so:

```python
from mosaic.models.boltz1 import Boltz1
from mosaic.structure_prediction import TargetChain
import mosaic.losses.structure_prediction as sp
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.proteinmpnn.mpnn import ProteinMPNN
from mosaic.optimizers import simplex_APGM
import jax
import numpy as np

boltz1 = Boltz1()
mpnn = ProteinMPNN.from_pretrained()

target_sequence = "DYSFSCYSQLEVNGSQHSLTCAFE..."
binder_length = 80

# Generate features for binder-target complex
boltz_features, _ = boltz1.binder_features(
    binder_length=binder_length,
    chains=[TargetChain(sequence=target_sequence)],
)

# Generate features for binder alone (monomer)
mono_features, _ = boltz1.binder_features(
    binder_length=binder_length,
    chains=[]
)

combined_loss = (
    boltz1.build_loss(
        loss=4 * sp.BinderTargetContact()
        + sp.RadiusOfGyration(target_radius=15.0)
        + sp.WithinBinderContact()
        + 0.3 * sp.HelixLoss()
        + 5.0 * InverseFoldingSequenceRecovery(mpnn, temp=jax.numpy.array(0.01)),
        features=boltz_features,
        recycling_steps=1,
    )
    + 0.5 * esm_loss
    + trigram_ll
    + 0.1 * stability_loss
    + 0.5
    * boltz1.build_loss(
        loss=0.2 * sp.PLDDTLoss()
        + sp.RadiusOfGyration(target_radius=15.0)
        + 0.3 * sp.HelixLoss(),
        features=mono_features,
        recycling_steps=1,
    )
)

_, PSSM = simplex_APGM(
    loss_function=combined_loss,
    n_steps=150,
    x=jax.nn.softmax(
        0.5 * jax.random.gumbel(
            key=jax.random.key(np.random.randint(100000)),
            shape=(binder_length, 20),
        )
    ),
    stepsize=0.1,
    momentum=0.9,
)

```

Here we're using ~5 different models to construct a loss function: the [Boltz-1](https://github.com/jwohlwend/boltz) structure prediction model (which is used _twice_: once to predict the binder-target complex and once to predict the binder as a monomer), ESM2, ProteinMPNN, an n-gram model, and a stability model trained on the [mega-scale](https://www.nature.com/articles/s41586-023-06328-6) dataset. 

It's super easy to define additional loss terms, which are JIT-compatible callable pytrees, e.g.

```python
class LogPCysteine(LossTerm):
    def __call__(self, soft_sequence: Float[Array, "N 20"], key = None):
        mean_log_p = jnp.log(soft_sequence[:, IDX_CYS] + 1E-8).mean()
        return mean_log_p, {"log_p_cys": mean_log_p}

```

There's no reason custom loss terms can't involve more expensive (differentiable) operations, e.g. running ProteinX, or an [EVOLVEpro-style fitness predictor](https://www.science.org/doi/10.1126/science.adr6006).

The [marimo notebooks](examples) give a few examples of how this can work.


It's very easy to swap in different optimizers. For instance, let's say we really wanted to try projected gradient descent on the hypercube $[0,1]^N$. We can implement that in a few lines of code:

```python
from mosaic.optimizers import _print_iter, _eval_loss_and_grad
def RSO_box(
    *,
    loss_function,
    x: Float[Array, "N 20"],
    n_steps: int,
    stepsize: float,
    max_grad_norm: float,
    key=None,
):
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 10000))
    
    for _iter in range(n_steps):
        (v, aux), g = _eval_loss_and_grad(
            x=x,
            loss_function=loss_function,
            key=key
        )
        g_norm = np.linalg.norm(g)
        if g_norm > max_grad_norm:
            g = g * (max_grad_norm / g_norm)
        x = (x - stepsize * g).clip(0,1)
        key = jax.random.fold_in(key, 0)
        _print_iter(_iter, aux, v)

    return x
```

Take a look at [optimizers.py](src/mosaic/optimizers.py) for a few examples of different optimizers.

---

#### Structure Prediction
---

We provide a simple interface in `mosaic.structure_prediction` and `mosaic.models.*` to five structure prediction models: `Boltz1`, `Boltz2`, `AF2`, `ProtenixMini,` `ProtenixTiny,` `ProtenixBase,` and `Protenix2025.`


To make a prediction or design a binder, you'll need to make a list of `mosaic.structure_prediction.TargetChain` objects. This is a simple dataclass that contains a protein (or DNA or RNA) sequence, a flag to tell the model if it should use MSAs (`use_msa`), and potentially a template structure (as a `gemmi.Chain`).

For example, we can make a prediction with Protenix for IL7Ra like so:

```python

import jax
from mosaic.structure_prediction import TargetChain
from mosaic.models.protenix import ProtenixMini


model = Protenix2025()

target_sequence = "DYSFSCYSQLEVNGSQHSLTCAFEDPDVNTTNLEFEICGALVEVKCLNFRKLQEIYFIETKKFLLIGKSNICVKVGEKSLTCKKIDLTTIVKPEAPFDLSVVYREGANDFVVTFNTSHLQKKYVKVLMHDVAYRQEKDENKWTHVNLSSTKLTLLQRKLQPAAMYEIKVRSIPDHYFKGFWSEWSPSYYFRT"


# generate features and a "writer" object that turns model output into a prediction wrapper
target_only_features, target_only_structure = model.target_only_features(
    [TargetChain(target_sequence)]
)

prediction = model.predict(
    features=target_only_features,
    writer=target_only_structure,
    key=jax.random.key(0),
    recycling_steps=10,
)

# prediction contains useful properties like `prediction.st`, `prediction.pae` etc.
```

This interface is the same for all structure prediction models, so in theory we should be able to replace `Protenix2025` above with `Boltz2` by changing only a single line of code!

We also define a collection of (model agnostic!) structure prediction related losses [here](src/mosaic/losses/structure_prediction.py). It's super easy to define your own using the provided interface.


> Internally we distinguish between three classes of losses: those that rely only on the trunk, structure module, or confidence module. For computational efficiency we only run the structure module or confidence module if required.


Continuing the example above, we can construct a loss and do design as follows:

```python
import mosaic.losses.structure_prediction as sp
from mosaic.optimizers import simplex_APGM
import numpy as np

binder_length = 80

design_features, design_structure = model.binder_features(
    binder_length=binder_length, chains=[TargetChain(target_sequence)]
)

loss = model.build_loss(
    loss=sp.BinderTargetContact() + sp.WithinBinderContact(), features=design_features, recycling_steps=2
)

PSSM = jax.nn.softmax(
    0.5
    * jax.random.gumbel(
        key=jax.random.key(np.random.randint(100000)),
        shape=(binder_length, 20),
    )
)

_, PSSM = simplex_APGM(
    loss_function=loss,
    x=PSSM,
    n_steps=50,
    stepsize=0.15,
    momentum=0.3,
)
```

> Every structure prediction model also support a lower-level feature + losses interfaces if you'd like to do something fancy (e.g. design a protein binder against a small molecule with Boltz or Protenix).

#### Protenix
---

See [protenij.py](examples/protenij.py) for an example of how to use this family of models. This loss function supports some advanced features to speed up hallucination, namely "pre-cycling" (running multiple recycling iterations on the target alone _before_ design).


#### ProteinMPNN
---

Load your prefered ProteinMPNN (soluble or vanilla) model using 

```python
from mosaic.proteinmpnn.mpnn import ProteinMPNN

mpnn = ProteinMPNN.from_pretrained()
```

In the simplest case we have a single-chain structure or complex where the protein we're designing occurs as the first chain (note this can be a prediction). We can then construct the (negative) log-likelihood of the designed sequence under ProteinMPNN as a loss term:
```python
from mosaic.losses.protein_mpnn import FixedStructureInverseFoldingLL
import gemmi

inverse_folding_LL = FixedStructureInverseFoldingLL.from_structure(gemmi.read_structure("scaffold.pdb"), mpnn)
```
This can then be added to whatever overall loss function you're constructing. 

Note that it is often helpful to clip the loss using, e.g.,  `ClippedLoss(inverse_folding_LL, 2, 100)`: over-optimizing ProteinMPNN likelihoods typically results in homopolymers. 

#### ProteinMPNN + structure prediction
---
ProteinMPNN can also be combined with live structure predictions. Mathematically this is
$-\log P_\theta(s | AF2(s)),$ the log-likelihood of the sequence $s$ under inverse folding _of the predicted structure for that sequence_.
This loss term is `ProteinMPNNLoss.`

Another very useful loss term is `InverseFoldingSequenceRecovery`: a continuous relaxation of sequence recovery after sampling with ProteinMPNN (roughly $\langle s, -E_{z \sim p_\theta(\cdot | AF2(s))} [z] \rangle$). We've found this term often speeds up design and increases filter pass rates.

```python
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery

# Include as part of a structure prediction loss
loss = model.build_loss(
    loss=sp.BinderTargetContact()
    + sp.WithinBinderContact()
    + 5.0 * InverseFoldingSequenceRecovery(mpnn, temp=jax.numpy.array(0.01)),
    features=features,
)
```


#### ESM
---

> Warning: due to python issues, it's impossible to use both ESM2 and ESMC in the same environment. 

Another useful loss term is the pseudolikelihood of the ESM2 protein language model (via [esm2quinox](https://github.com/patrick-kidger/esm2quinox/tree/main)); which is correlated with all kinds of useful properties (solubility, expressibility, etc).

This term can be constructed as follows:
```python
import esm
import esm2quinox
from mosaic.losses.esm import ESM2PseudoLikelihood

torch_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
ESM2PLL = ESM2PseudoLikelihood(esm2quinox.from_torch(torch_model))
```

In typical practice this loss should be clipped or squashed to avoid over-optimization (e.g. `ClippedLoss(ESM2PLL, 2, 100)`).

We also implement the corresponding loss for ESMC (via [esmj](https://github.com/escalante-bio/esmj)).
```python
from esmj import from_torch
from esm.models.esmc import ESMC as TORCH_ESMC
from mosaic.losses.esmc import ESMCPseudoLikelihood

esmc = from_torch(TORCH_ESMC.from_pretrained("esmc_300m").to("cpu"))
ESMCPLL = ESMCPseudoLikelihood(esmc)
```

#### Stability
---

A simple delta G predictor trained on the megascale dataset. Might be a nice example of how to train and add a simple regression head on a small amount of data: [train.py](src/mosaic/stability_model/train.py).

```python
from mosaic.losses.stability import StabilityModel

stability_loss = StabilityModel.from_pretrained(esm)
```

#### AbLang
---
[AbLang](https://github.com/oxpig/AbLang), a family of antibody-specific language models.

```python
import ablang
import jablang
from mosaic.losses.ablang import AbLangPseudoLikelihood

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
from mosaic.losses.trigram import TrigramLL

trigram_ll = TrigramLL.from_pkl()
```

### Optimizers and loss transformations
---

We include some standard [optimizers](src/mosaic/optimizers.py).


First, `simplex_APGM,` which is an accelerated proximal gradient algorithm on the probability simplex. One critical hyperparameter is the stepsize, a reasonable first guess is `0.1*np.sqrt(binder_length)`. Another useful keyword argument is `scale`, which corresponds to $\ell_2$ regularization. Values larger than `1.0` encourage sparse solutions; a typical binder design run might start with `scale=1.0` to get an initial, soft solution and then ramp up to something higher to get a discrete solution. 

`simplex_APGM` also accepts a keyword argument, `logspace,` to run the algorithm in logspace, e.g. as an accelerated proximal bregman method. In this case `scale` corresponds to (negative) entropic regularization: values greater than one encourage sparsity.

We also include a discrete optimization algorithm, `gradient_MCMC`, which is a variant of MCMC with a proposal distribution defined using a taylor approximation to the objective function (see [Plug & Play Directed Evolution of Proteins with Gradient-based Discrete MCMC](https://arxiv.org/abs/2212.09925).) This algorithm is especially useful for finetuning either existing designs or the result of continuous optimization.


#### Loss transformations

We also provide a few [common transformations of loss functions](src/mosaic/losses/transformations.py). Of note are `ClippedLoss`, which wraps and clips another loss term. 

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

