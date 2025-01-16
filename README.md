### Functional, multi-objective protein design using continuous relaxation.

This proof of concept combines two ideas from protein design into a simple interface:

- Gradient-based optimization over a continuous, _relaxed_ sequence space (as in [ColabDesign](https://github.com/sokrypton/ColabDesign), RSO, BindCraft, etc)
- A functional, modular interface to easily combine multiple learned or hand-crafted loss terms and optimization algorithms (as in [A high-level programming language for generative protein design](https://www.biorxiv.org/content/10.1101/2022.12.21.521526v1.full.pdf) etc)

The (not necessarily novel) observation here is that it's possible to use this continuous relaxation _simultaneously_ with multiple learned objective terms [^1]. 

The point is to allow us to easily construct objective functions that are combinations of multiple learned potentials and optimize them efficiently, like so:

```python
combined_loss = (
    StructurePrediction(
        model=model,
        name="ART2B",
        loss=4 * BinderTargetContact()
        + RadiusOfGyration(target_radius=15.0)
        + WithinBinderContact()
        + 0.3 * HelixLoss(),
        features=boltz_features,
        recycling_steps=0,
    )
    + 0.5 * esm_loss
    + trigram_ll
    + 0.5
    * StructurePrediction(
        model=model,
        name="mono",
        loss=0.2 * PLDDTLoss()
        + 0.1 * StabilityModel.from_pretrained(esm)
        + RadiusOfGyration(target_radius=15.0)
        + 0.3 * HelixLoss(),
        features=monomer_features,
        recycling_steps=0,
    )
)

logits_combined_objective = design_bregman_optax(
    loss_function=combined_loss,
    n_steps=150,
    x=np.random.randn(binder_length, 20) * 0.1,
    optim=optax.chain(
        optax.clip_by_global_norm(1.0), optax.sgd(np.sqrt(binder_length))
    ),
)

```

Here we're using ~4 different models to construct a loss function: the [Boltz-1](https://github.com/jwohlwend/boltz) structure prediction model (which is used _twice_: once to predict the binder-target complex and once to predict the binder as a monomer), ESM2, an n-gram model, and a stability model trained on the [mega-scale](https://www.nature.com/articles/s41586-023-06328-6) dataset. 

It's super easy to define additional loss terms, which are JIT-compatible callable pytrees, e.g. 

```
class LogPCysteine(LossTerm):
    def __call__(self, soft_sequence: Float[Array, "N 20"], key = None):
        mean_log_p = jnp.log(soft_sequence[:, IDX_CYS] + 1E-8).mean()
        return mean_log_p, {"log_p_cys": mean_log_p}

```

There's no reason loss terms can't involve more expensive (differentiable) operations, e.g. running AlphaFold, or an [EVOLVEpro-style fitness predictor](https://www.science.org/doi/10.1126/science.adr6006).

The [marimo notebook](example_notebook.py) gives an example of how this can work.

> You'll need a GPU or TPU-compatible version of JAX for structure prediction. You might need to install this manually, i.e. ` uv add jax[cuda12_local].`

> **WARNING**: ColabDesign, BindCraft, etc are well-tested and well-tuned methods; this is a position piece: it may require substantial hand-holding to work at all (tuning learning rates, etc), often produces proteins that fail simple in-silico tests, must be combined with standard filtering methods, hasn't been tested in any wetlab, etc.


#### Exhaustive discussion

Hallucination-based protein design workflows attempt to solve the following optimization problem:

$$\underset{s \in A^n}{\textrm{minimize}}~\ell(s).$$

Here $A$ is the set of amino acids, so the decision variable $s$ ranges over all protein sequences of length $n$. $~\ell: A^n \rightarrow \mathbf{R}$ is a loss functional that evaluates the quality of the protein $s$. In typical practice $\ell$ is some function of the output of a neural network; i.e. in [ColabDesign](https://github.com/sokrypton/ColabDesign) $\ell$ might be (negative) average pLDDT from AlphaFold. 

One challenge with naive approaches is that $A^n$ is extremely large and discrete optimization is difficult; while MCMC and other discrete algorithms have been used (see, e.g., [Rives et al](https://www.biorxiv.org/content/10.1101/2022.12.21.521526v1.full.pdf)) they are often *very* slow. 

ColabDesign, RSO, and BindCraft, among others, use the fact that $\ell$ has a particular structure that allows for a continuous relaxation of the original problem: almost every neural network first encodes the sequence $s$ into a one-hot matrix $P \in \mathbf{R}^{(n, c)}$. If we consider $\ell$ is a functional on $\mathbf{R}^{(n, c)}$ we can use automatic differentiation to do continuous optimization on either $\mathbf{R}^{(n, c)}$ or $\Delta_c^n$ ($n$ products of the probability simplex[^2]). 

Solutions to this relaxed optimization problem must then be translated into sequences; many different methods work here: RSO uses inverse folding of the predicted structure, BindCraft/ColabDesign uses a softmax with ramping temperature to encourage one-hot solutions, etc. 

By default we use a generalized proximal gradient method (mirror descent with entropic regularization) to do optimization over the simplex and to encourage sparse solutions, though it very easy to swap in other optimization algorithms (e.g. projected gradient descent or composition with a softmax as in ColabDesign). 

Typically $\ell$ is formed by a single neural network (or an ensemble of the same architecture), but in practice we're interested in simultaneously optimizing different properties predicted by different neural networks. This has the added benefit of reducing the chance of finding so-called adversarial sequences. 

[^1]: This requires us to treat neural networks as _simple parametric functions_ that can be combined programatically; **not** as complicated software packages that require large libraries (e.g. PyTorch lightning), bash scripts, or containers as is common practice in BioML. 

[^2]: This is related to the classic optimization trick of optimizing over distributions rather than single points. First, $\underset{x}{\textrm{minimize }}f(x)$ is relaxed to $\underset{p \in \Delta}{\textrm{minimize }}E_p f(x)$. Next, if it makes sense to take the expectation of $x$ (as in the one-hot sequence case), we can interchange $f$ and $E$ to get the final relaxation: $$\underset{p \in \Delta}{\textrm{minimize }} f( E_p x) = \underset{p \in \Delta}{\textrm{minimize }} f(p).$$


#### TODO:
- [ ] Additional loss terms:
    - [ ] AlphaFold2
    - [ ] ProteinMPNN
- [ ] Alternative optimization algorithms:
    - [ ] ColabDesign/BC-style logits + softmax
    - [ ] MCMC
    - [ ] Gradient-assisted MCMC
- [ ] Add per-term gradient normalization/clipping
- [ ] Clean up tokenization situation

