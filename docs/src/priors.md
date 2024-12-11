# Quasi-Bayes: Priors and Sampling

## Quasi-Bayes at a high level
Let $l_n(\theta)$ denote a scaled version of our GMM objective function. Additionally, let the transformation $e^{l_n(\theta)}$ denote the quasi-likelihood function and $\pi(\theta)$ denote a prior density. Then the quasi-posterior takes the form:

$$\pi(\theta|\text{data}) = \frac{e^{l_n(\theta)} \pi(\theta)}{\int e^{l_n(\theta)} \pi(\theta) d\theta }$$
$$\pi_0(\theta^*|\text{data})\propto e^{l_n(g^{-1}(\theta^*))} \bar{\pi}(\theta^*)$$

and is a valid density over $\theta$ [(Chernozhukov and Hong, 2003)](https://arxiv.org/abs/2301.07782). 

### Priors 
#### Linearly encoded priors
For problems without constraints and with only exchangeability constraints, we specify a simple and dispersed prior:

$$\theta \sim N(\vec 0, I * v),$$ 

where $I$ is the identity matrix and $v$ is a large prespecified constant. Whenever we impose linear constraints on our parameters (whether or we also impose constraints via SMC), we modify our prior to encode the constraint directly. For example, if our constraints imply that $\theta_1$ (the first element of our parameter vector) must be larger than $\theta_2$, then we introduce an auxiliary parameter $\theta^*_2$ and specify priors on $(\theta_1, \theta_2)$ as

$$\theta_1, \theta^*_2 \sim N(0,v)$$

$$\theta_2 = \theta_1 + exp(\theta^*_2)$$

We construct similar transformations for all linear constraints. 

#### More general priors
More generally, we encode constraints via "dogmatic priors" of the form 

$$\pi(\theta^*) \propto \bar{\pi}(\theta^*)\mathbf{1}_\mathcal{C}(\theta),$$

where $\mathcal{C}$ denotes the region of parameter space in which all of the desired constraints are satisfied in the data. With this prior, we then have to sample from a quasi-posterior of the form

$$ \pi(\theta^*|\text{data})\propto e^{\ell_n(g^{-1}(\theta^*))} \bar{\pi}(\theta^*)\mathbf{1}_\mathcal{C}(\theta) $$ 

We do this in two steps. First, we sample from the simpler prior described in the previous subsection, and then we run a "Sequentially Constrained Monte Carlo" (SMC) algorithm in order to sample from this more complex prior. We describe SMC on its own page in the documentation. 

## Sampling
By default, for our prior with linear constraints above, the `estimate` function uses the Metropolis-Hastings sampling from [Turing.jl](https://turinglang.org/) with proposal step sizes controlled by the `step` keyword. After importing Turing, NPDemand can also take in custom samplers from Turing.jl, including Hamiltonian Monte Carlo (HMC) and No U-Turn Samplers (NUTS) with auto-differentiation through [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl). 

For SMC (implemented in the `smc!` function), the user can stil control the step size via `step`, but we have only implemented Metropolis-Hastings sampling to-date. 
