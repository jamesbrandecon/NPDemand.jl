# Quasi-Bayes Usage 
Our implementation of quasi-Bayes (QB) tools includes two main functions: `estimate!` (the same function used to estimate problems via GMM) and `smc!`. Some of the underlying details are explained on prior pages. This page focuses on the usage of these functions

## Linearly imposed constraints
In order to use QB estimate a problem, a standard call to `estimate!` would look something like the following. 
```julia
using Turing

estimate!(problem, 
        quasi_bayes = true,
        burn_in = 0.50,     # Fraction of samples to discard as burn-in
        n_attempts = 100,   # Number of samples used to identify a valid starting point 
        n_samples = 15000,  # Total (including burn-in) samples to collect
        step = 0.01,        # Controls MH step size
        skip = 5,           # Controls chain thinning. E.g., skip=5 stores only every fifth sample
        sampler = MH())     # control the sampler used by Turing.jl
```

One way to speed up sampling is to use Hamiltonian Monte Carlo (`HMC()`) or No U-Turn Sampler (`NUTS()`). Another is to increase the `chunksize` used by the autodiff tools inside of Turing, as below:
```julia
HMC(0.01,
     5; 
     adtype = Turing.AutoForwardDiff(chunksize = 1000))
```
!!! warning 
    Although AutoZygote is an often-recommended alternative to AutoForwardDiff, it will not work in the current package due to choices we've made internally. 

After this has been run, the problem stores two key objects. One is the original chain (without any burn-in or thinning) in `problem.chain`, and the other is a "filtered" chain (*with* burn-in and thinning) in `problem.results.filtered_chain`. Post-estimation tools will use the filtered chain for calculating price elasticities and counterfactuals. 

!!! note
    The user can also input `sampler = "mh"` or `sampler = "hmc"` to use these samplers with options that have worked well for us during development and testing. 

## Nonlinearly imposed constraints (SMC)
In order to use SMC, you must `estimate!` a problem first, with `quasibayes` set to `true`. Then, you can use the following command: 
```julia
 smc!(problem, 
        burn_in = 0.50, 
        skip = 20,
        max_penalty = 6,            # Maximum allowed penalty term
        step = 0.01, 
        mh_steps = 5,               # Number of Metropolis-Hastings steps every iteration
        seed = 1021, 
        ess_threshold = 200,        # Minimum acceptable Effective Sample Size per iteration
        smc_method = :adaptive,     # Calculate the penalty sequence adaptively
        max_violations = 0.05,      # Maximum acceptable fraction of markets with any violations of constraints
        max_iter = 100)             # Maximum number of iterations 
```