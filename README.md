# NPDemand

[![Build Status](https://travis-ci.com/jamesbrandecon/NPDemand.jl.svg?branch=master)](https://travis-ci.com/jamesbrandecon/NPDemand.jl)
<!-- 
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jamesbrandecon/NPDemand.jl?svg=true)](https://ci.appveyor.com/project/jamesbrandecon/NPDemand-jl)
[![Coverage](https://codecov.io/gh/jamesbrandecon/NPDemand.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jamesbrandecon/NPDemand.jl)
[![Coverage](https://coveralls.io/repos/github/jamesbrandecon/NPDemand.jl/badge.svg?branch=master)](https://coveralls.io/github/jamesbrandecon/NPDemand.jl?branch=master) -->
This package is designed to take data on many markets of potentially many products (where all products are in each market), select small subgroups of products between which substitution is strongest, and estimate structural demand curves nonparametrically on those subgroups. It can currently handle cases in which each subgroup has only 2-4 products, and price is the only observed product characteristic, fairly well. More complicated cases quickly face a curse of dimensionality. See `Initial_Documentation.pdf` for the basics of the package. Documentation based on `Documenter.jl` is in progress. 

Please let me know if you have any suggestions.

## Installation
The package is extremely new, so it isn't registered. To install, use
```jl
pkg> add https://github.com/jamesbrandecon/NPDemand.jl
```

## Usage
There are three important functions included here so far: `inverse_demand`, `price_elasticity`, and `hierNet`/`hierNet_boot`:
- `inverse_demand(df::DataFrame)`: Takes as input a dataframe with columns s1-sJ, p1-pJ, z1-zJ for J products and estimates inverse demand functions separately for each good. Each inverse demand function is approximated by a Bernstein polynomial of user-chosen order. There are a number of options which are demonstrated in the examples. 
- `price_elasticity(inv_sigma, df , p_points, included)`: Takes the output of `inverse_demand`, and returns estimates of own- or cross-price elasticities, according to the matrix `included` which specifies which products are substitutes for each other, calculated either at a pre-specified vector of prices `p_points` or at the realized prices in the data. The former requires solving for counterfactual market shares, and is best for getting a sense of the shape of the demand curve. The latter can be used, for example, to calculate markups under traditional models of static pricing in industrial organization.  
- `hierNet`/`hierNet_boot`: These functions are for model selection. They take as an input a dataframe with columns s1-sJ, p1-pJ, z1-zJ for J products and run constrained lasso regressions (details in draft, coming soon) to select subsets of products which are strong substitutes. `hierNet_boot` runs this procedure multiple times on bootstrapped samples and takes the intersection of selected substitutes across bootstraps. This takes longer but is less likely to include extraneous substitutes. See examples for options. 

One helpful additional function is `simulate_logit`, which allows one to easily simulate data from a logit demand system with endogenous prices (and instruments for those prices) to test the `NPDemand` functions.

Rather than run through some short examples here, I've included example files in `/examples` which demonstrates the use of the most important functions and provides descriptions of the relevant inputs.

## Calling from Python/R 
Some researchers may wish to call this package directly from Python code. The easiest way to do this is through a Jupyter notebook. In IPython, after running the command `%load_ext julia.magic`, one can call Julia (assuming it is installed) by prefacing each line with `%julia`. In R, one can use the `JuliaCall` package.  

## To-do
- Very soon: Permit alternative model selection approaches, e.g. limiting the maximum number of substitutes for each product.
- Soon: The solver I'm using when demand is constrained to be monotonic is much slower (orders of magnitude) than the CVX solver in Matlab. In search of faster algorithms, and for now monotonicity constraints are not the default. 

