# NPDemand

[![Build Status](https://travis-ci.com/jamesbrandecon/NPDemand.jl.svg?branch=master)](https://travis-ci.com/jamesbrandecon/NPDemand.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jamesbrandecon/NPDemand.jl?svg=true)](https://ci.appveyor.com/project/jamesbrandecon/NPDemand-jl)
[![Coverage](https://codecov.io/gh/jamesbrandecon/NPDemand.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jamesbrandecon/NPDemand.jl)
[![Coverage](https://coveralls.io/repos/github/jamesbrandecon/NPDemand.jl/badge.svg?branch=master)](https://coveralls.io/github/jamesbrandecon/NPDemand.jl?branch=master)

This is currently being translated from Matlab to Julia (as I also learn how to use github). Do not attempt to use yet

## Installation
The package is extremely new, so it isn't registered yet. To install, use
```jl
Pkg.add("github.com/jamesbrandecon/NPDemand.jl")
```

## Usage
There are two important functions included here so far: `inverse_demand` and `price_elasticity_priceIndex`:
- `inverse_demand`: Takes as inputs matrices of market shares and prices, and estimates inverse demand functions separately for each good. Each inverse demand function is approximated by a Bernstein polynomial of user-chosen order.   
- `price_elasticity_priceIndex`: Takes the output of `inverse_demand` as an input, and returns estimates of own- or cross-price elasticities calculated either at a pre-specified vector of prices or at the realized prices in the data. The former requires solving for counterfactual market shares, and is best for getting a sense of the shape of the demand curve. The latter can be used, for example, to calculate optimal markups under traditional models of industrial organization.  

One helpful additional function is `simulate_logit`, which allows one to easily simulate data from a logit demand system to test the `NPDemand` functions.

Rather than run through some short examples here, I've included the file `example1.jl` here which demonstrates the use of the most important functions and provides descriptions of the relevant inputs.

## To-do
- Very soon: add option to control for additional market characteristics
- Very soon: The solver I'm using is much slower (orders of magnitude) than the CVX solver in Matlab. In search of faster algorithms.
- Next: Add model selection features to permit use in larger markets.
- Later: Permit models without price in the index
- Later: Allow for differing orders of Bernstein polynomials for different products
