# NPDemand

[![Build Status](https://travis-ci.com/jamesbrandecon/NPDemand.jl.svg?branch=master)](https://travis-ci.com/jamesbrandecon/NPDemand.jl)

This package is designed to take data on many markets of potentially many products (where all products are in each market), select small subgroups of products between which substitution is strongest, and estimate structural demand curves nonparametrically on those subgroups. It can currently handle cases in which each subgroup has ~8 or fewer products, and price is the only observed product characteristic, fairly well. More complicated cases quickly face a curse of dimensionality.

Please let me know if you have any suggestions for the package.

## Installation
The package is new, so it isn't registered. To install, use
```jl
pkg> add https://github.com/jamesbrandecon/NPDemand.jl
```

## Usage
There are three important functions included here so far: `inverse_demand`, `price_elasticity`, and `hierNet`/`hierNet_boot`:
- `inverse_demand(df::DataFrame)`: Takes as input a dataframe with columns `shares0-sharesJ-1`, `prices0 - pricesJ-1`, `demand_instruments0-demand_instrumentsJ-1` for J products and estimates inverse demand functions separately for each good. Each inverse demand function is approximated by a Bernstein polynomial of user-chosen order. There are a number of options which are demonstrated in the examples.
- `price_elasticity(inv_sigma, df, p_points, included)`: Takes `inv_sigma` (the output of `inverse_demand`), and returns estimates of own- or cross-price elasticities, according to the matrix `included` which specifies which products are substitutes for each other, calculated at a pre-specified vector of prices `p_points`, which can be user-chosen or the realized prices in the data. The former requires solving for counterfactual market shares and setting the option `trueS = false`, and is best for getting a sense of the shape of the demand curve. The latter can be used, for example, to calculate markups under traditional models of static pricing in industrial organization.  
- `hierNet`/`hierNet_boot`: These functions are for model selection. They take as an input the same DataFrame as above and run constrained lasso regressions (details in draft, coming soon) to select subsets of products which are strong substitutes. `hierNet_boot` runs this procedure multiple times on bootstrapped samples and takes the intersection of selected substitutes across bootstraps. This takes longer but is less likely to include extraneous substitutes. See examples for options.

One helpful additional function is `simulate_logit`, which allows one to easily simulate data from a logit demand system with endogenous prices (and instruments for those prices) to test the `NPDemand` functions.

Rather than run through some short examples here, I've included example files in `/examples` which demonstrates the use of the most important functions and provides descriptions of the relevant inputs.


## Model Selection Details
Model selection follows a modified version of [Bien et al (2013)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4527358/). Currently, I simply regress each product's `shares`j on all demand instruments. Changing this first-stage to a random forest is on the to-do list for the package. I then run J "hierarchical lasso" regressions of each `prices` on all predicted `shares` with either "weak" or "strong" hierarchy imposed. This regression approximates a product's price with an interacted quadratic function in all market shares.

I then define a product k as being a substitute for j if the linear coefficient on `shares`k is non-zero. When "strong" hierarchy is imposed, this is quite natural, as the quadratic term and any interactions involving `shares`k are nonzero only if the linear term is also nonzero. When "weak" hierarchy is imposed, interaction terms may be nonzero even if the linear term is zero, so this definition of substitutes implicitly treats any such cases as mistaken. Strong hierarchy is therefore better suited to the task at hand, but is far more time consuming and often yields very similar selected models in my simulations.

## Minimal Example with Model Selection
As described above, begin with a DataFrame which takes the following form:
```jl
julia> first(df, 5)
5×6 DataFrame
│ Row │ shares0  │ prices0  │ demand_instruments0 │ shares1  │ prices1  │ demand_instruments1 │
│     │ Float64  │ Float64  │ Float64             │ Float64  │ Float64  │ Float64             │
├─────┼──────────┼──────────┼─────────────────────┼──────────┼──────────┼─────────────────────┤
│ 1   │ 0.318976 │ 0.840209 │ 0.377864            │ 0.25169  │ 1.46536  │ 0.612532            │
│ 2   │ 0.221904 │ 1.41531  │ 0.705143            │ 0.312787 │ 1.88974  │ 0.762217            │
│ 3   │ 0.252167 │ 1.02818  │ 0.584668            │ 0.252472 │ 2.06294  │ 0.941736            │
│ 4   │ 0.252374 │ 1.79092  │ 0.788273            │ 0.279031 │ 1.49938  │ 0.672677            │
│ 5   │ 0.286579 │ 1.28321  │ 0.526458            │ 0.292947 │ 0.459311 │ 0.280061            │
```
In this case, there are two goods, so model selection is not necessary, but we can demonstrate the approach regardless. Selecting a model requires a call to `hierNet` or `hierNet_boot`, the latter of which takes an extra option `nboots` which controls the number of bootstrapped samples on which the selection procedure is performed:

```jl
included, included_symmetric = hierNet_boot(df; nfolds = 5, nlam = 10, strong = false, nboot = 5);
included, included_symmetric = hierNet(df; nfolds = 5, nlam = 10, strong = false);
````

The matrix `included` is a matrix of ones and zeros indicating which products are close substitutes, and `included_symmetric` is a similar matrix which enforces that k must substitute for j if j substitutes for k. Next, one can estimate demand and calculate price elasticites:

```jl
inv_sigma, designs = inverse_demand(df; included = included_symmetric);
elast, Jacobians, share_vec = price_elasticity(inv_sigma, df, p; whichProducts = [1,1],
             included = included_symmetric, trueS = true);
```
`whichProducts=[j,k]` indicates that you want `elast` to include elasticities of demand for product j with respect to price k, and `trueS = true` indicates that you want these elasticities evaluated at the realized market shares and prices.  

## Calling from Python/R
Some researchers may wish to call this package directly from Python code. The easiest way to do this is through a Jupyter notebook. In IPython, after running the command `%load_ext julia.magic`, one can call Julia (assuming it is installed) by prefacing each line with `%julia`. In R, one can use the `JuliaCall` package.  

## To-do
- Additional documentation for `hierNet_boot` and `price_elasticity` options.
- Give option to make first stage prediction in `hierNet` a random forest.
- Convert Compiani (2020) code to Julia in order to add exchangeability as an optional constraint.
