# NPDemand

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://jamesbrandecon.github.io/NPDemand.jl/dev/)

This package has been significantly re-designed as part of a work in progress with Giovanni Compiani and Adam Smith. I have removed the functionality for variable/model selection, but have significantly increased the performance of the package and improved the simplicity of use. The old code has been saved in the `copy_old_code` branch, if for some reason that code is useful for you.

Please let me know if you have any suggestions for the package or find any bugs.

## Installation
The package is not registered. To install, use
```jl
pkg> add https://github.com/jamesbrandecon/NPDemand.jl
```

## Main Functions
There are three important functions included here so far: `define_problem`, `estimate!`, and `price_elasticity`:  
- `define_problem(df::DataFrame; exchange = [], index_vars = ["prices"], FE = [], constraints = [], bO = 2, obj_tol = 1e-5, constraint_tol = 1e-5)`: Constructs a `problem::NPDProblem` using the provided problem characteristics. Inputs: 
    - `exchange`: A vector of (at most two) groups of products which are exchangeable. E.g., with 4 goods, if the first
    and second are exchangeable and so are the third and fourth, set `exchange` = [[1 2], [3 4]].
    - `index_vars`: String array listing column names in `df` which represent variables that enter the inverted index.
    - `FE`: String array listing column names in `df` which should be included as fixed effects.
    - `obj_tol`: Tolerance specifying the value of `g_abstol` within `Optim.Options` during optimization.
    - `constraint_tol`: Tolerance specifying tightness of constraints
    - `constraints`: A list of symbols of accepted constraints. Currently supported constraints are: 
        - :monotone: Demand is increasing in the index. 
        - :all_substitutes: All products are substitutes.
        - :diagonal_dominance_group: Diagonal dominance (see Compiani, 2022) within exchangeable groups. 
        - :diagonal_dominance_all: Diagonal dominance across all products.
        - :subs_in_group: All products within exchangeable groupings are substitutes.(**Note**: this constraint is the only available nonlinear constraint and will slow down estimation considerably)
- `estimate!(problem::NPDProblem; max_iterations=10000, show_trace = false)`: solves `problem` subject to provided constraints, and replaces `problem.results` with the resulting parameter vector. `max_iterations` is passed into Optim.Options for every optimization step.  
- `update_constraints!(problem::NPDProblem, new_constraints::Vector{Symbol})`: Replaces the constraints in `problem` with `new_constraints`. Because of the way `:exchangeability` is enforced, this function cannot be used to change the structure of exchangeable groupings. 
- `price_elasticities!(problem::NPDProblem)`: Takes the solved `problem` as first argument, a `DataFrame` as the second argument, and evaluates all price elasticities in-sample. Currently does not calculate out-of-sample price elasticities. For this, use the function `compute_demand_function!`.
- `compute_demand_function!(problem, df; max_iter = 1000, show_trace = false)`: estimates the demand function/curve using NPD estimates calculated via estimate!. The function takes in an estimated problem::NPDProblem and a dataframe with counterfactual values of the covariates in the utility function. One must specify all fields that were used in estimation (including shares). The function will change the values of df[!,r"shares"] to take on the value of the estimated demand function. Options: 
    - `max_iter`: controls the number of iterations for the nonlinear solver calculating market shares. Default is 1000 but well-estimated problems should converge faster.
    - `show_trace`: if `true`, Optim will print the trace for each iteration of the nonlinear solver. 



Two helpful additional functions are `simulate_logit` and `toDataFrame`, which allows one to easily simulate data from a logit demand system with endogenous prices (and instruments for those prices) to test the `NPDemand` functions. Usage shown in `/examples`.

## Example of constructing a problem and estimating price elasticities
As described above, begin with a DataFrame which takes the following form (with three products in this example):
```jl
julia> df
2000×16 DataFrame
  Row │ shares0    prices0   x0        share_iv0  price_iv0  shares1   prices1    x1        share_iv1  price_iv1  shares2    prices2   x2 ⋯
      │ Float64    Float64   Float64   Float64    Float64    Float64   Float64    Float64   Float64    Float64    Float64    Float64   Fl ⋯
──────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    1 │ 0.328229   0.66807   0.416167   0.416167  0.0717971  0.14538    1.89975   0.903872   0.903872  0.7109     0.236292   0.458123  0. ⋯
    2 │ 0.0994119  1.50148   0.367425   0.367425  0.899897   0.25597    1.48679   0.8222     0.8222    0.636302   0.208002   0.834816  0.
    3 │ 0.131559   1.60465   0.323835   0.323835  0.596641   0.320453   1.35984   0.84993    0.84993   0.429504   0.175882   0.87676   0.
  ⋮   │     ⋮         ⋮         ⋮          ⋮          ⋮         ⋮          ⋮         ⋮          ⋮          ⋮          ⋮         ⋮         ⋱
 1998 │ 0.252616   0.315564  0.456881   0.456881  0.319923   0.33775    0.65299   0.493729   0.493729  0.212751   0.0719814  1.96853   0.
 1999 │ 0.193392   1.51716   0.928404   0.928404  0.664692   0.428216  -0.120323  0.501273   0.501273  0.0640289  0.0655636  1.52594   0. ⋯
 2000 │ 0.373363   0.476147  0.850132   0.850132  0.398496   0.177178   1.27511   0.374716   0.374716  0.533122   0.0835981  1.69474   0.
                                                                                                            4 columns and 1994 rows omitted
```
Then, we can define a `problem`. In this package, a `problem` is constructed by combining data (in `df`) and constraints. A problem can be constructed simply using `define_problem`:
```jl
bO = 2; # Order of each dimension of Berntstein polynomial
exchange = [[1 2 3]] # All products are exchangeable
index_vars = ["prices", "x"] # endogenous prices and exogenous x enter the index

constraints = [:exchangeability, :monotone]; # Impose exchangeability and that demand for each product is increasing in the index (decreasing in price)
npd_problem = define_problem(df; 
                            exchange = exchange, 
                            index_vars = index_vars, 
                            constraints = constraints,
                            bO = bO);
show(npd_problem)
```
We can then estimate the problem and calculate in-sample price elasticities with two lines. 
```jl
estimate!(problem)
price_elasticities!(npd_problem);
```

## Fixed Effects
Fixed effects are estimated as parameters, not absorbed from the data. So, be careful including fixed effects with too many values, as this may both slow down the optimization and require more memory.

To include fixed effects (categorical variables), use the option `FE` to provide a vector of strings, where each element of the vector is a name of a column in the provided data `df`. Note however that at present, variables that are included as fixed effect must be constant across products within a market. The only exception to this rule is `"product"` which is a keyword which will produce product fixed-effects. There need not be a column named `product` in the data, and in fact the code will ignore it if it's there. 

