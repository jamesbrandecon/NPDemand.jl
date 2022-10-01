# NPDemand

[![Build Status](https://travis-ci.com/jamesbrandecon/NPDemand.jl.svg?branch=master)](https://travis-ci.com/jamesbrandecon/NPDemand.jl)

This package has been significantly re-designed. I have removed the functionality for variable/model selection, but have significantly increased the performance of the model and improved the simplicity of use. 

Please let me know if you have any suggestions for the package.

## Installation
The package is not registered. To install, use
```jl
pkg> add https://github.com/jamesbrandecon/NPDemand.jl
```

## Usage
There are three important functions included here so far: `define_problem`, `estimate!`, and `price_elasticity`:
- `define_problem(df::DataFrame; exchange = [], index_vars = ["prices"], FE = [], constraints = [], bO = 2, tol = 1e-5)`: Constructs a `problem::NPDProblem` using the provided problem characteristics. Inputs: 
    - `exchange`::Vector{Matrix{Int64}}: A vector of groups of products which are exchangeable. E.g., with 4 goods, if the first
    and second are exchangeable and so are the third and fourth, set `exchange` = [[1 2], [3 4]].
    - `index_vars`: String array listing column names in `df` which represent variables that enter the inverted index.
    - `FE`: String array listing column names in `df` which should be included as fixed effects.
    - `tol`: Tolerance specifying tightness of constraints
        - Note: All fixed effects are estimated as parameters by the minimizer, so be careful adding fixed effects for variables that take 
        many values.
    - `constraints`: A list of symbols of accepted constraints. Currently supported constraints are: 
        - :monotone  
        - :all_substitutes 
        - :diagonal\\_dominance\\_group 
        - :diagonal\\_dominance\\_all 
        - :subs\\_in\\_group (Note: this constraint is the only available nonlinear constraint and will slow down estimation considerably)
- `estimate!(problem::NPDProblem)`: `estimate!` solves `problem` subject to provided constraints, and replaces `problem.results` with the resulting parameter vector.
- `price_elasticity(problem::NPDProblem, df::DataFrame; at::Matrix, whichProducts = [1,1])`: Takes the solved `problem` as first argument, a `DataFrame` as the second argument, and evaluates price elasticities in-sample at prices `at`. Currently does not calculate out-of-sample price elasticities, though this will be added in the future. Returns four results, in order: 
    - (1) a vector of elasticities of demand for product `whichProducts[1]` with respect to `whichProducts[2]`
    - (2) the average of the price elasticity matrix across all markets
    - (3) the vector of shares at which the elasticities were calculated 
    - (4) a matrix for which each column `j` is a vector of own-price elasticities in all markets


Two helpful additional functions are `simulate_logit` and `toDataFrame`, which allows one to easily simulate data from a logit demand system with endogenous prices (and instruments for those prices) to test the `NPDemand` functions. Usage shown in `/examples`.

## Example of constructing a problem and estimating price elasticities
As described above, begin with a DataFrame which takes the following form (with three products in this example):
```jl
julia> first(df, 5)
5×12 DataFrame
 Row │ shares0    prices0  demand_instruments0  x0         shares1    prices1   demand_instruments1  x1         shares2   prices2    demand_instruments2  x2       
     │ Float64    Float64  Float64              Float64    Float64    Float64   Float64              Float64    Float64   Float64    Float64              Float64  
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 0.182075   1.02347             0.414823  0.183303   0.0828074  1.63728              0.832006  0.206252   0.358987  0.0313327            0.0682992  0.214006
   2 │ 0.316593   1.23523             0.439179  0.88636    0.178557   1.89351              0.835049  0.980622   0.130365  1.70814              0.69188    0.452745
   3 │ 0.270514   1.11647             0.414269  0.465882   0.126923   1.94381              0.87404   0.496037   0.137069  1.98788              0.908658   0.672715
   4 │ 0.115936   1.27626             0.725582  0.620084   0.225512   0.566409             0.168862  0.123414   0.344237  0.698677             0.106712   0.450302
   5 │ 0.0588137  2.38134             0.90554   0.0705893  0.176393   0.393806             0.31332   0.0280815  0.405084  0.429503             0.297496   0.907807
```
Then, we can define a `problem`. In this package, a `problem` is constructed by combining data (in `df`) and constraints. A problem can be constructed simply using `define_problem`:
```jl
bO = 2; 
exchange = [[1 2 3]] # All products are exchangeable
index_vars = ["prices", "x"]
normalization = [];
tol = 1e-5

constraints = [:exchangeability, :subs_in_group]; # Impose exchangeability and 
npd_problem = define_problem(df; 
                            exchange = exchange, 
                            index_vars = index_vars, 
                            constraints = constraints,
                            bO = bO,
                            tol = tol);
show(npd_problem)
```
Note that, for now, I do not include the raw data `df` in the `problem`. I may change this at some point, but for now this reduces the size of a problem and makes is easier to save and re-load problems later. We can then estimate the problem and calculate in-sample price elasticities with two lines. 
```jl
estimate!(problem)
elasticities, average_elast_mat, shares, all_own_elasts = price_elasticity(npd_problem, df::DataFrame; at = df[!,r"prices"], whichProducts = [1,1]);
```

## To-do
- Optional LoopVectorization.jl speed-ups
