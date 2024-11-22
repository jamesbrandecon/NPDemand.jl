module NPDemand

using StaticArrays, JuMP, Ipopt, OSQP
using LinearAlgebra, Statistics, Optim, Compat, NLopt, NLsolve, DataFrames, LineSearches, Combinatorics, Primes
using ForwardDiff, Strided
using ForwardDiff: GradientConfig, Chunk
using StatsBase
using StatsBase: weights
using Random

using Zygote
using Turing, AdvancedMH
using Roots: find_zero, Bisection
using ProgressBars: ProgressBar

using Printf

include("simulate_logit.jl")
include("toDataFrame.jl")
include("b.jl")
include("bern.jl")
include("db.jl")
include("dbern.jl")
include("prep_matrices.jl")
include("define_problem.jl")
include("objective_functions.jl")
# include("prep_inner_matrices.jl")
include("make_interactions.jl")
include("make_constraint.jl")
include("add_constraint.jl")
# include("inner_elasticity.jl")
include("price_elasticity.jl")
include("estimate.jl")
include("compute_demand_function.jl")
include("quasibayes.jl")
include("constraint_checks.jl")


# include("solve_s_nested_flexible.jl")

export estimate!, define_problem, show, price_elasticities!, toDataFrame, simulate_logit, update_constraints!
export bern, dbern, compute_demand_function!, summarize_elasticities, own_elasticities, are_constraints_satisfied
export elasticity_quantiles, report_constraint_violations, smc!

end
