module NPDemand

import LinearAlgebra
import Statistics
import Optim
import Compat
import NLopt
import DataFrames
import NLsolve
import Combinatorics
import LineSearches
import Primes
import Strided

using LinearAlgebra, Statistics, Optim, Compat, NLopt, NLsolve, DataFrames, LineSearches, Combinatorics, Primes
using ForwardDiff, Strided
using ForwardDiff: GradientConfig, Chunk
using StaticArrays

include("simulate_logit.jl")
include("toDataFrame.jl")
include("b.jl")
include("bern.jl")
include("db.jl")
include("dbern.jl")
include("prep_matrices.jl")
include("objective_functions.jl")
include("prep_inner_matrices.jl")
include("make_interactions.jl")
include("make_constraint.jl")
include("define_problem.jl")
include("add_constraint.jl")
include("inner_elasticity.jl")
include("price_elasticity.jl")
include("estimate.jl")


include("solve_s_nested_flexible.jl")

export estimate!,define_problem, show, price_elasticity, toDataFrame, simulate_logit, update_constraints!
export bern, dbern

end
