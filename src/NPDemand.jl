module NPDemand

using LinearAlgebra, Statistics, Optim, Compat
using GLM
using NLopt
using GLPK
#using Convex, SCS
using RCall, DataFrames
@rlibrary ggplot2


include("simulate_logit.jl")
include("fullInteraction.jl")
include("makeConstraint.jl")

end
