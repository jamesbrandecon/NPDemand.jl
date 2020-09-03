module NPDemand

import LinearAlgebra
import Statistics
import Optim
import Compat
import NLopt
import DataFrames
import NLsolve
import RCall

using LinearAlgebra, Statistics, Optim, Compat, NLopt, NLsolve, DataFrames, RCall

include("b.jl")
include("db.jl")
include("bern.jl")
include("dbern.jl")
include("solve_s_nested_flexible.jl")
include("simulate_logit.jl")
include("fullInteraction.jl")
include("makeConstraint.jl")
include("inverse_demand.jl")
include("price_elasticity.jl")
include("hierNet.jl")
include("hierNet_boot.jl")
#activate ~/.julia/dev/NPDemand.jl/Project.toml
end
