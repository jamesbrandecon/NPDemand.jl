module NPDemand

import LinearAlgebra
import Statistics
import Optim
import Compat
import NLopt
import DataFrames
import NLsolve

using LinearAlgebra, Statistics, Optim, Compat, NLopt, NLsolve, DataFrames

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
#activate ~/.julia/dev/NPDemand.jl/Project.toml
end
