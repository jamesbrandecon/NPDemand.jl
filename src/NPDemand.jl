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

"""
    bern(t, order)

Returns a univariate Bernstein polynomial of order `order` constructed from array/matrix `t`
"""

include("bern.jl")


"""
    dbern(t, order)

Returns the derivative of a univariate Bernstein polynomial of order `order` constructed from array/matrix `t`
"""

include("dbern.jl")


"""
    toDataFrame(s::Matrix, p::Matrix, z::Matrix, x::Matrix = zeros(size(s)))

Returns the derivative of a univariate Bernstein polynomial of order `order` constructed from array/matrix `t`
"""

include("toDataFrame.jl")


include("solve_s_nested_flexible.jl")

"""
    simulate_logit(J,T, beta, v)

Simulates logit demand for `J` products in `T` markets, with price preference parameter `beta` and market shocks with standard deviation `v`.
"""

include("simulate_logit.jl")


include("fullInteraction.jl")
include("makeConstraint.jl")

"""
    inverse_demand(df::DataFrame)

Returns estimates of inverse demand functions by regressing prices1-pricesJ on shares1-sharesJ, instrumenting with demand_instruments1-demand_instrumentsJ
"""

include("inverse_demand.jl")

"""
    price_elasticity(inv_sigma, df::DataFrame, p_points)

Takes results of `inverse_demand()` as first argument, the data in a `DataFrame` as the second argument, and evaluates price elasticities at prices `p_points`.
"""

include("price_elasticity.jl")


"""
    hierNet(df)

Selects relevant substitutes for each product. Returns a matrix where the (i,j) element is 1 if j is a strong substitute for i, 0 otherwise. The second output of the function is a similar matrix which imposes symmetry, i.e. that if the (i,j) is 1 then so is the (j,i)
element.
"""

include("hierNet.jl")

"""
    hierNet_boot(df)

Similar to hierNet(), but runs the selection procedure for `nboot` (default is 5) times for each product and takes the union of selected substitutes.
"""

include("hierNet_boot.jl")


export hierNet, hierNet_boot, inverse_demand, price_elasticity, toDataFrame, simulate_logit
export bern, dbern

end
