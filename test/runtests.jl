using NPDemand
using Test

@testset "NPDemand.jl" begin
    # Write your tests here.
    T = 100;
    J = 2;
    beta = -0.4;
    varxi = 0.1;
    G =10;
    s, pt, zt, xi = NPDemand.simulate_logit(J,T,beta,varxi);
    @test size(s,1) ==T

    included = ones(J,J);         # Matrix of ones and zeros, specifying which products are substitutes.
                                    # Each row corresponds to an inverse demand function.
                                    # Set element (j,k) to 1 if k is a substitute for j
                                    # Mostly helpful if you're incorporating model selection prior
                                    # to estimation. Otherwise set to ones(J).

    bernO = 2*ones(J,1);        # Order of Bernstein Polynomial
    iv=0;                       # Order of IV Polynomial = (bernO + iv)
    constrained = 1;            # Monotonicity Constraint (experience says you always want this on)
    xt = zeros(size(pt));       # No exogenous product characteristics
    trueS = 0;                    # Evaluate at true market shares or not
    own = [1,1];                # [derivative of j, with respect to price of k]
    cross = [1,2];
    inv_sigma, designs = NPDemand.inverse_demand(s, pt, zeros(size(pt[:,1])), zt, bernO, iv, J, constrained, included, nothing);
    deltas = -1*median(pt).*ones(G,J);
    p_points = range(quantile(pt[:,1],.25),stop = quantile(pt[:,1],.75),length = G);
    p_points = convert.(Float64, p_points)
    deltas[:,1] = -1*p_points;
    esep, Jmat, svec, numBadMarkets, all_own = NPDemand.price_elasticity_priceIndex(inv_sigma, s, p_points, deltas, bernO, own, included, trueS,[]);

end
