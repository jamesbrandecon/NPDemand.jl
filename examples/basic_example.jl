# ------------------------------------------------------------------
# Code to simulate logit data and estimate nonparametrically
# Plots estimates of own and cross-price elasticities
# Written by James Brand
# ------------------------------------------------------------------
using Statistics, NPDemand
using RCall, DataFrames
@rlibrary ggplot2

J=2; # # products
T =2000;
beta = -0.4; # price coefficient
sdxi = 0.15; # standard deviation of xi

S = 100;
G = 10; # size of grid on which to evaluate price elasticities
esep_own = zeros(S,G);
esep_cross = zeros(S,G);
esep_own_dist = zeros(S,T);
esepTrue = zeros(S,G);

s, p, z, xi = NPDemand.simulate_logit(J,T, beta, sdxi);
p_points = range(quantile(p[:,1],.25),stop = quantile(p[:,1],.75),length = G);
p_points = convert.(Float64, p_points)

# ------------------------------------------------------
# Set options for estimation and elasticity calculation
# ------------------------------------------------------
included = ones(J,J);         # Matrix of ones and zeros, specifying which products are substitutes.
                                # Each row corresponds to an inverse demand function.
                                # Set element (j,k) to 1 if k is a substitute for j
                                # Mostly helpful if you're incorporating model selection prior
                                # to estimation. Otherwise set to ones(J).

bernO = 2*ones(J,1);        # Order of Bernstein Polynomial
iv=0;                       # Order of IV Polynomial = (bernO + iv)
constrained = 0;            # Monotonicity Constraint (experience says this can help but much slower in current build)
x = zeros(size(p));       # No exogenous product characteristics
trueS = 0;                    # Evaluate at true market shares or not
own = [1,1];                # [derivative of j, with respect to price of k]
cross = [1,2];

# ------------------------------------------------------
# Simulation
# ------------------------------------------------------
for si = 1:1:S
    # Returns market shares, prices, instruments, and the market demand shock, respectively
    s, p, z  = NPDemand.simulate_logit(J,T, beta, sdxi);

    # Estimate demand nonparametrically
        # If you want to include an additional covariate in all demand
        # functions, add an additional argument "marketvars" after included. If it is an
        # additional product characteristic, marketvars should be T x J
    inv_sigma, designs = NPDemand.inverse_demand(s, p, x, z, bernO, iv, J, constrained, included, nothing);

    # Calculate price elasticities
    deltas = -1*median(p).*ones(G,J);
    deltas[:,1] = -1*p_points;

    esep, Jacobians, share_vec = NPDemand.price_elasticity_priceIndex(inv_sigma, s, p_points, deltas, bernO, own, included, trueS,[])
    trueEsep = beta.*p_points.*(1 .- share_vec[:,1])

    esep_own[si,:] = esep;
    esepTrue[si,:] = trueEsep;
end

# Now generate plots using RCall and ggplot
esep025 = zeros(G,1)
esep50 = zeros(G,1)
esep975 = zeros(G,1)
for i = 1:G
    esep025[i] = quantile(esep_own[:,i], 0.25)
    esep50[i] = quantile(esep_own[:,i], 0.5)
    esep975[i] = quantile(esep_own[:,i], 0.975)
end
esep025 = dropdims(esep025,dims = 2)
esep50 = dropdims(esep50,dims = 2)
esep975 = dropdims(esep975,dims = 2)

df = DataFrame(p = p_points, e025 = esep025, e50 = esep50, e975 = esep975)

# Plot 95% interval of price elasticities along interquartile range
ggplot(df, aes(x=:p, y=:e50)) + geom_line() + geom_line(aes(y=:e975), color = "gray", linetype = "dashed") +
    geom_line(aes(y=:e025), color = "gray", linetype = "dashed") +
    xlab("Price") + ylab("Own-Elasticity") + theme_light()

# ------------------------------------------------------
# ------------------------------------------------------
# You can also calculate price elasticities at the realized prices and market shares
        # Jacobians2 is particularly helpful here. This is an array, each element of which is the Jacobian
        # of demand with respect to prices. This can be used to calculate markups under many models
        # of competition.
s, p, z  = NPDemand.simulate_logit(J,T, beta, sdxi);
inv_sigma, designs = NPDemand.inverse_demand(s, p, x, z, bernO, iv, J, constrained, included, nothing);
trueS = 1;
inv_sigma, designs = NPDemand.inverse_demand(s, p, x, z, bernO, iv, J, constrained, included, nothing);
esep2, Jacobians2 = NPDemand.price_elasticity_priceIndex(inv_sigma, s, p, -1 .*p, bernO, own, included, trueS,[]);

df2 = DataFrame(Estimate = esep2, True = beta.*p[:,1].*(1 .- s[:,1]))
ggplot(df2, aes(x=:True))+
 geom_density(aes(x=:Estimate), color = "gray", linetype = "dashed") +
 geom_density(aes(x=:True), color = "black") + xlab("Elasticity") +
 ylab("Density")

# Note that the estimated density has a much longer tail than the truth. This is
    # largely due to over-fitting the inverse demand function in a large sample.
    # in small J settings, increasing the polynomial order helps. In very large T samples
    # you could also sample split (which is recommended anyway if you add model
    # selection as a pre-step)
