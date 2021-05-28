# ------------------------------------------------------------------
# Code to simulate logit data and estimate nonparametrically
# Plots estimates of own and cross-price elasticities
# Written by James Brand
# ------------------------------------------------------------------
using Statistics, NPDemand
using RCall, DataFrames
@rlibrary ggplot2

J=2; # # of products
T =2000; # # of markets
beta = -0.4; # price coefficient
sdxi = 0.15; # standard deviation of xi

S = 200;
G = 20; # size of grid on which to evaluate price elasticities
elast_own = zeros(S,G);
elast_cross = zeros(S,G);
elast_own_dist = zeros(S,T);
elastTrue = zeros(S,G);

s, p, z, xi = simulate_logit(J,T, beta, sdxi);
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
constrained = false;            # Monotonicity Constraint (experience says this can help but much slower in current build)
x = zeros(size(p));       # No exogenous product characteristics
trueS = true;                    # Evaluate at true market shares or not
own = [1,1];                # [derivative of j, with respect to price of k]
cross = [1,2];

# ------------------------------------------------------
# Simulation
# ------------------------------------------------------
for si = 1:1:S
    # Returns market shares, prices, instruments, and the market demand shock, respectively
    s, p, z  = simulate_logit(J,T, beta, sdxi);
    df = toDataFrame(s,p,z);
    # Estimate demand nonparametrically
        # If you want to include an additional covariate in all demand
        # functions, add an additional argument "marketvars" after included. If it is an
        # additional product characteristic, marketvars should be T x J
    inv_sigma, designs = inverse_demand(df; included = included);

    # Calculate price elasticities
    prices = median(p).*ones(G,J);
    prices[:,1] = p_points;

    elast, Jacobians, share_vec = price_elasticity(inv_sigma, df, prices; included = included,
        whichProducts = own, trueS = false)

    trueelast = beta.*p_points.*(1 .-  share_vec[:,1])

    elast_own[si,:] = elast;
    elastTrue[si,:] = trueelast;
end

# Now generate plots using RCall and ggplot
elast025 = zeros(G,1)
elast50 = zeros(G,1)
elast975 = zeros(G,1)
for i = 1:G
    elast025[i] = quantile(elast_own[:,i], 0.25)
    elast50[i] = quantile(elast_own[:,i], 0.5)
    elast975[i] = quantile(elast_own[:,i], 0.975)
end
elast025 = dropdims(elast025,dims = 2)
elast50 = dropdims(elast50,dims = 2)
elast975 = dropdims(elast975,dims = 2)

df = DataFrame(p = p_points, e025 = elast025, e50 = elast50, e975 = elast975)

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
s, p, z  = simulate_logit(J,T, beta, sdxi);
df = toDataFrame(s,p,z);
inv_sigma, designs = inverse_demand(df; included = included);
elast2, Jacobians2 = price_elasticity(inv_sigma, df, p, included = included; deltas = -1 .* p);

df2 = DataFrame(Estimate = elast2, True = beta.*p[:,1].*(1 .- s[:,1]))
ggplot(df2, aes(x=:True))+
 geom_density(aes(x=:Estimate), color = "gray", linetype = "dashed") +
 geom_density(aes(x=:True), color = "black") + xlab("Elasticity") +
 ylab("Density")
