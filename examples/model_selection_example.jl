# ------------------------------------------------------------------
# Code to simulate logit data and estimate nonparametrically
# Plots estimates of own and cross-price elasticities
# Written by James Brand
# ------------------------------------------------------------------
using Statistics, NPDemand
using RCall, DataFrames
@rlibrary ggplot2

J=4; # number of products
T =2000;
beta = -0.4; # price coefficient in utility function
sdxi = 0.15; # standard deviation of xi

S = 20; # number of simulations
G = 10; # size of grid on which to evaluate price elasticities
elast_own = zeros(S,G);
elast_cross = zeros(S,G);
elast_own_dist = zeros(S,T);
elastTrue = zeros(S,G);

s, pt, zt, xi = NPDemand.simulate_logit(J,T, beta, sdxi);

p_points = range(quantile(pt[:,1],.25),stop = quantile(pt[:,1],.75),length = G);
p_points = convert.(Float64, p_points)

# ------------------------------------------------------
# Set options for estimation and elasticity calculation
# ------------------------------------------------------
bernO = 2*ones(2J,1);        # Order of Bernstein Polynomial
iv=0;                       # Order of IV Polynomial = (bernO + iv)
trueS = 0;                    # Evaluate at true market shares or not
own = [1,1];                # [derivative of j, with respect to price of k]

nfolds = 5; # number of cross-validation folds
nlam = 10; # number of regularization parameters to try. Actual values chosen automatically by hierNet
strong = true; # boolean for whether or not to impose strong hierarchy constraint
# Note: "strong = true" takes much longer than "strong = false."
nboot = 1; # number of bootstrapped samples to run hierarchical lasso on

# ------------------------------------------------------
# Simulation
# ------------------------------------------------------
included_symmetric_pct = zeros(2J,2J)
included_pct = zeros(2J,2J)
for si = 1:1:S
    # Simulate demand in two groups -- each product only substitutes to J-1 others
    s, pt, zt = NPDemand.simulate_logit(J,T, beta, sdxi);
    s2, pt2, zt2  = NPDemand.simulate_logit(J,T, beta, sdxi);

    s = [s s2];
    s = s ./ 2;
    pt = [pt pt2];
    zt = [zt zt2];
    df = NPDemand.toDataFrame(s,pt,zt);

    # hierNet() Returns two matrices: one, the "raw" selected model, and another
    #   which imposes symmetry. I.e. if j is a substute for k, then k is
    #   a substitute for j as well (this can drastically increase the # parameters
    #    to estimate when s has many columns)
    included, included_symmetric = NPDemand.hierNet_boot(df; nfolds = nfolds, nlam, false, nboot);

    # Estimate demand nonparametrically
        # If you want to include an additional covariate in all demand
        # functions, add an additional argument "marketvars" after included. If it is an
        # additional product characteristic, marketvars should be T x J
    inv_sigma, designs = NPDemand.inverse_demand(df, bernO, iv, 2J, constrained, included_symmetric, nothing);

    # Calculate price elasticities
    deltas = -1*median(pt).*ones(G,2J);
    deltas[:,1] = -1*p_points;
    elast, Jacobians, share_vec = NPDemand.price_elasticity(inv_sigma, s, p_points, deltas, bernO, own, included_symmetric, trueS,[]);
    trueelast = beta.*p_points.*(1 .- 2 .* share_vec[:,1])

    elast_own[si,:] = elast;
    elastTrue[si,:] = trueelast;
    included_pct[:,:] += included./S; # summarizing selection patterns. See pure_model_selection.jl
    included_symmetric_pct[:,:] += included_symmetric./S;
end

elast025 = zeros(G,1)
elast50 = zeros(G,1)
elast975 = zeros(G,1)
for i = 1:G
    elast025[i] = quantile(elast_own[:,i], 0.1)
    elast50[i] = quantile(elast_own[:,i], 0.5)
    elast975[i] = quantile(elast_own[:,i], 0.9)
end
elast025 = dropdims(elast025,dims = 2)
elast50 = dropdims(elast50,dims = 2)
elast975 = dropdims(elast975,dims = 2)

df = DataFrame(p = p_points, e025 = elast025, e50 = elast50, e975 = elast975)
ggplot(df, aes(x=:p, y=:e50)) + geom_line() + geom_line(aes(y=:e975), color = "gray", linetype = "dashed") +
    geom_line(aes(y=:e025), color = "gray", linetype = "dashed") +
    xlab("Price") + ylab("Own-Elasticity") + theme_light()
