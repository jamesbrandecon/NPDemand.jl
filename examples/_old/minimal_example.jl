# ------------------------------------------------------------------
# Code to simulate logit data and estimate nonparametrically
# Simplest reasonable example of nonparametric demand estimation
# Written by James Brand
# ------------------------------------------------------------------
using Statistics, NPDemand
using RCall, DataFrames
@rlibrary ggplot2

J = 2; # number of products
T =2000; # number of markets
beta = -0.4; # price coefficient
sdxi = 0.15; # standard deviation of xi

# Returns market shares, prices, instruments, and the market demand shock, respectively
s, p, z  = simulate_logit(J, T, beta, sdxi);
df = toDataFrame(s,p,z);

# Estimate demand nonparametrically
inv_sigma, designs = inverse_demand(df);

# Calculate price elasticities at realized prices and market shares
elast, jacobians = price_elasticity(inv_sigma, df, p);
true_elast = beta.*p.*(1 .- s[:,1]) # equation for own-price elasticities in logit model

# Plot kernel densities of estimated and true own-price elasticities
df2 = DataFrame(Estimate = elast, True = true_elast[:,1])
ggplot(df2, aes(x=:True))+
 geom_density(aes(x=:Estimate), color = "gray", linetype = "dashed") +
 geom_density(aes(x=:True), color = "black") + xlab("Elasticity") +
 ylab("Density")
