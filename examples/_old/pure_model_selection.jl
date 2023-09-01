# ------------------------------------------------------------------
# Code to simulate logit data select relevant substitutes
# Written by James Brand
# ------------------------------------------------------------------
using Statistics, NPDemand
using RCall, DataFrames
@rlibrary ggplot2

J=4; # number of products
T =500;
beta = -0.4; # price coefficient in utility function
sdxi = 0.15; # standard deviation of xi

S = 50; # number of simulations
G = 10; # size of grid on which to evaluate price elasticities

# ------------------------------------------------------
# Set options for Model Selection
# ------------------------------------------------------
nfolds = 5; # number of cross-validation folds
nlam = 10; # number of regularization parameters to try. Actual values chosen automatically by hierNet
strong = false; # boolean for whether or not to impose strong hierarchy constraint
# Note: "strong = true" takes much longer than "strong = false." Differences described in Bien et al (2013)
nboot = 10;

# ------------------------------------------------------
# Simulation
# ------------------------------------------------------
included_symmetric_pct = zeros(2J,2J)
included_pct = zeros(2J,2J)
for si = 1:1:S
    # Simulate demand in two groups -- each product only substitutes to J-1 others
    s, pt, zt = simulate_logit(J,T, beta, sdxi);
    s2, pt2, zt2  = simulate_logit(J,T, beta, sdxi);

    s = [s s2];
    s = s ./ 2;
    pt = [pt pt2];
    zt = [zt zt2];
    df = toDataFrame(s,pt,zt);

    # hierNet_boot() Returns two matrices: one, the "raw" selected model, and another
    #   which imposes symmetry. I.e. if j is a substute for k, then k is
    #   a substitute for j as well (this can drastically increase the # parameters
    #    to estimate when s has many columns)
    included, included_symmetric = hierNet_boot(df; nfolds = nfolds, nlam = nlam, strong = strong, nboot = nboot);

    included_pct[:,:] += included./S;
    included_symmetric_pct[:,:] += included_symmetric./S;
end

# DataFrames where (i,j) shows the fraction of simulation runs in which
# product (j) was selected as a substitute for product (i)
# Perfect selection in this example would be a block matrix of 2 (JxJ) matrixes of ones on the diagonal

basic_selection = DataFrame(round.(included_pct,digits=2))
symmetric_selection = DataFrame(round.(included_symmetric_pct,digits=2))
