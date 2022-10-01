using ForwardDiff, Statistics, Plots
using LinearAlgebra, Statistics, Optim, DataFrames, LineSearches, Combinatorics
using Primes


J = 3; # of products
T = 5000; # # of markets
beta = -1; # price coefficient
sdxi = 0.25; # standard deviation of xi
s, p, z, x, xi  = simulate_logit(J, T, beta, sdxi);
df = toDataFrame(s,p,z,x);


bO = 2; 
exchange = [[1 2], [3 4]]
index_vars = ["prices", "x"]
normalization = [];
tol = 1e-5

constraints = [:exchangeability, :subs_in_group];
npd_problem = define_problem(df; 
                            exchange = exchange, 
                            index_vars = index_vars, 
                            constraints = constraints,
                            bO = bO,
                            tol = tol);
show(npd_problem)

estimate!(npd_problem)
elast_prod1, avg, svec, all_own = price_elasticity(npd_problem, df; whichProducts=[1,1]);
true_elast_prod1 = beta .* df.prices0 .* (1 .- df.shares0);

scatter(true_elast_prod1, elast_prod1, alpha = 0.3, ylims = (-4,1), 
            legend = false, xlabel = "Truth", ylabel = "Estimate")
plot!(true_elast_prod1, true_elast_prod1, linewidth = 2, linecolor = :black)

npd_problem2 = +(npd_problem, :monotone);
estimate!(npd_problem2)
elast_prod1, avg, svec, all_own = price_elasticity2(npd_problem2, df; whichProducts=[1,1]);
true_elast_prod1 = beta .* df.prices0 .* (1 .- df.shares0);

scatter(true_elast_prod1, elast_prod1, alpha = 0.3, ylims = (-4,1), 
            legend = false, xlabel = "Truth", ylabel = "Estimate")
plot!(true_elast_prod1, true_elast_prod1, linewidth = 2, linecolor = :black)
