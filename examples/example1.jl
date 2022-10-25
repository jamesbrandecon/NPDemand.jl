using Plots
using NPDemand

# Simulate data
J = 4; # of products
T = 5000; # # of markets
beta = -1; # price coefficient
sdxi = 0.25; # standard deviation of xi
s, p, z, x, xi  = simulate_logit(J, T, beta, sdxi);
df = toDataFrame(s,p,z,x);

# Specify estimation/model parameters
bO = 2; 
exchange = [[1 2], [3], [4]]
index_vars = ["prices", "x"]
constraint_tol = 1e-5;
obj_tol = 1e-5;

constraints = [:exchangeability, :monotone, :diagonal_dominance_all];
npd_problem = define_problem(df; 
                            exchange = exchange, 
                            index_vars = index_vars, 
                            constraints = constraints,
                            bO = bO,
                            FE = ["dummyFE", "product"],
                            constraint_tol = constraint_tol,
                            obj_tol = obj_tol);
show(npd_problem)

# Estimate problem and plot comparison of estimated and true own-price elasticities
estimate!(npd_problem, max_iterations = 2000)
elast_prod1, avg, shares, all_own = price_elasticity(npd_problem, df; whichProducts=[1,1]);
true_elast_prod1 = beta .* df.prices0 .* (1 .- df.shares0);

scatter(true_elast_prod1, elast_prod1, alpha = 0.3, ylims = (-4,1), 
            legend = false, xlabel = "Truth", ylabel = "Estimate")
plot!(true_elast_prod1, true_elast_prod1, linewidth = 2, linecolor = :black)

# Make copy of problem and drop all but exchangeability constraints
npd_problem2 = deepcopy(npd_problem)
update_constraints!(npd_problem2, [:exchangeability])

estimate!(npd_problem2)
elast_prod1, avg, shares, all_own = price_elasticity(npd_problem2, df; whichProducts=[1,1]);
true_elast_prod1 = beta .* df.prices0 .* (1 .- df.shares0);

scatter(true_elast_prod1, elast_prod1, alpha = 0.3, ylims = (-4,1), 
            legend = false, xlabel = "Truth", ylabel = "Estimate")
plot!(true_elast_prod1, true_elast_prod1, linewidth = 2, linecolor = :black)
