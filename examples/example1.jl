# Example which shows how to simulate data, estimate a model, and compare estimated to true own-price elasticities

using Plots
using NPDemand

# Simulate data
J = 5; # of products
T = 2000; # # of markets
beta = -1; # price coefficient
sdxi = 0.25; # standard deviation of xi
s, p, z, x, xi  = simulate_logit(J, T, beta, sdxi);
df = toDataFrame(s,p,z,x);

# Specify estimation/model parameters
bO = 2; 
exchange = [collect(1:J)]; # exchangeability groups
# Note: exchangability can be either 0-indexed or 1-indexed. This also works:
# exchange = [[0 1], [2 3]]; 
index_vars = ["prices", "x"]
constraint_tol = 1e-5;
obj_xtol = 1e-5;
obj_ftol = 1e-5;

approximation_details = Dict(
                        :order => 5, 
                        :max_interaction => 0,
                        :sieve_type => "polynomial"
                    )

constraints = [:exchangeability, :monotone, :diagonal_dominance_all];
@elapsed begin
    npd_problem = define_problem(df; 
                            exchange = [[1;2;3;4;5]], 
                            index_vars = index_vars, 
                            constraints = constraints,
                            FE = [], 
                            bO = approximation_details[:order],
                            approximation_details = approximation_details, 
                            verbose = true
                        );

    using Turing#, Profile
    estimate!(npd_problem, 
        approximation_details = approximation_details, 
        quasi_bayes = true, 
        sampler = Turing.NUTS(), 
        n_samples = 2000, 
        skip = 5); 

    price_elasticities!(npd_problem, 
                        sieve_type = "polynomial", 
                        approximation_details = approximation_details, 
                        CI = 0.95
    );
    summarize_elasticities(npd_problem,"matrix", "quantile").Value

    @time report_constraint_violations(npd_problem;
        verbose = true,
        output = "dict", 
        approximation_details = approximation_details)

    @time smc!(npd_problem, 
        approximation_details = approximation_details)
end

true_elast_prod1 = beta .* df.prices0 .* (1 .- df.shares0);

elast_q = elasticity_quantiles(
    npd_problem, 1, 1, 
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
    n_draws = 300)
  
scatter(true_elast_prod1, own_elasticities(npd_problem)[:,1], alpha = 0.3, ylims = (-4,1), 
            legend = false, xlabel = "Truth", ylabel = "Estimate")
plot!(true_elast_prod1, true_elast_prod1, linewidth = 2, linecolor = :black)

histogram(own_elasticities(npd_problem)[:,1], 
            bins = 50, 
            alpha = 0.5, 
            # legend = false, 
            xlabel = "Own Price Elasticity", 
            ylabel = "Density", 
            title = "Posterior Distribution of Own Price Elasticity", 
            normalize = :pdf)
histogram!(true_elast_prod1, 
            bins = 50, 
            alpha = 0.5, 
            # legend = false, 
            title = "True Own Price Elasticity", 
            normalize = :pdf)

# Make copy of problem and drop all but exchangeability constraints
npd_problem2 = deepcopy(npd_problem)
update_constraints!(npd_problem2, [:exchangeability])

estimate!(npd_problem2)
price_elasticities!(npd_problem2);

scatter(true_elast_prod1, own_elasticities(npd_problem2)[:,1], alpha = 0.3, ylims = (-4,1), 
            legend = false, xlabel = "Truth", ylabel = "Estimate")
plot!(true_elast_prod1, true_elast_prod1, linewidth = 2, linecolor = :black)


report_constraint_violations(npd_problem;
        verbose = true,
        output = "dict", 
        approximation_details = approximation_details)