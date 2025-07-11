# Example which shows how to simulate data, estimate a model, and compare estimated to true own-price elasticities

using Plots
using NPDemand
using DataFrames

# Simulate data
J = 4; # of products
T = 10000; # # of markets
beta = -1; # price coefficient
sdxi = 0.01; # standard deviation of xi
s, p, z, x, xi  = simulate_logit(J, T, beta, sdxi);
df = toDataFrame(s,p,z,x);

# Specify estimation/model parameters
exchange = [[1;2;3;4]]; # exchangeability groups
# Note: exchangability can be either 0-indexed or 1-indexed. This also works:
# exchange = [[0 1], [2 3]]; 
index_vars = ["prices", "x"]
constraint_tol = 1e-5;
obj_xtol = 1e-5;
obj_ftol = 1e-5;

constraints = [:exchangeability];

npd_problem_bernstein_full = define_problem(df; 
                        exchange = exchange, 
                        index_vars = index_vars, 
                        constraints = constraints,
                        FE = [], 
                        approximation_details = Dict(
                            :order => 2,  
                            :max_interaction => Inf, 
                            :sieve_type => "bernstein", 
                            :tensor => true # NOTE: tensor overrides max_interaction
                        ), 
                        verbose = true
                        );

estimate!(npd_problem_bernstein_full, quasi_bayes = false); 
price_elasticities!(npd_problem_bernstein_full);



npd_problem_bernstein_limited = define_problem(df; 
                        exchange = exchange, 
                        index_vars = index_vars, 
                        constraints = constraints,
                        FE = [], 
                        approximation_details = Dict(
                            :order => 2,  
                            :max_interaction => 0, 
                            :sieve_type => "bernstein", 
                            :tensor => false
                        ), 
                        verbose = true
                        );

estimate!(npd_problem_bernstein_limited, quasi_bayes = false); 
price_elasticities!(npd_problem_bernstein_limited);



npd_problem_polynomial_full = define_problem(df; 
                        exchange = exchange, 
                        index_vars = index_vars, 
                        constraints = constraints,
                        FE = [], 
                        approximation_details = Dict(
                            :order => 2,  
                            :max_interaction => 2, 
                            :sieve_type => "polynomial", 
                            :tensor => true
                        ), 
                        verbose = true
                        );

estimate!(npd_problem_polynomial_full, quasi_bayes = false); 
price_elasticities!(npd_problem_polynomial_full);


npd_problem_polynomial_limited = define_problem(df; 
                        exchange = exchange, 
                        index_vars = index_vars, 
                        constraints = constraints,
                        FE = [], 
                        approximation_details = Dict(
                            :order => 2,  
                            :max_interaction => 0, 
                            :sieve_type => "polynomial", 
                            :tensor => false
                        ), 
                        verbose = true
                        );

estimate!(npd_problem_polynomial_limited, quasi_bayes = false); 
price_elasticities!(npd_problem_polynomial_limited);


# --------------------------------------------------------------
# Compare elasticities from Bernstein and polynomial sieves
# --------------------------------------------------------------

@info "Median Elasticities, Bernstein polynomial"
print("$(npd_problem_bernstein_full.design_width) total parameters, $(sum(NPDemand.get_nbetas(npd_problem_bernstein_full))) unique")
display(summarize_elasticities(npd_problem_bernstein_full, "matrix", "quantile", q=0.5).Value)

@info "Median Elasticities, Bernstein polynomial"
print("$(npd_problem_bernstein_limited.design_width) total parameters, $(sum(NPDemand.get_nbetas(npd_problem_bernstein_limited))) unique")
display(summarize_elasticities(npd_problem_bernstein_limited, "matrix", "quantile", q=0.5).Value)

@info "Median Elasticities, regular polynomial"
print("$(npd_problem_polynomial_full.design_width) total parameters, $(sum(NPDemand.get_nbetas(npd_problem_polynomial_full))) unique")
display(summarize_elasticities(npd_problem_polynomial_full, "matrix", "quantile", q=0.5).Value)

@info "Median Elasticities, regular polynomial"
print("$(npd_problem_polynomial_limited.design_width) total parameters, $(sum(NPDemand.get_nbetas(npd_problem_polynomial_limited))) unique")
display(summarize_elasticities(npd_problem_polynomial_limited, "matrix", "quantile", q=0.5).Value)