using NPDemand
using DataFrames
using Turing
using Plots

function simulate_comp2g(T, beta, gamma_own, gamma_other, sdxi, J1, J2)
    # J -- num of goods
    # T -- num of markets
    # beta -- price coefficient
    # gamma_own -- coefficient on index of other goods within the own category
    # gamma_other -- coefficient on index of other goods in the other category
    # sdxi -- standard deviation of xi
    
    # z = 0.9 .* rand(T,J) .+ 0.05;
    # xi = randn(T,J).*sdxi;
    # p = 2 .*(z .+ rand(T,J)*0.1).+xi;
    J = J1+J2;
    z = rand(T,J);
    xi = randn(T,J).*sdxi;
    p = max.(2*z .+ 0.1*rand(T,J) .+ xi, 1e-2);
    
    x = 2 .* rand(T,J);
    # x = rand(T,J);
    delta = beta*p + x + xi;
    q = zeros(T,J);
    for j = 1:1:J1
        index_own = setdiff(1:J1,j);
        index_other = collect(J1+1:J);
        q[:,j] = exp.(delta[:,j] + gamma_own*mean(delta[:,index_own],dims=2) + gamma_other*mean(delta[:,index_other],dims=2));
    end
    temp_J2 = collect(J1+1:J);
    for j = 1:length(temp_J2)
        index_own = setdiff(J1+1:J,temp_J2[j]);
        index_other = collect(1:J1);
        q[:,temp_J2[j]] = exp.(delta[:,temp_J2[j]] + gamma_own*mean(delta[:,index_own],dims=2) + gamma_other*mean(delta[:,index_other],dims=2));
    end
    s = q./(maximum(sum(q,dims=2))*1.1);

    df = toDataFrame(s,p,z,x);
    return df
end

# Simulate logit data
J = 4; # of products
T = 500; # # of markets
beta = -1; # price coefficient
sdxi = 1; # standard deviation of xi
s, p, z, x, xi  = simulate_logit(J, T, beta, sdxi);
df = toDataFrame(s,p,z,x);
exchange = [collect(1:J)] 
constraints = [:exchangeability, :monotone, :all_substitutes, :diagonal_dominance_all]; 

# simulate complements data
J = 4; # of products
T = 500; # # of markets
beta = -1; # price coefficient
sdxi = 0.1; # standard deviation of xi
gamma_own = 0.25
gamma_other = 0.25
df = simulate_comp2g(T, beta, -abs(gamma_own), abs(gamma_other), sdxi, Int(J/2), Int(J/2));
exchange = [collect(1:Int(J/2)), collect(Int(J/2)+1:J)] 
constraints = [:exchangeability, :monotone, :subs_in_group, :complements_across_group]; 

approximation_details = Dict(
    :order => 3, 
    :max_interaction => Inf, 
    :sieve_type => "bernstein", # "bernstein" or "polynomial"
    :tensor => true # NOTE: tensor overrides max_interaction
);

# count parameters
NPDemand.count_params(; n_products=J, exchange=exchange, approximation_details=approximation_details)

# define problem
npd_problem = define_problem(df; 
                        exchange = exchange, 
                        index_vars = ["prices", "x"], 
                        constraints = constraints,
                        FE = [], 
                        approximation_details = approximation_details, 
                        verbose = true
                    );

# estimate baseline model
@time estimate!(npd_problem, 
    quasi_bayes = true, 
    sampler = HMC(0.01, 10), 
    # sampler = NUTS(1000, 0.65), 
    n_samples = 1000, 
    skip = 1); 

# trace plots
Plots.plot(npd_problem.results.filtered_chain[:,1:4])

# check constraint violations
report_constraint_violations(npd_problem; constraints=[:monotone, :all_substitutes, :diagonal_dominance_all])

price_elasticities!(npd_problem)
emat = summarize_elasticities(npd_problem, "matrix", "quantile"; q=0.5, integrate=true, CI=0.95)
emat.Posterior_Median

# run smc
@time smc!(npd_problem,
    seed = 1024,
    burn_in = 0.2,
    skip = 1,
    mh_steps = 20,
    max_iter = 10,
    step = 0.01,
    # max_penalty = 10,
    ess_threshold = 500,
    smc_method = :adaptive
)