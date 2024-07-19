"""
    compute_demand_function!(problem, df; max_iter = 1000, show_trace = false)

`compute_demand_function!` estimates the demand function/curve using NPD estimates calculated via estimate!.

The function takes in an estimated problem::NPDProblem and a dataframe with counterfactual values of the 
covariates in the utility function. One must specify all fields that were used in estimation (including shares). The function
will change the values of df[!,r"shares"] to take on the value of the estimated demand function.

Options: 
- `max_iter`: controls the number of iterations for the nonlinear solver calculating market shares. Default is 1000 but well-estimated problems should converge faster.
- `show_trace`: if `true`, Optim will print the trace for each iteration of the nonlinear solver. 
-  `compute_own_elasticities`: NOT yet implemented-- if `true`, will also generate columns called `own_elast` which will include estimated own-price elasticities at all counterfactual points.
"""
function compute_demand_function!(problem, df; 
    max_iter = 1000, 
    show_trace = false, 
    CI::Union{Vector{Any}, Real} = [], 
    n_draws::Union{Vector{Any}, Int} = [])

    println("Solving nonlinear problem for counterfactual market shares.....")
    
    if "xi0" ∉ names(df)
        println("No columns named `xi' found. Setting residual demand shifters to zero")
    else
        println("Using provided residual demand shifters in counterfactual calculations")
    end

    if problem.chain ==[]
        demand = compute_demand_function_inner(problem, df, max_iter = max_iter, show_trace = show_trace, β = problem.results.minimizer);
        df[!,r"shares"] .= demand;
    else
        burn_in     = problem.sampling_details.burn_in;
        skip        = problem.sampling_details.skip;
        J           = length(problem.Xvec);
        T           = size(problem.data,1);

        demand      = zeros(size(df[!,r"shares"]))
        demand_CI   = [];

        if n_draws == []
            n_draws = length(burn_in+1:skip:size(problem.chain,1))
        end
        try 
            @assert ((n_draws > 0) & (n_draws <= length(burn_in+1:skip:size(problem.chain,1))))
        catch 
            error("`n_draws` must be greater than 0 and weakly less than the total number of draws in the final chain")
        end
        println("Using $n_draws quasi-Bayes estimates. This may take a few minutes....")
        if CI == []
            nbetas = get_nbetas(problem);
            draw_order  = sample(1:size(problem.results.filtered_chain,1), n_draws, replace = false);

            for i in 1:n_draws
                print(".")
                sample_i    = problem.results.filtered_chain[draw_order[i],:];
                β_i = map_to_sieve(sample_i[1:sum(nbetas)], 
                                sample_i[sum(nbetas)+1:end], 
                                problem.exchange, 
                                nbetas, 
                                problem)
                demand_i = compute_demand_function_inner(problem, df; 
                max_iter = max_iter, 
                show_trace = show_trace,
                β = β_i)
                demand = demand .+ demand_i;
            end
            demand .= demand ./ n_draws; # Calculate the mean posterior elasticities
            df[!,r"shares"] .= demand; 
        else
            alpha = 1 - CI;

            nbetas      = get_nbetas(problem);
            draw_order  = sample(1:size(problem.results.filtered_chain,1), n_draws, replace = false);

            for i in 1:n_draws
                print(".")
                sample_i    = problem.results.filtered_chain[draw_order[i],:];
                β_i         = map_to_sieve(sample_i[1:sum(nbetas)], 
                                sample_i[sum(nbetas)+1:end], 
                                problem.exchange, 
                                nbetas, 
                                problem)
                demand_i    = compute_demand_function_inner(problem, df, 
                                max_iter = max_iter, 
                                show_trace = show_trace, 
                                β = β_i)
                
                demand      = demand .+ demand_i;
                push!(demand_CI, demand_i);
            end
            
            ub = [quantile(getindex.(demand_CI, t), 1-alpha/2) 
                    for t in 1:size(df,1)
                    ]
            lb = [quantile(getindex.(demand_CI, t), alpha/2) 
                    for t in 1:size(df,1)
                    ]
            demand .= demand ./ n_draws; # Calculate the mean posterior elasticities
            df[!,r"shares"] .= demand;
            
            for i in 0:(J-1)
                df[!,"shares$(i)_lb"] = lb
                df[!,"shares$(i)_ub"] = ub
            end
        end
    end
end

function compute_demand_function_inner(problem, df; 
    max_iter = 1000, show_trace = false,
    β = problem.results.minimizer)

    J = length(problem.Xvec);
    FE = problem.FE;

    try 
        @assert size(df[!,r"prices"],2) == J
        @assert size(df[!,r"shares"],2) == J
    catch 
        error("Fewer than J columns provided for either prices or shares")
    end

    # Grab estimated parameters
    θ = β[1:problem.design_width]
    γ = β[length(θ)+1:end]

    # Reshape FEs into matrix of dummy variables
    FEmat = [];
    if FE!=[]
        FEmat = [];
        for f ∈ FE
            if f != "product"
                unique_vals = unique(df[!,f]);
                unique_vals = unique_vals[1:end-1]; # Drop one category per FE dimension
                for fi ∈ unique_vals
                    if (f==FE[1]) & (fi==unique_vals[1])
                        FEmat = reshape((df[!,f] .== fi), size(df,1),1)
                    else
                        FEmat = hcat(FEmat, reshape((df[!,f] .== fi), size(df,1),1))
                    end
                end
            end
        end
    end    
    
    product_FEs = false;
    if "product" ∈ FE
        product_FEs = true;
    end
    s_func(shares_search) = inner_demand_function(shares_search, df, θ, γ, problem, FEmat, product_FEs)
    ans = nlsolve(s_func, 1/(2*J) .* ones(J * size(df,1)), show_trace = show_trace, iterations = max_iter)

    # df[!,r"shares"] .= reshape(ans.zero, size(df,1),J)
    reshape(ans.zero, size(df,1),J)
end

function inner_demand_function(shares_search, df, θ, γ, problem, FEmat, product_FEs)
    J = length(problem.Xvec);
    N = Int(length(shares_search)/J);
    bO = problem.bO;

    exchange = problem.exchange;
    index_vars = problem.index_vars;
    
    shares_search = reshape(shares_search, Int(N),J);
    
    df[!,r"shares"] .= shares_search;

    Xvec, ~, Bvec, ~, ~ = prep_matrices(df, exchange, index_vars, FEmat, product_FEs, bO; inner = true);

    deltas = [];
    for j = 1:J 
        if j==1
            B = Bvec[j];
            deltas = B * γ;
        else
            B = Bvec[j];
            deltas = hcat(deltas, B * γ);
        end

        if "xi$(j-1)" ∈ names(df)
            deltas[:,j] = deltas[:,j] .+ df[!,"xi$(j-1)"];
        end
    end

    theta_count = 0;
    objective = [];
    for j=1:J
        design_width_j = size(Xvec[j],2);
        j_ind = (theta_count + 1):(theta_count + design_width_j);
        @views θ_j = θ[j_ind]
        if j==1
            objective = deltas[:,j] - Xvec[j] * θ_j;
        else 
            objective = hcat(objective, deltas[:,j] - Xvec[j] * θ_j);
        end
        theta_count += design_width_j;
    end

    return dropdims(reshape(objective, N*J,1), dims=2)
end
