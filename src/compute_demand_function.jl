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
    n_draws::Union{Vector{Any}, Int} = [], 
    average_over::Vector{String} = [],
    drop_failed_solves = true)

    println("Solving nonlinear problem for counterfactual market shares.....")
    
    if "xi0" ∉ names(df)
        println("No columns named `xi' found. Setting residual demand shifters to zero")
    else
        println("Using provided residual demand shifters in counterfactual calculations")
    end

    if problem.chain ==[]
        inverted = compute_demand_function_inner(problem, df, 
            max_iter = max_iter, 
            show_trace = show_trace, 
            β = problem.results.minimizer,
            average_over = average_over)
        
        demand = inverted.solved_zero;
        if inverted.f_converged == false
            @warn("Nonlinear solver did not converge. Re-run with view_trace = true to check for convergence issues")
        end
        df[!,r"shares"] .= demand;
    else
        burn_in     = problem.sampling_details.burn_in;
        skip        = problem.sampling_details.skip;
        J           = length(problem.Xvec);
        T           = size(problem.data,1);

        demand      = zeros(size(df[!,r"shares"]))
        demand_CI   = [];
        converged_vec = [];

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

            for i in ProgressBar(1:n_draws)
                # print(".")
                sample_i    = problem.results.filtered_chain[draw_order[i],:];
                β_i = map_to_sieve(sample_i[1:sum(nbetas)], 
                                sample_i[sum(nbetas)+1:end], 
                                problem.exchange, 
                                nbetas, 
                                problem)
                inverted = compute_demand_function_inner(problem, df; 
                    max_iter = max_iter, 
                    show_trace = show_trace,
                    β = β_i,
                    average_over = average_over);
                
                demand_i    = inverted.solved_zero;
                converged_i = inverted.f_converged;

                if (converged_i) || (drop_failed_solves == false)
                    demand  = demand .+ demand_i;
                end
                push!(converged_vec, converged_i);
            end
            demand .= demand ./ n_draws; # Calculate the mean posterior elasticities
            df[!,r"shares"] .= demand; 
        else
            alpha = 1 - CI;

            nbetas      = get_nbetas(problem);
            draw_order  = sample(1:size(problem.results.filtered_chain,1), n_draws, replace = false);

            for i in ProgressBar(1:n_draws)
                # print(".")
                sample_i    = problem.results.filtered_chain[draw_order[i],:];
                β_i         = map_to_sieve(sample_i[1:sum(nbetas)], 
                                sample_i[sum(nbetas)+1:end], 
                                problem.exchange, 
                                nbetas, 
                                problem)
                inverted    = compute_demand_function_inner(problem, df, 
                                max_iter = max_iter, 
                                show_trace = show_trace, 
                                β = β_i,
                                average_over = average_over)
                
                demand_i    = inverted.solved_zero;
                converged_i = inverted.f_converged;

                if (converged_i) || (drop_failed_solves == false)
                    demand  = demand .+ demand_i;
                end

                push!(demand_CI, demand_i);
                push!(converged_vec, converged_i);
            end

            # Report share of non-converged draws if greater than 0
            if sum(converged_vec) < n_draws
                println("$(n_draws - sum(converged_vec)) of $n_draws failed to converge")
            end
            
            if drop_failed_solves == true
                indexes = findall(converged_vec .== true)
            else
                indexes = 1:n_draws
            end

            ub = [quantile(getindex.(demand_CI[indexes], t), 1-alpha/2) 
                    for t in 1:size(df,1)
                    ]
            lb = [quantile(getindex.(demand_CI[indexes], t), alpha/2) 
                    for t in 1:size(df,1)
                    ]
            
            demand .= demand ./ length(demand_CI[indexes]); # Calculate the mean posterior elasticities
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
    β = problem.results.minimizer, 
    average_over::Vector{String} = [])

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


    # Make a matrix which can be substituted in lieu of original FEmat
    # All zeros, then replace the selected FE columns/values with 1s
    gamma_length = length(γ);
    num_product_fes = "product" ∈ FE ? J - length(problem.exchange) : 0;
    FEmat = zeros(size(df,1), gamma_length - length(problem.index_vars) - num_product_fes);

    if average_over !=[] # if the user provides any FEs to average over
        try 
            @assert "product" ∉ average_over 
        catch 
            error("Cannot average over product FEs")
        end

        # Average over the elements of γ corresponding to each FE in average_over, 
        # replacing the first column with the average
        # and all others with zeros
        for fe_name in average_over
            # find the elements of γ corresponding to this FE
            inds = findall([problem.fe_param_mapping[i].name for i in eachindex(problem.fe_param_mapping)] .== fe_name);

            # We'll pack the average value of these elements into this index and replace all others with zero
            # (zeros are just in case indexing somewhere else is messed up)
            first_element = inds[1]; 

            γ[first_element] = mean(γ[inds]);
            γ[setdiff(inds, first_element)] .= 0;

            # Then we'll replace the corresponding column in FEmat with 1s to assign the average
            FEmat[:,first_element] .= 1;
        end
    else
        # If not averaging, then we need to construct a matrix that will line up with γ
        for i in axes(FEmat, 2)
            name = problem.fe_param_mapping[i].name;
            value = problem.fe_param_mapping[i].value;
            FEmat[:,i] = (df[!,name] .== value);
        end  
    end
    
    product_FEs = false;
    if "product" ∈ FE
        product_FEs = true;
    end
    s_func(shares_search) = inner_demand_function(shares_search, df, θ, γ, problem, FEmat, product_FEs)
    ans = nlsolve(s_func, 1/(2*J) .* ones(J * size(df,1)), show_trace = show_trace, iterations = max_iter)

    return (; solved_zero = reshape(ans.zero, size(df,1), J), 
            f_converged = ans.f_converged,
            x_converged = ans.x_converged,
            iterations = ans.iterations)

end

function inner_demand_function(shares_search, df, θ, γ, problem, FEmat, product_FEs)
    J = length(problem.Xvec);
    N = Int(length(shares_search)/J);
    bO = problem.bO;

    exchange = problem.exchange;
    index_vars = problem.index_vars;
    
    shares_search = reshape(shares_search, Int(N),J);
    
    # Compute sieves using the candidate shares
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
