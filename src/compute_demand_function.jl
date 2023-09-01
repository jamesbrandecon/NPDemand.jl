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
function compute_demand_function!(problem, df; max_iter = 1000, show_trace = false)
    J = length(problem.Xvec);
    FE = problem.FE;

    try 
        @assert size(df[!,r"prices"],2) == J
        @assert size(df[!,r"shares"],2) == J
    catch 
        error("Fewer than J columns provided for either prices or shares")
    end

    find_prices = findall(problem.index_vars .== "prices");
    price_index = find_prices[1];

    # Grab estimated parameters
    β = problem.results.minimizer;
    θ = β[1:problem.design_width]
    γ = β[length(θ)+1:end]

    # Normalize own-price coefficient
    γ[price_index] = 1;

    # Fixed-effect normalizations
    for i ∈ problem.normalization
        γ[i] =0; 
    end

    # Enforce equality constraints directly
    for i∈eachindex(problem.mins)
        θ[problem.mins[i]] = θ[problem.maxs[i]]
    end
    @assert minimum(problem.Aeq * θ .== 0)
    if !isempty(problem.Aineq)
        @assert maximum(problem.Aineq * θ .< problem.constraint_tol)
    end

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
    
    # new_linear_matrices = prep_matrices(df, problem.exchange, problem.index_vars, FEmat, product_FEs, problem.bO; linear_only = true);

    # deltas = new_linear_matrices[1] * γ;
    # for j = 2:J 
    #     deltas = hcat(deltas, new_linear_matrices[j]*γ)
    # end

    println("Beginning to solve for counterfactual market shares.....")
    println("Assumes residual demand shifters set to zero")
    s_func(shares_search) = inner_demand_function(shares_search, df, θ, γ, problem, FEmat, product_FEs)
    ans = nlsolve(s_func, 1/(2*J) .* ones(J * size(df,1)), show_trace = show_trace, iterations = max_iter)

    df[!,r"shares"] .= reshape(ans.zero, size(df,1),J)
end


function inner_demand_function(shares_search, df, θ, γ, problem, FEmat, product_FEs)
    J = length(problem.Xvec);
    N = Int(length(shares_search)/J);
    bO = problem.bO;
    # all_combos = problem.combos;
    # all_redundant = problem.redundant;

    exchange = problem.exchange;
    index_vars = problem.index_vars;
    
    shares_search = reshape(shares_search, Int(N),J);
    
    df[!,r"shares"] .= shares_search;

    Xvec, Avec, Bvec, syms, combos = prep_matrices(df, exchange, index_vars, FEmat, product_FEs, bO; inner = true);

    deltas = [];
    for j = 1:J 
        if j==1
            B = Bvec[j];
            deltas = B * γ;
        else
            B = Bvec[j];
            deltas = hcat(deltas, B * γ);
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
