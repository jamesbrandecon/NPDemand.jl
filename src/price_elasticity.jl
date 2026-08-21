"""
    price_elasticities!(problem; 
        CI::Union{Vector{Any}, Real} = [], 
        n_draws::Union{Vector{Any}, Int} = [])

Takes the solved `problem` as first argument, a `DataFrame` as the second argument, and evaluates all price elasticities in-sample. 
Currently does not calculate out-of-sample price elasticities. For this, use the function `compute_demand_function!`. 

Results of this function are stored as a `DataFrame` in problem.all_elasticities. Results can be summarized by hand or using the `summarize_elasticities` function. 
We also store the Jacobian of the demand function with respect to prices, which can be used to calculate other quantities of interest.
"""
function price_elasticities!(problem; stat="median")

    statfun = stat == "mean" ? mean :
              stat == "median" ? median :
              error("Unsupported stat: $stat (use \"mean\" or \"median\")")

    max_interaction = problem.approximation_details[:max_interaction]
    sieve_type = problem.approximation_details[:sieve_type]

    if problem.sampling_details ==[]

        elast = price_elasticities_inner(
            problem; 
            β = problem.estimates.minimizer,
            sieve_type = sieve_type, 
            max_interaction = max_interaction)
        problem.all_elasticities = elast.all_elast_mat
        problem.all_jacobians = elast.Jmat

    else

        J = size(problem.data[:,r"shares"],2)
        T = size(problem.data,1)
        ndraws = size(problem.chain_params, 1)
        nbetas = get_nbetas(problem)
        elast_draws = Array{Float64}(undef, J, J, T, ndraws)
        jacob_draws = Array{Float64}(undef, J, J, T, ndraws)

        for i in ProgressBar(1:ndraws)

            β_i = map_to_sieve(
                problem.chain_params[i,1:sum(nbetas)],
                problem.chain_params[i,(sum(nbetas)+1):end],
                problem.exchange,
                nbetas,
                problem
            )
            elast = price_elasticities_inner(
                problem; 
                β = β_i, 
                sieve_type = sieve_type, 
                max_interaction = max_interaction
            )
            elast_i = elast.all_elast_mat
            jacob_i = elast.Jmat
            elast_draws[:,:,:,i] = reshape(reduce(hcat, elast_i), J, J, T)
            jacob_draws[:,:,:,i] = reshape(reduce(hcat, jacob_i), J, J, T)
        
        end

        problem.all_elasticities = [statfun(elast_draws[:,:,t,:]; dims=3)[:,:,1] for t in 1:T]
        problem.all_jacobians = [statfun(jacob_draws[:,:,t,:]; dims=3)[:,:,1] for t in 1:T]

    end

end

function price_elasticities_inner(npd_problem; 
    β = npd_problem.estimates.minimizer, 
    sieve_type = "bernstein", 
    max_interaction = 1)

    # Add a market ID column if not already present
    if !(:market_ids ∈ names(npd_problem.data))
        npd_problem.data[!,"market_ids"] .= 1:size(npd_problem.data,1);
    end

    df = npd_problem.data;
    at = df[!,r"prices"];
    
    # Unpack results
    θ = β[1:npd_problem.design_width]
    γ = β[length(θ)+1:end]

    X = npd_problem.Xvec;
    B = npd_problem.Bvec;
    bO = npd_problem.bO;
    exchange = npd_problem.exchange;
    nbetas = size.(npd_problem.Xvec,2);

    # Check inputs
    if (at!=[]) & (size(at,2) != size(df[:,r"prices"],2))
        error("Argument `at` must be a matrix of prices with J columns if provided")
    end
    if typeof(at) ==DataFrame
        at = Matrix(at);
    end

    s = Matrix(df[:, r"shares"]);
    J = size(s,2);
    bernO = convert.(Integer, bO);

    # Construct index 
        # Note: Currently evaluates at realized prices. Have to edit to allow for counterfactual prices
    index = zeros(size(X[1],1),J)
    for j = 1:J
        index[:,j] = B[j]*γ;
    end
    
    # Share Jacobian
    tempmats = []
    dsids = zeros(J,J,size(index,1)) # initialize matrix of ∂s^{-1}/∂s
    for j1 = 1:J
        θ_j1, permuted_shares, permutations = get_params_one_equation(j1; exchange = exchange, s = s, θ = θ, nbetas = nbetas)
        
        for j2 = 1:J 
            # tempmat_s = calc_derivative_sieve(j1, j2; 
            #     exchange = exchange, 
            #     shares = s, 
            #     permuted_shares = permuted_shares, 
            #     perm = permutations, 
            #     bernO = bernO, 
            #     sieve_type = sieve_type, 
            #     max_interaction = max_interaction)
            tempmat_s = npd_problem.tempmats[j1,j2];
            dsids[j1,j2,:] = tempmat_s * θ_j1;
            push!(tempmats, tempmat_s)
        end
    end
    
    
    Jmat = []; # vector of derivatives of inverse shares
    Jmat = Vector{Matrix}(undef, length(dsids[1,1,:]));

    all_elast_mat = Vector{Matrix}(undef, length(dsids[1,1,:]));
    temp = [];

    for ii in axes(dsids,3) # 1:length(dsids[1,1,:])
        J_s = zeros(J,J);
        for j1 = 1:J
            for j2 = 1:J
                J_s[j1,j2] = dsids[j1,j2,ii]
            end
        end
        try 
            temp = -1*inv(J_s);
        catch
            temp = -1*pinv(J_s);
        end
        
        # push!(Jmat, temp)
        Jmat[ii] = J_s;
    
        # Market vector of prices/shares
        ps_mat = zeros(J,J)
        for j1 = 1:J, j2 = 1:J 
            ps_mat[j1,j2] = at[ii,j2]/s[ii,j1];
        end
        
        all_elast_mat[ii] = temp .* ps_mat;
    end

    (;all_elast_mat, Jmat)
end

function get_params_one_equation(j1; exchange = [], s = [], θ = [], nbetas = [])
    which_group                               = findall(j1 .∈ exchange)[1];
    first_product_in_group                    = exchange[which_group][1];
    J                                         = maximum(maximum.(exchange));

    permutations                              = collect(1:J);
    permutations[first_product_in_group]      = j1; 
    permutations[j1]                          = first_product_in_group;

    permuted_shares                           = copy(s);
    permuted_shares[:,first_product_in_group] = s[:,j1]; 
    permuted_shares[:,j1]                     = s[:,first_product_in_group];
    
    if j1 ==1 
        init_ind = 0;
    else
        init_ind = sum(nbetas[1:j1-1])
    end
    θ_j1 = θ[init_ind+1:init_ind+nbetas[j1]];
    
    return θ_j1, permuted_shares, permutations
end

function calc_derivative_sieve(j1, j2; 
        exchange = [], shares = [], 
        permuted_shares = [], perm = [], 
        bernO = 2, sieve_type = "bernstein", max_interaction = 1, tensor = true,
        recipe = nothing, 
        constraints = nothing)

    basis_function = sieve_type == "bernstein" ? bern : poly
    dbasis_function = sieve_type == "bernstein" ? dbern : dpoly
    
    if (sieve_type == "bernstein") & (tensor == true)
        # When using this code, we need to invert j1 and j2 via perm, bc they have already been passed through perm 
        j1_orig = perm[j1]; 
        j2_orig = perm[j2];  
        tempmat_s = zeros(size(shares,1),1);
        J         = maximum(maximum.(exchange));
        for j_loop = 1:1:J
            stemp = permuted_shares[:,j_loop]; # j_loop = 1 -> stemp == perm_s[:,1] = s[:,2];
            # j1=3, j2=4. j_loop = 4 -> stemp = perm_s[:,4] = s[:,4]
            if j2_orig == perm[j_loop] # j2==2, perm[1] ==2, so s[:,2] added as derivative 
                tempmat_s = [tempmat_s dbern(stemp, bernO)];
            else 
                tempmat_s = [tempmat_s bern(stemp, bernO)];
            end
        end
        tempmat_s = tempmat_s[:,2:end] # remove zeros used to initialize the matrix
        tempmat_s, _, _ = make_interactions(tempmat_s, exchange, bernO, j1_orig, perm);
    elseif tensor ==false
        # which_group = findall(j1 .∈ exchange)[1];
        # exchange_for_poly = length(exchange) == size(shares,2) ? [] : adjust_exchange(exchange, j1);
        tempmat_s = NPDemand.poly_features_derivative(
            permuted_shares; 
            order = bernO, 
            max_interaction = max_interaction, 
            exchange = exchange, 
            var_index = j2, 
            recipe = recipe, 
            basis_type = sieve_type);
    else 
        # otherwise we are going to use fully generic tensor products
        tempmat_s = NPDemand.tensor_features_derivative(
            permuted_shares, 
            var_index = j2,  # Find which column in permuted space contains original column j2
            basis_orders = bernO .* ones(Int, size(shares,2)), 
            exchange = exchange, 
            basis_function = basis_function,
            dbasis_function = dbasis_function
            );
    end
    return tempmat_s
end

"""
    summarize_elasticities(problem::Union{NPDProblem, NPDResults}, which_elasticities::String, stat::String;
        q = 0.5)

Convenience function for summarizing the market-level price elasticities stored in `problem.all_elasticities`
(populated by running `price_elasticities!(problem)`). Accepts either an `NPDProblem` or the lightweight
`NPDResults` produced by `define_results`, since both store `all_elasticities` in the same form.

`which_elasticities` must be one of:
- `"matrix"`: returns a JxJ matrix, with `stat` applied across markets to each entry of the elasticity matrix
- `"own"`: returns a single number, pooling all own-price elasticities across products and markets
- `"cross"`: returns a single number, pooling all cross-price elasticities across products and markets

`stat` must be one of `"mean"`, `"median"`, or `"quantile"`. If `stat == "quantile"`, `q` gives the quantile of interest
(e.g., 0.75 for the 75th percentile).
"""
function summarize_elasticities(problem::Union{NPDProblem, NPDResults}, which_elasticities::String, stat::String; q = 0.5)

    if which_elasticities ∉ ["own", "cross", "matrix"]
        error("which_elasticities must be in ['own', 'cross', 'matrix']")
    end
    if stat ∉ ["mean", "median", "quantile"]
        error("stat must be in ['mean', 'median', 'quantile']")
    end
    if problem.all_elasticities == []
        error("No price elasticities calculated yet -- run price_elasticities!(problem)")
    end

    statfun = stat == "mean"   ? mean :
              stat == "median" ? median :
              x -> quantile(x, q)

    J = size(problem.all_elasticities[1], 1)

    if which_elasticities == "matrix"
        return [statfun(getindex.(problem.all_elasticities, j1, j2)) for j1 in 1:J, j2 in 1:J]
    elseif which_elasticities == "own"
        return statfun(reduce(vcat, [getindex.(problem.all_elasticities, j, j) for j in 1:J]))
    elseif which_elasticities == "cross"
        return statfun(reduce(vcat, [getindex.(problem.all_elasticities, j1, j2) for j1 in 1:J for j2 in setdiff(1:J, j1)]))
    end
end