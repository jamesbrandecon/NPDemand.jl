"""
    price_elasticities!(problem::NPDProblem;
         CI = [], 
         n_draws = [])

Takes the solved `problem` as first argument, a `DataFrame` as the second argument, and evaluates all price elasticities in-sample. 
Currently does not calculate out-of-sample price elasticities. For this, use the function `compute_demand_function!`. 

Results of this function are stored as a `DataFrame` in problem.all_elasticities. Results can be summarized by hand or using the `summarize_elasticities` function. 
We also store the Jacobian of the demand function with respect to prices, which can be used to calculate other quantities of interest.
"""
function price_elasticities!(problem; 
        CI::Union{Vector{Any}, Real} = [], 
        n_draws::Union{Vector{Any}, Int} = [])
    try 
        @assert (CI==[]) | (CI isa Real)
    catch 
        error("CI must be a real number between 0 and 1. Use 0.95 for a 95% credible interval")
    end

    if problem.chain ==[]
        elast = price_elasticities_inner(problem);
        problem.all_elasticities = DataFrame(market_ids = problem.data.market_ids, all_elasticities = elast)
    else
        burn_in = problem.sampling_details.burn_in;
        skip    = problem.sampling_details.skip;
        J       = length(problem.Xvec);
        T       = size(problem.data,1);

        elast_CI = [];
        if n_draws == []
            n_draws = size(problem.results.filtered_chain,1)
        end

        if CI == []
            elast   = [zeros(J,J) for i in 1:T];
            nbetas = get_nbetas(problem);
            for i in 1:n_draws
                sample_i = problem.results.filtered_chain[i,:];
                β_i = map_to_sieve(sample_i[1:sum(nbetas)], 
                                sample_i[sum(nbetas)+1:end], 
                                problem.exchange, 
                                nbetas, 
                                problem)
                elast_i = price_elasticities_inner(problem, β = β_i)
                elast = elast .+ elast_i;
            end
            elast .= elast ./ ndraws; # Calculate the mean posterior elasticities
            problem.all_elasticities = DataFrame(market_ids = problem.data.market_ids, all_elasticities = elast)
        else
            elast   = [zeros(J,J) for i in 1:T];
            alpha = 1 - CI;
            try 
                @assert ((n_draws > 0) & (n_draws <= size(problem.results.filtered_chain,1)))
            catch 
                error("`n_draws` must be greater than 0 and weakly less than the total number of draws in the final chain")
            end
            nbetas      = get_nbetas(problem);
            # draw_order  = sample(1:size(problem.results.filtered_chain,1), n_draws, replace = false);

            for i in 1:n_draws
                sample_i    = problem.results.filtered_chain[i,:];
                β_i         = map_to_sieve(sample_i[1:sum(nbetas)], 
                                sample_i[sum(nbetas)+1:end], 
                                problem.exchange, 
                                nbetas, 
                                problem)
                elast_i     = price_elasticities_inner(problem, β = β_i)
                elast       = elast + elast_i;
                push!(elast_CI, elast_i);
            end
            ub = [
                [quantile(getindex.(getindex.(elast_CI, t),j,j), 1-alpha/2) for j = 1:J, j = 1:J] 
                for t in 1:T
                    ]
            lb = [
                [quantile(getindex.(getindex.(elast_CI, t),j,j), alpha/2) for j = 1:J, j = 1:J] 
                for t in 1:T
                    ]
            elast .= elast ./ n_draws; # Calculate the mean posterior elasticities
            problem.all_elasticities = DataFrame(
                market_ids = problem.data.market_ids, 
                all_elasticities = elast, 
                ub = ub, 
                lb = lb)
        end
    end
end

function price_elasticities_inner(npd_problem; β = npd_problem.results.minimizer)
    # Add a market ID column if not already present
    if !(:market_ids ∈ names(npd_problem.data))
        npd_problem.data[!,"market_ids"] .= 1:size(npd_problem.data,1);
    end

    df = npd_problem.data;
    at = df[!,r"prices"];
    
    # Unpack results
    # β = npd_problem.results.minimizer
    θ = β[1:npd_problem.design_width]
    γ = β[length(θ)+1:end]

    # γ[1] = 1;
    # for i ∈ npd_problem.normalization
    #     γ[i] =0; 
    # end
    # for i ∈ eachindex(npd_problem.mins)
    #     θ[npd_problem.mins[i]] = θ[npd_problem.maxs[i]]
    # end

    X = npd_problem.Xvec;
    B = npd_problem.Bvec;
    bO = npd_problem.bO;
    exchange = npd_problem.exchange;

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
    # order = bernO;
    
    # design_width = sum(size.(X,2));
    # θ = β[1:design_width]
    # γ = β[length(θ)+1:end]

    # Construct index 
        # Note: Currently evaluates at realized prices. Have to edit to allow for counterfactual prices
    index = zeros(size(X[1],1),J)
    for j = 1:J
        index[:,j] = B[j]*γ;
    end

    # Shares to evaluate derivatives -- bad form, holdover from old code
    # svec = s;
    
    # Share Jacobian
    tempmats = []
    # tempmats = Vector{Matrix}(undef, J^2);
    dsids = zeros(J,J,size(index,1)) # initialize matrix of ∂s^{-1}/∂s
    # ji = 1; # count the cell of tempmats to fill
    for j1 = 1:J
        which_group = findall(j1 .∈ exchange)[1];
        first_product_in_group = exchange[which_group][1];

        perm = collect(1:J);
        perm[first_product_in_group] = j1; perm[j1] = first_product_in_group;
    
        perm_s = copy(s);
        perm_s[:,first_product_in_group] = s[:,j1]; perm_s[:,j1] = s[:,first_product_in_group];
        
        if j1 ==1 
            init_ind = 0;
        else
            init_ind = sum(size.(X[1:j1-1],2))
        end
        θ_j1 = θ[init_ind+1:init_ind+size(X[j1],2)];
        
        # @show θ_j1
        for j2 = 1:J 
            tempmat_s = zeros(size(index,1),1)
            for j_loop = 1:1:J
                stemp = perm_s[:,j_loop]; # j_loop = 1 -> stemp == perm_s[:,1] = s[:,2];
                # j1=3, j2=4. j_loop = 4 -> stemp = perm_s[:,4] = s[:,4]
                if j2 == perm[j_loop] # j2==2, perm[1] ==2, so s[:,2] added as derivative 
                    tempmat_s = [tempmat_s dbern(stemp, bernO)];
                else 
                    tempmat_s = [tempmat_s bern(stemp, bernO)];
                end
            end
            tempmat_s = tempmat_s[:,2:end]
            tempmat_s, a, b = make_interactions(tempmat_s, exchange, bernO, j1, perm);
            
            dsids[j1,j2,:] = tempmat_s * θ_j1;
            push!(tempmats, tempmat_s)
            # tempmats[ji] = tempmat_s;
            # ji +=1
        end
    end
    
    
    Jmat = []; # vector of derivatives of inverse shares
    Jmat = Vector{Matrix}(undef, length(dsids[1,1,:]));

    # J_sp = zeros(size(svec[:,1]));
    # all_own = zeros(size(svec,1),J);
    # svec2 = svec;
    # avg_elast_mat = zeros(J,J);

    # all_elast_mat = [];
    all_elast_mat = Vector{Matrix}(undef, length(dsids[1,1,:]));
    temp = [];

    for ii = 1:length(dsids[1,1,:])
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
        Jmat[ii] = temp;
    
        # Market vector of prices/shares
        ps = at[ii,:]./s[ii,:];
        # ps_mat = repeat(at[ii,:]', J,1) ./ repeat(svec2[ii,:], 1,J);
        ps_mat = zeros(J,J)
        for j1 = 1:J, j2 = 1:J 
            ps_mat[j1,j2] = at[ii,j2]/s[ii,j1];
        end
        
        # push!(all_elast_mat, temp .* ps_mat)
        all_elast_mat[ii] = temp .* ps_mat;
    end
    # npd_problem.all_elasticities = all_elast_mat; #DataFrame(product1 = prod1, product2 = prod2, elasticity = elas_ijj, market_ids = market);
    # npd_problem.all_elasticities = DataFrame(market_ids = df.market_ids, all_elasticities = npd_problem.all_elasticities)
    
    # npd_problem.all_jacobians = Jmat;
    all_elast_mat
end




"""
    summarize_elasticities(problem::NPDProblem, stat::String; q = [])

Convenience function for summarizing price elasticities. `problem` should be a solved `NPDProblem` after running `price_elasticities!`. 
`stat` should be in ["mean", "median", "quantile"], and if `stat`=="quantile", the option `q` should include the quantile of interest (e.g., 0.75 for the 75th percentile price elasticities).
"""
function summarize_elasticities(problem::NPDProblem, stat::String; q = [])
    @assert stat ∈ ["mean", "median", "quantile"]

    J = length(problem.Xvec);
    output = zeros(J,J);
    for j1 ∈ 1:size(problem.all_elasticities[!,:all_elasticities][1],1)
        for j2 ∈ 1:size(problem.all_elasticities[!,:all_elasticities][1],1)
            if stat=="median"
                output[j1,j2] = median(getindex.(problem.all_elasticities[!,:all_elasticities], j1, j2));
            elseif stat=="mean"
                output[j1,j2] = mean(getindex.(problem.all_elasticities[!,:all_elasticities], j1, j2));
            elseif stat =="quantile" 
                if q==[]
                    println("Quantile q not specified -- assuming median")
                    q = 0.5;
                end
                output[j1,j2] = quantile(getindex.(problem.all_elasticities[!,:all_elasticities], j1, j2), q);
            end
        end
    end
    return output
end

function elasticity_cdf(problem::NPDProblem, ind1, ind2)
    tempfunc(x) = summarize_elasticities(problem, "quantile", q = x)[ind1,ind2];
    return 0.01:0.01:0.99, [tempfunc(x) for x ∈ 0.01:0.01:0.99]
end


function own_elasticities(problem::NPDProblem)
    try
        @assert problem.all_elasticities !=[]
    catch
        error("No price elasticities calculated yet -- run price_elasticity!(problem)")
    end
    J = length(problem.Xvec);
    N = size(problem.Xvec[1],1);
    output = zeros(N,J);
    for j1 ∈ 1:J
        output[:,j1] = getindex.(problem.all_elasticities[!,:all_elasticities], j1, j1);
    end
    output = DataFrame(output, :auto);
    for j1 ∈ 1:J
        rename!(output, Symbol("x$j1") => "product_"*string(j1));
    end
    return output
end

# function price_elasticities(β, npd_problem)
#     # Add a market ID column if not already present
#     if !(:market_ids ∈ names(npd_problem.data))
#         npd_problem.data[!,"market_ids"] .= 1:size(npd_problem.data,1);
#     end

#     df = npd_problem.data;
#     at = df[!,r"prices"];
    
#     # Unpack results
#     θ = β[1:npd_problem.design_width]
#     γ = β[length(θ)+1:end]

#     γ[1] = 1;
#     for i ∈ npd_problem.normalization
#         γ[i] =0; 
#     end
#     for i ∈ eachindex(npd_problem.mins)
#         θ[npd_problem.mins[i]] = θ[npd_problem.maxs[i]]
#     end
#     for i ∈ eachindex(npd_problem.mins)
#         θ[npd_problem.mins[i]] = θ[npd_problem.maxs[i]]
#     end

#     X = npd_problem.Xvec;
#     B = npd_problem.Bvec;
#     bO = npd_problem.bO;
#     exchange = npd_problem.exchange;

#     # Check inputs
#     if (at!=[]) & (size(at,2) != size(df[:,r"prices"],2))
#         error("Argument `at` must be a matrix of prices with J columns if provided")
#     end
#     if typeof(at) ==DataFrame
#         at = Matrix(at);
#     end

#     s = Matrix(df[:, r"shares"]);
#     J = size(s,2);
#     bernO = convert.(Integer, bO);
#     order = bernO;
    
#     design_width = sum(size.(X,2));
#     θ = β[1:design_width]
#     γ = β[length(θ)+1:end]

#     # Construct index 
#         # Note: Currently evaluates at realized prices. Have to edit to allow for counterfactual prices
#     index = zeros(size(X[1],1),J)
#     for j = 1:J
#         index[:,j] = B[j]*γ;
#     end

#     # Shares to evaluate derivatives -- bad form, holdover from old code
#     svec = s;
    
#     # Share Jacobian
#     tempmats = []
#     dsids = zeros(J,J,size(npd_problem.data,1)) # initialize matrix of ∂s^{-1}/∂s
#     for j1 = 1:J
#         which_group = findall(j1 .∈ exchange)[1];
#         first_product_in_group = exchange[which_group][1];

#         perm = collect(1:J);
#         perm[first_product_in_group] = j1; perm[j1] = first_product_in_group;
    
#         perm_s = copy(s);
#         perm_s[:,first_product_in_group] = s[:,j1]; perm_s[:,j1] = s[:,first_product_in_group];
        
#         if j1 ==1 
#             init_ind = 0;
#         else
#             init_ind = sum(size.(X[1:j1-1],2))
#         end
#         θ_j1 = θ[init_ind+1:init_ind+size(X[j1],2)];
        
#         # @show θ_j1
#         for j2 = 1:J 
#             tempmat_s = zeros(size(npd_problem.data,1),1)
#             for j_loop = 1:1:J
#                 stemp = perm_s[:,j_loop]; # j_loop = 1 -> stemp == perm_s[:,1] = s[:,2];
#                 # j1=3, j2=4. j_loop = 4 -> stemp = perm_s[:,4] = s[:,4]
#                 if j2 == perm[j_loop] # j2==2, perm[1] ==2, so s[:,2] added as derivative 
#                     tempmat_s = [tempmat_s dbern(stemp, bernO)];
#                 else 
#                     tempmat_s = [tempmat_s bern(stemp, bernO)];
#                 end
#             end
#             tempmat_s = tempmat_s[:,2:end]
#             tempmat_s, a, b = make_interactions(tempmat_s, exchange, bernO, j1, perm);
#             dsids[j1,j2,:] = tempmat_s * θ_j1;
#             push!(tempmats, tempmat_s)
#         end
#     end
    
#     Jmat = []; # vector of derivatives of inverse shares
#     # J_sp = zeros(size(svec[:,1]));
#     # all_own = zeros(size(svec,1),J);
#     svec2 = svec;
#     # avg_elast_mat = zeros(J,J);
#     all_elast_mat = [];

#     for ii = 1:length(dsids[1,1,:])
#         J_s = zeros(J,J);
#         for j1 = 1:J
#             for j2 = 1:J
#                 J_s[j1,j2] = dsids[j1,j2,ii]
#             end
#         end
#         temp = -1*inv(J_s);
#         push!(Jmat, temp)
    
#         # Market vector of prices/shares
#         ps = at[ii,:]./svec2[ii,:];
#         # ps_mat = repeat(at[ii,:]', J,1) ./ repeat(svec2[ii,:], 1,J);
#         ps_mat = zeros(J,J)
#         for j1 = 1:J, j2 = 1:J 
#             ps_mat[j1,j2] = at[ii,j2]/svec2[ii,j1];
#         end
    
#         push!(all_elast_mat, temp .* ps_mat) #, temp .* ps_mat
#     end
#     return all_elast_mat
# end