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
            elast .= elast ./ n_draws; # Calculate the mean posterior elasticities
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
    θ = β[1:npd_problem.design_width]
    γ = β[length(θ)+1:end]

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
        end
    end
    
    
    Jmat = []; # vector of derivatives of inverse shares
    Jmat = Vector{Matrix}(undef, length(dsids[1,1,:]));

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
        ps_mat = zeros(J,J)
        for j1 = 1:J, j2 = 1:J 
            ps_mat[j1,j2] = at[ii,j2]/s[ii,j1];
        end
        
        all_elast_mat[ii] = temp .* ps_mat;
    end

    all_elast_mat
end


"""
    summarize_elasticities(problem::NPDProblem, stat::String; q = [])

Convenience function for summarizing price elasticities. `problem` should be a solved `NPDProblem` after running `price_elasticities!`. 
`stat` should be in ["mean", "median", "quantile"], and if `stat`=="quantile", the option `q` should include the quantile of interest (e.g., 0.75 for the 75th percentile price elasticities).
"""
function summarize_elasticities(problem, which_elasticities::String, stat::String; 
    q = [], integrate = false, n_draws::Int = 100, CI::Real = 0.95)

    # Add input checks
    if stat ∉ ["mean", "quantile"]; error("stat must be in ['mean', 'quantile']"); end
    if which_elasticities ∉ ["own","cross","matrix"]; error("which_elasticities must be in ['own', 'cross', 'matrix']"); end
    if ((CI > 1) | (CI<0)); error("CI must be a real number between 0 and 1. Use 0.95 for a 95% credible interval"); end
    if problem.all_elasticities ==[]; error("No price elasticities calculated yet -- run price_elasticity!(problem)"); end

    J = length(problem.Xvec);

    # If we don't yet have a Markov chain, we can't integrate!
    if (problem.chain == [])
        if integrate == true
            @warn "No Markov chain available to integrate over, ignoring integration request"
        end
        integrate = false;
    end
    nbetas = get_nbetas(problem);
    if which_elasticities ∈ ["own","cross"]
        # elast_vec = [];
        if integrate == false # if we are running an aggregation on the pre-integrated elasticities
            if which_elasticities == "own"
                tmp = zeros(0)
                for j ∈ 1:J
                    append!(tmp, getindex.(problem.all_elasticities[!,:all_elasticities], j, j));
                end
            end
    
            if which_elasticities == "cross"
                tmp = zeros(0)
                for j1 ∈ 1:J
                    for j2 ∈ collect(setdiff(1:J, j1))
                        append!(tmp, getindex.(problem.all_elasticities[!,:all_elasticities], j1, j2));
                    end
                end
            end

            if stat == "mean"
                output = mean(tmp);

            elseif stat =="quantile" 
                if q==[]
                    println("Quantile q not specified -- assuming median")
                    q = 0.5;
                end
                output = quantile(tmp, q);
            end
        else # If we are calculating a statistic and doing inference on the aggregation via the posterior
            stat_vec = [];
            statprint = "";
            for i in 1:n_draws
                elast_i = zeros(0);
                sample_i    = problem.results.filtered_chain[i,:];
                β_i         = map_to_sieve(sample_i[1:sum(nbetas)], 
                                sample_i[sum(nbetas)+1:end], 
                                problem.exchange, 
                                nbetas, 
                                problem)
                elast_i_matrix     = price_elasticities_inner(problem, β = β_i);
                if which_elasticities == "own"
                    for j ∈ 1:J
                        append!(elast_i, getindex.(elast_i_matrix, j, j));
                    end
                else # if which_elasticities == "cross"
                    for j1 ∈ 1:J
                        for j2 ∈ collect(setdiff(1:J, j1))
                            append!(elast_i, getindex.(elast_i_matrix, j1, j2));
                        end
                    end
                end
                # elast_vec = vcat(elast_vec, elast_i);
                if stat == "mean"
                    push!(stat_vec, mean(elast_i));
                    if i==1; statprint = stat; end
                elseif stat =="quantile" 
                    if q==[]
                        println("Quantile q not specified -- assuming median")
                        q = 0.5;
                    end
                    push!(stat_vec, quantile(elast_i, q));
                    if i==1; statprint = string(stat, " ", q); end
                end
            end
            output = (;Statistic = statprint, 
                Posterior_Mean = mean(stat_vec), 
                Posterior_Median = median(stat_vec), 
                Posterior_CI = (quantile(stat_vec, (1-CI)/2), quantile(stat_vec, 1 - (1-CI)/2)))
        end
    end

    if which_elasticities == "matrix"
        statprint = "";
        if integrate == false
            output = zeros(J,J);
            for j1 ∈ 1:J
                for j2 ∈ 1:J
                    if stat=="mean"
                        statprint = stat;
                        output[j1,j2] = mean(getindex.(problem.all_elasticities[!,:all_elasticities], j1, j2));
                    elseif stat =="quantile" 
                        if q==[]
                            println("Quantile q not specified -- assuming median")
                            q = 0.5;
                        end
                        output[j1,j2] = quantile(getindex.(problem.all_elasticities[!,:all_elasticities], j1, j2), q);
                        statprint = string(stat, " ", q);
                    end
                end
            end
        else
            output = zeros(J,J,n_draws);
            for i in 1:n_draws
                sample_i    = problem.results.filtered_chain[i,:];
                β_i         = map_to_sieve(sample_i[1:sum(nbetas)], 
                                sample_i[sum(nbetas)+1:end], 
                                problem.exchange, 
                                nbetas, 
                                problem)
                elast_i_matrix  = price_elasticities_inner(problem, β = β_i);
                for j1 ∈ 1:J
                    for j2 ∈ 1:J
                        if stat=="mean"
                            output[j1,j2,i] = mean(getindex.(elast_i_matrix, j1, j2));
                            if i==1; statprint = stat; end
                        elseif stat =="quantile" 
                            if i==1; statprint = string(stat, " ", q); end
                            if q==[]
                                println("Quantile q not specified -- assuming median")
                                q = 0.5;
                            end
                            output[j1,j2,i] = quantile(getindex.(elast_i_matrix, j1, j2), q);
                        end
                    end
                end
            end
        end
        if length(size(output)) ==3
            output = (;Statistic = statprint, 
                    Posterior_Mean = dropdims(mean(output, dims = 3), dims=3), 
                    Posterior_Median = dropdims(median(output, dims = 3), dims=3), 
                    Posterior_CI = ([quantile(output[j1,j2,:], (1-CI)/2) for j1=1:J, j2=1:J], [quantile(output[j1,j2,:], 1 - (1-CI)/2) for j1=1:J, j2=1:J]))
        else
            output = (;Statistic = statprint, 
                    Value = output)
        end
    end
    return output
end

function elasticity_quantiles(problem::NPDProblem, ind1::Int, ind2::Int; 
    quantiles = collect(0.01:0.01:0.99),
    n_draws::Int = 100)
    if problem.chain ==[]
        tempfunc(x) = summarize_elasticities(problem, "matrix", "quantile", integrate = false, q = x).Value[ind1,ind2];
        return quantiles, [tempfunc(x) for x ∈ quantiles]
    else
        tempfunc2(x) = summarize_elasticities(problem, "matrix", "quantile", q = x, integrate = true, n_draws = n_draws);
        median_temp_vec = [];
        ub_temp_vec = [];
        lb_temp_vec = [];
        for x ∈ ProgressBar(quantiles)
            summarized_elasticities = tempfunc2(x);
            push!(median_temp_vec, summarized_elasticities.Posterior_Median[ind1, ind2])
            push!(ub_temp_vec, summarized_elasticities.Posterior_CI[2][ind1, ind2])
            push!(lb_temp_vec, summarized_elasticities.Posterior_CI[1][ind1, ind2])
        end
        return quantiles, median_temp_vec, lb_temp_vec, ub_temp_vec
    end
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
