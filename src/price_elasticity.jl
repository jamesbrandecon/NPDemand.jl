"""
    price_elasticity!(problem::NPDProblem, df::DataFrame; at::Matrix, whichProducts = [1,1])

Takes the solved `problem` as first argument, a `DataFrame` as the second argument, and evaluates price elasticities in-sample at prices `at`. 
Currently does not calculate out-of-sample price elasticities. This will be added to compute_demand_function!. 

Results of this function are stored as a `DataFrame` in problem.all_elasticities. Results can be summarized by hand or with summary functions: 
- `mean_elasticity(problem::NPDProblem)`
- `median_elasticity(problem::NPDProblem)`
- quantile_elasticiy(problem::NPDProblem, q ∈ [0,1])
"""
function price_elasticities!(npd_problem; whichProducts = [1,1])

    df = npd_problem.data;
    at = df[!,r"prices"];

    # Unpack results
    β = npd_problem.results.minimizer
    θ = β[1:npd_problem.design_width]
    γ = β[length(θ)+1:end]

    γ[1] = 1;
    for i ∈ npd_problem.normalization
        γ[i] =0; 
    end
    for i∈eachindex(npd_problem.mins)
        θ[npd_problem.mins[i]] = θ[npd_problem.maxs[i]]
    end

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
    order = bernO;
    
    design_width = sum(size.(X,2));
    # θ = β[1:design_width]
    # γ = β[length(θ)+1:end]

    # Construct index 
        # Note: Currently evaluates at realized prices. Have to edit to allow for counterfactual prices
    index = zeros(size(X[1],1),J)
    for j = 1:J
        index[:,j] = B[j]*γ;
    end

    # Shares to evaluate derivatives -- bad form, holdover from old code
    svec = s;
    
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
            init_ind=0;
        else
            init_ind = sum(size.(X[1:j1-1],2))
        end
        θ_j1 = θ[init_ind+1:init_ind+size(X[j1],2)];
        # @show θ_j1
        for j2 = 1:J 
            tempmat_s = zeros(size(index,1),1)
            for j_loop = 1:1:J
                stemp = perm_s[:,j_loop]; # j_loop = 1 -> stemp == perm_s[:,1] = s[:,2]
                if j2==perm[j_loop] # j2==2, perm[1] ==2, so s[:,2] added as derivative 
                    tempmat_s = [tempmat_s dbern(stemp, bernO)];
                else 
                    tempmat_s = [tempmat_s bern(stemp, bernO)];
                end
            end
            tempmat_s = tempmat_s[:,2:end]
            tempmat_s, a, b = make_interactions(tempmat_s, exchange, bernO, first_product_in_group, perm);
            dsids[j1,j2,:] = tempmat_s * θ_j1;
            push!(tempmats, tempmat_s)
        end
    end
    Jmat = []; # vector of derivatives of inverse shares
    J_sp = zeros(size(svec[:,1]));
    all_own = zeros(size(svec,1),J);
    svec2 = svec;
    avg_elast_mat = zeros(J,J);
    all_elast_mat = [];

    for ii = 1:length(dsids[1,1,:])
        J_s = zeros(J,J);
        for j1 = 1:J
            for j2 = 1:J
                J_s[j1,j2] = dsids[j1,j2,ii]
            end
        end
        temp = -1*inv(J_s);
        push!(Jmat, temp)
    
        # Market vector of prices/shares
        ps = at[ii,:]./svec2[ii,:];
        # ps_mat = repeat(at[ii,:]', J,1) ./ repeat(svec2[ii,:], 1,J);
        ps_mat = zeros(J,J)
        for j1=1:J, j2 = 1:J 
            ps_mat[j1,j2] = at[ii,j2]/svec2[ii,j1];
        end
        avg_elast_mat += (temp .* ps_mat) ./ size(at,1); # take average over 
        push!(all_elast_mat, temp .* ps_mat)

        # All own-price elasticities
        all_own[ii,:] = -1*Diagonal(inv(J_s))*ps;
    
        # Save to calculate desired own-price elasticities
        J_sp[ii,1] = temp[whichProducts[1],whichProducts[2]];
    end
        
    # New code, uses matrix of index, which are equal to -1 .* prices
    esep = J_sp.* (at[:,whichProducts[2]]./svec2[:,whichProducts[1]]); # own-price varying
    
    # return esep, Jmat, svec, all_own
    # return esep, avg_elast_mat, svec, all_own, all_elast_mat, Jmat
    npd_problem.all_elasticities = all_elast_mat; #DataFrame(product1 = prod1, product2 = prod2, elasticity = elas_ijj, market_ids = market);
end
    

function summarize_elasticities(problem::NPDProblem, stat::String; q = [])
    @assert stat ∈ ["mean", "median", "quantile"]

    J = length(problem.Xvec);
    output = zeros(J,J);
    for j1 ∈ 1:size(problem.all_elasticities[1],1)
        for j2 ∈ 1:size(problem.all_elasticities[1],1)
            if stat=="median"
                output[j1,j2] = median(getindex.(problem.all_elasticities, j1, j2));
            elseif stat=="mean"
                output[j1,j2] = mean(getindex.(problem.all_elasticities, j1, j2));
            elseif stat =="quantile" 
                if q==[]
                    println("Quantile q not specified -- assuming median")
                    q = 0.5;
                end
                output[j1,j2] = quantile(getindex.(problem.all_elasticities, j1, j2), q);
            end
        end
    end
    return output
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
        output[:,j1] = getindex.(problem.all_elasticities, j1, j1);
    end

    return output
end