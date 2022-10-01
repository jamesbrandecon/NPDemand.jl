"""
    price_elasticity(problem::NPDProblem, df::DataFrame; at::Matrix, whichProducts = [1,1])

Takes the solved `problem` as first argument, a `DataFrame` as the second argument, and evaluates price elasticities in-sample at prices `at`. 
Currently does not calculate out-of-sample price elasticities, though this will be added in the future. 
"""
function price_elasticity(npd_problem, df::DataFrame; at = df[!,r"prices"], whichProducts = [1,1])
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
    trueS=true;

    design_width = sum(size.(X,2));
    # θ = β[1:design_width]
    # γ = β[length(θ)+1:end]

    # Construct index 
        # Note: Currently evaluates at realized prices. Have to edit to allow for counterfactual prices
    index = zeros(size(X[1],1),J)
    for j = 1:J
        index[:,j] = B[j]*γ;
    end

    # Declare prices and shares to evaluate derivatives
    svec = similar(index)
    if trueS == false
        for p_i = 1:size(index,1)
            s!(sj) = solve_s_nested_flexible(sj, inv_sigma, index[p_i,:]', J, bernO, included, maxes, nothing);
            ans = nlsolve(s!, 1/(2*J) .* ones(J))
            svec[p_i,:] =  ans.zero;
        end
    else
        svec = s;
    end
    
    # Share Jacobian
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
        end
    end
    Jmat = [];
    J_sp = zeros(size(svec[:,1]));
    all_own = zeros(size(svec,1),J);
    svec2 = svec;
    avg_elast_mat = zeros(J,J);

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
        ps_mat = repeat(at[ii,:]', J,1) ./ repeat(svec2[ii,:], 1,J) ;
        avg_elast_mat += (temp .* ps_mat) ./ size(at,1);
    
        # All own-price elasticities
        all_own[ii,:] = -1*Diagonal(inv(J_s))*ps;
    
        # Save to calculate desired own-price elasticities
        J_sp[ii,1] = temp[whichProducts[1],whichProducts[2]];
    end
        
    # New code, uses matrix of index, which are equal to -1 .* prices
    esep = J_sp.* (at[:,whichProducts[2]]./svec2[:,whichProducts[1]]); # own-price varying
    
    # if trueS==1
    #     print("There were $numBadMarkets bad markets")
    # end
    # return esep, Jmat, svec, all_own
    return esep, avg_elast_mat, svec, all_own
end
    