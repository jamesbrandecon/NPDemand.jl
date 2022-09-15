function price_elasticity2(β, df::DataFrame, bO; X =[], B = [], at = df[!,r"prices"])
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
    
    θ = β[1:design_width]
    γ = β[length(θ)+1:end]

    # Construct index 
        # Note: Currently evaluates at realized prices. Have to edit to allow for counterfactual prices
    index = zeros(size(X[1],1),J)
    for j = 1:J
        index[:,j] = B[j]*γ;
    end

    ## Declare prices and shares to evaluate derivatives
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
        perm = collect(1:J);
        perm[1] = xj; perm[j1] = 1;
    
        perm_s = copy(s);
        perm_s[:,1] = s[:,j1]; perm_s[:,j1] = s[:,1];

        if j1 ==1 
            init_ind=0;
        else
            init_ind = sum(size.(X[1:j1-1],2))
        end
        θ_j1 = θ[init_ind+1:init_ind+size(X[j1],2)];

        for j2 = 1:J
            tempmat_s = zeros(size(index,1),1)
            for j_loop = 1:1:J
                stemp = perm_s[:,j_loop];
                    if j2==perm[j_loop]
                        tempmat_s = [tempmat_s dbern(stemp, bernO)];
                    else
                        tempmat_s = [tempmat_s bern(stemp, bernO)];
                    end
            end

            tempmat_s = tempmat_s[:,2:end]
            tempmat_s = make_interactions(tempmat_s, exchange, bernO, j1, perm);
            dsids[j1,j2,:] = tempmat_s * θ_j1;
        end
    end
    
    Jmat = [];
    J_sp = zeros(size(svec[:,1]));
    all_own = zeros(size(svec,1),J);
    svec2 = svec;

    for ii = 1:length(dsids[1,1,:])
        J_s = [];
        for j1 = 1:J
            J_sj1 = [];
            for j2 = 1:J
                if j2==1
                    J_sj1 = dsids[j1,j2,ii]
                else
                    J_sj1 = [J_sj1 dsids[j1,j2,ii]]
                end
            end
            if j1==1
                J_s = J_sj1;
            else
                J_s = vcat(J_s, J_sj1);
            end
        end
        temp = -1*inv(J_s);
        push!(Jmat, temp)
    
        # Market vector of prices/shares
        ps = at[ii,:]./svec2[ii,:];
    
        # All own-price elasticities
        all_own[ii,:] = -1*Diagonal(inv(J_s))*ps;
    
        # Save to calculate desired own-price elasticities
        J_sp[ii,1] = temp[whichProducts[1],whichProducts[2]];
    end
    
    #esep = J_sp.*at[:,whichProducts[2]]./svec2[:,whichProducts[1]]; # old code, called at by mistake
    
    # New code, uses matrix of index, which are equal to -1 .* prices
    esep = J_sp.* (at[:,whichProducts[2]]./svec2[:,whichProducts[1]]); # own-price varying
    numBadMarkets = 0
    
    # if trueS==1
    #     print("There were $numBadMarkets bad markets")
    # end
    return esep, Jmat, svec, all_own
end
    