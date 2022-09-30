function elast_penalty(exchange, elast_mats; elast_prices)
    at = elast_prices;

    # Check inputs
    if (at!=[]) & (size(at,2) != size(df[:,r"prices"],2))
        error("Argument `at` must be a matrix of prices with J columns if provided")
    end
    if typeof(at) ==DataFrame
        at = Matrix(at);
    end

    svec = elast_mats;
    J = size(s,2);
    
    # Share Jacobian
    dsids = zeros(J,J,size(index,1)) # initialize matrix of ∂s^{-1}/∂s
    for j1 = 1:J
        for j2 = 1:J
            if j1 ==1 
                init_ind=0;
            else
                init_ind = sum(size.(X[1:j1-1],2))
            end
            θ_j1 = θ[init_ind+1:init_ind+size(X[j1],2)];
            dsids[j1,j2,:] = tempmat_s[j1,j2] * θ_j1;
        end
    end
    
    Jmat = [];
    J_sp = zeros(size(svec[:,1]));
    all_own = zeros(size(svec,1),J);
    svec2 = copy(svec);
    jacobian_vec = [];

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
        push!(jacobian_vec, temp .* ps_mat);
    end
    
    # Convert jacobian_vec to Penalty
        # Have to calculate sign matrix from exchange

    return 
end

function subset_for_elast_const(npd_problem, df::DataFrame; grid_size=2)
    s = Matrix(df[:, r"shares"]);
    J = size(s,2);

    max_s = maximum(s, dims=1);
    min_s = minimum(s, dims=1);

    temp = [];
    for j = 1:J
        push!(temp, collect(range(min_s[j], max_s[j], length = grid_size)))
    end
    temp = collect.(Iterators.product(temp...));
    temp = reshape(temp, length(temp));
    
    use_row = zeros(size(s,1));
    for i ∈ eachindex(temp)
        distances = sum((s .- temp[i]').^2, dims=2);
        ind = getindex.(findall(distances .== minimum(distances)),1)[1];
        use_row[ind] = 1;
    end
    
    subset = df[(use_row .==1),:];
    return subset
end

function make_elasticity_mat(npd_problem, df::DataFrame)
    X = npd_problem.Xvec;
    B = npd_problem.Bvec;
    bO = npd_problem.bO;
    exchange = npd_problem.exchange;

    s = Matrix(df[:, r"shares"]);
    J = size(s,2);
    bernO = convert.(Integer, bO);
    order = bernO;
    trueS=true;

    design_width = sum(size.(X,2));
    
    # Share Jacobian
    dsids = zeros(J,J,size(s,1)) # initialize matrix of ∂s^{-1}/∂s
    elast_mats = [];
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
        
        for j2 = 1:J 
            tempmat_s = zeros(size(s,1),1)
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
            push!(elast_mats, tempmat_s)
        end
    end
    elast_mats = reshape(elast_mats, J,J);
    return elast_mats
end