function elast_penalty(θ_packed::AbstractArray, exchange::Array, elast_mats::Matrix{Any}, elast_prices::Matrix{Float64}, 
    lambda::Real, conmat::Matrix{Float64}; 
    J = maximum(maximum.(exchange)),
    J_sp = zeros(eltype(θ_packed),size(elast_mats[:,1])), during_obj = false)
    
    dsids = zeros(eltype(θ_packed),J,J,size(elast_mats[1,1],1))
    at = elast_prices;
    
    if during_obj == false
        θ = unpack(θ_packed, exchange, size.(elast_mats, 2), grad=false);
    else
        θ = θ_packed;
    end

    # Check inputs
    if typeof(at) ==DataFrame
        at = Matrix(at);
    end
    # svec = elast_mats;
    # J = maximum(maximum.(exchange));

    Threads.@threads for j1 = 1:J
        for j2 = 1:J
            if j1 ==1 
                init_ind = 0;
            else
                init_ind = sum(size.(elast_mats[1:j1-1,1],2))
            end
                θ_j1 = θ[init_ind+1:init_ind+size(elast_mats[j1,1],2)];
            try
                @views dsids[j1,j2,:] = elast_mats[j1,j2] * θ_j1; 
            catch
                @show size(θ_j1) size(elast_mats[j1,j2])
                @show j1 j2
            end
        end
    end
    
    Jmat = [];
    # svec2 = copy(svec);
    jacobian_vec = [];
    failed_inverse = false;

    penalty = zero(eltype(θ));
    temp = zeros(eltype(θ), J,J);
    J_s = zeros(eltype(θ),J,J);
    Threads.@threads for ii = 1:length(dsids[1,1,:])
        for j1 = 1:J
            for j2 = 1:J
                J_s[j1,j2] = dsids[j1,j2,ii]
            end
        end
        
        try
            temp = -1*inv(J_s);
        catch
            failed_inverse = true;
            break
        end         
        penalty += sum((temp .< conmat) .* abs.(temp).^2 .* lambda) + abs(log(cond(J_s))) * (cond(J_s) < 1e-3);
    end
    if failed_inverse 
        # penalty += 1e10;
    # else
        # for i ∈ eachindex(jacobian_vec)
        #     penalty += sum((jacobian_vec[i] .< conmat) .* abs.(jacobian_vec[i]).^2 .* lambda);
        # end
    end
    return penalty
end

function subset_for_elast_const(npd_problem, df::DataFrame; grid_size=10)
    s = Matrix(df[:, r"shares"]);
    J = size(s,2);

    # max_s = maximum(s, dims=1);
    # min_s = minimum(s, dims=1);
    max_s = quantile(s[:,1], 0.9);
    min_s = quantile(s[:,1], 0.1);
    for j = 2:J
        max_s = vcat(max_s, quantile(s[:,1], 0.9));
        min_s = vcat(min_s, quantile(s[:,1], 0.1));
    end

    temp = [];
    for j = 1:J
        if grid_size ==1
            push!(temp, [(min_s[j] + max_s[j])/2])
        else
            push!(temp, collect(range(min_s[j], max_s[j], length = grid_size)))
        end
        # push!(temp, collect(range((max_s[j] + min_s[j])/2 - 0.05, (max_s[j] + min_s[j])/2 + 0.05, length=2)))
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
            tempmat_s, a, b = make_interactions(tempmat_s, exchange, bernO, j1, perm);
            push!(elast_mats, tempmat_s)
        end
    end
    elast_mats = reshape(elast_mats, J,J);
    # Make a transpose to fix issue in other functions -- 
        # reshape above makes transpose of desired matrix
    temp_elast_mats = deepcopy(elast_mats);
    for j1 = 1:J
        for j2 = 1:J
            temp_elast_mats[j1,j2] = elast_mats[j2,j1];
        end
    end
    elast_mats = temp_elast_mats;
    elast_prices = Matrix(df[!,r"prices"]);
    return elast_mats, elast_prices 
end


function elast_penaltyrev(θ::AbstractArray, exchange::Array, elast_mats::Matrix{Any}, elast_prices::Matrix{Float64}, 
    lambda::Real, conmat::Matrix{Float64}; 
    J = maximum(maximum.(exchange)),
    during_obj = false)

    at = elast_prices;

    dsids = zeros(eltype(θ),J,J,size(elast_mats[1,1],1));
    temp = zeros(eltype(θ), J,J);
    J_s = zeros(eltype(θ),J,J);

    for j1 = 1:J
        for j2 = 1:J
            if j1 ==1 
                init_ind = 0;
            else
                init_ind = sum(size.(elast_mats[1:j1-1,1],2))
            end
            
            θ_j1 = θ[init_ind+1:init_ind+size(elast_mats[j1,1],2)];
            
            try
                dsids[j1,j2,:] .= elast_mats[j1,j2] * θ_j1; 
            catch
                # @show size(θ_j1) size(elast_mats[j1,j2])
                # @show j1 j2
                error("Error in inner_elasticity functions")
            end
        end
    end
    
    Jmat = eltype(θ)[];
    jacobian_vec = eltype(θ)[];
    failed_inverse = false;

    penalty = zero(eltype(θ));

    for ii = 1:length(dsids[1,1,:])
        for j1 = 1:J
            for j2 = 1:J
                J_s[j1,j2] = dsids[j1,j2,ii]
            end
        end

        try
            temp = -1*inv(J_s);
        catch
            failed_inverse = true;
            break
        end
        
        DET = det(J_s)^2;
        penalty += sum((temp .< conmat) .* log.((abs.(temp).+1).^2) .* lambda) + 10 * lambda * ((DET + 1e-12) / DET - 1); 
        
    end
    # @show penalty
    if failed_inverse 
        penalty += 1e10;
    elseif isnan(penalty)
        # penalty = 1e10;
    end
    
    return penalty
end