function elast_penalty(θ_packed::AbstractArray, exchange::Array, elast_mats::Matrix{Any}, elast_prices::Matrix{Float64}, 
    lambda::Real, 
    conmat::Dict{Symbol,Matrix{Float64}}; 
    J = maximum(maximum.(exchange)),
    J_sp = zeros(eltype(θ_packed),size(elast_mats[:,1])), 
    during_obj = false)
    
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
            @views dsids[j1,j2,:] = elast_mats[j1,j2] * θ_j1; 
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
        
        # try
            temp = -1*inv(J_s);
        # catch
        #     failed_inverse = true;
        #     break
        # end         
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

    ub = 0.6;
    lb = 0.4;

    max_s = quantile(s[:,1], ub);
    min_s = quantile(s[:,1], lb);
    for j = 2:J
        max_s = vcat(max_s, quantile(s[:,j], ub));
        min_s = vcat(min_s, quantile(s[:,j], lb));
    end

    temp = [];
    for j = 1:J
        midpoint = (min_s[j] + max_s[j])/2;
        if grid_size ==1
            push!(temp, [midpoint])
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
    lambda::Real, 
    conmat::Dict; 
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
            
            # try
                dsids[j1,j2,:] .= elast_mats[j1,j2] * θ_j1; 
            # catch
            #     error("Error in inner_elasticity functions")
            # end
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

        # try
            temp = -1*inv(J_s);
        # catch
        #     failed_inverse = true;
        #     break
        # end
        
        DET = det(J_s)^2; 
        if conmat[:subs] !=[]
            penalty += sum((temp .< conmat[:subs]) .* log.((abs.(temp).+1).^2) .* lambda); # + 10 * lambda * ((DET + 1e-12) / DET - 1); 
        end
        if conmat[:complements] !=[]
            penalty += sum((temp .> conmat[:complements]) .* log.((abs.(temp).+1).^2) .* lambda); # + 10 * lambda * ((DET + 1e-12) / DET - 1); 
        end
        
    end
    # @show penalty
    if failed_inverse 
        penalty += 1e10;
    elseif isnan(penalty)
        # penalty = 1e10;
    end
    
    return penalty
end

function elast_penalty_all(x, exchange, elast_mats, 
    elast_prices, lambda, conmat; J = maximum(maximum.(exchange)),
    during_obj = false,
    quantile_vec = [0.5], 
    problem_details_dict = Dict(),
    nonlinear_method = "grid")

    if nonlinear_method =="grid" 
        out = elast_penaltyrev(x, exchange, elast_mats, elast_prices, lambda, conmat; J = maximum(maximum.(exchange)),
        during_obj = false)
    else
        out = elast_penalty_quantile(x, exchange, elast_mats, elast_prices, lambda, conmat; J = maximum(maximum.(exchange)),
        during_obj = false,
        quantile_vec = quantile_vec, 
        problem_details_dict = problem_details_dict,
        tempmat_storage = problem_details_dict["tempmat_storage"])
    end
    return out
end

function calc_tempmats(problem)
    J = length(problem.Xvec);
    #     ----------------------------
    # JMB need to fix args here and fix references to this function
    s = problem.data[:, r"shares"];
    exchange = problem.exchange;
    bO = problem.bO;
    bernO = convert.(Integer, bO);
    
    tempmats = []
    perm_s = zeros(size(s));
    dsids = zeros(J,J,size(s,1)) # initialize matrix of ∂s^{-1}/∂s

    for j1 = 1:J
        which_group = findall(j1 .∈ exchange)[1];
        first_product_in_group = exchange[which_group][1];

        perm = collect(1:J);
        perm[first_product_in_group] = j1; perm[j1] = first_product_in_group;
    
        perm_s .= s;
        perm_s[:,first_product_in_group] = s[:,j1]; perm_s[:,j1] = s[:,first_product_in_group];
                
        for j2 = 1:J 
            tempmat_s = zeros(size(s,1),1)
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
            push!(tempmats, tempmat_s);
        end
    end
    temp_storage_mat = reshape(tempmats, J,J);
    temp_elast_mats = deepcopy(temp_storage_mat);
    for j1 = 1:J
        for j2 = 1:J
            temp_elast_mats[j1,j2] = temp_storage_mat[j2,j1];
        end
    end
    temp_storage_mat = temp_elast_mats;
    return temp_storage_mat
end

function elast_penalty_quantile(θ, exchange::Array, 
    elast_mats::Matrix{Any}, elast_prices::Matrix{Float64}, 
    lambda::Real, 
    conmat::Dict; 
    J = maximum(maximum.(exchange)),
    during_obj = false,
    quantile_vec = [0.5], 
    problem_details_dict = Dict(), 
    tempmat_storage = [])

    # print("Elasticity penalty")
    # @show problem_details_dict
    df = problem_details_dict["data"];
    at = df[!,r"prices"];

    X = problem_details_dict["Xvec"];
    J = length(X);
    B = problem_details_dict["Bvec"];
    bO = problem_details_dict["bO"];
    exchange = problem_details_dict["exchange"];

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

    # Shares to evaluate derivatives -- bad form, holdover from old code
    svec = s;
    
    # Share Jacobian
    dsids = zeros(eltype(θ), J,J, size(s,1)) # initialize matrix of ∂s^{-1}/∂s

    for j1 = 1:J 
        if j1 ==1 
            init_ind = 0;
        else
            init_ind = sum(size.(X[1:j1-1],2))
        end
        
        θ_j1 = θ[init_ind+1:init_ind+size(X[j1],2)];
        for j2 = 1:J 
            dsids[j1,j2,:] .= tempmat_storage[j1,j2] * θ_j1;
        end
    end
    
    Jmat = eltype(θ)[]; # vector of derivatives of inverse shares
    J_sp = zeros(eltype(θ), size(svec[:,1]));
    all_own = zeros(eltype(θ), size(svec,1),J);
    svec2 = svec;
    temp = zeros(eltype(θ), J,J);
    all_elast_mat = zeros(eltype(θ), J,J,length(s[:,1]));
    penalty = zero(eltype(θ));

    for ii = 1:length(dsids[1,1,:])
        J_s = zeros(eltype(θ), J,J);
        for j1 = 1:J
            for j2 = 1:J
                J_s[j1,j2] = dsids[j1,j2,ii]
            end
        end
        try 
            temp = -1*inv(J_s);
        catch
            temp = J_s;
            penalty += 1e5;
        end
    
        # Market vector of prices/shares
        ps = at[ii,:] ./svec2[ii,:];
        ps_mat = zeros(J,J)
        for j1 = 1:J, j2 = 1:J 
            ps_mat[j1,j2] = at[ii,j2]/svec2[ii,j1];
        end
         
        all_elast_mat[:,:,ii] = temp .* ps_mat
    end
        
    # J = length(problem.Xvec);
    output_leq = zeros(eltype(θ), J,J,length(quantile_vec));
    output_geq = zeros(eltype(θ), J,J,length(quantile_vec));
    for qi in eachindex(quantile_vec)
        for j1 ∈ 1:size(all_elast_mat,1)
            for j2 ∈ 1:size(all_elast_mat,1)
                # output[j1,j2,q] = quantile(getindex.(all_elast_mat, j1, j2), q);
                output_leq[j1,j2,qi] = quantile(all_elast_mat[j1,j2,:], 1.0 - quantile_vec[qi]);
                output_geq[j1,j2,qi] = quantile(all_elast_mat[j1,j2,:], quantile_vec[qi]);
                # mean(all_elast_mat[j1,j2,:]); #
            end
        end
    end

    # calculate penalty for each provided element of quantile_vec 
    if conmat[:subs] !=[]
        for q = 1:length(quantile_vec)
            penalty += sum((output_leq[:,:,q] .< conmat[:subs]) .* abs.(output_leq[:,:,q]).^2 .* lambda); # + 10 * lambda * ((DET + 1e-12) / DET - 1); 
        end
    end
    if conmat[:complements] !=[]
        for q = 1:length(quantile_vec)
            penalty += sum((output_geq[:,:,q] .> conmat[:complements]) .* abs.(output_geq[:,:,q]).^2 .* lambda); # + 10 * lambda * ((DET + 1e-12) / DET - 1); 
        end
    end

    if conmat[:monotone] !=[]
        for q = 1:length(quantile_vec)
            penalty += sum((output_geq[:,:,q] .> conmat[:monotone]) .* abs.(output_geq[:,:,q]).^2 .* lambda); # + 10 * lambda * ((DET + 1e-12) / DET - 1); 
        end
        # @show output_geq[:,:,1] penalty
    end
    
    return penalty; #output_geq[:,:,1]
end

function elast_penalty_JUMP(θ, exchange::Array, 
    elast_mats::Matrix{Any}, elast_prices::Matrix{Float64}, 
    lambda::Real, 
    conmat::Dict; 
    J = maximum(maximum.(exchange)),
    during_obj = false,
    quantile_vec = [0.5], 
    problem_details_dict = Dict(), 
    tempmat_storage = [])

    # print("Elasticity penalty")
    # @show problem_details_dict
    df = problem_details_dict["data"];
    at = df[!,r"prices"];

    X = problem_details_dict["Xvec"];
    J = length(X);
    B = problem_details_dict["Bvec"];
    bO = problem_details_dict["bO"];
    exchange = problem_details_dict["exchange"];

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

    # Shares to evaluate derivatives -- bad form, holdover from old code
    svec = s;
    
    # Share Jacobian
    dsids = zeros(eltype(θ), J,J, size(s,1)) # initialize matrix of ∂s^{-1}/∂s

    for j1 = 1:J 
        if j1 ==1 
            init_ind = 0;
        else
            init_ind = sum(size.(X[1:j1-1],2))
        end
        
        # θ_j1 = θ[init_ind+1:init_ind+size(X[j1],2)];
        for j2 = 1:J 
            temp_j1_j2 = tempmat_storage[j1,j2];
            deriv = zeros(eltype(θ), size(s,1));
            # Build the derivative by column, for JuMP 
            # for i = 1:size(X[j1],2)
            #     deriv += temp_j1_j2[:,i] .* θ[init_ind+i];
            # end
            for i = 1:size(X[j1],2)
                for k = 1:length(deriv)
                    deriv[k] += temp_j1_j2[k,i] * θ[init_ind+i];
                end
            end
            dsids[j1,j2,:] =  deriv;
        end
    end
    
    Jmat = eltype(θ)[]; # vector of derivatives of inverse shares
    J_sp = zeros(eltype(θ), size(svec[:,1]));
    all_own = zeros(eltype(θ), size(svec,1),J);
    svec2 = svec;
    temp = zeros(eltype(θ), J,J);
    all_elast_mat = zeros(eltype(θ), J,J,length(s[:,1]));
    penalty = zero(eltype(θ));

    for ii = 1:length(dsids[1,1,:])
        J_s = zeros(eltype(θ), J,J);
        for j1 = 1:J
            for j2 = 1:J
                J_s[j1,j2] = dsids[j1,j2,ii]
            end
        end
        # try 
            temp = -1*inv(J_s);
        # catch
        #     temp = J_s;
        #     penalty += 1e5;
        # end
    
        # Market vector of prices/shares
        ps = at[ii,:] ./svec2[ii,:];
        ps_mat = zeros(J,J)
        for j1 = 1:J, j2 = 1:J 
            ps_mat[j1,j2] = at[ii,j2]/svec2[ii,j1];
        end
         
        all_elast_mat[:,:,ii] = temp;# .* ps_mat
    end
        
    # J = length(problem.Xvec);
    output_leq = zeros(eltype(θ), J,J,length(quantile_vec));
    output_geq = zeros(eltype(θ), J,J,length(quantile_vec));
    for qi in eachindex(quantile_vec)
        for j1 ∈ 1:size(all_elast_mat,1)
            for j2 ∈ 1:size(all_elast_mat,1)
                # output[j1,j2,q] = quantile(getindex.(all_elast_mat, j1, j2), q);
                output_leq[j1,j2,qi] = quantile(all_elast_mat[j1,j2,:], 1 - quantile_vec[qi]);
                output_geq[j1,j2,qi] = quantile(all_elast_mat[j1,j2,:], quantile_vec[qi]);
                # mean(all_elast_mat[j1,j2,:]); #
            end
        end
    end
    
    if conmat[:subs] !=[]
        for q = 1:length(quantile_vec)
            penalty += sum((output_leq[:,:,q] .< conmat[:subs]) .* abs.(output_leq[:,:,q]).^2 .* lambda); # + 10 * lambda * ((DET + 1e-12) / DET - 1); 
        end
    end
    if conmat[:complements] !=[]
        for q = 1:length(quantile_vec)
            penalty += sum((output_geq[:,:,q] .> conmat[:complements]) .* abs.(output_geq[:,:,q]).^2 .* lambda); # + 10 * lambda * ((DET + 1e-12) / DET - 1); 
        end
    end
    # @show conmat[:monotone] output[:,:,1] (output[:,:,1].> conmat[:monotone])

    if conmat[:monotone] !=[]
        for q = 1:length(quantile_vec)
            penalty += sum((output_geq[:,:,q] .> conmat[:monotone]) .* abs.(output_geq[:,:,q]).^2 .* lambda); # + 10 * lambda * ((DET + 1e-12) / DET - 1); 
        end
        # @show output_geq[:,:,1] penalty
    end
    
    return penalty #output_geq[:,:,1]
end

function elasticities_on_grid(problem::NPDProblem)
    # θ::AbstractArray, exchange::Array, elast_mats::Matrix{Any}, elast_prices::Matrix{Float64}, 
    # lambda::Real, conmat::Matrix{Float64}; 
    # J = maximum(maximum.(exchange)),
    # during_obj = false

    β = problem.results.minimizer
    θ = β[1:problem.design_width]
    γ = β[length(θ)+1:end]
    γ[1] = 1;
    for i ∈ problem.normalization
        γ[i] =0; 
    end
    for i∈eachindex(problem.mins)
        θ[problem.mins[i]] = θ[problem.maxs[i]]
    end

    at = problem.elast_prices;
    elast_mats = problem.elast_mats;
    J = maximum(maximum.(problem.exchange));

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
            
            # try
                dsids[j1,j2,:] .= elast_mats[j1,j2] * θ_j1; 
            # catch
                # @show size(θ_j1) size(elast_mats[j1,j2])
                # @show j1 j2
                # error("Error in inner_elasticity functions")
            # end
        end
    end
    
    Jmat = eltype(θ)[];
    jacobian_vec = eltype(θ)[];
    failed_inverse = false;

    penalty = zero(eltype(θ));

    vector_of_elast_mats = [];
    for ii = 1:length(dsids[1,1,:])
        for j1 = 1:J
            for j2 = 1:J
                J_s[j1,j2] = dsids[j1,j2,ii]
            end
        end

        # try
            temp = -1*inv(J_s);
        # catch
            # failed_inverse = true;
            # break
        # end
        
        push!(vector_of_elast_mats, temp)
    end

    return vector_of_elast_mats
end