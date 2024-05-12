function calc_tempmats(problem::NPDProblem)
    J = length(problem.Xvec);

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
            tempmat_s, a, b = NPDemand.make_interactions(tempmat_s, exchange, bernO, j1, perm);
            push!(tempmats, tempmat_s);
        end
    end
    # Take transpose of temp_storage so that it's correct for future use
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

function inner_elast_loop(dsids_i, J, at, svec; type = "jacobian")
    J_s = [dsids_i[j1,j2] for j1 in 1:J, j2 in 1:J]
    temp = -1*inv(J_s);

    if type =="jacobian"
        return temp 
    else
        return temp .* [at[j2]/svec[j1] for j1 in 1:J, j2 in 1:J]
    end
end

function elast_mat_zygote(θ, problem,
    tempmat_storage = []; 
    at = [], s = [], type = "jacobian")

    J = length(problem.Xvec);
    exchange = problem.exchange;
    
    design_width = sum(size.(problem.Xvec,2));
    indexes = [0;cumsum(size.(problem.Xvec,2))];

    dsids_raw = [tempmat_storage[j1,j2] * θ[indexes[j1]+1:indexes[j1+1]] for j1 = 1:J, j2 = 1:J];
    
    dsids = [dsids_raw[j1,j2][i] for j1 = 1:J, j2 = 1:J, i = 1:length(dsids_raw[1,1])]
    all_elast_mat = [inner_elast_loop(dsids[:,:,ii], J, at[ii,:], s[ii,:]; type = type) for ii in 1:length(dsids[1,1,:])];
    return all_elast_mat
end

function gmm(x::Matrix{T}, problem::NPDemand.NPDProblem, bigA::Vector{Matrix{Float64}}) where T<:Real
    design_width::Int64 = problem.design_width;
    Bvec::Vector{Matrix{Float64}} = problem.Bvec;
    Xvec::Vector{Matrix{Float64}} = problem.Xvec;
    Avec::Vector{Matrix{Float64}} = problem.Avec;
    J::Int = length(Avec);
    β = x[1:design_width];
    γ = x[design_width+1:end];

    indexes = vcat(0,cumsum(size.(Xvec,2)));
    out = zero(T);
    for i in 1:J
        out += convert(T, (Bvec[i]*γ - Xvec[i] * β[(indexes[i]+1:indexes[i+1])])'*bigA[i]*(Bvec[i] * γ - Xvec[i] * β[indexes[i]+1:indexes[i+1]]))
    end
    return out
end

function sieve_to_betas_index(problem)
    starts = [1;cumsum(size.(problem.Xvec,2))[1:end-1] .+ 1]
    ends = cumsum(size.(problem.Xvec,2))
    cols = []
    for i in 1:length(problem.exchange)
        push!(cols, starts[minimum.(problem.exchange)[i]]:ends[minimum.(problem.exchange)[i]])
    end
    return reduce(vcat, cols)
end

function get_nbetas(problem)
    sieve_widths = size.(problem.Xvec,2);
    first_products = first.(problem.exchange);
    nbetas = [getindex(sieve_widths, i) for i in first_products]
    return nbetas
end

function get_lower_bounds(problem)
    A = problem.Aineq[:,sieve_to_betas_index(problem)]
    lbs = []
    for j in 1:size(A,2)
        if findall(A[:,j] .== -1) == []
            push!(lbs, nothing)
        else
            push!(lbs, findall(vec(sum(A[findall(A[:,j] .== -1),:], dims=1)) .== 1))
        end
    end
    return lbs
end

function get_parameter_order(lbs)
    assigned = []
    unassigned = [1:length(lbs);]
    while length(unassigned) > 0
        revisit = []
        for i in unassigned
            if isnothing(lbs[i]) || count(x->x in lbs[i], assigned) == length(lbs[i])
                push!(assigned, i)
            else 
                if any(x->i<x, lbs[i]) || any(x->x in lbs[i], revisit)
                    push!(revisit, i)
                else 
                    push!(assigned, i)
                end        
            end
        end
        unassigned = revisit
    end
    return assigned
end

function reparameterization(betastar, lbs, parameter_order)
    nbeta = length(betastar)
    buffer_beta = Zygote.Buffer(betastar); # Have to define a "Buffer" to make an editable object for Zygote
    for i in parameter_order
        if isnothing(lbs[i])
            buffer_beta[i] = betastar[i]
        else
            wchlb::eltype(betastar) = findmax(buffer_beta[lbs[i]])[1]
            buffer_beta[i] = wchlb + exp(betastar[i])
        end
    end
    return copy(buffer_beta)
end

function reparameterization_draws(betastar_draws, lbs, parameter_order)
    nbeta = size(betastar_draws, 2)
    ndraws = size(betastar_draws, 1)
    beta_draws = zeros(ndraws,nbeta)
    for r in 1:ndraws
        for i in parameter_order
            if isnothing(lbs[i])
                beta_draws[r,i] = betastar_draws[r,i]
            else
                beta_draws[r,i] = findmax(beta_draws[r, lbs[i]])[1] + exp(betastar_draws[r,i])
            end
        end
    end
    return beta_draws
end

function map_to_sieve(beta, gamma, exchange, nbetas, problem)
    @assert sum(nbetas) == length(beta)
    
    nbeta = sum(nbetas); # number of parameters in each unique sieve
    J = length(problem.Xvec);

    # indexes in sieve space 
    starts_sieve = [1;cumsum(size.(problem.Xvec,2))[1:end-1] .+ 1]; 
    ends_sieve = cumsum(size.(problem.Xvec,2))

    # indexes in base parameter space
    starts_params = [1;nbetas[1].+1];
    ends_params = cumsum(nbetas);

    # Transform 
    sieve_params = zeros(eltype(beta), problem.design_width);     
    for j in 1:J
        which_group = findfirst(j .∈  exchange); # find the group corresponding to this product
        sieve_params[starts_sieve[j]:ends_sieve[j]] = beta[starts_params[which_group]:ends_params[which_group]]
    end

    all_params = [sieve_params; 1.0; gamma];
    all_params = reshape(all_params, 1, length(all_params));

    return all_params
end

function pick_step_size(problem, prior, tempmats, bigA; target = 0.2, n_samples = 100)
    step_grid = collect(range(0.001, 3, length = 100)) ./ sqrt(problem.design_width)
    accept = [];
    for x in step_grid
        step = x;
        chain = Turing.sample(mysample(problem, prior, tempmats, bigA), MH(
            :gamma => AdvancedMH.RandomWalkProposal(Normal(0, step)),
            :betastar =>  AdvancedMH.RandomWalkProposal(MvNormal(zeros(sum(nbetas)), diagm(step*ones(sum(nbetas)))))
            ), n_samples, initial_params = start); 
        push!(accept, mean(chain["betastar[1]"][2:end,:] - chain["betastar[1]"][1:(end-1),:] .!= 0))
    end
    return step_grid[findmin(abs.(accept .- target))[2]], step_grid, accept
end

@model function sample_quasibayes(problem::NPDemand.NPDProblem, 
    prior::Dict, 
    tempmats=[], 
    weight_matrices::Vector{Matrix{Float64}}=[],
    prices::Matrix{Float64} = Matrix(problem.data[!,r"prices"]), 
    shares::Matrix{Float64} = Matrix(problem.data[:, r"shares"]); penalty = 1_000)
    
    # prior
    betabar = prior["betabar"]
    gammabar = prior["gammabar"]
    vbeta = prior["vbeta"]
    vgamma = prior["vgamma"]    
    lbs = prior["lbs"]
    parameter_order = prior["parameter_order"]
    nbetas = prior["nbetas"]
    
    gamma_length = size(problem.Bvec[1],2);

    gamma ~ MvNormal(gammabar,vgamma*diagm(ones(gamma_length-1)));
    betastar ~ MvNormal(betabar, diagm(vbeta))
    
    # Apply reparameterization
    beta = reparameterization(betastar, lbs, parameter_order)

    # Format parameter vec so that gmm can use it
    all_params = map_to_sieve(beta, gamma, problem.exchange, nbetas, problem)

    if tempmats!=[]
        J = length(problem.Xvec);
        elasts = elast_mat_zygote(all_params, problem, tempmats; at = prices, s = shares);
        reshaped_elasts = [elasts[i][j1,j2] for j1=1:J, j2 = 1:J, i = 1:size(problem.data,1)];

        elasticity_check = run_elasticity_check(reshaped_elasts, problem.constraints)

        if elasticity_check[1]
            # Quasi-Likelihood
            Turing.@addlogprob! -0.5*gmm(all_params, problem, weight_matrices)
            return
        else
            Turing.@addlogprob! (-0.5*gmm(all_params, problem, weight_matrices)- penalty)
            return
        end
    else
        # Quasi-Likelihood
        Turing.@addlogprob! -0.5*gmm(all_params, problem, weight_matrices)
        return
    end
end

function posterior_elasticities(j, k, betadraws, gammadraws, tempmats, problem)
    ndraws = min(size(betadraws,1), 1_000);
    tmpout = zeros(size(problem.data,1), ndraws)
    for i in 1:ndraws
        params_i = map_to_sieve(betadraws[i,:], gammadraws[i,:], problem.exchange, nbetas, problem)
        tmpout[:,i] = getindex.(elast_mat_zygote(params_i, problem, tempmats; 
            at = Matrix(problem.data[!,r"prices"]), s = Matrix(problem.data[!,r"shares"])), j,k); 
    end
    return tmpout
end


function find_starting_point(problem, prior, 
    tempmats, weight_matrices; 
    n_attempts = 1000)
    
    prior_chain = Turing.sample(sample_quasibayes(problem, prior, tempmats, weight_matrices), Prior(), n_attempts)

    nbetas = prior["nbetas"];
    lbs = prior["lbs"];
    parameter_order = prior["parameter_order"];
    gamma_length = size(problem.Bvec[1],2);

    betastardraws = hcat([prior_chain["betastar[$i]"] for i in 1:sum(nbetas)]...)
    betadraws = reparameterization_draws(betastardraws, lbs, parameter_order)
    gammadraws = hcat([prior_chain["gamma[$i]"] for i in 1:gamma_length-1]...);
    J = length(problem.Xvec);

    # Initialize output
    param_out = zeros(sum(prior["nbetas"]) + size(gammadraws,2))

    # Start looping over samples parameters
    i=1;
    constraints_satisfied = false
    while (i < n_attempts) & !(constraints_satisfied) 
        sieve_params = map_to_sieve(betadraws[i,:], gammadraws[i,:], problem.exchange, prior["nbetas"], problem)
        
        # Calculate elasticities at these parameters 
        elasts_i = elast_mat_zygote(sieve_params, problem, tempmats; 
            at = Matrix(problem.data[!,r"prices"]), s = Matrix(problem.data[!,r"shares"]));
        elasts_i = [elasts_i[i][j1,j2] for j1=1:J, j2 = 1:J, i = 1:size(problem.data,1)];
        
        # Then check if constraints are satisfied
        constraints_satisfied = run_elasticity_check(elasts_i, problem.constraints);

        i+=1;
    end

    if constraints_satisfied
        exit_flag = "success";
        param_out = [betastardraws[i-1,:];gammadraws[i-1,:]]
    else 
        exit_flag = "failed";
    end
    
    return param_out, exit_flag
end
