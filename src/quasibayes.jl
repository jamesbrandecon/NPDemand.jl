function calc_tempmats(problem::NPDProblem)
    J = length(problem.Xvec);

    s = problem.data[:, r"shares"];
    exchange = problem.exchange;
    bO = problem.bO;
    bernO = convert.(Integer, bO);
    
    tempmats = Matrix{Float64}[]
    perm_s = zeros(size(s));
    # dsids = zeros(J,J,size(s,1)) # initialize matrix of ∂s^{-1}/∂s

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
            tempmat_s, ~, ~ = NPDemand.make_interactions(tempmat_s, exchange, bernO, j1, perm);
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

function inner_elast_loop(dsids_i::Matrix{T}, J::Int, at::Vector{Float64}, svec::Vector{Float64}; type::String = "jacobian") where T
    
    J_s = [dsids_i[j1,j2] for j1 in 1:J, j2 in 1:J]
    temp = -1*inv(J_s);

    if type == "jacobian"
        return temp 
    else
        return temp .* [at[j2]/svec[j1] for j1 in 1:J, j2 in 1:J]
    end
end

function elast_mat_zygote(θ::AbstractArray{T}, 
    problem::NPDemand.NPDProblem,
    tempmat_storage::Matrix{Matrix{Float64}} = []; 
    at::Matrix = [], s::Matrix = [], 
    type::String = "jacobian") where T <:Real

    J = length(problem.Xvec);
    indexes = [0;cumsum(size.(problem.Xvec,2))];

    dsids_raw = [tempmat_storage[j1,j2] * θ[indexes[j1]+1:indexes[j1+1]] for j1 = 1:J, j2 = 1:J];
    
    dsids = [dsids_raw[j1,j2][i] for j1 = 1:J, j2 = 1:J, i = 1:length(dsids_raw[1,1])]
    all_elast_mat::Vector{Matrix{Real}} = [inner_elast_loop(dsids[:,:,ii], J, at[ii,:], s[ii,:]; type = type) for ii in 1:length(dsids[1,1,:])];
    return all_elast_mat
end

function gmm_fast(x::Matrix{T}, problem::NPDemand.NPDProblem, 
    yZX::Vector{LinearAlgebra.Adjoint{Float64, Vector{Float64}}},
    XZy::Vector{Vector{Float64}},
    XX::Vector{Matrix{Float64}},
    design_width::Int,
    J::Int) where T<:Real

    γ2 = x[design_width+2:end]; 
    indexes = vcat(0,cumsum(size.(problem.Xvec,2)));
    out = zero(eltype(x));
    for i in 1:J
        θi = [x[indexes[i]+1:indexes[i+1]];γ2];
        out += -1 * yZX[i] * θi - θi' * XZy[i] + θi' * XX[i] * θi
    end
    return out[1]
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
    if problem.Aineq != []
        A = problem.Aineq[:,sieve_to_betas_index(problem)]
        lbs = []
        for j in 1:size(A,2)
            if findall(A[:,j] .== -1) == []
                push!(lbs, nothing)
            else
                push!(lbs, findall(vec(sum(A[findall(A[:,j] .== -1),:], dims=1)) .== 1))
            end
        end
    else 
        lbs = 5000 .* ones(Int, size(problem.Aeq[:,sieve_to_betas_index(problem)], 2))
    end
    return lbs
end

function get_parameter_order(lbs)
    if !(all(lbs .== 5000))
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
    else 
        assigned = collect(1:length(lbs));
    end
    return assigned
end

function reparameterization(betastar::Vector{T}, lbs::Vector, parameter_order::Vector) where T<:Real
    # buffer_beta = similar(betastar); #
    buffer_beta = Zygote.Buffer(betastar); # Have to define a "Buffer" to make an editable object for Zygote
    if all(lbs .== 5000)
        buffer_beta = betastar;
    else
        nbeta = length(betastar)
        for i in parameter_order
            if isnothing(lbs[i])
                buffer_beta[i] = betastar[i]
            else
                wchlb::eltype(betastar) = findmax(buffer_beta[lbs[i]])[1]
                buffer_beta[i] = wchlb + exp(betastar[i])
            end
        end
        # return copy(buffer_beta)
    end
    # return buffer_beta
    return copy(buffer_beta)
end

function reparameterization_draws(betastar_draws, lbs, parameter_order)
    nbeta = size(betastar_draws, 2)
    ndraws = size(betastar_draws, 1)
    beta_draws = zeros(ndraws,nbeta)
    if all(lbs .==5000)
        beta_draws .= betastar_draws;
    else
        for r in 1:ndraws
            for i in parameter_order
                if isnothing(lbs[i])
                    beta_draws[r,i] = betastar_draws[r,i]
                else
                    beta_draws[r,i] = findmax(beta_draws[r, lbs[i]])[1] + exp(betastar_draws[r,i])
                end
            end
        end
    end
    return beta_draws
end

function map_to_sieve(beta::AbstractArray{T}, gamma::AbstractArray{T}, exchange::Vector{Matrix{Int64}}, nbetas::Vector{Int64}, problem::NPDemand.NPDProblem) where T
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
    sieve_params = Zygote.Buffer(zeros(T, problem.design_width));     
    for j in 1:J
        which_group = findfirst(j .∈  exchange); # find the group corresponding to this product
        sieve_params[starts_sieve[j]:ends_sieve[j]] = beta[starts_params[which_group]:ends_params[which_group]]
    end

    all_params = [copy(sieve_params); 1.0; gamma];
    all_params = reshape(all_params, 1, length(all_params));

    return all_params
end

function pick_step_size(problem, prior, tempmats, bigA; target = 0.2, n_samples = 100)
    step_grid = collect(range(0.001, 3, length = 100)) ./ sqrt(problem.design_width)
    accept = [];
    for x in step_grid
        step = x;
        chain = Turing.sample(sample_quasibayes(problem, prior, tempmats, bigA), MH(
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
    shares::Matrix{Float64} = Matrix(problem.data[:, r"shares"]); 
    penalty = 1_000,
    matrix_storage_dict = Dict())
    
    # prior
    betabar         = prior["betabar"]
    gammabar        = prior["gammabar"]
    vbeta           = prior["vbeta"]
    vgamma          = prior["vgamma"]    
    lbs             = prior["lbs"]
    parameter_order = prior["parameter_order"]
    nbetas          = prior["nbetas"]
    
    gamma_length::Int = size(problem.Bvec[1],2);

    betastar ~ MvNormal(betabar, diagm(vbeta))
    gamma ~ MvNormal(gammabar, vgamma*diagm(ones(gamma_length-1)));
    
    # Apply reparameterization
    beta = reparameterization(betastar, lbs, parameter_order)

    # Format parameter vec so that gmm can use it
    all_params = map_to_sieve(beta, gamma, problem.exchange, nbetas, problem)
    
    # Define objective function 
    if matrix_storage_dict == Dict()
        objective = x -> gmm(x, problem, weight_matrices)
    else 
        yZX = matrix_storage_dict["yZX"]
        XZy = matrix_storage_dict["XZy"]
        XX = matrix_storage_dict["XX"]
        objective = x -> gmm_fast(x, problem, yZX, XZy, XX, problem.design_width, length(problem.Avec));
    end

    if !((penalty == 0) | (problem.constraints == [:exchangeability]))
        J = length(problem.Xvec);
        elasts = elast_mat_zygote(all_params, problem, tempmats; at = prices, s = shares);
        reshaped_elasts = [elasts[i][j1,j2] for j1=1:J, j2 = 1:J, i = 1:size(problem.data,1)];
        elasticity_check = run_elasticity_check(reshaped_elasts, problem.constraints, problem.exchange)

        if elasticity_check[1]
            # Quasi-Likelihood
            # print(".")
            Turing.@addlogprob! -0.5 * objective(all_params)
            return
        else
            # print("-")
            # rate = cdf(Normal(0,1), -1 * penalty * d[:any].^2) * 2
            Turing.@addlogprob! (-0.5 * objective(all_params) - penalty)
            return
        end
    else
        # Quasi-Likelihood
        Turing.@addlogprob! -0.5 * objective(all_params)
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
        constraints_satisfied = run_elasticity_check(elasts_i, problem.constraints, problem.exchange);

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

function smc(problem::NPDemand.NPDProblem; 
    grid_points::Int    = 50, 
    max_penalty::Real   = 5, 
    ess_threshold::Real = 100, 
    step_size::Real     = 0.1, 
    skip::Int           = 5,
    burn_in::Int        = 5000, 
    mh_steps            = 10, 
    seed                = 4132)

    # Define inputs to quasi-bayes sampling 
    prior           = problem.sampling_details.prior;
    lbs             = prior["lbs"]
    parameter_order = prior["parameter_order"]
    nbetas          = prior["nbetas"]
    gamma_length    = size(problem.Bvec[1],2);

    # 1. Run MCMC on unconstrained model
    start_row       = burn_in+1;
    skiplen         = skip;

    particles       = problem.chain;
    nbetas          = NPDemand.get_nbetas(problem);
    nbeta           = length(lbs)

    # report_constraint_violations(problem)
    
    betastardraws   = hcat([particles["betastar[$i]"] for i in 1:sum(nbetas)]...)[start_row:end,:]
    betadraws       = NPDemand.reparameterization_draws(betastardraws, lbs, parameter_order)
    gammadraws      = hcat([particles["gamma[$i]"] for i in 1:gamma_length-1]...)[start_row:end,:]

    # thin the markov chain
    L               = size(betastardraws,1);
    skip_inds       = 1:skiplen:L
    betadraws       = betadraws[skip_inds,:];
    gammadraws      = gammadraws[skip_inds,:];
    betastardraws   = betastardraws[skip_inds,:];
    nparticles      = size(betastardraws,1);

    # Format parameter vec so that gmm can use it
    thetas          = [betastardraws gammadraws]
    betas           = NPDemand.reparameterization_draws(thetas[:,1:nbeta], lbs, parameter_order)
    thetas_sieve    = vcat([NPDemand.map_to_sieve(betas[i,:], gammadraws[i,:], problem.exchange, nbetas, problem) for i in 1:nparticles]...)
    
    # 2. Set initial weights
    smc_weights     = fill(1.0 / nparticles, nparticles)
    penalty         = range(0, max_penalty, length = grid_points);

    # violation_dict = report_constraint_violations(problem, 
    #         params = mean(thetas_sieve, StatsBase.weights(smc_weights), dims = 1))
    violation_dict_array = [
        report_constraint_violations(problem, 
            params = thetas_sieve[i,:], verbose = false) for i in axes(thetas_sieve,1)
    ];
    violation_dict = Dict{Symbol, Float64}()
    for k in keys(violation_dict_array[1])
        push!(violation_dict, k => mean([violation_dict_array[i][k] for i in axes(thetas_sieve,1)]));
    end
    t = 1;

    viol_store = [];
    ess_store  = [];
    Sigma = cov(thetas); # covariance matrix parameters, used to improve proposals
    
    Random.seed!(seed)
    while (t < length(penalty)) & (violation_dict[:any] > 0.01)
        t = t+1
        print("\n Iteration "*string(t-1)*"...\r")
    
        # 3.1 Calculate importance weights
        new_weights = similar(smc_weights)
        for i in axes(thetas,1)
            particle_i          = thetas[i,:];
            betastar_i          = particle_i[1:nbeta];

            beta_i              = NPDemand.reparameterization_draws(reshape(betastar_i,1,nbeta), lbs, parameter_order);
            gamma_i             = particle_i[(nbeta+1):end];

            logprior_t          = logprior_smc(beta_i, gamma_i, penalty[t], problem);
            logprior_t_minus_1  = logprior_smc(beta_i, gamma_i, penalty[t-1], problem);

            prior_ratio         = exp(logprior_t - logprior_t_minus_1)
            new_weights[i]      = smc_weights[i] * prior_ratio
        end
        
        # 3.2 Normalize weights
        smc_weights = new_weights ./ sum(new_weights)
        
        # 3.3 Check ESS and resample if necessary
        ESS = 1 / sum(smc_weights.^2)
        push!(ess_store, ESS)
        if ESS < ess_threshold
            print("\n ESS below threshold -- Re-sampling")
            indices = wsample(1:nparticles, smc_weights, nparticles)
            thetas = thetas[indices,:]
            smc_weights .= 1.0 / nparticles
        end
        
        # 3.4 Perturb particles via MH step(s)
        n_accept = zeros(nparticles);
        print("\n Metropolis-Hasting steps: ") 
        for mh_iter in 1:mh_steps
            for i in axes(thetas,1)
                # proposal = thetas[i,:] + rand(step_size.*Normal(0,1), size(thetas[i,:]))
                proposal = thetas[i,:] + rand(MvNormal(zeros(length(thetas[1,:])),step_size .* Sigma))

                logprior_proposal   = logprior_smc(proposal[1:nbeta], proposal[(nbeta+1):end], penalty[t], problem)
                logprior_theta      = logprior_smc(thetas[i,1:nbeta], thetas[i,(nbeta+1):end], penalty[t], problem)
                
                loglik_proposal     = loglikelihood(problem, proposal[1:nbeta], proposal[nbeta+1:end], nbetas)
                loglik_theta        = loglikelihood(problem, thetas[i,1:nbeta], thetas[i,nbeta+1:end], nbetas)

                # Calculate MH acceptance ratio
                logratio = loglik_proposal + logprior_proposal - loglik_theta - logprior_theta
                if rand() < exp(logratio)
                    thetas[i,:]  = proposal
                    n_accept[i] += 1
                end
            end
            print("$mh_iter..")
        end
        n_accept = n_accept ./ mh_steps
        accept_rate = round(mean(n_accept), digits = 2);
        println("\n Average MH acceptance rate: $(accept_rate)")
        # println("Acceptance rate: "*string(n_accept/nparticles))

        # thetas = thetas .+ rand(step_size.*Normal(0,1), nparticles, nbeta + 1)
        betas = NPDemand.reparameterization_draws(thetas[:,1:nbeta], lbs, parameter_order)
        thetas_sieve = vcat([NPDemand.map_to_sieve(betas[i,:], thetas[i,(nbeta+1):end], problem.exchange, nbetas, problem) for i in 1:nparticles]...)
    
        # Check constraints 
        # violation_dict = report_constraint_violations(problem, params = mean(thetas_sieve, StatsBase.weights(smc_weights), dims = 1));
        violation_dict_array = [
        report_constraint_violations(problem, 
            params = thetas_sieve[i,:], verbose = false) for i in axes(thetas_sieve,1)
        ];
        violation_dict = Dict{Symbol, Float64}()
        for k in keys(violation_dict_array[1])
            push!(violation_dict, k => 
                round(
                    mean([violation_dict_array[i][k] for i in axes(thetas_sieve,1)]),
                    digits = 3)
            );
        end
        push!(viol_store, violation_dict[:any])
        
        display(violation_dict)
        println("Increasing penalty") 
    end

    return (; thetas, smc_weights, violations = viol_store, ess = ess_store);
end

function logprior_smc(particle_betastar::Array{T}, particle_gamma::Array{T}, penalty::Real, problem::NPDemand.NPDProblem) where T
    prior   = problem.sampling_details.prior;
    betabar = prior["betabar"]

    betabar         = prior["betabar"]
    gammabar        = prior["gammabar"]
    vbeta           = prior["vbeta"]
    vgamma          = prior["vgamma"]    
    lbs             = prior["lbs"]
    parameter_order = prior["parameter_order"]
    nbetas          = prior["nbetas"]

    # betastar ~ MvNormal(betabar, diagm(vbeta))
    # gamma ~ MvNormal(gammabar, vgamma*diagm(ones(gamma_length-1)));
    # @show size(problem.sampling_details.prior["betabar"]) size(particle_betastar)
    # @show size(diagm(vbeta))
    # @show logpdf(MvNormal(betabar, diagm(vbeta)), dropdims(particle_betastar, dims=1))
    # @show logpdf(MvNormal(gammabar, vgamma*diagm(ones(size(particle_gamma,1)))), particle_gamma)
    original_prior = logpdf(MvNormal(betabar, diagm(vbeta)), reshape(particle_betastar, length(particle_betastar))) + logpdf(MvNormal(gammabar, vgamma*diagm(ones(size(particle_gamma,1)))), particle_gamma)

    particle_beta = NPDemand.reparameterization_draws(reshape(particle_betastar,1,sum(nbetas)), lbs, parameter_order);
    particle = NPDemand.map_to_sieve(particle_beta, particle_gamma, problem.exchange, nbetas, problem);

    D = report_constraint_violations(problem, params = particle, verbose = false, output = "count");
    
    distance = D; 
    penalty_prior = 1 * sum(log.(cdf.(Normal(0,1), -1 * penalty * distance)));

    return penalty_prior + original_prior
end

function loglikelihood(problem::NPDemand.NPDProblem, particle_betastar::Vector{T}, particle_gamma::Vector{T}, nbetas) where T
    nbetas = NPDemand.get_nbetas(problem)
    lbs = NPDemand.get_lower_bounds(problem)
    parameter_order = NPDemand.get_parameter_order(lbs)
    particle_beta = NPDemand.reparameterization_draws(reshape(particle_betastar,1,sum(nbetas)), lbs, parameter_order);

    x = NPDemand.map_to_sieve(particle_beta, particle_gamma, problem.exchange, nbetas, problem);
    return -0.5 * gmm(x, problem, problem.weight_matrices)
end