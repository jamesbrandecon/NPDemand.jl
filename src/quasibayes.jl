function calc_tempmats(problem::NPDProblem;
    recipe = nothing)

    J = length(problem.Xvec);

    s        = Matrix(problem.data[:, r"shares"]);
    exchange = problem.exchange;
    bO       = problem.approximation_details[:order];
    bernO    = convert.(Integer, bO);
    
    tempmats = Matrix{Float64}[]
    perm_s   = zeros(size(s));
    nbetas   = size.(problem.Xvec,2);
    approximation_details = problem.approximation_details;

    if approximation_details[:sieve_type] == "polynomial" && isnothing(recipe)
        recipes = [ begin
            ex2 = length(exchange)==J ? [] : adjust_exchange(exchange, j1)
            build_poly_recipe(J;
                order           = approximation_details[:order],
                max_interaction = approximation_details[:max_interaction],
                exchange        = ex2)
          end for j1 in minimum.(exchange)]
      end

    if isempty(exchange) exchange = []; end # if exchange is empty, set it to empty vector
    for j1 = 1:J
        which_group = findfirst(j1 .∈  exchange); # find the group corresponding to this product
        first_product_in_group = minimum(exchange[which_group]);
        _, permuted_shares, permutations = get_params_one_equation(j1; 
            exchange = exchange, 
            s = s, 
            θ = 1:problem.design_width, # not needed --providing a dummy vector
            nbetas = nbetas)
        
        for j2 = 1:J 
            # println("Calculating tempmat for j1 = ", j1, " and j2 = ", j2)
            tempmat_s = calc_derivative_sieve(permutations[j1], permutations[j2];
                exchange          = ((exchange==[]) | (approximation_details[:sieve_type] == "bernstein")) ? exchange : adjust_exchange(exchange, first_product_in_group),
                shares            = permuted_shares,             
                permuted_shares   = permuted_shares,
                perm              = permutations,
                bernO             = bernO,
                sieve_type        = approximation_details[:sieve_type],
                recipe            = approximation_details[:sieve_type] == "polynomial" ? recipes[which_group] : nothing
                )
            push!(tempmats, tempmat_s)
        end
    end
    # Take transpose of temp_storage so that it's correct for future use
    temp_storage_mat = reshape(tempmats, J,J);
    temp_elast_mats  = deepcopy(temp_storage_mat);
    for j1 = 1:J
        for j2 = 1:J
            temp_elast_mats[j1,j2] = temp_storage_mat[j2,j1];
        end
    end
    temp_storage_mat = temp_elast_mats;
    return temp_storage_mat
end

function inner_elast_loop(dsids_i::AbstractMatrix{T}, J::Int, at::AbstractVector{Float64}, svec::AbstractVector{Float64}; type::String = "jacobian") where T
    J_s = [dsids_i[j1,j2] for j1 in 1:J, j2 in 1:J]
    temp = -1*pinv(J_s);

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

    J           = length(problem.Xvec);
    indexes     = [0;cumsum(size.(problem.Xvec,2))];
    temp_length = size(problem.data,1); #length(dsids[1,1,:]);

    dsids_raw     = [tempmat_storage[j1, j2] * θ[indexes[j1]+1:indexes[j1+1]] for j1 in 1:J, j2 in 1:J]
    dsids         = [dsids_raw[j1, j2][i] for j1 in 1:J, j2 in 1:J, i in 1:length(dsids_raw[1, 1])]
    all_elast_mat = [inner_elast_loop(dsids[:, :, ii], J, at[ii, :], s[ii, :]; type = type) for ii in 1:temp_length]

    return all_elast_mat
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

function get_nbetas(problem::NPDemand.NPDProblem)
    sieve_widths = size.(problem.Xvec,2);
    first_products = first.(problem.exchange);
    nbetas = [getindex(sieve_widths, i) for i in first_products]
    return nbetas
end

function get_lower_bounds(problem)
    if problem.Aineq != []
        A = problem.Aineq[:,sieve_to_betas_index(problem)]
        lbs = []
        for j in axes(A,2)
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

function reparameterization(betastar::Vector{T}, lbs::Vector, parameter_order::Vector; buffer_beta = similar(betastar)) where T<:Real

    # buffer_beta = Zygote.Buffer(betastar); # Have to define a "Buffer" to make an editable object for Zygote
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
    return buffer_beta
end

function reparameterization_draws(betastar_draws, lbs, parameter_order)
    nbeta = size(betastar_draws, 2)
    ndraws = size(betastar_draws, 1)
    beta_draws = zeros(eltype(betastar_draws), ndraws,nbeta)
    if all(lbs .==5000) | (lbs == [])
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

function map_to_sieve(beta::AbstractArray{T}, gamma::AbstractArray{T}, exchange::Vector, 
    nbetas::Vector{Int64}, problem::NPDemand.NPDProblem; sieve_type = "bernstein") where T
    
    if sieve_type == "polynomial"
        # –– build product‐wise index (one block of size=size(Xvec[j],2))
        J            = length(problem.Xvec)
        counts       = size.(problem.Xvec,2)
        starts_prod  = [1; cumsum(counts)[1:end-1] .+ 1]
        ends_prod    = cumsum(counts)
        # –– build group‐wise index in β
        starts_grp   = [1; cumsum(nbetas)[1:end-1] .+ 1]
        ends_grp     = cumsum(nbetas)
        # –– fill each product j with its group’s β‐block
        sieve_params = zeros(T, sum(counts))
        for j in 1:J
            g = findfirst(x->j in x, exchange)
            sieve_params[starts_prod[j]:ends_prod[j]] .=
                beta[starts_grp[g]:ends_grp[g]]
        end
        # –– append constant and γ, reshape as before
        allp = [sieve_params; 1.0; gamma]
        return reshape(allp, 1, length(allp))
    else
        nbeta = sum(nbetas); # number of parameters in each unique sieve
        J = length(problem.Xvec);

        # indexes in sieve space 
        starts_sieve  = [1;cumsum(size.(problem.Xvec,2))[1:end-1] .+ 1]; 
        ends_sieve    = cumsum(size.(problem.Xvec,2))

        # indexes in base parameter space
        starts_params = [1;cumsum(nbetas)[1:end-1] .+ 1];
        ends_params   = cumsum(nbetas);

        # Transform 
        # sieve_params = Zygote.Buffer(zeros(T, problem.design_width));     
        sieve_params = zeros(T, problem.design_width);
        for j in 1:J
            which_group = findfirst(j .∈  exchange); # find the group corresponding to this product
            sieve_params[starts_sieve[j]:ends_sieve[j]] = beta[starts_params[which_group]:ends_params[which_group]]
        end

        # all_params = [copy(sieve_params); 1.0; gamma];
        all_params = [sieve_params; 1.0; gamma];
        all_params = reshape(all_params, 1, length(all_params));

        return all_params
    end
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
    tempmats::Matrix{Matrix{T}}=[], 
    weight_matrices::Vector{Matrix{T}}=[],
    prices::Matrix{T} = Matrix(problem.data[!,r"prices"]), 
    shares::Matrix{T} = Matrix(problem.data[:, r"shares"]); 
    penalty = 0,
    matrix_storage_dict = Dict(), 
    sieve_type = "bernstein") where T <: Real
    
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
    
    # @show length(betastar)

    # Apply reparameterization
    beta = sieve_type == "bernstein" ? reparameterization(betastar, lbs, parameter_order) : betastar;

    # Format parameter vec so that gmm can use it
    sieve_type = get(prior, "sieve_type", "bernstein")
    
    # print("-")
    all_params = map_to_sieve(
        beta, 
        gamma,   
        problem.exchange, 
        nbetas, 
        problem;
        sieve_type = sieve_type
        )
    
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
        # print("E")
        elasts = elast_mat_zygote(all_params, problem, tempmats; at = prices, s = shares);
        reshaped_elasts = [elasts[i][j1,j2] for j1=1:J, j2 = 1:J, i = 1:size(problem.data,1)];
        elasticity_check = run_elasticity_check(reshaped_elasts, problem.constraints, problem.exchange)
        # print("/")
        if elasticity_check[1]
            # Quasi-Likelihood
            # print(".")
            Turing.@addlogprob! -0.5 * size(problem.data,1) * objective(all_params)
            return
        else
            # print("-")
            Turing.@addlogprob! (-0.5 * size(problem.data,1) * objective(all_params) - penalty)
            return
        end
    else
        # Quasi-Likelihood
        Turing.@addlogprob! -0.5 * size(problem.data,1) * objective(all_params)
        return
    end
end

function posterior_elasticities(j, k, betadraws, gammadraws, tempmats, problem)
    ndraws = min(size(betadraws,1), 1_000);
    tmpout = zeros(eltype(betadraws), size(problem.data,1), ndraws)
    for i in 1:ndraws
        st       = problem.sampling_details.approximation_details[:sieve_type]
        params_i = map_to_sieve(betadraws[i,:], gammadraws[i,:],
                            problem.exchange, nbetas, problem;
                            sieve_type=st)
        tmpout[:,i] = getindex.(elast_mat_zygote(params_i, problem, tempmats; 
            at = Matrix(problem.data[!,r"prices"]), s = Matrix(problem.data[!,r"shares"])), j,k); 
    end
    return tmpout
end


function find_starting_point(problem, prior, 
    tempmats, weight_matrices; 
    n_attempts = 1000)
    
    prior_chain = Turing.sample(
        sample_quasibayes(
            problem, 
            prior, 
            tempmats, 
            weight_matrices), 
        Prior(), 
        n_attempts)

    nbetas          = prior["nbetas"];
    lbs             = prior["lbs"];
    parameter_order = prior["parameter_order"];
    gamma_length    = size(problem.Bvec[1],2);

    betastardraws   = hcat([prior_chain["betastar[$i]"] for i in 1:sum(nbetas)]...)
    betadraws       = reparameterization_draws(betastardraws, lbs, parameter_order)
    gammadraws      = hcat([prior_chain["gamma[$i]"] for i in 1:gamma_length-1]...);
    J               = length(problem.Xvec);

    # Initialize output
    param_out       = zeros(sum(prior["nbetas"]) + size(gammadraws,2))

    # Start looping over samples parameters
    i = 1;
    constraints_satisfied = false
    while (i < n_attempts) & !(constraints_satisfied) 
        st           = problem.sampling_details.approximation_details[:sieve_type]
        sieve_params = map_to_sieve(betadraws[i,:], gammadraws[i,:],
                               problem.exchange, prior["nbetas"], problem;
                               sieve_type=st)
        
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
    smc_method          = :grid, 
    seed                = 4132, 
    max_iter            = 1000, 
    adaptive_tolerance  = false, 
    max_violations      = 0.01,
    modulo_num          = 1, 
    approximation_details::Dict{Symbol, Any} = Dict()
    )

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
    nbeta           = length(lbs) == 0 ? sum(nbetas) : length(lbs);
    
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
    # betas           = NPDemand.reparameterization_draws(
    #     thetas[:,1:nbeta], 
    #     lbs, 
    #     parameter_order
    #     )
    st            = approximation_details[:sieve_type]
    thetas_sieve  = vcat([map_to_sieve(
                            betadraws[i,:], 
                            gammadraws[i,:],
                            problem.exchange, 
                            nbetas, 
                            problem;
                            sieve_type=st)
                         for i in 1:nparticles]...)
    
    # 2. Set initial weights
    smc_weights     = fill(1.0 / nparticles, nparticles)
    penalty         = range(0, max_penalty, length = grid_points);

    # violation_dict = report_constraint_violations(problem, 
    #         params = mean(thetas_sieve, StatsBase.weights(smc_weights), dims = 1))
    violation_dict_array = [
        report_constraint_violations_inner(problem, 
            params = thetas_sieve[i,:], verbose = false) for i in axes(thetas_sieve,1)
    ];
    violation_dict = Dict{Symbol, Float64}()
    for k in keys(violation_dict_array[1])
        push!(violation_dict, k => mean([violation_dict_array[i][k] for i in axes(thetas_sieve,1)]));
    end
    t = 1;

    viol_store = [];
    ess_store  = [];
    penalty_vec = [];
    
    Random.seed!(seed)
    prev_penalty = 0.01;
    new_penalty  = 1e-6;
    x_pen = range(1e-3, Float64.(max_penalty), length = Int(grid_points)); 

    if smc_method == :linear_grid
        penalty_list = x_pen;
    elseif smc_method == :logit_grid
        penalty_list = maximum(x_pen) ./ (1 .+ exp.(-2 .*(x_pen .- median(x_pen))));
    elseif smc_method == :geometric_grid
        penalty_list = geometric_grid(1e-3, Float64.(max_penalty), Int(grid_points));
    end

    t = 1;
    beta_dist, gamma_dist = make_prior_dists(prior, gamma_length);
    beta_μ = beta_dist.μ;
    beta_Σ = Matrix(beta_dist.Σ);
    gamma_μ = gamma_dist.μ;
    gamma_Σ = Matrix(gamma_dist.Σ);
    logprior_t_minus_1 = zeros(nparticles);

    failure_count = 0;
    while (violation_dict[:any] > max_violations) & (prev_penalty < max_penalty) & (t < max_iter)
        t = t+1
        print("\n Iteration "*string(t-1)*"...\r")

        if (smc_method == :adaptive) & (mod(t,modulo_num) == 0)
            # Solve for next step penalty
            print("\n Optimizing penalty... \r")
            # try 
                if adaptive_tolerance 
                    new_penalty = find_zero(f_ess, (prev_penalty, max_penalty), Bisection(); xatol = get_tolerance(prev_penalty))
                else
                    if t>1
                        ~, logprior_t_minus_1 = get_importance_weights(thetas_sieve, smc_weights, Float64(prev_penalty), Float64(prev_penalty), problem, logprior_t_minus_1 = logprior_t_minus_1);
                    end
                    custom_ub = max(prev_penalty * 10.0, prev_penalty + 0.01);
                    xatol = 0.0002;
                    try 
                        new_penalty = find_zero(x -> f_ess(x, thetas_sieve, smc_weights, prev_penalty, problem, ess_threshold, logprior_t_minus_1=logprior_t_minus_1), (prev_penalty, min(custom_ub, max_penalty)), Bisection(); xatol = xatol, verbose = false)
                        new_penalty = new_penalty - xatol;
                    catch 
                        new_penalty = custom_ub;
                    end
                    if abs(prev_penalty - new_penalty) > 2 * xatol # adjustment to make sure tolerance isn't an issue
                        new_penalty = new_penalty - xatol;
                    end
                end
            # catch
                # error("Solving for penalty failed. Try reducing `ess_threshold` or increasing the number of particles (by reducing `burn_in` or `skip`)")
            # end
        elseif (smc_method == :adaptive) # every other iteration, use the previous penalty and re-sample again
            new_penalty = prev_penalty;
        else
            new_penalty = penalty_list[t]
        end
        
        # Calculate normalized importance weights
        log_smc_weights, ~ = get_importance_weights(thetas_sieve, smc_weights, prev_penalty, new_penalty, problem)

        smc_weights .= exp.(log_smc_weights .- maximum(log_smc_weights))
        smc_weights .= smc_weights ./ sum(smc_weights)
        ess = 1 / sum(smc_weights.^2)
        push!(ess_store, ess)

        # Always resample under adaptive approach
        indices = 1:nparticles;
        try 
            indices = wsample(1:nparticles, smc_weights, nparticles)
        catch
            failure_count +=1;
            if failure_count == 1
                @warn "Resampling failed. Running new MH steps without resampling..."
                indices = 1:nparticles;
            else
                @warn "Resampling failed twice. Exiting..."
                continue
            end
        end

        thetas = thetas[indices,:]
        thetas_sieve = thetas_sieve[indices,:]
        smc_weights .= 1.0 / nparticles

        # Stoage
        logprior_storage = zeros(eltype(thetas), nparticles)
        loglike_storage = zeros(eltype(thetas), nparticles)

        # 3.4 Perturb particles via MH step(s)
        n_accept = zeros(nparticles);
        proposal_distribution = [];
        try 
            Sigma = cov(thetas); # covariance matrix parameters, used to improve proposals
            Sigma = Sigma .* 2.38^2 ./ size(Sigma,1); # scale covariance matrix
            proposal_distribution = MvNormal(zeros(length(thetas[1,:])), step_size .* Sigma);
        catch
            @warn "Covariance matrix is not positive definite. Using identity matrix instead. If using a non-adaptive grid, you may wish to increase `grid_points`."
            Sigma = diagm(ones(size(cov(thetas),1)))
            proposal_distribution = MvNormal(zeros(length(thetas[1,:])), step_size .* Sigma);
        end
        
        print("\n Running Metropolis-Hasting steps....\n")
        Threads.@threads for i in ProgressBar(axes(thetas,1))
            for mh_iter in 1:mh_steps
                reparameterization_storage = zeros(eltype(thetas_sieve), sum(nbetas));
                # println(i)
                seed = MersenneTwister(3*Threads.threadid() + i);

                # Propose new values + reparameterize + map to sieve
                thetai_new       = thetas[i,:] + rand(seed, proposal_distribution)
                betai_new        = NPDemand.reparameterization(thetai_new[1:nbeta], lbs, parameter_order, buffer_beta = reparameterization_storage)
                thetai_sieve_new = NPDemand.map_to_sieve(betai_new, thetai_new[(nbeta+1):end], problem.exchange, nbetas, problem, sieve_type=st)
            
                # Evaluate prior
                # logprior_new     = logprior_smc(thetai_new[1:nbeta], thetai_new[(nbeta+1):end], beta_dist, gamma_dist) + logpenalty_smc(thetai_sieve_new, new_penalty, problem)
                logprior_new     = logprior_smc(thetai_new[1:nbeta], thetai_new[(nbeta+1):end], beta_μ, beta_Σ, gamma_μ, gamma_Σ) + logpenalty_smc(thetai_sieve_new, new_penalty, problem)
                if mh_iter == 1
                    logprior_old = logprior_smc(thetas[i,1:nbeta], thetas[i,(nbeta+1):end], beta_μ, beta_Σ, gamma_μ, gamma_Σ) + logpenalty_smc(thetas_sieve[i,:], new_penalty, problem)
                else
                    logprior_old = logprior_storage[i]
                end

                # # Evaluate likelihood
                loglike_new     = -0.5 * size(problem.data,1) * gmm(reshape(thetai_sieve_new, size(thetas_sieve,2), 1), problem, problem.weight_matrices)
                if mh_iter == 1
                    loglike_old = -0.5 * size(problem.data,1) * gmm(reshape(thetas_sieve[i,:], size(thetas_sieve,2), 1), problem, problem.weight_matrices)
                else
                    loglike_old = loglike_storage[i]
                end

                # # Calculate MH acceptance ratio
                logratio = loglike_new + logprior_new - loglike_old - logprior_old
                if rand(seed) < exp(logratio)
                    thetas[i,:]         = thetai_new
                    thetas_sieve[i,:]   = thetai_sieve_new
                    logprior_storage[i] = logprior_new
                    loglike_storage[i]  = loglike_new
                    n_accept[i]        += 1
                end
                GC.safepoint()
            end
        end
        n_accept = n_accept ./ mh_steps
        accept_rate = round(mean(n_accept), digits = 2);
        # println("\n Average MH acceptance rate: $(accept_rate)")

        # Check constraints 
        violation_dict_array = [report_constraint_violations(problem, params = thetas_sieve[i,:], verbose = false) for i in axes(thetas_sieve,1)];
        violation_dict = Dict{Symbol, Float64}()
        for k in keys(violation_dict_array[1])
            push!(violation_dict, k => 
                round(
                    mean([violation_dict_array[i][k] for i in axes(thetas_sieve,1)]),
                    digits = 3)
            );
        end
        push!(viol_store, violation_dict[:any])

        # Store and update penalty
        push!(penalty_vec, new_penalty)
        prev_penalty = new_penalty;

        println("|--------------------------------|--------|")
        println("| Iteration results              |        |")
        println("|--------------------------------|--------|")
        println(@sprintf("| %-30s | %.4f   |", "Current Penalty", new_penalty))
        if !(isnan(ess) | isinf(ess))
            println(@sprintf("| %-30s | %.2f |", "ESS", Int(floor(ess))))
        else
            println(@sprintf("| %-30s | %.2f |", "ESS", NaN))
        end
        println(@sprintf("| %-30s | %.2f   |", "Average MH Acceptance rate", accept_rate))
        println("| Violations                     |        |")
        for (key, value) in violation_dict
            println(@sprintf("| %-30s | %.3f  |", key, value))
        end
    end

    return (; thetas, smc_weights, violations = viol_store, ess = ess_store, penalties = penalty_vec);
end

function logpdf_mvn(mu::Vector{T}, Sigma::Matrix{T}, theta::Vector{T}) where T<:Real
    n = length(mu)
    
    # Ensure that Sigma is positive definite
    L = cholesky(Sigma).L
    diff = theta - mu
    quadratic_form = sum((L \ diff) .^ 2)
    
    logdetSigma = 2.0 * sum(log.(diag(L))) # log determinant of Sigma
    logpdf = -0.5 * (n * log(2 * π) + logdetSigma + quadratic_form)
    
    return logpdf
end

# function f_ess(p::T, thetas_sieve::Matrix{T}, smc_weights::Vector{T}, 
#     prev_penalty::Real, problem::NPDemand.NPDProblem, 
#     ess_threshold::Real;
#     logprior_t_minus_1 = zeros(T, size(thetas_sieve,1))) where T<:Real

#     if sum(smc_weights)==0 
#         ess = 0
#     else 
#         wts, ~ = get_importance_weights(thetas_sieve, smc_weights, prev_penalty, p[1], problem, logprior_t_minus_1 = logprior_t_minus_1)
#         if sum(wts) == 0
#             ess = Inf; # 0
#         else 
#             ess = (1 / sum(wts.^2))
#         end
#     end
#     return ess - ess_threshold
# end

function f_ess(p::T, thetas_sieve::Matrix{T}, smc_weights::Vector{T}, 
    prev_penalty::Real, problem::NPDemand.NPDProblem, 
    ess_threshold::Real;
    logprior_t_minus_1 = zeros(T, size(thetas_sieve,1))) where T

    logwts, ~ = get_importance_weights(thetas_sieve, smc_weights, prev_penalty, p[1], 
            problem, 
            logprior_t_minus_1 = logprior_t_minus_1, 
            multithread = true)

    num = 2*maximum(logwts) + 2*log(sum(exp.(logwts .- maximum(logwts))))
    denom = maximum(2*logwts) + log(sum(exp.(2*logwts .- maximum(2*logwts))))
    log_ess = num - denom
    exp_log_ess = exp(log_ess) 
    if isfinite(exp_log_ess)
        return exp_log_ess - ess_threshold
    else
        return -ess_threshold;
    end
end

function get_importance_weights(thetas_sieve::Matrix{T}, smc_weights::Vector{T}, 
    penalty_prev::Real, penalty_new::Float64, problem::NPDemand.NPDProblem; 
    new_log_weights     = similar(smc_weights),
    logprior_t          = zeros(T, size(thetas_sieve,1)),
    logprior_t_minus_1  = zeros(T, size(thetas_sieve,1)),
    multithread         = false) where T<:Real

    if multithread == false 
        for i in axes(thetas_sieve,1)
            particle_sieve_i          = thetas_sieve[i,:]
            logprior_t[i]             = logpenalty_smc(particle_sieve_i, penalty_new, problem)
            if logprior_t_minus_1     == zeros(T, size(thetas_sieve,1))
                logprior_t_minus_1_i  = logpenalty_smc(particle_sieve_i, penalty_prev, problem)
                log_prior_ratio           = logprior_t[i] - logprior_t_minus_1_i
            else
                log_prior_ratio           = logprior_t[i] - logprior_t_minus_1[i]
            end
            new_log_weights[i]            = log(smc_weights[i]) + log_prior_ratio;
        end
    else
        Threads.@threads for i in axes(thetas_sieve,1)
            particle_sieve_i          = thetas_sieve[i,:]
            logprior_t[i]             = logpenalty_smc(particle_sieve_i, penalty_new, problem)
            if logprior_t_minus_1     == zeros(T, size(thetas_sieve,1))
                logprior_t_minus_1_i  = logpenalty_smc(particle_sieve_i, penalty_prev, problem)
                log_prior_ratio           = logprior_t[i] - logprior_t_minus_1_i
            else
                log_prior_ratio           = logprior_t[i] - logprior_t_minus_1[i]
            end
            new_log_weights[i]            = log(smc_weights[i]) + log_prior_ratio;
        end
    end
    # println("weights before normalizing")
    # println(mean(new_weights))
    # if sum(new_weights) > 0 
    #     new_weights = new_weights ./ sum(new_weights);
    # end
    # println("weights after normalizing")
    # println(mean(new_weights))

    return new_log_weights, logprior_t
end

function geometric_grid(A::Float64, B::Float64, n::Int)
    r = (B / A)^(1 / (n - 1))  # Common ratio for geometric progression
    grid = [A * r^(i - 1) for i in 1:n]
    return grid
end

function make_prior_dists(prior, gamma_length)
    betabar     = prior["betabar"]
    gammabar    = prior["gammabar"]
    vbeta       = prior["vbeta"]
    vgamma      = prior["vgamma"]    
    ngamma      = gamma_length-1;
    
    beta_dist   = MvNormal(betabar, diagm(vbeta))
    gamma_dist  = MvNormal(gammabar, vgamma*diagm(ones(ngamma)))

    return beta_dist, gamma_dist
end

# function logprior_smc(particle_betastar::AbstractArray{T}, particle_gamma::AbstractArray{T}, beta_dist, gamma_dist) where T
function logprior_smc(particle_betastar, particle_gamma, beta_μ, beta_Σ, gamma_μ, gamma_Σ)  
    # out_beta    = logpdf(beta_dist, particle_betastar)
    # out_gamma   = logpdf(gamma_dist, particle_gamma)
    out_beta    = logpdf_mvn(beta_μ, beta_Σ, particle_betastar)
    out_gamma   = logpdf_mvn(gamma_μ, gamma_Σ, particle_gamma)
    return out_beta + out_gamma
end

function approx_cdf_normal01(x::Real)::Float64
    # Constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p  = 0.3275911
    
    sign = x < 0 ? -1 : 1
    abs_x = abs(x) / sqrt(2.0)
    
    # Approximation of the error function using a series expansion
    t = 1.0 / (1.0 + p * abs_x)
    y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    erf_approx = 1.0 - y * exp(-abs_x^2)
    
    return 0.5 * (1.0 + sign * erf_approx)
end

function get_tolerance(p::Float64)
    mintol = 1e-4
    if p==0
        out = mintol
    else
        ndecimals = floor(log10(p))
        out = max(mintol, floor(10^ndecimals, digits=Int(abs(ndecimals))))
    end
    return out
end

function logpenalty_smc(particle_sieve::Array{T}, penalty::Real, problem::NPDemand.NPDProblem) where T
    distance    = report_constraint_violations_inner(problem, params = particle_sieve, verbose = false, output = "frac")
    # out         = 1 * sum(log.(cdf.(Normal(0,1), -2 * penalty * distance)))
    out         = 1 * sum(log.(approx_cdf_normal01.(-2 * penalty * distance)))
    return out
end

function loglikelihood(problem::NPDemand.NPDProblem, particle_betastar::Vector{T}, particle_gamma::Vector{T}, nbetas) where T
    nbetas          = NPDemand.get_nbetas(problem)
    lbs             = NPDemand.get_lower_bounds(problem)
    parameter_order = NPDemand.get_parameter_order(lbs)
    particle_beta   = NPDemand.reparameterization_draws(reshape(particle_betastar,1,sum(nbetas)), lbs, parameter_order);

    x = NPDemand.map_to_sieve(particle_beta, particle_gamma, problem.exchange, nbetas, problem);
   
    return -0.5 * gmm(x, problem, problem.weight_matrices)
end


function fe_posteriors(problem; FE::Union{Array, String} = [])
    if FE==[] 
        error("Please provide name of FE (the `FE` keyword argument)")
    end
    if (problem.chain ==[])
        error("No Markov chain found in the problem")
    end

    coef_names = [problem.fe_param_mapping[i].name for i in 1:length(problem.fe_param_mapping)]
    coefs_for_this_fe = findall(coef_names .== FE)
    all_gammas = problem.results.filtered_chain[:,end-size(problem.Bvec[1],2)+1:end];
    num_index_vars = length(problem.index_vars);
    
    df_fe = DataFrame()
    for (~, i) in enumerate(coefs_for_this_fe)
        val = problem.fe_param_mapping[i].value
        column_name = "Value$val"
        if i ==1 
            df_fe[!, column_name] = all_gammas[:, num_index_vars + i]
        else 
            df_fe[!, column_name] = all_gammas[:, num_index_vars + i]
        end
    end
    return df_fe
end
