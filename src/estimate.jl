function estimate_fast!(problem::NPDProblem; 
    linear_solver = "Ipopt", 
    verbose = true)

    β, γ = jmp_obj(problem, # was JMP_obj_constrained
        linear_solver = linear_solver, 
        verbose = verbose);
        
    problem.results = NPD_parameters([β;γ], []);
end

function jmp_obj(npd_problem::NPDProblem; linear_solver = "Ipopt", verbose = true)
    # Unpack tolerances, even if not using
    constraint_tol = npd_problem.constraint_tol;
    obj_xtol = npd_problem.obj_xtol;
    obj_ftol = npd_problem.obj_ftol;

    # Unpack data
    Avec = npd_problem.Avec;
    Xvec = npd_problem.Xvec;
    Bvec = npd_problem.Bvec;
    indexes = vcat(0,cumsum(size.(Xvec,2)));
    J = length(Xvec);

    # Define JuMP problem 
    if linear_solver =="Ipopt"
        verbose_int = 0;
        if verbose ==true
            verbose_int = 5;
        end
        model = Model(optimizer_with_attributes(Ipopt.Optimizer, 
            "constr_viol_tol" => constraint_tol,
            "print_level" => verbose_int));
    elseif linear_solver =="OSQP"
        model = Model(optimizer_with_attributes(OSQP.Optimizer, 
            "check_termination" => 20000,
            "max_iter" => 20000));
    end

    @variable(model, γ[1:size(npd_problem.Bvec[1],2)]);
    @variable(model, β[1:npd_problem.design_width]);
    verbose && println("Setting up problem in JuMP ....")
    @objective(model, Min,
    sum((Bvec[i]*γ - Xvec[i] * β[(indexes[i]+1:indexes[i+1])])'*Avec[i]*pinv(Avec[i]'*Avec[i])*Avec[i]'*(Bvec[i] * γ - Xvec[i] * β[indexes[i]+1:indexes[i+1]]) for i ∈ 1:J));

    # Add constraints
    @constraint(model, γ[1]==1); # Price coefficient normalized
    @constraint(model, [i = 1:size(npd_problem.Aineq,1)], # Enforcing inequality constraints
        sum(npd_problem.Aineq[i,:] .* β) <= 0)
    @constraint(model, [i = 1:size(npd_problem.Aeq,1)], # Enforcing exchangeability
        sum(npd_problem.Aeq[i,:] .* β) == 0)
    
    verbose && println("Solving problem in JuMP ....")

    # Solve problem and store results
    JuMP.optimize!(model);
    β_solved = value.(β);
    γ_solved = value.(γ);
    return β_solved, γ_solved
end

function make_conmat(problem)
    exchange = problem.exchange;
    J = length(problem.Xvec);

    conmat_monotone = [];
    if :monotone_nonlinear in problem.constraints 
        conmat_monotone = zeros(Float64,J,J);
        conmat_monotone .= Inf;
        for j = 1:J    
            conmat_monotone[j,j] = 0.0;
        end
    end

    if maximum(x ∈[:subs_in_group, :all_substitutes_nonlinear, :subs_across_group] for x ∈ problem.constraints)
        conmat_subs = zeros(Float64,J,J);
        conmat_subs .= -Inf;
    else
        conmat_subs=[];
    end
    if maximum(x∈[:complements_in_group, :complements_across_group] for x ∈ problem.constraints)
        conmat_complements = zeros(Float64,J,J);
        conmat_complements .= Inf;
    else
        conmat_complements = [];
    end
    if (:subs_in_group ∈ problem.constraints) | (:all_substitutes_nonlinear ∈ problem.constraints)
        # If subs_in_group, then only need to constrain within groups
        # All 
        for j1 = 1:J
            ej = getindex.(findall(j1 .∈ exchange),1)[1];
            for j2 = 1:J
                if (j2!=j1) | (j2 ∈ exchange[ej])
                    conmat_subs[j1,j2] = 0;
                end                
            end
        end
    end
    if (:subs_across_group ∈ problem.constraints) | (:all_substitutes_nonlinear ∈ problem.constraints)
        # All products in different groups should have conmat[j1,j2] = 0
        for j1 = 1:J
            ej = getindex.(findall(j1 .∈ exchange),1)[1];
            for j2 = 1:J
                if (j2 ∉ exchange[ej]) & (j1!=j2)
                    conmat_subs[j1,j2] = 0;
                end                
            end
        end
    end

    # For complements, sign of infinities are reversed
    if :complements_in_group ∈ problem.constraints
        for j1 = 1:J
            ej = getindex.(findall(j1 .∈ exchange),1)[1];
            for j2 = 1:J
                if (j2 ∈ exchange[ej])
                    conmat_complements[j1,j2] = 0;
                end                
            end
        end
    end
    if :complements_across_group ∈ problem.constraints
        # All products in different groups should have conmat[j1,j2] = 0
        for j1 = 1:J
            ej = getindex.(findall(j1 .∈ exchange),1)[1];
            for j2 = 1:J
                if (j2 ∉ exchange[ej])
                    conmat_complements[j1,j2] = 0;
                end                
            end
        end
    end   

    # conmat_subs = [-Inf 0.0; 0.0 -Inf];
    conmat = Dict(
        :subs => conmat_subs,
        :complements => conmat_complements, 
        :monotone => conmat_monotone
    )
    return conmat
end


"""
    estimate!(problem::NPDProblem;
        verbose = true,
        linear_solver = "Ipopt", 
        quasi_bayes = false,
        sampler = [], 
        n_samples::Int = 50_000,
        burn_in::Real = 0.25, 
        skip::Int = 5,
        n_attempts = 0,
        penalty = 0, 
        step::Union{Real, Symbol} = 0.01)

Estimates the problem using the specified parameters.

## Arguments
- `problem::NPDProblem`: The problem to be estimated.
- `verbose::Bool`: Whether to print verbose output. Default is `true`.
- `linear_solver::String`: The linear solver to use. Must be either "Ipopt" or "OSQP". Default is "Ipopt".
- `quasi_bayes::Bool`: Whether to use quasi-bayes sampling. Default is `false`.
- `sampler`: The sampler to use for quasi-bayes sampling. Default is an empty array.
- `n_samples::Int`: The number of samples to draw. Default is 50,000.
- `burn_in::Real`: The fraction of samples to drop as burn-in. Must be less than 1. Default is 0.25.
- `skip::Int`: The number of samples to skip between saved samples. Default is 5.
- `n_attempts`: The number of attempts to find a valid starting point for the sampler. Default is 0.
- `penalty`: The penalty value for the objective function. Default is 0.
- `step::Union{Real, Symbol}`: The step size for the sampler. Can be a real number or the symbol `:auto` to automatically calculate the step size. Default is 0.01.

"""
function estimate!(problem::NPDProblem;
    verbose = true,
    linear_solver = "Ipopt", 
    quasi_bayes = false,
    sampler = [], 
    n_samples::Int = 50_000,
    burn_in::Real = 0.25, 
    skip::Int = 5,
    n_attempts = 0,
    penalty = 0, 
    step::Union{Real, Symbol} = 0.01)

    try 
        @assert (step isa Real) | (step == :auto)
    catch 
        error("`step` must either be a Real or the Symbol :auto, which indicates a desire to have the step size automatically calculated")
    end

    # Check that linear solver is Ipopt or OSQP 
    if linear_solver ∉ ["Ipopt", "OSQP"]
        error("Linear solver must be Ipopt or OSQP")
    end

    # Unpack problem 
    df = problem.data;
    matrices = problem.matrices;
    Xvec = problem.Xvec;
    Bvec = problem.Bvec;
    Avec = problem.Avec;
    Aineq = problem.Aineq; 
    Aeq = problem.Aeq;
    mins = problem.mins;  
    maxs = problem.maxs;
    normalization = problem.normalization;
    design_width = problem.design_width;
    elast_mats = problem.elast_mats; 
    elast_prices = problem.elast_prices;
    constraint_tol = problem.constraint_tol;
    obj_xtol = problem.obj_xtol;
    obj_ftol = problem.obj_ftol;
    bO = problem.bO;
    exchange = problem.exchange;
    cfg = problem.cfg;
    weight_matrices = problem.weight_matrices;

    find_prices = findall(problem.index_vars .== "prices");
    price_index = find_prices[1];

    problem.tempmats = calc_tempmats(problem);

    # Estimate the problem only with linear constraints if not using quasi-bayes
    if !quasi_bayes
        verbose && println("Estimating problem in JuMP without nonlinear constraints....")
        estimate_fast!(problem, 
            linear_solver = linear_solver, 
            verbose = verbose); 
    end

    # Otherwise skip the linear solver and jump to MCMC
    if quasi_bayes 
        try 
            @assert burn_in < 1 
        catch
            error("`burn_in` denotes the fraction of samples to drap. Must be less than 1")
        end
        
        burn_in_fraction = burn_in;
        burn_in = Int(burn_in * n_samples);
        gamma_length = size(Bvec[1],2);

        # Define inputs to quasi-bayes sampling 
        lbs             = get_lower_bounds(problem)
        parameter_order = get_parameter_order(lbs)
        nbetas          = get_nbetas(problem)
        vbetastar       = 10;
        vbeta           = zeros(sum(nbetas))

        if (sampler == "mh")
            sampler = MH(
                :gamma => AdvancedMH.RandomWalkProposal(MvNormal(zeros(gamma_length-1), diagm(step*ones(gamma_length-1)))),
                :betastar =>  AdvancedMH.RandomWalkProposal(MvNormal(zeros(sum(nbetas)), diagm(step*ones(sum(nbetas))))))
        elseif (sampler ==[]) | (sampler == "hmc")
            sampler = HMC(0.01, 1; adtype = AutoForwardDiff())
        end
        
        for j in 1:sum(nbetas)
            if isnothing(lbs[j])
                vbeta[j] = vbetastar
            else 
                vbeta[j] = sqrt(log(1 + vbetastar))
            end
        end
        
        prior = Dict(
            "betabar" => zeros(sum(nbetas)), 
            "vbeta" => vbeta,
            "gammabar" => zeros(gamma_length-1),
            "vgamma" => 10,
            "lbs" => lbs,
            "parameter_order" => parameter_order,
            "nbetas" => nbetas
        )

        problem.tempmats = calc_tempmats(problem);
        
        # Find a starting point for sampling
        if ((problem.results != []) & (n_attempts == 0))
            println("Using existing minimizer as the initial point")
            start = [problem.results.minimizer[NPDemand.sieve_to_betas_index(problem)]; problem.results.minimizer[problem.design_width+2:end]]
        elseif (n_attempts == 0)
            start, start_exit = find_starting_point(problem, prior, calc_tempmats(problem), weight_matrices, n_attempts = 1);
        else
            verbose && println("Finding a valid starting point for sampler....")
            start, start_exit = find_starting_point(problem, prior, calc_tempmats(problem), weight_matrices, n_attempts = n_attempts);
            if start_exit == "success"
                println("Valid starting point found")
            else
                println("Did not find a valid starting point. Running the sampler anyway, but you may wish to increase `n_attempts` to find a better starting point.")
            end
        end

        if step == :auto
            verbose && println("Sampling small chains with different step sizes to target 20% acceptance rate....")
            step, step_grid, accept = pick_step_size(problem, prior, tempmats, weight_matrices; n_samples = n_samples);
        end

        J = length(Xvec);
        matrix_storage_dict = Dict();
        augmented_X = [hcat(Xvec[i], -1 .* Bvec[i][:,2:end]) for i in 1:J]
        matrix_storage_dict["yZX"] = [(-1 .* df[!,"prices$i"])' * weight_matrices[i+1] * augmented_X[i+1] for i in 0:J-1]
        matrix_storage_dict["XX"] = [augmented_X[i+1]'* weight_matrices[i+1] * augmented_X[i+1] for i in 0:J-1]
        matrix_storage_dict["XZy"] = [augmented_X[i+1]' * weight_matrices[i+1]' * (-1 .* df[!,"prices$i"]) for i in 0:J-1]

        # Sample 
        verbose && println("Beginning sampling....")
        chain = Turing.sample(
                sample_quasibayes(problem, prior, problem.tempmats, weight_matrices; 
                penalty = penalty, matrix_storage_dict = matrix_storage_dict), 
                sampler, n_samples,
                initial_params = start
                ); 

        # Convert thh chain into the parameter sieve
        start_row = burn_in+1;
        betastardraws = hcat([chain["betastar[$i]"] for i in 1:sum(nbetas)]...)[start_row:end,:]
        betadraws = reparameterization_draws(betastardraws, lbs, parameter_order)
        gammadraws = hcat([chain["gamma[$i]"] for i in 1:gamma_length-1]...)[start_row:end,:]
        
        # thin the markov chain
        L = size(betastardraws,1);
        skip_inds = 1:skip:L
        betadraws = betadraws[skip_inds,:];
        gammadraws = gammadraws[skip_inds,:];

        # calculate posterior mean parameters
        qpm = map_to_sieve(mean(betadraws, dims=1)', mean(gammadraws, dims=1)', problem.exchange, nbetas, problem)

        problem.sampling_details = (; burn_in = burn_in_fraction, skip = skip, smc = false, prior = prior)
        problem.results  = NPD_parameters(qpm, hcat(betadraws, gammadraws));
        problem.chain    = chain;
    end
end

"""
    smc!(problem::NPDemand.NPDProblem;
        grid_points::Int    = 50, 
        max_penalty::Real   = 5, 
        ess_threshold::Real = 100, 
        step::Real          = 0.1, 
        skip::Int           = 5,
        burn_in::Real       = 0.25, 
        mh_steps            = max(5, floor(size(problem.results.filtered_chain, 2))/10),
        seed                = 4132,
        smc_method          = :adaptive,
        max_iter            = 1000,
        adaptive_tolerance  = false, 
        max_violations      = 0.01)

Run sequentially constrained Monte Carlo (SMC) on the problem.

# Arguments
- `problem::NPDemand.NPDProblem`: The problem object on which SMC will be run.

# Optional Arguments
- `grid_points::Int`: The number of grid points for the SMC grid. Default is 50.
- `max_penalty::Real`: The maximum penalty value for the SMC algorithm. Default is 5.
- `ess_threshold::Real`: The effective sample size threshold for the SMC algorithm. Default is 100.
- `step::Real`: The step size for the SMC algorithm. Default is 0.1.
- `skip::Int`: The thinning factor for the SMC chain. Default is 5.
- `burn_in::Real`: The fraction of samples to be discarded as burn-in. Default is 0.25.
- `mh_steps`: The number of Metropolis-Hastings steps per iteration. Default is calculated based on the size of the filtered chain.
- `seed`: The random seed for the SMC algorithm. Default is 4132.
- `smc_method`: The method for choosing the SMC grid. Default is :adaptive. Other options are [:linear\\_grid, :geometric\\_grid, and :logit\\_grid], which specify grids of each form between zero and the maximum penalty.
- `max_iter`: The maximum number of iterations for the SMC algorithm. Default is 1000.
- `adaptive_tolerance`: Whether to use adaptive tolerance for the SMC algorithm. Default is false.
- `max_violations`: The maximum allowed fraction of markets with violations. Default is 0.01.

The function will overwrite the results in the problem object with the resulting chain.

For harder or slower problems, it may be necessary to increase the number of Metropolis-Hastings steps per iteration (`mh_steps`), the number of iterations (`max_iter`), or the maximum allowed fraction markets with violations (`max_violations`).

`burn_in` and `skip` control the number of samples to drop and the thinning of the chain, respectively.
"""
function smc!(problem::NPDemand.NPDProblem;
    grid_points::Int    = 50, 
    max_penalty::Real   = 5, 
    ess_threshold::Real = 100, 
    step::Real          = 0.1, 
    skip::Int           = 5,
    burn_in::Real       = 0.25, 
    mh_steps            = max(5, floor(size(problem.results.filtered_chain, 2))/10),
    seed                = 4132,
    smc_method          = :adaptive,
    max_iter            = 1000,
    adaptive_tolerance  = false, 
    max_violations      = 0.01,
    extra_mh_loops      = 0)

    try 
        @assert smc_method ∈ [:adaptive, :linear_grid, :geometric_grid, :logit_grid]
    catch
        error("`smc_method` must be one of [:adaptive, :linear_grid, :geometric_grid, :logit_grid]")
    end

    burn_in_int = Int(burn_in * size(problem.chain,1));
    modulo_num = Int(1 + extra_mh_loops);

    # Add smc_results to problem
    problem.smc_results = smc(problem::NPDemand.NPDProblem; 
        grid_points         = grid_points, 
        max_penalty         = max_penalty, 
        ess_threshold       = ess_threshold, 
        step_size           = step, 
        skip                = skip,
        burn_in             = burn_in_int, 
        mh_steps            = mh_steps,
        seed                = seed,
        smc_method          = smc_method, 
        max_iter            = max_iter, 
        adaptive_tolerance  = adaptive_tolerance, 
        max_violations      = max_violations, 
        modulo_num          = modulo_num);
    
    # Calculate new posterior mean and replace problem results 
    lbs             = get_lower_bounds(problem)
    parameter_order = get_parameter_order(lbs)
    nbetas          = get_nbetas(problem)
    nbeta           = length(lbs)

    nparticles      = size(problem.smc_results.thetas,1);
    
    betas           = reparameterization_draws(problem.smc_results.thetas[:,1:nbeta], lbs, parameter_order)
    gammas          = problem.smc_results.thetas[:,(nbeta+1):end]        
    thetas_sieve    = vcat([map_to_sieve(betas[i,:], problem.smc_results.thetas[i,(nbeta+1):end], problem.exchange, nbetas, problem) for i in 1:nparticles]...)

    problem.results.minimizer       = mean(thetas_sieve, dims = 1);
    problem.results.filtered_chain  = hcat(betas, gammas)
    problem.sampling_details        = (; burn_in = burn_in, 
        skip = skip, smc = true, prior = problem.sampling_details.prior);
end
