function estimate_fast!(problem::NPDProblem;
    linear_solver = "Ipopt",
    verbose = true)

    β, γ = jmp_obj(problem,
        linear_solver = linear_solver,
        verbose = verbose);

    problem.results = NPD_parameters([β;γ]);
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
        for j1 = 1:J
            ej = getindex.(findall(j1 .∈ exchange),1)[1];
            for j2 = 1:J
                if (j2 ∉ exchange[ej])
                    conmat_complements[j1,j2] = 0;
                end
            end
        end
    end

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
        n_samples::Int = 50_000,
        burn_in::Real = 0.25,
        skip::Int = 5,
        step::Real = 0.01)

Estimates the problem using the specified parameters.

## Arguments
- `problem::NPDProblem`: The problem to be estimated.
- `verbose::Bool`: Whether to print verbose output. Default is `true`.
- `linear_solver::String`: The linear solver to use. Must be either "Ipopt" or "OSQP". Default is "Ipopt".
- `quasi_bayes::Bool`: Whether to use quasi-bayes sampling. Default is `false`.
- `n_samples::Int`: The number of samples to draw. Default is 50,000.
- `burn_in::Real`: The fraction of samples to use for HMC adaptation and then discard. Must be less than 1. Default is 0.25.
- `skip::Int`: The thinning factor for the saved chain. Default is 5.
- `step::Real`: The HMC step size. Default is 0.01.
- `n_leapfrog::Int`: Number of leapfrog steps per HMC proposal. Default is 10.
"""
function estimate!(problem::NPDProblem;
    verbose = true,
    linear_solver = "Ipopt",
    quasi_bayes = false,
    n_samples::Int = 50_000,
    burn_in::Real = 0.25,
    skip::Int = 5,
    sampler = HMC(0.01, 10),
    custom_prior::Union{Dict, Nothing} = nothing,
    regularization::String = "normal",
    depth_decay::Bool = false
    )

    # Check that linear solver is Ipopt or OSQP
    if linear_solver ∉ ["Ipopt", "OSQP"]
        error("Linear solver must be Ipopt or OSQP")
    end

    # Check that regularization is normal or horseshoe
    if regularization ∉ ["normal", "horseshoe"]
        error("`regularization` must be either \"normal\" or \"horseshoe\"")
    end
    horseshoe = regularization == "horseshoe"

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
    approximation_details = problem.approximation_details;
    sieve_type = approximation_details[:sieve_type]
    order = approximation_details[:order]
    max_interaction = approximation_details[:max_interaction]

    find_prices = findall(problem.index_vars .== "prices");
    price_index = find_prices[1];

    # Estimate the problem only with linear constraints if not using quasi-bayes
    if !quasi_bayes
        verbose && println("Estimating problem in JuMP without nonlinear constraints....")
        estimate_fast!(problem,
            linear_solver = linear_solver,
            verbose = verbose);
    end

    # Otherwise skip the linear solver and jump to HMC
    if quasi_bayes
        try
            @assert burn_in < 1
        catch
            error("`burn_in` denotes the fraction of samples to discard. Must be less than 1")
        end

        burn_in_fraction = burn_in;
        burn_in = round(Int, burn_in * n_samples);
        gamma_length = size(Bvec[1],2);

        # Define prior
        nbetas          = get_nbetas(problem)
        lbs             = sieve_type == "bernstein" ? get_lower_bounds(problem) : []
        parameter_order = lbs != []                 ? get_parameter_order(lbs)  : 1:sum(nbetas)
        vbetastarsq     = 100;
        vbetasq         = zeros(sum(nbetas))
        betabar         = zeros(sum(nbetas))

        # lbs is either a per-coefficient vector of `nothing`/dependency-index-lists, or
        # (when there are no Aineq/Aeq constraints at all) a whole-vector sentinel of
        # typemax(Int) meaning "no dependency structure" -- must be checked before treating
        # entries as real dependencies (typemax(Int) is not `nothing`, so isnothing-based
        # checks below would otherwise misread every coefficient as constrained).
        lbs_trivial = all(lbs .== typemax(Int))

        # define sets of parameter dependencies (through constraint bounds)
        dep_sets = lbs_trivial ? [Int[] for _ in eachindex(lbs)] : [sort(collect(all_dependencies(i, lbs))) for i in eachindex(lbs)]

        if sieve_type == "bernstein" && !lbs_trivial
            for j in 1:sum(nbetas)
                if isnothing(lbs[j])
                    vbetasq[j] = vbetastarsq
                else
                    vbetasq[j] = log(1 + vbetastarsq)
                    depth      = length(dep_sets[j])
                    # m          = max((1.0 + depth)^(-2), 1e-3)
                    betabar[j] = 0
                    # betabar[j] = -log(1 + length(dep_sets[j]))
                    # betabar[j] = log(m) - vbetasq[j] / 2
                end
            end
        else
            vbetasq .= vbetastarsq
        end

        # τ_0: global-scale prior for the horseshoe τ ~ Half-Cauchy(0, τ_0), with τ_0 set
        # from the maximum depth of the constrained-parameter tree (fixed decay rate 2,
        # mirroring the original convergent-series argument, applied once to the global
        # scale rather than per coefficient). Defaults to 1 (no adjustment) unless
        # `depth_decay` is set.
        is_constrained_for_tau0 = lbs_trivial ? falses(length(lbs)) : .!isnothing.(lbs)
        if depth_decay && any(is_constrained_for_tau0)
            pathlen  = compute_pathlen(lbs, parameter_order)
            depth_max = maximum(pathlen[is_constrained_for_tau0])
            tau0      = (1.0 + depth_max)^(-4)
        else
            tau0 = 1.0
        end

        prior = Dict(
            "betabar"  => !isnothing(custom_prior) && haskey(custom_prior, "betabar")  ? custom_prior["betabar"] .+ zeros(sum(nbetas))  : betabar,
            "vbetasq"  => !isnothing(custom_prior) && haskey(custom_prior, "vbetasq")  ? custom_prior["vbetasq"].*ones(size(vbetasq))    : vbetasq,
            "gammabar" => !isnothing(custom_prior) && haskey(custom_prior, "gammabar") ? custom_prior["gammabar"] .+ zeros(gamma_length-1) : zeros(gamma_length-1),
            "vgammasq" => !isnothing(custom_prior) && haskey(custom_prior, "vgammasq") ? custom_prior["vgammasq"]                         : 10,
            "lbs"            => lbs,
            "parameter_order" => collect(parameter_order),
            "nbetas"         => nbetas,
            "horseshoe"      => horseshoe,
            "tau0"           => tau0
        )

        J = length(Xvec);
        matrix_storage_dict = gmm_fast_blocks(problem, nbetas)

        # Sample (z_init defaults to zeros(n) inside analytical_hmc)
        verbose && println("Beginning sampling....")
        chain = analytical_hmc(prior, matrix_storage_dict, J;
            n_samples  = n_samples,
            step_size  = sampler.ε,
            n_leapfrog = sampler.n_leapfrog,
            n_adapt    = burn_in,
            thin       = skip,
            verbose    = verbose,
            horseshoe  = horseshoe)

        # Convert chain from NCP (z) space back to parameter space
        # chain already excludes adaptation steps and is thinned by skip
        z_betadraws   = hcat([chain["z_beta[$i]"]  for i in 1:sum(nbetas)]...)
        z_gammadraws  = hcat([chain["z_gamma[$i]"] for i in 1:gamma_length-1]...)
        gammadraws    = prior["gammabar"]' .+ sqrt(prior["vgammasq"]) .* z_gammadraws

        if horseshoe && !all(lbs .== typemax(Int))
            # Global-scale horseshoe: increments are τ_0*τ*λ_i*|z| for constrained
            # coefficients (shrinking the increment itself toward 0, not its log); τ_0 is
            # a fixed depth-based constant (1 unless `depth_decay`), τ is shared/learned
            # across all constrained coefficients, λ_i is per-coefficient (always on).
            is_constrained  = .!isnothing.(lbs)
            tau_raw_draws   = tan.(0.5*π .* cdf.(Normal(), vec(chain["u_tau"])))
            tau_draws       = prior["tau0"] .* tau_raw_draws  # effective global scale actually used
            lambda_draws    = Dict(i => tan.(0.5*π .* cdf.(Normal(), vec(chain["u_lambda[$i]"]))) for i in findall(is_constrained))
            increment_draws = similar(z_betadraws)
            for i in 1:sum(nbetas)
                if is_constrained[i]
                    increment_draws[:,i] = tau_draws .* lambda_draws[i] .* abs.(z_betadraws[:,i])
                else
                    increment_draws[:,i] = prior["betabar"][i] .+ sqrt(prior["vbetasq"][i]) .* z_betadraws[:,i]
                end
            end
            betadraws     = reparameterization_increments_draws(increment_draws, lbs, parameter_order)
            betastardraws = increment_draws
        else
            betastardraws = prior["betabar"]' .+ sqrt.(prior["vbetasq"]') .* z_betadraws
            betadraws     = reparameterization_draws(betastardraws, lbs, parameter_order)
        end

        # calculate posterior mean parameters
        qpm = map_to_sieve(mean(betadraws, dims=1)', mean(gammadraws, dims=1)', problem.exchange, nbetas, problem)

        starparams_names = vcat(
            [Symbol("betastar[$i]")  for i in 1:sum(nbetas)],
            [Symbol("gammastar[$i]") for i in 1:gamma_length-1])
        starparams = hcat(betastardraws, gammadraws)

        if horseshoe && !all(lbs .== typemax(Int))
            starparams_names = vcat(starparams_names, [Symbol("tau")])
            starparams       = hcat(starparams, tau_draws)
            lambda_idx = findall(.!isnothing.(lbs))
            starparams_names = vcat(starparams_names, [Symbol("lambda[$i]") for i in lambda_idx])
            starparams       = hcat(starparams, hcat([lambda_draws[i] for i in lambda_idx]...))
        end

        problem.sampling_details  = (; burn_in = burn_in_fraction, skip = skip, smc = false, prior = prior)
        problem.results           = NPD_parameters(qpm);
        problem.chain_params      = hcat(betadraws, gammadraws);
        problem.chain_starparams  = MCMCChains.Chains(
            reshape(starparams, size(starparams,1), size(starparams,2), 1),
            starparams_names);
    end
end

"""
    smc!(problem::NPDemand.NPDProblem;
        grid_points::Int    = 50,
        max_penalty::Real   = 100,
        ess_threshold::Real = 100,
        step::Real          = 0.1,
        mh_steps            = 10,
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
- `max_penalty::Real`: The maximum penalty value for the SMC algorithm. Default is 100.
- `ess_threshold::Real`: The effective sample size threshold for the SMC algorithm. Default is 100.
- `step::Real`: The step size for the SMC algorithm. Default is 0.1.
- `mh_steps`: The number of Metropolis-Hastings steps per iteration. Default is 10.
- `seed`: The random seed for the SMC algorithm. Default is 4132.
- `smc_method`: The method for choosing the SMC grid. Default is :adaptive. Other options are [:linear\\_grid, :geometric\\_grid, and :logit\\_grid], which specify grids of each form between zero and the maximum penalty.
- `max_iter`: The maximum number of iterations for the SMC algorithm. Default is 1000.
- `adaptive_tolerance`: Whether to use adaptive tolerance for the SMC algorithm. Default is false.
- `max_violations`: The maximum allowed fraction of markets with violations. Default is 0.01.

The function will overwrite the results in the problem object with the resulting chain.

For harder or slower problems, it may be necessary to increase the number of Metropolis-Hastings steps per iteration (`mh_steps`), the number of iterations (`max_iter`), or the maximum allowed fraction markets with violations (`max_violations`).
"""
function smc!(problem::NPDemand.NPDProblem;
    grid_points::Int    = 50,
    max_penalty::Real   = 100,
    ess_threshold::Real = 100,
    step::Real          = 0.1,
    mh_steps            = 10,
    seed                = 4132,
    smc_method          = :adaptive,
    max_iter            = 1000,
    adaptive_tolerance  = false,
    max_violations      = 0.01,
    extra_mh_loops      = 0,
    penalty_type        = :frac
    )

    try
        @assert smc_method ∈ [:adaptive, :linear_grid, :geometric_grid, :logit_grid]
    catch
        error("`smc_method` must be one of [:adaptive, :linear_grid, :geometric_grid, :logit_grid]")
    end
    try
        @assert penalty_type ∈ [:frac, :count, :magnitude]
    catch
        error("`penalty_type` must be one of [:frac, :count, :magnitude]")
    end

    modulo_num = Int(1 + extra_mh_loops);
    approximation_details = problem.approximation_details;
    sieve_type = approximation_details[:sieve_type]

    # Add smc_results to problem
    problem.smc_results = smc(problem::NPDemand.NPDProblem;
        grid_points         = grid_points,
        max_penalty         = max_penalty,
        ess_threshold       = ess_threshold,
        step_size           = step,
        mh_steps            = mh_steps,
        seed                = seed,
        smc_method          = smc_method,
        max_iter            = max_iter,
        adaptive_tolerance  = adaptive_tolerance,
        max_violations      = max_violations,
        modulo_num          = modulo_num,
        penalty_type        = penalty_type,
        approximation_details = approximation_details
        );

    # Calculate new posterior mean and replace problem results
    lbs             = sieve_type == "bernstein" ? get_lower_bounds(problem) : []
    parameter_order = sieve_type == "bernstein" ? get_parameter_order(lbs) : 1:sum(get_nbetas(problem))
    nbetas          = get_nbetas(problem)
    nbeta           = sum(nbetas)

    prior_hs   = problem.sampling_details.prior
    use_hs     = get(prior_hs, "horseshoe", false) && !all(lbs .== typemax(Int))
    nparticles = size(problem.smc_results.thetas,1);

    if use_hs
        gamma_length    = size(problem.Bvec[1],2)
        ngamma          = gamma_length - 1
        is_constrained  = .!isnothing.(lbs)
        constrained_idx = findall(is_constrained)
        local_pos       = zeros(Int, nbeta)
        local_pos[constrained_idx] .= 1:length(constrained_idx)

        tau0            = get(prior_hs, "tau0", 1.0)
        _betabar        = prior_hs["betabar"]; _vbetasq = prior_hs["vbetasq"]
        zdraws          = problem.smc_results.thetas[:, 1:nbeta]
        gammas          = problem.smc_results.thetas[:, (nbeta+1):(nbeta+ngamma)]
        tau_draws       = exp.(problem.smc_results.thetas[:, nbeta+ngamma+1])
        lambda_draws    = exp.(problem.smc_results.thetas[:, (nbeta+ngamma+2):end])
        increment_draws = similar(zdraws)
        for i in 1:nbeta
            if is_constrained[i]
                λ_i = lambda_draws[:, local_pos[i]]
                increment_draws[:,i] = tau0 .* tau_draws .* λ_i .* abs.(zdraws[:,i])
            else
                increment_draws[:,i] = _betabar[i] .+ sqrt(_vbetasq[i]) .* zdraws[:,i]
            end
        end
        betas = reparameterization_increments_draws(increment_draws, lbs, parameter_order)
    else
        betas  = reparameterization_draws(problem.smc_results.thetas[:,1:nbeta], lbs, parameter_order)
        gammas = problem.smc_results.thetas[:,(nbeta+1):end]
    end
    thetas_sieve = vcat([map_to_sieve(betas[i,:], gammas[i,:], problem.exchange, nbetas, problem) for i in 1:nparticles]...)

    problem.results.minimizer    = mean(thetas_sieve, dims = 1);
    problem.chain_starparams     = problem.smc_results.thetas
    problem.chain_params         = hcat(betas, gammas)
    problem.sampling_details     = (; smc = true, prior = problem.sampling_details.prior);
end
