struct HMC
    ε::Float64
    n_leapfrog::Int
end

function calc_tempmats(problem::NPDProblem;
    recipe = nothing)

    J = length(problem.Xvec);

    s        = Matrix(problem.data[:, r"shares"]);
    exchange = problem.exchange;
    bO       = problem.approximation_details[:order];
    tensor   = haskey(problem.approximation_details, :tensor) ? problem.approximation_details[:tensor] : true;
    bernO    = convert.(Integer, bO);

    tempmats = Matrix{Float64}[]
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

    if isempty(exchange) exchange = []; end
    for j1 = 1:J
        which_group = findfirst(j1 .∈  exchange);
        first_product_in_group = minimum(exchange[which_group]);
        _, permuted_shares, permutations = get_params_one_equation(j1;
            exchange = exchange,
            s = s,
            θ = 1:problem.design_width,
            nbetas = nbetas)

        for j2 = 1:J
            tempmat_s = calc_derivative_sieve(permutations[j1], permutations[j2];
                exchange          = ((exchange==[]) || ((approximation_details[:sieve_type] == "bernstein") && (tensor == true))) ?
                    exchange : adjust_exchange(exchange, first_product_in_group),
                shares            = s,
                permuted_shares   = permuted_shares,
                perm              = permutations,
                bernO             = bernO,
                sieve_type        = approximation_details[:sieve_type],
                recipe            = approximation_details[:sieve_type] == "polynomial" ? recipes[which_group] : nothing,
                max_interaction   = approximation_details[:max_interaction],
                constraints       = problem.constraints,
                tensor            = haskey(approximation_details, :tensor) ? approximation_details[:tensor] : true
                )
            push!(tempmats, tempmat_s)
        end
    end
    temp_storage_mat = reshape(tempmats, J,J);
    temp_elast_mats  = deepcopy(temp_storage_mat);
    for j1 = 1:J
        for j2 = 1:J
            temp_elast_mats[j1,j2] = temp_storage_mat[j2,j1];
        end
    end
    return temp_elast_mats
end

function inner_elast_loop(dsids_i::AbstractMatrix{T}, J::Int, at::AbstractVector{Float64}, svec::AbstractVector{Float64}; type::String = "jacobian") where T
    temp = try
        -inv(dsids_i)
    catch
        -pinv(dsids_i)
    end

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
    temp_length = size(problem.data,1);

    dsids_raw     = [tempmat_storage[j1, j2] * θ[indexes[j1]+1:indexes[j1+1]] for j1 in 1:J, j2 in 1:J]
    all_elast_mat = [inner_elast_loop([dsids_raw[j1,j2][i] for j1 in 1:J, j2 in 1:J], J, view(at, i, :), view(s, i, :); type = type) for i in 1:temp_length]

    return all_elast_mat
end

function find_starting_point(problem, prior, tempmats;
    n_attempts = 1000)

    nbetas          = prior["nbetas"]
    lbs             = prior["lbs"]
    parameter_order = prior["parameter_order"]
    gamma_length    = size(problem.Bvec[1], 2)
    betabar         = prior["betabar"]
    gammabar        = prior["gammabar"]
    vbeta           = prior["vbeta"]
    vgamma          = prior["vgamma"]
    J               = length(problem.Xvec)

    param_out = (z_beta = zeros(sum(nbetas)), z_gamma = zeros(gamma_length - 1))

    for _ in 1:n_attempts
        z_beta_i   = randn(sum(nbetas))
        z_gamma_i  = randn(gamma_length - 1)
        betastar_i = betabar .+ sqrt.(vbeta) .* z_beta_i
        gamma_i    = gammabar .+ sqrt(vgamma) .* z_gamma_i
        beta_i     = reparameterization(betastar_i, lbs, parameter_order)
        st         = problem.approximation_details[:sieve_type]
        sieve_params = map_to_sieve(beta_i, gamma_i, problem.exchange, nbetas, problem; sieve_type = st)

        elasts_i = elast_mat_zygote(sieve_params, problem, tempmats;
            at = Matrix(problem.data[!, r"prices"]),
            s  = Matrix(problem.data[!, r"shares"]))
        elasts_i = [elasts_i[ii][j1, j2] for j1 in 1:J, j2 in 1:J, ii in 1:size(problem.data, 1)]

        if run_elasticity_check(elasts_i, problem.constraints, problem.exchange)
            return (z_beta = z_beta_i, z_gamma = z_gamma_i), "success"
        end
    end

    return param_out, "failed"
end

function posterior_elasticities(j, k, betadraws, gammadraws, tempmats, problem)
    ndraws = min(size(betadraws,1), 1_000)
    nbetas = get_nbetas(problem)
    tmpout = zeros(eltype(betadraws), size(problem.data,1), ndraws)
    for i in 1:ndraws
        st       = problem.approximation_details[:sieve_type]
        params_i = map_to_sieve(betadraws[i,:], gammadraws[i,:],
                            problem.exchange, nbetas, problem;
                            sieve_type = st)
        tmpout[:,i] = getindex.(elast_mat_zygote(params_i, problem, tempmats;
            at = Matrix(problem.data[!,r"prices"]), s = Matrix(problem.data[!,r"shares"])), j, k)
    end
    return tmpout
end

# Leapfrog HMC using fully analytical gradients
function analytical_hmc(prior::Dict, msd::Dict, J::Int;
    n_samples::Int      = 1000,
    step_size::Real     = 0.01,
    n_leapfrog::Int     = 10,
    n_adapt::Int        = 0,
    thin::Int           = 1,
    target_accept::Real = 0.8,
    adapt_L::Bool       = true,
    z_init              = nothing,
    seed::Union{Int,Nothing} = nothing,
    verbose::Bool       = true)

    betabar  = prior["betabar"];  vbeta    = prior["vbeta"]
    gammabar = prior["gammabar"]; vgamma   = prior["vgamma"]
    lbs      = prior["lbs"];      parameter_order = prior["parameter_order"]
    lbs_trivial = all(lbs .== typemax(Int))

    nbeta  = length(betabar);   ngamma = length(gammabar)
    n      = nbeta + ngamma

    sqrt_vbeta  = sqrt.(vbeta);  sqrt_vgamma = sqrt(vgamma)

    yZX_β     = msd["yZX_β"];    XZy_β     = msd["XZy_β"]
    XX_ββ     = msd["XX_ββ"];    XX_βγ     = msd["XX_βγ"]
    yZX_γ_sum = msd["yZX_γ_sum"]; XZy_γ_sum = msd["XZy_γ_sum"]
    XX_γγ_sum = msd["XX_γγ_sum"]
    starts    = msd["starts_params_v2"]; ends = msd["ends_params_v2"]
    gfp       = msd["group_for_product"]

    nbeta_per_g = ends .- starts .+ 1
    nb_max      = maximum(nbeta_per_g)
    buf_ββ  = zeros(nb_max)
    buf_βγ  = zeros(nb_max)
    buf_γ   = zeros(ngamma)
    beta    = zeros(nbeta);   gamma  = zeros(ngamma)
    ∂beta   = zeros(nbeta);   ∂gamma = zeros(ngamma)

    # Computes log-posterior and fills grad in-place. Zero heap allocations for lbs_trivial.
    function logpost_grad!(grad, z)
        z_β = @view z[1:nbeta];  z_γ = @view z[nbeta+1:end]

        @. beta  = betabar + sqrt_vbeta * z_β
        @. gamma = gammabar + sqrt_vgamma * z_γ

        repar_pb = nothing
        if !lbs_trivial
            betastar = copy(beta)
            beta_out, repar_pb = ChainRulesCore.rrule(reparameterization, betastar, lbs, parameter_order)
            beta .= beta_out
        end

        mul!(buf_γ, XX_γγ_sum, gamma)
        val_γ = dot(gamma, buf_γ)
        @. ∂gamma = -yZX_γ_sum - XZy_γ_sum + 2*buf_γ
        val = -dot(yZX_γ_sum, gamma) - dot(gamma, XZy_γ_sum) + val_γ
        fill!(∂beta, 0)

        for i in 1:J
            g   = gfp[i]
            nb  = nbeta_per_g[g]
            β_i  = @view beta[starts[g]:ends[g]]
            ∂β_g = @view ∂beta[starts[g]:ends[g]]
            ββ_v = @view buf_ββ[1:nb]
            βγ_v = @view buf_βγ[1:nb]

            mul!(ββ_v, XX_ββ[i], β_i)
            mul!(βγ_v, XX_βγ[i], gamma)

            @. ∂β_g += -yZX_β[i] - XZy_β[i] + 2*ββ_v + 2*βγ_v
            val    += -dot(yZX_β[i], β_i) - dot(β_i, XZy_β[i]) +
                       dot(β_i, ββ_v) + 2*dot(β_i, βγ_v)

            mul!(buf_γ, XX_βγ[i]', β_i)
            @. ∂gamma += 2*buf_γ
        end

        if lbs_trivial
            @. grad[1:nbeta]    = -z_β - 0.5*∂beta*sqrt_vbeta
        else
            _, ∂betastar, _, _ = repar_pb(∂beta)
            @. grad[1:nbeta]    = -z_β - 0.5*∂betastar*sqrt_vbeta
        end
        @. grad[nbeta+1:end] = -z_γ - 0.5*∂gamma*sqrt_vgamma

        return -0.5*(dot(z_β, z_β) + dot(z_γ, z_γ)) - 0.5*val
    end

    # Leapfrog HMC with optional dual-averaging adaptation (Hoffman & Gelman 2014, Algorithm 5)
    rng      = isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed)
    n_total   = n_samples
    n_post    = n_samples - n_adapt
    n_stored  = ceil(Int, n_post / thin)
    samples  = Matrix{Float64}(undef, n_stored, n)
    z        = isnothing(z_init) ? zeros(n) : float(vec(z_init))[1:n]
    z_prop   = similar(z);  grad = zeros(n);  grad_prop = zeros(n);  p = similar(z)

    logp      = logpost_grad!(grad, z)
    n_accept  = 0
    store_idx = 0
    ε         = float(step_size)
    L         = n_leapfrog
    τ_target  = ε * L

    μ      = log(10 * ε)
    H̄      = 0.0
    log_ε̄  = log(ε)
    t₀     = 10;  γ_da = 0.05;  κ = 0.75

    iter = verbose ? ProgressBar(1:n_total) : 1:n_total
    for s in iter
        randn!(rng, p)
        H_old = -logp + 0.5*dot(p, p)
        z_prop .= z;  grad_prop .= grad
        logp_prop = logp

        p .+= 0.5 .* ε .* grad_prop
        for l in 1:L
            z_prop    .+= ε .* p
            logp_prop  = logpost_grad!(grad_prop, z_prop)
            p .+= (l < L ? ε : 0.5*ε) .* grad_prop
        end

        H_new = -logp_prop + 0.5*dot(p, p)
        α = (isfinite(H_new) && isfinite(logp_prop)) ? min(1.0, exp(H_old - H_new)) : 0.0
        if log(rand(rng)) < log(α + eps())
            z .= z_prop;  grad .= grad_prop
            logp = logp_prop;  n_accept += 1
        end

        post = s - n_adapt
        if post > 0 && (post - 1) % thin == 0 && store_idx < n_stored
            store_idx += 1
            samples[store_idx, :] .= z
        end

        if n_adapt > 0 && s <= n_adapt
            m     = float(s)
            H̄     = (1 - 1/(m + t₀)) * H̄ + (1/(m + t₀)) * (target_accept - α)
            log_ε = μ - (sqrt(m) / γ_da) * H̄
            log_ε̄ = m^(-κ) * log_ε + (1 - m^(-κ)) * log_ε̄
            ε     = clamp(exp(log_ε), 1e-6, 10.0)
            adapt_L && (L = clamp(round(Int, τ_target / ε), 1, 200))
        elseif n_adapt > 0 && s == n_adapt + 1
            ε = clamp(exp(log_ε̄), 1e-6, 10.0)
            adapt_L && (L = clamp(round(Int, τ_target / ε), 1, 200))
        end

        if verbose
            phase = (n_adapt > 0 && s <= n_adapt) ? "adapt" : "sample"
            set_description(iter, @sprintf("HMC [%s] ε=%.3g L=%d acc=%.0f%%",
                phase, ε, L, n_accept/s*100))
        end
    end

    verbose && @info "analytical_hmc acceptance rate: $(round(n_accept/n_total*100, digits=1))%"

    param_names = vcat(
        [Symbol("z_beta[$i]")  for i in 1:nbeta],
        [Symbol("z_gamma[$i]") for i in 1:ngamma])
    return MCMCChains.Chains(
        reshape(samples, n_stored, n, 1),
        param_names)
end

function loglikelihood(problem::NPDemand.NPDProblem, particle_betastar::Vector{T}, particle_gamma::Vector{T}) where T
    nbetas          = get_nbetas(problem)
    lbs             = get_lower_bounds(problem)
    parameter_order = get_parameter_order(lbs)
    particle_beta   = reparameterization_draws(reshape(particle_betastar, 1, sum(nbetas)), lbs, parameter_order)
    x = map_to_sieve(particle_beta, particle_gamma, problem.exchange, nbetas, problem)
    return -0.5 * gmm(x, problem, problem.weight_matrices)
end
