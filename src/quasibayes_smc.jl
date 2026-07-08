function logpdf_mvn(mu::Vector{T}, Sigma::Matrix{T}, theta::Vector{T}) where T<:Real
    n = length(mu)
    L = cholesky(Sigma).L
    diff = theta - mu
    quadratic_form = sum((L \ diff) .^ 2)
    logdetSigma = 2.0 * sum(log.(diag(L)))
    logpdf = -0.5 * (n * log(2 * π) + logdetSigma + quadratic_form)
    return logpdf
end

function logpdf_mvn(mu::Vector, chol::Cholesky, logdet_sigma::Real, theta::AbstractVector)
    diff = theta .- mu
    quadratic_form = sum((chol.L \ diff) .^ 2)
    return -0.5 * (length(mu) * log(2π) + logdet_sigma + quadratic_form)
end

function logpdf_diag_mvn(mu::AbstractVector, inv_var::AbstractVector, logdet_sigma::Real, theta::AbstractVector)
    q = zero(promote_type(eltype(theta), eltype(inv_var)))
    @inbounds for i in eachindex(mu, inv_var, theta)
        d = theta[i] - mu[i]
        q += d * d * inv_var[i]
    end
    return -0.5 * (length(mu) * log(2π) + logdet_sigma + q)
end

function approx_cdf_normal01(x::Real)::Float64
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p  = 0.3275911

    sign = x < 0 ? -1 : 1
    abs_x = abs(x) / sqrt(2.0)

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

function geometric_grid(A::Float64, B::Float64, n::Int)
    r = (B / A)^(1 / (n - 1))
    grid = [A * r^(i - 1) for i in 1:n]
    return grid
end

function make_prior_dists(prior, gamma_length)
    betabar  = prior["betabar"]
    gammabar = prior["gammabar"]
    vbetasq  = prior["vbetasq"]
    vgammasq = prior["vgammasq"]
    ngamma   = gamma_length-1;

    beta_dist  = MvNormal(betabar, Diagonal(vbetasq))
    gamma_dist = MvNormal(gammabar, Diagonal(fill(vgammasq, ngamma)))

    return beta_dist, gamma_dist
end

function logprior_smc(particle_betastar, particle_gamma,
                      beta_μ, beta_inv_var::AbstractVector, beta_logdet::Real,
                      gamma_μ, gamma_inv_var::AbstractVector, gamma_logdet::Real)
    out_beta  = logpdf_diag_mvn(beta_μ, beta_inv_var, beta_logdet, particle_betastar)
    out_gamma = logpdf_diag_mvn(gamma_μ, gamma_inv_var, gamma_logdet, particle_gamma)
    return out_beta + out_gamma
end

# log-density of log(τ) when τ ~ Half-Cauchy(0,1): a hyperbolic-secant distribution,
# f(u) = (1/π) sech(u). Computed via a numerically stable log(cosh(u)).
function _logcosh(u::Real)
    au = abs(u)
    return au + log1p(exp(-2*au)) - log(2)
end
log_halfcauchy_logscale(u::Real) = -_logcosh(u) - log(π)

# Horseshoe-shrinkage version of logprior_smc: `particle_z` is the standard-normal
# latent for every beta coefficient (both constrained and unconstrained — the
# depth-dependent betabar/vbetasq only shape the *deterministic* map to β for
# unconstrained coefficients, not the prior on z itself, exactly mirroring
# analytical_hmc's NCP structure), and `log_tau` is the (unconstrained) log of the
# shared global horseshoe scale, log(τ) ~ hyperbolic-secant (i.e. τ ~ Half-Cauchy(0,1)).
function logprior_smc(particle_z, particle_gamma, log_tau::Real,
                      gamma_μ, gamma_inv_var::AbstractVector, gamma_logdet::Real)
    out_z     = -0.5*(dot(particle_z, particle_z) + length(particle_z)*log(2π))
    out_gamma = logpdf_diag_mvn(gamma_μ, gamma_inv_var, gamma_logdet, particle_gamma)
    out_tau   = log_halfcauchy_logscale(log_tau)
    return out_z + out_gamma + out_tau
end

# Local+global horseshoe: as above, plus a per-constrained-coefficient log(λ_j),
# each also log(λ_j) ~ hyperbolic-secant (λ_j ~ Half-Cauchy(0,1)), independent of τ
# and of each other.
function logprior_smc(particle_z, particle_gamma, log_tau::Real, log_lambda::AbstractVector,
                      gamma_μ, gamma_inv_var::AbstractVector, gamma_logdet::Real)
    out_z      = -0.5*(dot(particle_z, particle_z) + length(particle_z)*log(2π))
    out_gamma  = logpdf_diag_mvn(gamma_μ, gamma_inv_var, gamma_logdet, particle_gamma)
    out_tau    = log_halfcauchy_logscale(log_tau)
    out_lambda = sum(log_halfcauchy_logscale, log_lambda)
    return out_z + out_gamma + out_tau + out_lambda
end

function logpenalty_smc(distance::AbstractVector, penalty::Real)
    return sum(log.(2 .* approx_cdf_normal01.(-penalty .* distance)))
end

function logpenalty_smc(particle_sieve::Array{T}, penalty::Real, problem::NPDemand.NPDProblem;
    penalty_type = :frac) where T
    distance = report_constraint_violations_inner(problem, params = particle_sieve, verbose = false, output = String(penalty_type))
    return logpenalty_smc(distance, penalty)
end

# Allocation-efficient penalty evaluation for use inside the MH inner loop
function _fill_neg_inv!(out::Matrix{Float64}, dsids::Matrix{Float64}, J::Int)
    try
        if J == 2
            out .= -inv(SMatrix{2,2,Float64,4}(dsids))
        elseif J == 3
            out .= -inv(SMatrix{3,3,Float64,9}(dsids))
        elseif J == 4
            out .= -inv(SMatrix{4,4,Float64,16}(dsids))
        elseif J == 5
            out .= -inv(SMatrix{5,5,Float64,25}(dsids))
        elseif J == 6
            out .= -inv(SMatrix{6,6,Float64,36}(dsids))
        else
            out .= -inv(dsids)
        end
    catch
        out .= -pinv(dsids)
    end
end

# Zero-allocation constraint counter
function _count_violations_fast(elast::Matrix{Float64}, constraints, exchange, J::Int)
    n_viol = 0

    if :monotone in constraints
        ok = true
        @inbounds for j in 1:J
            if elast[j, j] > 0.0; ok = false; break; end
        end
        n_viol += !ok
    end

    if :all_substitutes in constraints
        ok = true
        @inbounds for j1 in 1:J
            ok || break
            for j2 in 1:J
                j1 == j2 && continue
                if elast[j1, j2] < 0.0; ok = false; break; end
            end
        end
        n_viol += !ok
    end

    if :diagonal_dominance_all in constraints
        ok = true
        @inbounds for j1 in 1:J
            s = 0.0
            for j2 in 1:J; s += abs(elast[j1, j2]); end
            if 2.0 * abs(elast[j1, j1]) < s; ok = false; break; end
        end
        n_viol += !ok
    end

    if :subs_in_group in constraints
        ok = true
        for grp in exchange
            for a in grp
                ok || break
                for b in grp
                    a == b && continue
                    if elast[a, b] < 0.0; ok = false; break; end
                end
            end
            ok || break
        end
        n_viol += !ok
    end

    if :subs_across_group in constraints
        ok = true
        for g1 in 1:length(exchange)
            ok || break
            for g2 in 1:length(exchange)
                g1 == g2 && continue
                for a in exchange[g1]
                    ok || break
                    for b in exchange[g2]
                        if elast[a, b] < 0.0; ok = false; break; end
                    end
                end
                ok || break
            end
        end
        n_viol += !ok
    end

    if :all_complements in constraints
        ok = true
        @inbounds for j1 in 1:J
            ok || break
            for j2 in 1:J
                j1 == j2 && continue
                if elast[j1, j2] > 0.0; ok = false; break; end
            end
        end
        n_viol += !ok
    end

    if :complements_in_group in constraints
        ok = true
        for grp in exchange
            for a in grp
                ok || break
                for b in grp
                    a == b && continue
                    if elast[a, b] > 0.0; ok = false; break; end
                end
            end
            ok || break
        end
        n_viol += !ok
    end

    if :complements_across_group in constraints
        ok = true
        for g1 in 1:length(exchange)
            ok || break
            for g2 in 1:length(exchange)
                g1 == g2 && continue
                for a in exchange[g1]
                    ok || break
                    for b in exchange[g2]
                        if elast[a, b] > 0.0; ok = false; break; end
                    end
                end
                ok || break
            end
        end
        n_viol += !ok
    end

    return n_viol
end

# Inline magnitude violations
function _magnitude_violations_fast(elast::Matrix{Float64}, constraints, exchange, J::Int)
    total    = 0.0
    n_active = 0

    if :monotone in constraints
        n_active += 1
        num = 0.0; den = 0.0
        @inbounds for j in 1:J
            v = elast[j, j]; num += max(0.0, v); den += abs(v)
        end
        total += (num / J) / (den / J + 1e-10)
    end

    if :all_substitutes in constraints
        n_active += 1
        num = 0.0; den = 0.0; n_off = 0
        @inbounds for j1 in 1:J, j2 in 1:J
            j1 == j2 && continue
            v = elast[j1, j2]; num += max(0.0, -v); den += abs(v); n_off += 1
        end
        n_off > 0 && (total += (num / n_off) / (den / n_off + 1e-10))
    end

    if :diagonal_dominance_all in constraints
        n_active += 1
        num = 0.0; den = 0.0
        @inbounds for j1 in 1:J
            col_sum = 0.0
            for j2 in 1:J; j2 == j1 && continue; col_sum += elast[j2, j1]; end
            num += max(0.0, abs(col_sum) - abs(elast[j1, j1]))
            den += abs(elast[j1, j1])
        end
        total += (num / J) / (den / J + 1e-10)
    end

    if :subs_in_group in constraints
        n_active += 1
        grp_sum = 0.0
        for grp in exchange
            num = 0.0; den = 0.0; n_off = 0
            for a in grp, b in grp
                a == b && continue
                v = elast[a, b]; num += max(0.0, -v); den += abs(v); n_off += 1
            end
            n_off > 0 && (grp_sum += (num / n_off) / (den / n_off + 1e-10))
        end
        total += grp_sum / length(exchange)
    end

    if :subs_across_group in constraints
        n_active += 1
        num = 0.0; den = 0.0; n_cross = 0
        for g1 in 1:length(exchange), g2 in 1:length(exchange)
            g1 == g2 && continue
            for a in exchange[g1], b in exchange[g2]
                v = elast[a, b]; num += max(0.0, -v); den += abs(v); n_cross += 1
            end
        end
        n_cross > 0 && (total += (num / n_cross) / (den / n_cross + 1e-10))
    end

    if :all_complements in constraints
        n_active += 1
        num = 0.0; den = 0.0; n_off = 0
        @inbounds for j1 in 1:J, j2 in 1:J
            j1 == j2 && continue
            v = elast[j1, j2]; num += max(0.0, v); den += abs(v); n_off += 1
        end
        n_off > 0 && (total += (num / n_off) / (den / n_off + 1e-10))
    end

    if :complements_in_group in constraints
        n_active += 1
        grp_sum = 0.0
        for grp in exchange
            num = 0.0; den = 0.0; n_off = 0
            for a in grp, b in grp
                a == b && continue
                v = elast[a, b]; num += max(0.0, v); den += abs(v); n_off += 1
            end
            n_off > 0 && (grp_sum += (num / n_off) / (den / n_off + 1e-10))
        end
        total += grp_sum / length(exchange)
    end

    if :complements_across_group in constraints
        n_active += 1
        num = 0.0; den = 0.0; n_cross = 0
        for g1 in 1:length(exchange), g2 in 1:length(exchange)
            g1 == g2 && continue
            for a in exchange[g1], b in exchange[g2]
                v = elast[a, b]; num += max(0.0, v); den += abs(v); n_cross += 1
            end
        end
        n_cross > 0 && (total += (num / n_cross) / (den / n_cross + 1e-10))
    end

    return n_active > 0 ? total / n_active : 0.0
end

# Log-penalty using BLAS mul! for cache-friendly Phase 1, SMatrix inv for zero-heap Phase 2
function logpenalty_fast(θ, problem::NPDemand.NPDProblem, tempmats::Matrix{Matrix{Float64}},
        penalty::Real;
        penalty_type::Symbol = :frac)

    J        = length(problem.Xvec)
    indexes  = [0; cumsum(size.(problem.Xvec, 2))]
    T        = size(problem.data, 1)
    ncon     = length(problem.constraints)
    θ_slices = [view(θ, indexes[j]+1:indexes[j+1]) for j in 1:J]

    dsids_all = Matrix{Float64}(undef, T, J*J)
    dsids_buf = Matrix{Float64}(undef, J, J)
    elast_buf = Matrix{Float64}(undef, J, J)

    # Phase 1: one BLAS dgemv per (j1,j2) pair
    @inbounds for j1 in 1:J, j2 in 1:J
        mul!(view(dsids_all, :, (j1-1)*J + j2), tempmats[j1, j2], θ_slices[j1])
    end

    # Phase 2: per-market inversion and constraint check
    log_pen = 0.0
    for i in 1:T
        @inbounds for j1 in 1:J, j2 in 1:J
            dsids_buf[j1, j2] = dsids_all[i, (j1-1)*J + j2]
        end
        _fill_neg_inv!(elast_buf, dsids_buf, J)
        dist = if penalty_type === :magnitude
            _magnitude_violations_fast(elast_buf, problem.constraints, problem.exchange, J)
        elseif penalty_type === :count
            Float64(_count_violations_fast(elast_buf, problem.constraints, problem.exchange, J))
        else  # :frac (default)
            ncon > 0 ? Float64(_count_violations_fast(elast_buf, problem.constraints, problem.exchange, J)) / ncon : 0.0
        end
        log_pen += log(2.0 * approx_cdf_normal01(-penalty * dist))
    end
    return log_pen
end

function smc_penalty_distances(thetas_sieve::Matrix, problem::NPDemand.NPDProblem;
    multithread = false,
    penalty_type = :frac)
    out    = Matrix{Float64}(undef, size(thetas_sieve, 1), size(problem.data, 1))
    loop   = axes(thetas_sieve, 1)
    output = String(penalty_type)
    if multithread
        Threads.@threads for i in loop
            out[i,:] .= report_constraint_violations_inner(problem, params = thetas_sieve[i,:], verbose = false, output = output)
        end
    else
        for i in loop
            out[i,:] .= report_constraint_violations_inner(problem, params = thetas_sieve[i,:], verbose = false, output = output)
        end
    end
    return out
end

function particle_logpenalty(thetas_sieve, penalty_distances, i, penalty, problem;
    penalty_type = :frac)
    if penalty_distances === nothing
        return logpenalty_smc(thetas_sieve[i,:], penalty, problem; penalty_type = penalty_type)
    end
    return logpenalty_smc(view(penalty_distances, i, :), penalty)
end

function get_importance_weights(thetas_sieve::Matrix{T}, smc_weights::Vector{T},
    penalty_prev::Real, penalty_new::Real, problem::NPDemand.NPDProblem;
    new_log_weights    = similar(smc_weights),
    logprior_t         = zeros(T, size(thetas_sieve,1)),
    logprior_t_minus_1 = zeros(T, size(thetas_sieve,1)),
    multithread        = false,
    penalty_distances  = nothing,
    penalty_type       = :frac) where T<:Real

    has_old_logprior = any(!iszero, logprior_t_minus_1)
    if multithread == false
        for i in axes(thetas_sieve,1)
            logprior_t[i]      = particle_logpenalty(thetas_sieve, penalty_distances, i, penalty_new, problem; penalty_type = penalty_type)
            logprior_old       = has_old_logprior ? logprior_t_minus_1[i] : particle_logpenalty(thetas_sieve, penalty_distances, i, penalty_prev, problem; penalty_type = penalty_type)
            new_log_weights[i] = log(smc_weights[i]) + logprior_t[i] - logprior_old
        end
    else
        Threads.@threads for i in axes(thetas_sieve,1)
            logprior_t[i]      = particle_logpenalty(thetas_sieve, penalty_distances, i, penalty_new, problem; penalty_type = penalty_type)
            logprior_old       = has_old_logprior ? logprior_t_minus_1[i] : particle_logpenalty(thetas_sieve, penalty_distances, i, penalty_prev, problem; penalty_type = penalty_type)
            new_log_weights[i] = log(smc_weights[i]) + logprior_t[i] - logprior_old
        end
    end

    return new_log_weights, logprior_t
end

function f_ess(p::T, thetas_sieve::Matrix{T}, smc_weights::Vector{T},
    prev_penalty::Real, problem::NPDemand.NPDProblem,
    ess_threshold::Real;
    logprior_t_minus_1 = zeros(T, size(thetas_sieve,1)),
    penalty_distances  = nothing,
    penalty_type       = :frac) where T

    logwts, _ = get_importance_weights(thetas_sieve, smc_weights, prev_penalty, p[1],
            problem,
            logprior_t_minus_1 = logprior_t_minus_1,
            multithread = true,
            penalty_distances = penalty_distances,
            penalty_type = penalty_type)

    num          = 2*maximum(logwts) + 2*log(sum(exp.(logwts .- maximum(logwts))))
    denom        = maximum(2*logwts) + log(sum(exp.(2*logwts .- maximum(2*logwts))))
    exp_log_ess  = exp(num - denom)
    return isfinite(exp_log_ess) ? exp_log_ess - ess_threshold : -ess_threshold
end

function smc(problem::NPDemand.NPDProblem;
    grid_points::Int    = 50,
    max_penalty::Real   = 100,
    ess_threshold::Real = 100,
    step_size::Real     = 0.1,
    mh_steps            = 10,
    smc_method          = :grid,
    seed                = 4132,
    max_iter            = 1000,
    adaptive_tolerance  = false,
    max_violations      = 0.01,
    modulo_num          = 1,
    penalty_type        = :frac,
    approximation_details::Dict{Symbol, Any} = Dict()
    )

    prior           = problem.sampling_details.prior;
    lbs             = prior["lbs"]
    parameter_order = prior["parameter_order"]
    gamma_length    = size(problem.Bvec[1],2);

    particles = problem.chain_starparams;
    nbetas    = get_nbetas(problem);
    nbeta     = length(lbs) == 0 ? sum(nbetas) : length(lbs);

    _betabar      = prior["betabar"]
    _gammabar     = prior["gammabar"]
    _vbetasq      = prior["vbetasq"]
    _vgammasq     = prior["vgammasq"]
    tau0            = get(prior, "tau0", 1.0)
    ngamma_smc      = gamma_length - 1
    use_hs          = get(prior, "horseshoe", false) && !all(lbs .== typemax(Int))
    use_local       = use_hs && get(prior, "local_shrinkage", false)
    is_constrained  = use_hs ? .!isnothing.(lbs) : falses(sum(nbetas))
    constrained_idx = use_local ? findall(is_constrained) : Int[]
    n_constrained   = length(constrained_idx)
    local_pos       = zeros(Int, sum(nbetas))
    use_local && (local_pos[constrained_idx] .= 1:n_constrained)

    if particles isa MCMCChains.Chains
        gammadraws = hcat([particles["gammastar[$i]"] for i in 1:ngamma_smc]...)
        if use_hs
            # HMC's chain_starparams stores the increment itself (τλz²) under "betastar[i]"
            # and the *effective* shared/local scales (i.e. already including τ_0) under
            # "tau"/"lambda[i]" (see estimate.jl) — divide τ_0 back out to recover this
            # sampler's own τ_raw = τ/τ_0 representation.
            # Recover z (only its square matters, so the positive root is as good as any)
            # and log(τ), log(λ_i).
            increment_or_beta = hcat([particles["betastar[$i]"] for i in 1:sum(nbetas)]...)
            tau_init   = vec(particles["tau"]) ./ tau0
            lambda0    = use_local ? Dict(i => vec(particles["lambda[$i]"]) for i in constrained_idx) : nothing
            zdraws     = similar(increment_or_beta)
            for i in 1:sum(nbetas)
                if is_constrained[i]
                    scale_i     = tau0 .* (use_local ? tau_init .* lambda0[i] : tau_init)
                    zdraws[:,i] = sqrt.(max.(increment_or_beta[:,i], 0.0) ./ scale_i)
                else
                    zdraws[:,i] = (increment_or_beta[:,i] .- _betabar[i]) ./ sqrt(_vbetasq[i])
                end
            end
            logtau0    = log.(tau_init)
            loglambda0 = use_local ? hcat([log.(lambda0[i]) for i in constrained_idx]...) : nothing
        else
            betastardraws = hcat([particles["betastar[$i]"] for i in 1:sum(nbetas)]...)
        end
    else
        # `chain_starparams` was set from a previous `smc!` call (a plain matrix, see
        # estimate.jl); it already stores columns in this sampler's own internal
        # representation ([z gamma log(τ) log(λ)] if local, [z gamma log(τ)] if global
        # horseshoe, [betastar gamma] otherwise).
        gammadraws = particles[:, sum(nbetas)+1:sum(nbetas)+ngamma_smc]
        if use_hs
            zdraws = particles[:, 1:sum(nbetas)]
            if use_local
                logtau0    = particles[:, sum(nbetas)+ngamma_smc+1]
                loglambda0 = particles[:, (sum(nbetas)+ngamma_smc+2):end]
            else
                logtau0 = particles[:, end]
            end
        else
            betastardraws = particles[:, 1:sum(nbetas)]
        end
    end

    nparticles = use_hs ? size(zdraws,1) : size(betastardraws,1);

    if use_hs
        tau_init_draws  = exp.(logtau0)
        increment_draws = similar(zdraws)
        for i in 1:sum(nbetas)
            if is_constrained[i]
                if use_local
                    lambda_i_draws = exp.(loglambda0[:, local_pos[i]])
                    increment_draws[:,i] = tau0 .* tau_init_draws .* lambda_i_draws .* zdraws[:,i].^2
                else
                    increment_draws[:,i] = tau0 .* tau_init_draws .* zdraws[:,i].^2
                end
            else
                increment_draws[:,i] = _betabar[i] .+ sqrt(_vbetasq[i]) .* zdraws[:,i]
            end
        end
        betadraws = reparameterization_increments_draws(increment_draws, lbs, parameter_order)
        thetas    = use_local ? [zdraws gammadraws logtau0 loglambda0] : [zdraws gammadraws logtau0]
    else
        betadraws = reparameterization_draws(betastardraws, lbs, parameter_order)
        thetas    = [betastardraws gammadraws]
    end

    st           = approximation_details[:sieve_type]
    thetas_sieve = vcat([map_to_sieve(
                            betadraws[i,:],
                            gammadraws[i,:],
                            problem.exchange,
                            nbetas,
                            problem;
                            sieve_type=st)
                         for i in 1:nparticles]...)
    matrix_storage_dict = gmm_fast_blocks(problem, nbetas)
    nproducts = length(problem.Avec)
    yZX_β, XZy_β = matrix_storage_dict["yZX_β"], matrix_storage_dict["XZy_β"]
    XX_ββ, XX_βγ = matrix_storage_dict["XX_ββ"], matrix_storage_dict["XX_βγ"]
    yZX_γ_sum, XZy_γ_sum = matrix_storage_dict["yZX_γ_sum"], matrix_storage_dict["XZy_γ_sum"]
    XX_γγ_sum = matrix_storage_dict["XX_γγ_sum"]
    starts_params     = matrix_storage_dict["starts_params_v2"]
    ends_params       = matrix_storage_dict["ends_params_v2"]
    group_for_product = matrix_storage_dict["group_for_product"]
    gmm_loglike(beta, gamma) = -0.5 * gmm_fast_v2(beta, gamma, yZX_β, XZy_β, XX_ββ, XX_βγ,
        yZX_γ_sum, XZy_γ_sum, XX_γγ_sum, starts_params, ends_params, group_for_product, nproducts)

    smc_weights = fill(1.0 / nparticles, nparticles)

    violation_dict_array = [
        report_constraint_violations_inner(problem,
            params = thetas_sieve[i,:], verbose = false) for i in axes(thetas_sieve,1)
    ];
    violation_dict = Dict{Symbol, Float64}()
    for k in keys(violation_dict_array[1])
        push!(violation_dict, k => mean([violation_dict_array[i][k] for i in axes(thetas_sieve,1)]));
    end

    viol_store  = [];
    ess_store   = [];
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
    beta_μ, gamma_μ = _betabar, _gammabar
    beta_inv_var, beta_logdet   = inv.(_vbetasq), sum(log.(_vbetasq))
    gamma_inv_var, gamma_logdet = fill(inv(_vgammasq), gamma_length-1), (gamma_length-1) * log(_vgammasq)
    n_kernel_steps = Int(mh_steps)
    ngamma          = gamma_length - 1
    _sqrt_vbetasq   = sqrt.(_vbetasq)

    failure_count = 0;
    while (violation_dict[:any] > max_violations) & (prev_penalty < max_penalty) & (t < max_iter)
        t = t+1
        print("\n Iteration "*string(t-1)*"...\r")
        penalty_distances = smc_penalty_distances(thetas_sieve, problem;
            multithread = true, penalty_type = penalty_type)
        current_logprior = zeros(eltype(thetas_sieve), nparticles)

        if (smc_method == :adaptive) & (mod(t,modulo_num) == 0)
            print("\n Optimizing penalty... \r")
            _, current_logprior = get_importance_weights(thetas_sieve, smc_weights, Float64(prev_penalty), Float64(prev_penalty), problem,
                penalty_distances = penalty_distances, penalty_type = penalty_type);
            if adaptive_tolerance
                new_penalty = find_zero(x -> f_ess(x, thetas_sieve, smc_weights, prev_penalty, problem, ess_threshold,
                    logprior_t_minus_1 = current_logprior, penalty_distances = penalty_distances,
                    penalty_type = penalty_type),
                    (prev_penalty, max_penalty), Bisection(); xatol = get_tolerance(prev_penalty))
            else
                custom_ub = max(prev_penalty * 10.0, prev_penalty + 0.01);
                xatol = 0.0002;
                try
                    new_penalty = find_zero(x -> f_ess(x, thetas_sieve, smc_weights, prev_penalty, problem, ess_threshold,
                        logprior_t_minus_1 = current_logprior, penalty_distances = penalty_distances,
                        penalty_type = penalty_type),
                        (prev_penalty, min(custom_ub, max_penalty)), Bisection(); xatol = xatol, verbose = false)
                    new_penalty = new_penalty - xatol;
                catch
                    new_penalty = min(custom_ub, max_penalty);
                end
                if abs(prev_penalty - new_penalty) > 2 * xatol
                    new_penalty = new_penalty - xatol;
                end
            end
        elseif (smc_method == :adaptive)
            new_penalty = prev_penalty;
        else
            new_penalty = penalty_list[t]
        end

        log_smc_weights, _ = get_importance_weights(thetas_sieve, smc_weights, prev_penalty, new_penalty, problem,
            logprior_t_minus_1 = current_logprior, penalty_distances = penalty_distances,
            penalty_type = penalty_type)

        smc_weights .= exp.(log_smc_weights .- maximum(log_smc_weights))
        smc_weights .= smc_weights ./ sum(smc_weights)
        ess = 1 / sum(smc_weights.^2)
        push!(ess_store, ess)

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

        thetas            = thetas[indices,:]
        thetas_sieve      = thetas_sieve[indices,:]
        penalty_distances = penalty_distances[indices,:]
        smc_weights      .= 1.0 / nparticles

        n_accept = zeros(nparticles);
        Sigma    = cov(thetas);
        Sigma    = (Sigma + Diagonal(fill(1e-6, size(Sigma, 1)))) .* 2.38^2 ./ size(Sigma,1);
        proposal_distribution = MvNormal(zeros(length(thetas[1,:])), step_size .* Sigma);

        # One RNG per thread — independent, reproducible streams; mh_iter mixed into seed
        thread_rngs             = [MersenneTwister(seed + tid) for tid in 1:Threads.maxthreadid()]
        reparameterization_bufs = [zeros(eltype(thetas_sieve), sum(nbetas)) for _ in 1:Threads.maxthreadid()]
        x_hs_bufs                = use_hs ? [zeros(eltype(thetas_sieve), sum(nbetas)) for _ in 1:Threads.maxthreadid()] : Vector{Float64}[]
        tau_pos                  = nbeta + ngamma + 1

        # Horseshoe forward map: build the per-coefficient increment (τλz² for
        # constrained, the usual affine map otherwise) and push it through the
        # max-chain, exactly mirroring analytical_hmc's horseshoe path.
        # `log_lambda` is empty when local_shrinkage is off (λ≡1 for every coefficient).
        hs_beta(z, log_tau, log_lambda, x_buf, beta_buf) = begin
            τ = tau0*exp(log_tau)
            @inbounds for k in eachindex(z)
                if is_constrained[k]
                    λk = use_local ? exp(log_lambda[local_pos[k]]) : 1.0
                    x_buf[k] = τ*λk*z[k]^2
                else
                    x_buf[k] = _betabar[k] + _sqrt_vbetasq[k]*z[k]
                end
            end
            reparameterization_increments(x_buf, lbs, parameter_order, buffer_beta = beta_buf)
        end

        prog           = Threads.Atomic{Int}(0)
        monitor_active = Threads.Atomic{Bool}(true)

        monitor = @async begin
            while monitor_active[] && (n_done = prog[]) < nparticles
                print(@sprintf("\r  MH steps: %d/%d (%.0f%%)", n_done, nparticles, 100.0*n_done/nparticles))
                sleep(0.05)
            end
            println(@sprintf("\r  MH steps: %d/%d (100%%)  ", nparticles, nparticles))
        end

        try
            Threads.@threads for i in axes(thetas,1)
                tid  = Threads.threadid()
                rng  = thread_rngs[tid]
                reparameterization_storage = reparameterization_bufs[tid]

                if use_hs
                    x_buf         = x_hs_bufs[tid]
                    z_old         = @view thetas[i,1:nbeta]
                    gamma_old     = @view thetas[i,(nbeta+1):(nbeta+ngamma)]
                    logtau_old    = thetas[i,tau_pos]
                    loglambda_old = use_local ? (@view thetas[i,(tau_pos+1):end]) : Float64[]
                    betai_old     = hs_beta(z_old, logtau_old, loglambda_old, x_buf, reparameterization_storage)
                    logprior_old  = use_local ?
                        logprior_smc(z_old, gamma_old, logtau_old, loglambda_old, gamma_μ, gamma_inv_var, gamma_logdet) :
                        logprior_smc(z_old, gamma_old, logtau_old, gamma_μ, gamma_inv_var, gamma_logdet)
                    logprior_old += logpenalty_smc(view(penalty_distances, i, :), new_penalty)
                else
                    betai_old    = reparameterization(thetas[i,1:nbeta], lbs, parameter_order, buffer_beta = reparameterization_storage)
                    gamma_old    = @view thetas[i,(nbeta+1):size(thetas,2)]
                    logprior_old = logprior_smc(thetas[i,1:nbeta], gamma_old, beta_μ, beta_inv_var, beta_logdet, gamma_μ, gamma_inv_var, gamma_logdet) +
                                   logpenalty_smc(view(penalty_distances, i, :), new_penalty)
                end
                loglike_old  = gmm_loglike(betai_old, gamma_old)

                for mh_iter in 1:n_kernel_steps
                    Random.seed!(rng, seed + i + mh_iter * nparticles)

                    thetai_new = thetas[i,:] + rand(rng, proposal_distribution)

                    if use_hs
                        z_new            = @view thetai_new[1:nbeta]
                        gamma_new        = @view thetai_new[(nbeta+1):(nbeta+ngamma)]
                        logtau_new       = thetai_new[tau_pos]
                        loglambda_new    = use_local ? (@view thetai_new[(tau_pos+1):end]) : Float64[]
                        betai_new        = hs_beta(z_new, logtau_new, loglambda_new, x_buf, reparameterization_storage)
                        thetai_sieve_new = map_to_sieve(betai_new, gamma_new, problem.exchange, nbetas, problem, sieve_type=st)
                        logprior_new     = use_local ?
                            logprior_smc(z_new, gamma_new, logtau_new, loglambda_new, gamma_μ, gamma_inv_var, gamma_logdet) :
                            logprior_smc(z_new, gamma_new, logtau_new, gamma_μ, gamma_inv_var, gamma_logdet)
                        logprior_new    += logpenalty_fast(thetai_sieve_new, problem, problem.tempmats, new_penalty; penalty_type = penalty_type)
                    else
                        betai_new        = reparameterization(thetai_new[1:nbeta], lbs, parameter_order, buffer_beta = reparameterization_storage)
                        gamma_new        = @view thetai_new[(nbeta+1):length(thetai_new)]
                        thetai_sieve_new = map_to_sieve(betai_new, gamma_new, problem.exchange, nbetas, problem, sieve_type=st)
                        logprior_new     = logprior_smc(thetai_new[1:nbeta], gamma_new, beta_μ, beta_inv_var, beta_logdet, gamma_μ, gamma_inv_var, gamma_logdet) +
                                           logpenalty_fast(thetai_sieve_new, problem, problem.tempmats, new_penalty; penalty_type = penalty_type)
                    end

                    loglike_new  = gmm_loglike(betai_new, gamma_new)

                    logratio = loglike_new + logprior_new - loglike_old - logprior_old
                    if log(rand(rng)) < logratio
                        thetas[i,:]       = thetai_new
                        thetas_sieve[i,:] = thetai_sieve_new
                        logprior_old      = logprior_new
                        loglike_old       = loglike_new
                        n_accept[i]      += 1
                    end
                    GC.safepoint()
                end
                Threads.atomic_add!(prog, 1)
            end
            wait(monitor)
        finally
            monitor_active[] = false
        end
        n_accept    = n_accept ./ n_kernel_steps
        accept_rate = round(mean(n_accept), digits = 2);

        violation_dict_array = [report_constraint_violations_inner(problem, params = thetas_sieve[i,:], verbose = false) for i in axes(thetas_sieve,1)];
        violation_dict = Dict{Symbol, Float64}()
        for k in keys(violation_dict_array[1])
            push!(violation_dict, k =>
                round(
                    mean([violation_dict_array[i][k] for i in axes(thetas_sieve,1)]),
                    digits = 3)
            );
        end
        push!(viol_store, violation_dict[:any])

        push!(penalty_vec, new_penalty)
        prev_penalty = new_penalty;

        println("|--------------------------------|----------|")
        println("| Iteration results              |          |")
        println("|--------------------------------|----------|")
        println(@sprintf("| %-30s | %8.4g |", "Current Penalty", new_penalty))
        if !(isnan(ess) | isinf(ess))
            println(@sprintf("| %-30s | %8d |", "ESS", Int(floor(ess))))
        else
            println(@sprintf("| %-30s | %8s |", "ESS", "NaN"))
        end
        println(@sprintf("| %-30s | %8.2f |", "Average MH Acceptance rate", accept_rate))
        println("| Violations                     |          |")
        for (key, value) in violation_dict
            println(@sprintf("| %-30s | %8.3f |", key, value))
        end
    end

    return (; thetas, smc_weights, violations = viol_store, ess = ess_store, penalties = penalty_vec);
end

function fe_posteriors(problem; FE::Union{Array, String} = [])
    if FE==[]
        error("Please provide name of FE (the `FE` keyword argument)")
    end
    if (problem.chain_starparams ==[])
        error("No Markov chain found in the problem")
    end

    coef_names = [problem.fe_param_mapping[i].name for i in 1:length(problem.fe_param_mapping)]
    coefs_for_this_fe = findall(coef_names .== FE)
    all_gammas = problem.chain_params[:,end-size(problem.Bvec[1],2)+1:end];
    num_index_vars = length(problem.index_vars);

    df_fe = DataFrame()
    for (_, i) in enumerate(coefs_for_this_fe)
        val = problem.fe_param_mapping[i].value
        column_name = "Value$val"
        df_fe[!, column_name] = all_gammas[:, num_index_vars + i]
    end
    return df_fe
end
