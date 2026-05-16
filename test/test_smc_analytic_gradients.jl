using NPDemand
using Test
using LinearAlgebra
using Random

@testset "smooth constraint analytic gradients" begin
    exchange = [[1, 2], [3]]
    smoothness = 20.0
    abs_eps = 1e-8
    jacobian = [
        -1.2 -0.3  0.2
         0.4 -0.8 -0.5
        -0.2  0.1 -1.0
    ]
    constraint_sets = [
        [:monotone],
        [:all_substitutes],
        [:subs_in_group],
        [:subs_across_group],
        [:all_complements],
        [:complements_in_group],
        [:complements_across_group],
        [:diagonal_dominance_all],
    ]

    for constraints in constraint_sets
        grad = zeros(size(jacobian))
        value = NPDemand.smooth_constraint_distance_grad!(grad, jacobian,
            constraints, exchange, smoothness, abs_eps)
        f = x -> begin
            mat = reshape(x, size(jacobian))
            tmp = zeros(eltype(x), size(jacobian))
            NPDemand.smooth_constraint_distance_grad!(tmp, mat,
                constraints, exchange, smoothness, abs_eps)
        end
        fd = NPDemand.ForwardDiff.gradient(f, vec(jacobian))
        @test isfinite(value)
        @test isapprox(vec(grad), fd; rtol = 1e-8, atol = 1e-8)
    end

    sign_cases = [
        ([:monotone], CartesianIndex(1, 1), >),
        ([:all_substitutes], CartesianIndex(1, 2), <),
        ([:subs_in_group], CartesianIndex(1, 2), <),
        ([:subs_across_group], CartesianIndex(1, 3), <),
        ([:all_complements], CartesianIndex(2, 1), >),
        ([:complements_in_group], CartesianIndex(2, 1), >),
        ([:complements_across_group], CartesianIndex(1, 3), >),
        ([:diagonal_dominance_all], CartesianIndex(2, 2), >),
    ]
    for (constraints, idx, sign_test) in sign_cases
        grad = zeros(size(jacobian))
        NPDemand.smooth_constraint_distance_grad!(grad, jacobian,
            constraints, exchange, smoothness, abs_eps)
        @test sign_test(grad[idx], 0)
    end
end

function smc_gradient_problem(J)
    Random.seed!(100 + J)
    T = 10
    s, p, z, x, xi = simulate_logit(J, T, -1.0, 0.1)
    df = toDataFrame(s, p, z, x)
    details = Dict(:order => 2, :max_interaction => 2, :sieve_type => "bernstein", :tensor => true)
    problem = define_problem(df;
        exchange = [collect(1:J)],
        index_vars = ["prices", "x"],
        constraints = [:exchangeability, :monotone, :diagonal_dominance_all],
        FE = [],
        approximation_details = details,
        verbose = false)
    return problem, details
end

@testset "SMC smooth logtarget analytic gradient" begin
    for J in (2, 3)
        problem, details = smc_gradient_problem(J)
        nbetas = NPDemand.get_nbetas(problem)
        lbs = NPDemand.get_lower_bounds(problem)
        parameter_order = NPDemand.get_parameter_order(lbs)
        beta_mean = zeros(sum(nbetas))
        gamma_mean = zeros(size(problem.Bvec[1], 2) - 1)
        sqrt_vbeta = fill(sqrt(10.0), sum(nbetas))
        sqrt_vgamma = sqrt(10.0)
        msd = NPDemand.gmm_fast_blocks(problem, nbetas)
        prices = Matrix(problem.data[!, r"prices"])
        shares = Matrix(problem.data[!, r"shares"])
        z0 = 0.05 .* randn(length(beta_mean) + length(gamma_mean))
        penalty = 0.7
        smoothness = 20.0

        gmm_loglike(beta, gamma) = -0.5 * NPDemand.gmm_fast_v2(beta, gamma,
            msd["yZX_β"], msd["XZy_β"], msd["XX_ββ"], msd["XX_βγ"],
            msd["yZX_γ_sum"], msd["XZy_γ_sum"], msd["XX_γγ_sum"],
            msd["starts_params_v2"], msd["ends_params_v2"], msd["group_for_product"], length(problem.Avec))
        f = zz -> NPDemand.smc_logtarget_z(zz, problem, nbetas, lbs, parameter_order,
            beta_mean, sqrt_vbeta, gamma_mean, sqrt_vgamma, gmm_loglike,
            penalty, smoothness, details[:sieve_type], prices, shares)

        grad = similar(z0)
        value = NPDemand.smc_logtarget_grad!(grad, z0, problem, nbetas, lbs, parameter_order,
            beta_mean, sqrt_vbeta, gamma_mean, sqrt_vgamma,
            msd["yZX_β"], msd["XZy_β"], msd["XX_ββ"], msd["XX_βγ"],
            msd["yZX_γ_sum"], msd["XZy_γ_sum"], msd["XX_γγ_sum"],
            msd["starts_params_v2"], msd["ends_params_v2"], msd["group_for_product"], length(problem.Avec),
            penalty, smoothness, details[:sieve_type], prices, shares)
        fd = NPDemand.ForwardDiff.gradient(f, z0)

        @test isapprox(value, f(z0); rtol = 1e-10, atol = 1e-10)
        @test isapprox(grad, fd; rtol = 1e-8, atol = 1e-8)
    end
end
