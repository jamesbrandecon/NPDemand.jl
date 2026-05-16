"""
Simple branch-compatible SMC benchmark.

Run from the package root:

    OPENBLAS_NUM_THREADS=1 julia --project=. --threads=auto examples/benchmark_smc.jl

Options are passed as `key=value` pairs:

    mode=fastest          Portable default. Uses legacy/count-MH SMC with
                          `mh_steps=1`; this runs on main, analytic-gradients,
                          and this branch. It is also the fastest setting found
                          on this branch in the realistic benchmark.
    mode=count_mh         Alias for `fastest`.
    mode=smooth_mh        Uses `smc_kernel=:mh, penalty_type=:smooth`.
                          Requires this branch's SMC API.
    mode=smooth_hmc       Uses `smc_kernel=:hmc, penalty_type=:smooth`.
                          Requires this branch's SMC API.
    mode=all              Runs `fastest`, then tries `smooth_mh` and
                          `smooth_hmc`, skipping modes unsupported by a branch.

Other useful options and defaults:

    J=4 T=500 n_samples=400 max_iter=8 max_penalty=10
    ess_threshold=100 max_violations=0.01 mh_steps=1
    hmc_step_size=0.02 hmc_leapfrog_steps=1 blas_threads=1
"""

using LinearAlgebra
using NPDemand
using Printf
using Random

function parse_options(args)
    options = Dict{String, String}()
    for arg in args
        if occursin("=", arg)
            key, value = split(arg, "="; limit = 2)
            options[key] = value
        else
            error("Arguments must be key=value pairs. Got `$arg`.")
        end
    end
    return options
end

const OPTIONS = parse_options(ARGS)
const MODE = Symbol(get(OPTIONS, "mode", "fastest"))
const J = parse(Int, get(OPTIONS, "J", "4"))
const T_MARKETS = parse(Int, get(OPTIONS, "T", "500"))
const N_SAMPLES = parse(Int, get(OPTIONS, "n_samples", "400"))
const MAX_ITER = parse(Int, get(OPTIONS, "max_iter", "8"))
const MAX_PENALTY = parse(Float64, get(OPTIONS, "max_penalty", "10.0"))
const ESS_THRESHOLD = parse(Float64, get(OPTIONS, "ess_threshold", "100.0"))
const MAX_VIOLATIONS = parse(Float64, get(OPTIONS, "max_violations", "0.01"))
const MH_STEPS = parse(Int, get(OPTIONS, "mh_steps", "1"))
const HMC_STEP_SIZE = parse(Float64, get(OPTIONS, "hmc_step_size", "0.02"))
const HMC_LEAPFROG_STEPS = parse(Int, get(OPTIONS, "hmc_leapfrog_steps", "1"))
const BLAS_THREADS = parse(Int, get(OPTIONS, "blas_threads", Threads.nthreads() > 1 ? "1" : string(BLAS.get_num_threads())))

BLAS.set_num_threads(BLAS_THREADS)

function build_problem()
    Random.seed!(20260512)
    shares, prices, instruments, x, _xi = simulate_logit(J, T_MARKETS, -1.0, 0.1)
    data = toDataFrame(shares, prices, instruments, x)
    details = Dict(
        :order => 2,
        :max_interaction => 2,
        :sieve_type => "bernstein",
        :tensor => true)

    return define_problem(data;
        exchange = [collect(1:J)],
        index_vars = ["prices", "x"],
        constraints = [:exchangeability, :monotone, :diagonal_dominance_all],
        FE = [],
        approximation_details = details,
        verbose = false)
end

function smc_common_kwargs()
    return (;
        seed = 1024,
        burn_in = 0.25,
        skip = 1,
        mh_steps = MH_STEPS,
        max_iter = MAX_ITER,
        max_penalty = MAX_PENALTY,
        ess_threshold = ESS_THRESHOLD,
        step = 0.01,
        smc_method = :adaptive,
        max_violations = MAX_VIOLATIONS)
end

function run_smc!(problem, mode::Symbol)
    common = smc_common_kwargs()
    if mode in (:fastest, :count_mh)
        smc!(problem; common...)
    elseif mode == :smooth_mh
        smc!(problem; common..., smc_kernel = :mh, penalty_type = :smooth)
    elseif mode == :smooth_hmc
        smc!(problem; common..., smc_kernel = :hmc, penalty_type = :smooth,
            hmc_step_size = HMC_STEP_SIZE,
            hmc_leapfrog_steps = HMC_LEAPFROG_STEPS)
    else
        error("Unknown mode `$mode`.")
    end
end

function run_case(base_problem, mode::Symbol)
    problem = deepcopy(base_problem)
    seconds = @elapsed run_smc!(problem, mode)
    violations = report_constraint_violations(problem; verbose = false)
    smc = problem.smc_results
    return (;
        mode,
        seconds,
        post_any = get(violations, :any, missing),
        penalties = join(round.(Float64.(smc.penalties), digits = 4), ","),
        ess = join(round.(Float64.(smc.ess), digits = 2), ","))
end

function try_case(base_problem, mode::Symbol)
    try
        return run_case(base_problem, mode)
    catch err
        if mode in (:smooth_mh, :smooth_hmc) && err isa MethodError
            @warn "Skipping `$mode`; this branch does not support the newer SMC keyword API."
            return nothing
        end
        rethrow()
    end
end

function format_post_any(value)
    return value isa Missing ? "missing" : @sprintf("%.4f", value)
end

println("SMC benchmark")
println("  mode=$MODE")
println("  threads=$(Threads.nthreads()), BLAS threads=$(BLAS.get_num_threads())")
println("  J=$J, T=$T_MARKETS, n_samples=$N_SAMPLES, max_iter=$MAX_ITER")
println("  max_penalty=$MAX_PENALTY, ess_threshold=$ESS_THRESHOLD, max_violations=$MAX_VIOLATIONS")

problem = build_problem()
estimate_seconds = @elapsed estimate!(problem;
    quasi_bayes = true,
    sampler = NPDemand.Turing.HMC(0.02, 2),
    n_samples = N_SAMPLES,
    burn_in = 0.25,
    skip = 1,
    verbose = false,
    custom_prior = Dict("vbeta" => 10, "vgamma" => 10.0, "betabar" => 0.0, "gammabar" => 0.0))

pre_violations = report_constraint_violations(problem; verbose = false)
println("  estimate_seconds=$(round(estimate_seconds, digits = 3))")
println("  retained_particles=$(size(problem.results.filtered_chain, 1))")
println("  pre_any=$(get(pre_violations, :any, missing))")
println()

modes = MODE == :all ? [:fastest, :smooth_mh, :smooth_hmc] : [MODE]
results = filter(!isnothing, [try_case(problem, mode) for mode in modes])

@printf("%-12s %10s %10s  %-24s  %s\n", "mode", "seconds", "post_any", "penalties", "ess")
for result in results
    @printf("%-12s %10.3f %10s  %-24s  %s\n",
        string(result.mode), result.seconds, format_post_any(result.post_any),
        result.penalties, result.ess)
end
