"""
    NPDResults

Lightweight container for the outputs of estimating an `NPDProblem` (coefficients,
elasticities, chains, sampling metadata), without the large per-product matrices
(`Xvec`, `Avec`, `Bvec`, `weight_matrices`, `Aineq`, `Aeq`, `tempmats`) that are
only needed while running the optimizer/sampler. Build one with `define_results`.

Note: because those matrices are dropped, an `NPDResults` cannot be passed back
into `estimate!` or `price_elasticities!` -- rebuild the `NPDProblem` with
`define_problem` (same data/specification) if you need to resume estimation or
compute new elasticities.
"""
struct NPDResults
    data
    index_vars
    constraints
    syms
    combos
    mins
    maxs
    FE
    normalization
    exchange
    design_width
    obj_xtol
    obj_ftol
    constraint_tol
    bO
    estimates
    all_elasticities
    all_jacobians
    converged
    chain_starparams
    chain_params
    smc_results
    sampling_details
    approximation_details
end

"""
    define_results(problem::NPDProblem; keep_data = true)

Extracts the results of an estimated `NPDProblem` into a lightweight `NPDResults`
object suitable for saving to disk. Drops the large matrices that are only used
during estimation -- `Xvec`, `Avec`, `Bvec`, `weight_matrices`, `Aineq`, `Aeq`,
`tempmats`, `matrices`, `cfg`, `elast_mats`, `elast_prices` -- several of which
scale combinatorially with the number of products and can be far larger than
everything else in the problem combined. Serializing `problem` directly (e.g.
via JLD2) will include all of these; serializing the output of `define_results`
will not.

Set `keep_data = false` to also drop the underlying `DataFrame` (`problem.data`),
e.g. if it is itself large and not needed to interpret the saved results.
"""
function define_results(problem::NPDProblem; keep_data::Bool = true)
    return NPDResults(
        keep_data ? problem.data : nothing,
        problem.index_vars,
        problem.constraints,
        problem.syms,
        problem.combos,
        problem.mins,
        problem.maxs,
        problem.FE,
        problem.normalization,
        problem.exchange,
        problem.design_width,
        problem.obj_xtol,
        problem.obj_ftol,
        problem.constraint_tol,
        problem.bO,
        problem.estimates,
        problem.all_elasticities,
        problem.all_jacobians,
        problem.converged,
        problem.chain_starparams,
        problem.chain_params,
        problem.smc_results,
        problem.sampling_details,
        problem.approximation_details,
    )
end
