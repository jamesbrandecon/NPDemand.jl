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
    elseif problem.Aeq != []
        lbs = typemax(Int) .* ones(Int, size(problem.Aeq[:,sieve_to_betas_index(problem)], 2))
    else
        lbs = typemax(Int) .* ones(Int, sum(size.(problem.Xvec,2)))
    end
    return lbs
end

function get_parameter_order(lbs)
    if !(all(lbs .== typemax(Int)))
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

function reparameterization(betastar::AbstractVector{T}, lbs::AbstractVector, parameter_order::AbstractVector; buffer_beta = similar(betastar)) where T<:Real
    if all(lbs .== typemax(Int))
        return betastar
    else
        for i in parameter_order
            if isnothing(lbs[i])
                buffer_beta[i] = betastar[i]
            else
                buffer_beta[i] = maximum(buffer_beta[lbs[i]]) + exp(betastar[i])
            end
        end
        return copy(buffer_beta)
    end
end

function ChainRulesCore.rrule(::typeof(reparameterization), betastar::AbstractVector, lbs::AbstractVector, parameter_order::AbstractVector)
    if all(lbs .== typemax(Int))
        trivial_pullback(ȳ) = ChainRulesCore.NoTangent(), ChainRulesCore.unthunk(ȳ), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
        return betastar, trivial_pullback
    else
        beta = similar(betastar)
        jmax = zeros(Int, length(betastar))
        for i in parameter_order
            if isnothing(lbs[i])
                beta[i] = betastar[i]
            else
                am      = argmax(beta[lbs[i]])
                jmax[i] = lbs[i][am]
                beta[i] = beta[jmax[i]] + exp(betastar[i])
            end
        end
        beta_out = copy(beta)
        function ordering_pullback(ȳ)
            ȳ_work    = copy(ChainRulesCore.unthunk(ȳ))
            ∂betastar = zeros(eltype(betastar), length(betastar))
            for i in Iterators.reverse(parameter_order)
                if isnothing(lbs[i])
                    ∂betastar[i] = ȳ_work[i]
                else
                    ∂betastar[i]    = ȳ_work[i] * exp(betastar[i])
                    ȳ_work[jmax[i]] += ȳ_work[i]
                end
            end
            return ChainRulesCore.NoTangent(), ∂betastar, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
        end
        return beta_out, ordering_pullback
    end
end

function reparameterization_draws(betastar_draws, lbs, parameter_order)
    nbeta  = size(betastar_draws, 2)
    ndraws = size(betastar_draws, 1)
    beta_draws = zeros(eltype(betastar_draws), ndraws, nbeta)
    if all(lbs .== typemax(Int)) || (lbs == [])
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
        J            = length(problem.Xvec)
        counts       = size.(problem.Xvec,2)
        starts_prod  = [1; cumsum(counts)[1:end-1] .+ 1]
        ends_prod    = cumsum(counts)
        starts_grp   = [1; cumsum(nbetas)[1:end-1] .+ 1]
        ends_grp     = cumsum(nbetas)
        sieve_params = zeros(T, sum(counts))
        for j in 1:J
            g = findfirst(x->j in x, exchange)
            sieve_params[starts_prod[j]:ends_prod[j]] .=
                beta[starts_grp[g]:ends_grp[g]]
        end
        allp = [sieve_params; 1.0; gamma]
        return reshape(allp, 1, length(allp))
    else
        J = length(problem.Xvec);

        starts_sieve  = [1;cumsum(size.(problem.Xvec,2))[1:end-1] .+ 1];
        ends_sieve    = cumsum(size.(problem.Xvec,2))

        starts_params = [1;cumsum(nbetas)[1:end-1] .+ 1];
        ends_params   = cumsum(nbetas);

        sieve_params = zeros(T, problem.design_width);
        for j in 1:J
            which_group = findfirst(j .∈  exchange);
            sieve_params[starts_sieve[j]:ends_sieve[j]] = beta[starts_params[which_group]:ends_params[which_group]]
        end

        all_params = [sieve_params; 1.0; gamma];
        all_params = reshape(all_params, 1, length(all_params));

        return all_params
    end
end
