# normalize exchange → Vector{Vector{Int}}
function norm_exchange(exchange)
    groups = Vector{Vector{Int}}()
    if !isempty(exchange)
        if all(x->isa(x,Int), exchange)
            push!(groups, exchange)
        else
            append!(groups, exchange)
        end
    end
    return groups
end

# build the exponent‐keys for poly_features
function feature_keys(p, order, max_interaction, exchange)
    # p = # of vars (size(X,2)), we don’t actually need X here
    function weak_comps(d,k)
        if k==1 return [[d]] end
        out = Vector{Vector{Int}}()
        for i in 0:d
            for tail in weak_comps(d-i, k-1)
                push!(out, vcat(i,tail))
            end
        end
        return out
    end

    exps_list = Vector{Vector{Int}}()
    push!(exps_list, zeros(Int,p))   # constant
    for d in 1:order, exps in weak_comps(d,p)
        if count(>(0),exps) <= max_interaction
            push!(exps_list, exps)
        end
    end

    groups = norm_exchange(exchange)
    # canonicalize
    keys = Tuple{Vararg{Int}}[]
    for e in exps_list
        key = copy(e)
        for g in groups
            key[g] = sort(e[g], rev=true)
        end
        push!(keys, Tuple(key))
    end
    return sort(unique(keys))
end

# build the exponent‐keys for poly_features_derivative
function derivative_keys(p, order, max_interaction, var_index, exchange)
    function weak_comps(d,k)
        if k==1 return [[d]] end
        out = Vector{Vector{Int}}()
        for i in 0:d
            for tail in weak_comps(d-i, k-1)
                push!(out, vcat(i,tail))
            end
        end
        return out
    end

    exps_list = Vector{Vector{Int}}()
    push!(exps_list, zeros(Int,p))   # constant term

    for d in 1:order, exps in weak_comps(d,p)
        if count(>(0),exps) <= max_interaction
            push!(exps_list, exps)
        end
    end

    groups = norm_exchange(exchange)
    keys = Tuple{Vararg{Int}}[]
    for e in exps_list
        key = copy(e)
        for g in groups
            key[g] = sort(e[g], rev=true)
        end
        push!(keys, Tuple(key))
    end
    return sort(unique(keys))
end

# now pick your settings
p = 4
order = 2
max_interaction = 2
exchange = [[1],[2],[3],[4]]
var_index = 1

# compute
Zkeys = feature_keys(p, order, max_interaction, exchange)
Dkeys = derivative_keys(p, order, max_interaction, var_index, exchange)

println("level cols = ", length(Zkeys))
println("deriv cols = ", length(Dkeys))
println("in levels ⧵ deriv = ", setdiff(Zkeys, Dkeys))
println("in deriv  ⧵ levels = ", setdiff(Dkeys, Zkeys))