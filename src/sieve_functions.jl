"""
  A recipe for polynomial‐feature / derivative‐feature construction:
  exps_groups[i] is the list of raw exponent‐vectors that map to
  the i’th symmetrized column; keys_sorted[i] is the “canonical”
  exponent‐tuple for that column.
"""
struct PolyRecipe
  exps_groups::Vector{Vector{Vector{Int}}}
  keys_sorted::Vector{Tuple{Vararg{Int}}}
end

"""
    dbern(t, order)

Returns the derivative of a univariate Bernstein polynomial of order `order` constructed from array/matrix `t`
"""
function dbern(t, order)
    out = [];
    for o = 0:1:order
    if o ==0
        out = db(t,order,o)
    else
        out = [out db(t,order,o)];
    end

    end
    return out
end

function db(t,n,k)
    out = binomial(n,k).*(k.*t.^(k.-1).*(1 .-t).^(n-k) .- t.^k.*(n-k).*(1 .-t).^(n-k-1));
    return out
end

function b(t,n,k)
    out = binomial(n,k).*(t.^k).*((1 .- t).^(n-k))
    return out
end

"""
    bern(t, order)

Returns a univariate Bernstein polynomial of order `order` constructed from array/matrix `t`
"""
function bern(t, order)
    # construct bernstein polynomial of order ``order"
    out = zeros(size(t,1),1)
    for o = 0:order
    if o ==0
        out = b(t,order,o)
    else
        out = [out b(t,order,o)];
    end

    end
    return out
end

# """
#     poly_features(X; order=2, max_interaction=1, exchange=[])

# Generate all polynomial features of `X` up to total degree `order`,
# with at most `max_interaction+1` distinct variables per monomial,
# but then _symmetrize_ over the columns in `exchange` by merging
# any monomials whose exponents on `exchange` are permutations of each other.

# Arguments
# - X::AbstractMatrix     – n×p data matrix
# - order::Integer        – maximum total degree
# - max_interaction::Integer – max # of other vars interacting
# - exchange::Vector{Int} – 1‑based column indices to symmetrize over

# Returns
# - Z_sym::Matrix{Float64} – n×m′ feature matrix with exchangeability built in
# """
"""
    poly_features(X; order, max_interaction, exchange, recipe=nothing)

If `recipe==nothing`, builds one via `build_poly_recipe`.  Otherwise
applies the given recipe directly.
"""
function poly_features(X::AbstractMatrix;
                       order::Integer=2,
                       max_interaction::Integer=1,
                       exchange=Int[][],
                       recipe::Union{PolyRecipe, Nothing}=nothing)
  n,p = size(X)
  if recipe == nothing
    recipe = build_poly_recipe(p;
                order=order,
                max_interaction=max_interaction,
                exchange=exchange)
  end

  # now apply the recipe: each exps_group → one column
  cols = Vector{Vector{Float64}}(undef, length(recipe.exps_groups))
  for (i, group) in enumerate(recipe.exps_groups)
    col = zeros(n)
    for e in group
      mon = ones(n)
      for j in 1:p
        if e[j] > 0
          mon .*= X[:,j].^e[j]
        end
      end
      col .+= mon
    end
    cols[i] = col
  end
  return hcat(cols...)
end

"""
    poly_features_derivative(X; var_index, order, max_interaction, exchange, recipe=nothing)

Analogous to `poly_features`, but computes the analytic partial w.r.t.
`X[:, var_index]` and then merges via the same `recipe`.
"""
function poly_features_derivative(X::AbstractMatrix;
                       var_index::Integer, 
                       order::Integer=2,
                       max_interaction::Integer=1,
                       exchange=Int[][],
                       recipe::Union{PolyRecipe, Nothing}=nothing)
  n,p = size(X)
  @assert 1 ≤ var_index ≤ p

  if recipe == nothing
    recipe = build_poly_recipe(p;
                order=order,
                max_interaction=max_interaction,
                exchange=exchange)
  end

  cols = Vector{Vector{Float64}}(undef, length(recipe.exps_groups))
  for (i, group) in enumerate(recipe.exps_groups)
    col = zeros(n)
    for e in group
      if e[var_index] == 0
        continue
      end
      mon = ones(n)
      for j in 1:p
        u = e[j]
        if j == var_index
          mon .*= u .* X[:,j].^(u-1)
        elseif u > 0
          mon .*= X[:,j].^u
        end
      end
      col .+= mon
    end
    cols[i] = col
  end
  return hcat(cols...)
end
# function poly_features(X::AbstractMatrix;
#                                   order::Integer=2,
#                                   max_interaction::Integer=1,
#                                   exchange=[])
#     n,p = size(X)

#     # 1) build full basis: list of exponent‐vectors + columns
#     function weak_comps(d, k)
#         if k == 1
#             return [[d]]
#         end
#         out = Vector{Vector{Int}}()
#         for i in 0:d
#             for tail in weak_comps(d - i, k - 1)
#                 push!(out, vcat(i, tail))
#             end
#         end
#         return out
#     end

#     exps_list = Vector{Vector{Int}}()
#     cols_list = Vector{Vector{Float64}}()

#     # constant term
#     push!(exps_list, zeros(Int,p))
#     push!(cols_list, ones(n))

#     for d in 1:order
#         for exps in weak_comps(d, p)
#             nnz = count(>(0), exps)
#             if nnz <= max_interaction
#                 col = ones(n)
#                 for j in 1:p
#                     if exps[j] > 0
#                         col .*= X[:,j].^exps[j]
#                     end
#                 end
#                 push!(exps_list, exps)
#                 push!(cols_list, col)
#             end
#         end
#     end

#     n,p = size(X)

#     # normalize exchange so it's always a Vector{Vector{Int}}
#     groups = Vector{Vector{Int}}()
#     if !isempty(exchange)
#       if all(x->isa(x,Int), exchange)
#         push!(groups, exchange)        # single group
#       else
#         append!(groups, exchange)      # already vector of vectors
#       end
#     end

#      # 2) merge orbits
#      dict = Dict{Tuple{Vararg{Int}}, Vector{Float64}}()
#      for (e, col) in zip(exps_list, cols_list)
#          key = copy(e)
#         for group in groups
#             key[group] = sort(e[group], rev=true)
#         end
#          t = Tuple(key)
#          dict[t] = get(dict, t, zeros(n)) .+ col
#      end

#     # 3) collect in a stable order
#     keys_sorted = sort(collect(keys(dict)))
#     Z_sym = hcat([dict[k] for k in keys_sorted]...)
#     return Z_sym
# end

# """
#     poly_features_derivative(X; order=2, max_interaction=1, var_index, exchange=[])

# Compute the analytical derivative of each polynomial feature (as generated by `poly_features`)
# w.r.t. `X[:, var_index]`, and then _symmetrize_ over `exchange` just like in `poly_features`.
# Returns an n×m′ matrix whose columns correspond exactly to those of
# `poly_features(X; order, max_interaction, exchange)`.

# Arguments
# - X::AbstractMatrix
# - order::Integer
# - max_interaction::Integer
# - var_index::Integer      – 1‐based index of the column to differentiate
# - exchange::Vector{Int}   – columns to symmetrize over

# """
# """
#     poly_features_derivative(X; var_index, order, max_interaction, exchange, recipe=nothing)

# Analogous to `poly_features`, but computes the analytic partial w.r.t.
# `X[:, var_index]` and then merges via the same `recipe`.
# """
# function poly_features_derivative(X::AbstractMatrix;
#                        var_index::Integer;
#                        order::Integer=2,
#                        max_interaction::Integer=1,
#                        exchange=Int[][],
#                        recipe::PolyRecipe=nothing)
#   n,p = size(X)
#   @assert 1 ≤ var_index ≤ p

#   if recipe == nothing
#     recipe = build_poly_recipe(p;
#                 order=order,
#                 max_interaction=max_interaction,
#                 exchange=exchange)
#   end

#   cols = Vector{Vector{Float64}}(undef, length(recipe.exps_groups))
#   for (i, group) in enumerate(recipe.exps_groups)
#     col = zeros(n)
#     for e in group
#       if e[var_index] == 0
#         continue
#       end
#       mon = ones(n)
#       for j in 1:p
#         u = e[j]
#         if j == var_index
#           mon .*= u .* X[:,j].^(u-1)
#         elseif u > 0
#           mon .*= X[:,j].^u
#         end
#       end
#       col .+= mon
#     end
#     cols[i] = col
#   end
#   return hcat(cols...)
# end
# function poly_features_derivative(X::AbstractMatrix;
#     order::Integer=2,
#     max_interaction::Integer=1,
#     var_index::Integer,
#     exchange=Int[][])
#     n,p = size(X)
#     @assert 1 ≤ var_index ≤ p "var_index must be between 1 and $p"

#     # normalize exchange into a Vector{Vector{Int}}
#     groups = Vector{Vector{Int}}()
#     if !isempty(exchange)
#         if all(x->isa(x,Int), exchange)
#             push!(groups, exchange)
#         else
#             append!(groups, exchange)
#         end 
#     end

#     # dict will accumulate merged derivative‐columns
#     dict = Dict{Tuple{Vararg{Int}}, Vector{Float64}}()
#     e = zeros(Int, p)  # exponent vector builder

#     # recursive generator with early pruning on nnz(e)
#     function rec!(idx::Int, rem::Int, nnz::Int)
#         if idx > p
#             # build the derivative column for this e
#         col = if e[var_index]==0
#         zeros(n)
#         else
#             tmp = ones(n)
#             for j in 1:p
#                 u = e[j]
#                 if j == var_index
#                     tmp .*= u .* X[:,j].^(u-1)
#                 elseif u > 0
#                     tmp .*= X[:,j].^u
#                 end
#             end
#             tmp
#         end

#     # canonicalize key by sorting within each exchange‐group
#     key = copy(e)
#     for g in groups
#         key[g] = sort(e[g], rev=true)
#     end
#     t = Tuple(key)
#     dict[t] = get(dict, t, zeros(n)) .+ col
#     return
#     end

#     # choose exponent at position idx from 0…rem
#     for u in 0:rem
#         new_nnz = nnz + (u>0 ? 1 : 0)
#         if new_nnz ≤ max_interaction
#             e[idx] = u
#             rec!(idx+1, rem - u, new_nnz)
#         else
#         break   # further u only increases nnz
#         end
#     end
#         e[idx] = 0
#     end

#     # start the recursion
#     rec!(1, order, 0)

#     # collect in stable order
#     keys_sorted = sort(collect(keys(dict)))
#     return hcat([dict[k] for k in keys_sorted]...)
# end
# function poly_features_derivative(X::AbstractMatrix;
#                                   order::Integer=2,
#                                   max_interaction::Integer=1,
#                                   var_index::Integer,
#                                   exchange=[])
#     n, p = size(X)
#     @assert 1 ≤ var_index ≤ p "var_index must be between 1 and $p"

#     # helper: all weak compositions of d into k parts
#     function weak_comps(d, k)
#         if k == 1
#             return [[d]]
#         end
#         out = Vector{Vector{Int}}()
#         for i in 0:d, tail in weak_comps(d - i, k - 1)
#             push!(out, vcat(i, tail))
#         end
#         return out
#     end

#     # 1) build full list of exponent‐vectors + derivative columns
#     exps_list  = Vector{Vector{Int}}()
#     deriv_list = Vector{Vector{Float64}}()

#     # constant term: exponent = zeros, derivative = zeros
#     push!(exps_list, zeros(Int, p))
#     push!(deriv_list, zeros(n))

#     for d in 1:order
#         for exps in weak_comps(d, p)
#             # only include if interaction ≤ max_interaction
#             if count(>(0), exps) ≤ max_interaction
#                 # build derivative for this monomial
#                 if exps[var_index] == 0
#                     col = zeros(n)
#                 else
#                     col = ones(n)
#                     for j in 1:p
#                         e = exps[j]
#                         if j == var_index
#                             col .*= e .* X[:,j].^(e - 1)
#                         elseif e > 0
#                             col .*= X[:,j].^e
#                         end
#                     end
#                 end
#                 push!(exps_list, exps)
#                 push!(deriv_list, col)
#             end
#         end
#     end
#     n,p = size(X)

#         # normalize exchange so it's always a Vector{Vector{Int}}
#         groups = Vector{Vector{Int}}()
#         if !isempty(exchange)
#           if all(x->isa(x,Int), exchange)
#             push!(groups, exchange)        # single group
#           else
#             append!(groups, exchange)      # already vector of vectors
#           end
#         end
    
#          # 2) merge orbits
#          dict = Dict{Tuple{Vararg{Int}}, Vector{Float64}}()
#          for (e, col) in zip(exps_list, deriv_list)
#              key = copy(e)
#             for group in groups
#                 key[group] = sort(e[group], rev=true)
#             end
#              t = Tuple(key)
#              dict[t] = get(dict, t, zeros(n)) .+ col
#          end

#     # 3) collect in the same stable order as poly_features
#     keys_sorted = sort(collect(keys(dict)))
#     D = hcat([dict[k] for k in keys_sorted]...)
#     return D
# end

"""
    build_poly_recipe(p; order=2, max_interaction=1, exchange=[])

Generate the exponent‐groups and canonical keys for a p‐variate
polynomial of total degree ≤ order, with ≤ max_interaction+1
distinct vars per monomial, symmetrized over each `exchange` group.
Returns a `PolyRecipe`.
"""
function build_poly_recipe(p::Integer;
    order::Integer=2,
    max_interaction::Integer=1,
    exchange=Int[][])

  # normalize exchange → Vector{Vector{Int}}
  groups = Vector{Vector{Int}}()
  if !isempty(exchange)
    if all(x->isa(x,Int), exchange)
      push!(groups, exchange)
    else
      append!(groups, exchange)
    end
  end

  # collect raw exponent‐vectors
  exps_list = Vector{Vector{Int}}()
  function rec_comps!(e::Vector{Int}, idx::Int, rem::Int, nnz::Int)
    if idx > p
      push!(exps_list, copy(e))
      return
    end
    for u in 0:rem
      new_nnz = nnz + (u>0 ? 1 : 0)
      if new_nnz ≤ max_interaction+1
        e[idx] = u
        rec_comps!(e, idx+1, rem-u, new_nnz)
      else
        break
      end
    end
    e[idx] = 0
  end
  rec_comps!(zeros(Int,p), 1, order, 0)

  # group by canonical key
  dict = Dict{Tuple{Vararg{Int}}, Vector{Vector{Int}}}()
  for e in exps_list
    key = copy(e)
    for g in groups
      key[g] = sort(e[g], rev=true)
    end
    kt = Tuple(key)
    push!(get!(dict, kt, Vector{Vector{Int}}()), e)
  end

  keys_sorted = sort(collect(keys(dict)))
  exps_groups = [dict[k] for k in keys_sorted]
  return PolyRecipe(exps_groups, keys_sorted)
end