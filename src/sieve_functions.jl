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
    poly(x, order)
Returns a univariate polynomial of order `order` constructed from array/matrix `x`
"""
function poly(x, order)
    # construct polynomial of chosen order
    out = hcat([x.^o for o = 0:order]...)
    return out
end

"""
    dpoly(t, order)
Returns the derivative of a univariate polynomial of order `order` constructed from array/matrix `t`
"""
function dpoly(x, order)
    out = hcat([o .* x.^(o-1) for o = 0:order]...)
    return out
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
                       recipe::Union{PolyRecipe, Nothing}=nothing, 
                       basis_type::String = "bernstein")
  n,p = size(X)
  if isnothing(recipe)
    recipe = build_poly_recipe(p;
                order=order,
                max_interaction=max_interaction,
                exchange=exchange)
  end

  # Pre-compute bases if using Bernstein
  Bs = basis_type == "bernstein" ? [bern(X[:,j], order) for j in 1:p] : nothing 

  # now apply the recipe: each exps_group → one column
  cols = Vector{Vector{Real}}(undef, length(recipe.exps_groups))
  for (i, group) in enumerate(recipe.exps_groups)
    col = zeros(n)
    for e in group
      mon = ones(n)
      for j in 1:p
        if e[j] > 0
          if basis_type == "bernstein"
            mon .*= Bs[j][:, e[j]+1]  # Select e[j]'th Bernstein basis
          else
            mon .*= X[:,j].^e[j]      # Standard polynomial power
          end
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
                       recipe::Union{PolyRecipe, Nothing}=nothing,
                       basis_type::String="bernstein")
  n,p = size(X)
  @assert 1 ≤ var_index ≤ p

  if recipe == nothing
    recipe = build_poly_recipe(p;
                order=order,
                max_interaction=max_interaction,
                exchange=exchange)
  end
  
  # Pre-compute bases if using Bernstein
  if basis_type == "bernstein"
    Bs  = [bern(X[:,j], order)  for j in 1:p]
    dBs = [dbern(X[:,j], order) for j in 1:p]
  end

  cols = Vector{Vector{Real}}(undef, length(recipe.exps_groups))
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
          if basis_type == "bernstein"
            mon .*= dBs[j][:, u+1]  # Use precomputed derivative
          else
            mon .*= u .* X[:,j].^(u-1)  # Standard polynomial derivative
          end
        elseif u > 0
          if basis_type == "bernstein"
            mon .*= Bs[j][:, u+1]  # Use precomputed basis
          else
            mon .*= X[:,j].^u  # Standard polynomial
          end
        end
      end
      col .+= mon
    end
    cols[i] = col
  end
  return hcat(cols...)
end


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

# -------------------------------------------------------
# Tensor‐product sieve: basis × derivative
# -------------------------------------------------------

struct TensorRecipe
  basis_orders::Vector{Int}
  exps_groups::Vector{Vector{Vector{Int}}}
  keys_sorted::Vector{Tuple{Vararg{Int}}}
  basis_function::Function
  dbasis_function::Function
end

# default univariate basis & its derivative
# function basis(t, order)
#   bern(t, order)
# end
# function dbasis(t, order)
#   dbern(t, order)
# end

function build_tensor_recipe(;
  basis_orders = basis_orders::Vector{Int}, 
  exchange=Int[][], 
  basis_function::Function=bern,
  dbasis_function::Function=dbern)

  # normalize exchange groups
  groups = Vector{Vector{Int}}()
  if !isempty(exchange)
    if all(x->isa(x,Int), exchange)
      push!(groups, exchange)
    else
      append!(groups, exchange)
    end
  end

  # collect all exponent‐vectors u where 0 ≤ u[j] ≤ basis_orders[j]
  p = length(basis_orders)
  exps_list = Vector{Vector{Int}}()
  function rec_idx!(e::Vector{Int}, idx::Int)
    if idx > p
      push!(exps_list, copy(e)); return
    end
    for u in 0:basis_orders[idx]
      e[idx] = u
      rec_idx!(e, idx+1)
    end
    e[idx] = 0
  end
  rec_idx!(zeros(Int,p), 1)

  # symmetrize via exchange groups → canonical key
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
  return TensorRecipe(basis_orders, exps_groups, keys_sorted, 
                      basis_function, dbasis_function)
end

function tensor_features(X::AbstractMatrix;
                         basis_orders::Vector{Int},
                         exchange=Int[][],
                         recipe::Union{TensorRecipe, Nothing}=nothing, 
                         basis_function::Function=bern,
                         dbasis_function::Function=dbern)
  n,p = size(X)
  recipe === nothing && 
    (recipe = build_tensor_recipe(
      basis_orders = basis_orders,
      exchange = exchange, 
      basis_function = basis_function,
      dbasis_function = dbasis_function))

  # precompute univariate bases
  Bs = [recipe.basis_function(X[:,j], basis_orders[j]) for j in 1:p]

  cols = Vector{Vector{Real}}(undef, length(recipe.exps_groups))
  for (i, group) in enumerate(recipe.exps_groups)
    col = zeros(n)
    for e in group
      mon = ones(n)
      for j in 1:p
        mon .*= Bs[j][:, e[j]+1]
      end
      col .+= mon
    end
    cols[i] = col
  end
  return hcat(cols...)
end

function tensor_features_derivative(X::AbstractMatrix;
                                    var_index::Integer,
                                    basis_orders::Vector{Int},
                                    exchange=Int[][],
                                    recipe::Union{TensorRecipe, Nothing}=nothing, 
                                    basis_function::Function=bern,
                                    dbasis_function::Function=dbern)
  
    n,p = size(X)
    recipe === nothing && 
      (recipe = build_tensor_recipe(
        basis_orders = basis_orders,
        exchange = exchange, 
        basis_function = basis_function,
        dbasis_function = dbasis_function))
  
    # precompute univariate bases
    Bs = [recipe.basis_function(X[:,j], basis_orders[j]) for j in 1:p]
    dBs = [recipe.dbasis_function(X[:,j], basis_orders[j]) for j in 1:p]
    Bs[var_index] = dBs[var_index]  # replace the var_index basis with its derivative
  
    cols = Vector{Vector{Real}}(undef, length(recipe.exps_groups))
    for (i, group) in enumerate(recipe.exps_groups)
      col = zeros(n)
      for e in group
        mon = ones(n)
        for j in 1:p
          mon .*= Bs[j][:, e[j]+1]
        end
        col .+= mon
      end
      cols[i] = col
    end
    return hcat(cols...)
  # # n,p = size(X)
  # # @assert 1 ≤ var_index ≤ p

  # # # Always symmetrize using the original exchange groups,
  # # # so zero‐derivative columns get combined like in tensor_features
  # # if recipe === nothing
  # #   recipe = build_tensor_recipe(basis_orders; exchange=exchange)
  # # end

  # # # precompute bases & derivatives
  # # Bs  = [basis(X[:,j], basis_orders[j])  for j in 1:p]
  # # dBs = [dbasis(X[:,j], basis_orders[j]) for j in 1:p]

  # # cols = Vector{Vector{Real}}(undef, length(recipe.exps_groups))
  # # for (i, group) in enumerate(recipe.exps_groups)
  # #   col = zeros(n)
  # #   for e in group
  # #     # e[var_index] == 0 && continue
  #     mon = ones# (n)
  #     for j in # 1:p
  #       idx = e[j] # + 1
  #       mon .*= (j == var_index ? dBs[j][:,idx] : Bs[j][:,id# x])
  #     # end
  #     col .+= # mon
  #   # end
  #   cols[i] = # col
  # # end
  # return hcat(cols...)
end


function count_params(;
  n_products::Int,
  approximation_details::Dict{Symbol, Any},
  exchange::Vector{Vector{Int}} = [collect(1:n_products)]) 

  # simulate data with n_products and 3 markets 
  s, p, z, x, xi  = simulate_logit(n_products, 5, -1, 0.01);
  df = toDataFrame(s,p,z,x);

  Xvec, ~, ~, ~, ~ = NPDemand.prep_matrices(
        df, exchange, ["prices", "x"], [], false, approximation_details[:order]; 
        price_iv = ["prices"], verbose = false, 
        approximation_details = approximation_details, 
        constraints = [:exchangeability], inner = true
    );

  # count the number of unique parameters and total paramterers 
  total_params = sum(size.(Xvec,2)) 
  unique_params = sum(size.(Xvec,2)[first.(exchange)]) 

  return(
    Dict(
      :total_params => total_params,
      :unique_params => unique_params,
      :sieve_type => approximation_details[:sieve_type],
      :order => approximation_details[:order],
      :max_interaction => approximation_details[:max_interaction],
      :tensor => approximation_details[:tensor],
      :exchange => exchange, 
      :n_products => n_products
    )
  )
end

