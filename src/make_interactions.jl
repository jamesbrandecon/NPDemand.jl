function make_interactions(X::Matrix, exchange_vec, m, this_j, perm)
    # X is a collection of univariate bernstein polynomials of dimension m
    # X = collect(1:18)'
    # exchange_vec = [[1 2 3 4],[5 6]]
    ncols = size(X,2);
    J = Int(ncols/(m+1));
    input_string = ""
    input_string2 = ""
    G = [];
    G2 = [];
    prime_vec = primes(1000);
    order_prime_factors = prime_vec[1:m+1]; 
    r = order_prime_factors;
    for j = 1:J
        # Make G[j] contain groups of (m+1) columns
        push!(G, r)
        push!(G2, (j-1)*(m+1)+1:j*(m+1))
    end

    orders = collect(Iterators.product(G...));
    orders = reshape(orders, length(orders));
    orders = collect.(orders);
    
    combos = collect(Iterators.product(G2...));
    combos = reshape(combos, length(combos));
    combos = collect.(combos);

    # Make symbol vector to apply combos to so that constriaints can be applied by name
    sym_vec = [Symbol("shares$i") for i in perm]
    temp = [];
    for j = 1:J 
        for o = 0:m 
            tempsym = Symbol(string(sym_vec[j]) * "_$o")
            temp = vcat(temp, tempsym)
        end
    end
    sym_vec = temp;
    

    # Apply remaining combos to symbolic vector
    sym_combos = []
    for i ∈ eachindex(combos)
        push!(sym_combos, prod(string.(sym_vec[combos[i]])))
    end
    
        inds = Array{Array}(undef, maximum(size(combos)))
        for exchange ∈ exchange_vec
                ex = setdiff(exchange, this_j)
                not_in_group = setdiff(1:J, ex)' # own-shares and cross-shares that are not in ex group
                not_in_group_or_own = setdiff(not_in_group, this_j)' # own-shares and cross-shares that are not in ex groups

                orders_not_in_group = getindex.(orders, not_in_group_or_own)
                orders_in_group = getindex.(orders, ex')
                
                prod_order_in_group = prod(orders_in_group, dims=2)
                prod_order_not_in_group = prod(orders_not_in_group, dims=2)
                own_order = getindex.(orders, this_j)
                order_vec = reduce(hcat, orders)';
                
                @assert this_j ∉ ex
                # l = Threads.SpinLock()
                if exchange == exchange_vec[1]
                    Threads.@threads for i ∈ eachindex(combos)
                        combotemp = getindex.(findall((own_order .== own_order[i]) .&
                            (prod_order_in_group .== prod_order_in_group[i]) .& 
                            (prod_order_not_in_group .== prod_order_not_in_group[i])),1)
                        
                        combotemp = combotemp[combotemp .> i]
                        inds[i] = [i,combotemp]
                    end
                else
                    indstemp = deepcopy(inds);
                    Threads.@threads for i ∈ collect(eachindex(combos))
                            combotemp = getindex.(findall((own_order .== own_order[i]) .&
                                (prod_order_in_group .== prod_order_in_group[i]) .& 
                                (prod_order_not_in_group .== prod_order_not_in_group[i])),1)
                            
                            combotemp = combotemp[combotemp .> i]
                            inds[i][2] = intersect(inds[i][2], combotemp)
                            
                        end
                        # inds = indstemp;
                end
        end
        duplicates = zeros(length(combos));
        full_interaction = [];
        skip_inds = zeros(size(duplicates))
        for i ∈ eachindex(combos)
            if skip_inds[i] ==0
                redundant = inds[i][2];
                # if length(exchange_vec)>1
                #     for e = 2:length(exchange_vec)
                #         redundant = union(redundant, collect(inds[length(combos)*(e-1)+i][2]))
                #     end
                #     redundant = unique(collect(redundant))
                # end
                redundant = sort(unique(redundant));
                # @assert redundant[1] == i
                duplicates[redundant[1:end]] .= 1;
                skip_inds[redundant[1:end]] .= 1;
                if i ==1
                    first_column = prod(X[:,combos[i]], dims=2);
                    if length(redundant) > 0
                        for j ∈ 1:length(redundant)
                            first_column += prod(X[:,combos[redundant[j]]], dims=2);
                        end
                    end
                    full_interaction = first_column;
                else
                    new_column = prod(X[:,combos[i]], dims=2);
                    if length(redundant) > 0
                        for j ∈ 1:length(redundant)
                            new_column += prod(X[:,combos[redundant[j]]], dims=2);
                        end
                    end
                    full_interaction = hcat(full_interaction, new_column)
                end
            end
        end

        # @show size(sym_combos)
        sym_combos = sym_combos[duplicates .==0]
        
        combos = combos[duplicates .==0]
        return full_interaction, sym_combos, combos
end
