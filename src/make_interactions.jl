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
        # if j==1
        #     input_string = string("G[$(j)]")
        #     input_string2 = string("G2[$(j)]")
        # else
        #     input_string = string(input_string, ",G[$(j)]")
        #     input_string2 = string(input_string2, ",G2[$(j)]")
        # end
    end
    # input_string = string("collect(Iterators.product(", input_string, "))");
    # input_string2 = string("collect(Iterators.product(", input_string2, "))");

    # orders = eval(Meta.parse(input_string))
    # combos = eval(Meta.parse(input_string2))
    orders = collect(Iterators.product(G...));
    orders = reshape(orders, length(orders));
    orders = collect.(orders);
    
    combos = collect(Iterators.product(G2...));
    combos = reshape(combos, length(combos));
    combos = collect.(combos);
    # orders = collect(Iterators.product(G[1], G[2]));
    # combos = collect(Iterators.product(G2[1], G2[2]));
    # if J>2
    #     for gi = 3:J 
    #         orders =  collect(Iterators.product(orders, G[gi]));
    #         combos =  collect(Iterators.product(combos, G[gi]));
    #     end
    # end
    # orders = reshape(orders, length(orders));
    # combos = reshape(combos, length(combos));
    
    # order_vec = [orders[i,:] for i in 1:size(orders,1)];
    # orders = order_vec;

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
    
    # Calculate the order of the bernstein polynomial in each share
    # if J<10
    #     orders = -1 .* ones(size(combos,1), J)
    #     for i ∈eachindex(combos)
    #         inds = getindex.(collect.(findall("_", sym_combos[i])),1) .+1
    #         share_ind = getindex.(collect.(findall("_", sym_combos[i])),1) .-1
    #         for j = 1:length(inds) 
    #             share = parse.(Int,sym_combos[i][share_ind[j]]);
    #             order = parse.(Int,sym_combos[i][inds[j]]);
    #             orders[i,share] = order;
    #         end
    #     end
    # else
    #     orders = -1 .* ones(size(combos,1), J)
    #     for i ∈eachindex(combos)
    #         inds = getindex.(collect.(findall("_", sym_combos[i])),1) .+1
    #         share_ind = getindex.(collect.(findall("_", sym_combos[i])),1) .-1
    #         for j = 1:length(inds) 
    #             order = parse.(Int,sym_combos[i][inds[j]]);
    #             if j <10
    #                 share = parse.(Int,sym_combos[i][share_ind[j]]);
    #             else
    #                 share = parse.(Int,sym_combos[i][share_ind[j]-1:share_ind[j]]);
    #             end
    #             orders[i,share] = order;
    #         end
    #     end
    # end
        # inds = []
        inds = Array{Array}(undef, maximum(size(combos)))
        for exchange ∈ exchange_vec
                ex = setdiff(exchange, this_j)
                not_in_group = setdiff(1:J, ex)' # "not in group" means own-shares and cross-shares that are not in ex group
                orders_not_in_group = getindex.(orders, not_in_group)
                orders_in_group = getindex.(orders, ex')
                prod_order_in_group = prod(orders_in_group, dims=2)
                own_order = getindex.(orders, this_j)
                order_vec = reduce(hcat, orders)';
                
                @assert this_j ∉ ex
                # l = Threads.SpinLock()
                if exchange == exchange_vec[1]
                    Threads.@threads for i ∈ eachindex(combos)
                        combotemp = getindex.(findall((own_order .== own_order[i]) .&
                            (prod_order_in_group .== prod_order_in_group[i])),1)
                        
                        combotemp = combotemp[combotemp .> i]
                        inds[i] = [i,combotemp]
                    end
                else
                    indstemp = deepcopy(inds);
                    Threads.@threads for i ∈ collect(eachindex(combos))
                            combotemp = getindex.(findall((own_order .== own_order[i]) .&
                                (prod_order_in_group .== prod_order_in_group[i])),1)
                            
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

    # full_interaction = prod(X[:,combos[1]], dims=2)
    # for i ∈ eachindex(combos)[2:end]
    #     full_interaction = hcat(full_interaction, prod(X[:,combos[i]], dims=2))
    # end
    
# end


# for o1 = 1:m, o2 = 1:m 
#     for j1 ∈ other_in_group, j2 ∈ other_in_group
#         findall(contains.(sym_combos, "shares$(j1)_$o1") .& contains.(sym_combos, "shares$(j2)_$o2"))
#     end
# end


# Impose exchangeability 
# Find where there are duplicates of parameters under exchangeability
    # E.g., [0 1 2 2] = [0 2 1 1] = [0 2 1 2]
# found_j1 = [try sym_combos[i][findfirst.("shares$(j1)_", sym_combos)[i][end]+1]; catch nothing; end; for i ∈ eachindex(sym_combos)]

# if exchange !=[]
#    for i ∈ eachindex(combos)
#         r"(?<=_)(\d+)"
#         # reg = Regex("shares"*string(j1)*"_?");
#         sym_combos[4000][getindex.(collect.(findall("_", sym_combos[4000])),1) .+1]
#         my_matches = parse.(Int, )

#    end
# end