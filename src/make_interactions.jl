function make_interactions(X::Matrix, exchange_vec, m, this_j, perm)
    # X is a collection of univariate bernstein polynomials of dimension m
    ncols = size(X,2);
    combos = Combinatorics.combinations(collect(1:ncols), Int(ncols/(m+1)));

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


    # Drop combinations that are combining two columns from the same polynomial
    bad_combo = []
    # for j = 1:J
    #     for i ∈ eachindex(combos)
    #         temp_i = string.(sym_vec[combos[i]])
    #         numj = sum(contains.(temp_i, "shares$j"))
    #         if (numj > 1)
    #             push!(bad_combo, combos[i])
    #         end
    #     end
    # end
    for j = 1:J
        for (i,v) ∈ enumerate(combos)
            temp_i = string.(sym_vec[v])
            numj = sum(contains.(temp_i, "shares$j"))
            if (numj > 1)
                push!(bad_combo, v)
            end
        end
        @show j
    end 

    combos = setdiff(combos, bad_combo)
    combos = combos[getindex.(size.(combos),1) .==J]

    # Apply remaining combos to symbolic vector
    sym_combos = []
    for i ∈ eachindex(combos)
        push!(sym_combos, prod(string.(sym_vec[combos[i]])))
    end

    # Calculate the order of the bernstein polynomial in each share
    orders = -1 .* ones(size(combos,1), J)
    for i ∈eachindex(combos)
        inds = getindex.(collect.(findall("_", sym_combos[i])),1) .+1
        for j = 1:length(inds) 
            orders[i,j] = parse.(Int,sym_combos[i][inds[j]])
        end
    end

    # for each element of sym_combos
    #     learn order of all shares
    #     find element with same order of shares but with exchanged shares flipped
    #     mark anything later than current element as a duplicate
    # end
    # then drop duplicates
    duplicates = zeros(size(combos,1))
    # exchange_vec = [[1,2,3,4],[5,6]]
    for exchange ∈ exchange_vec
        for j1 ∈ exchange, j2 ∈ exchange
            if (j1!=j2) & !(max(j1 ==this_j, j2==this_j))
                for i ∈ eachindex(sym_combos)
                    other_orders = setdiff(1:J, [j1, j2])
                    if (orders[i,j1] !=-1) & (orders[i,j2] !=-1)
                        rows = findall(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& (orders[:,j1] .== orders[i,j2]) .& (orders[:,j2] .== orders[i,j1]));
                        rows = getindex.(rows, 1);
                        rows = rows[rows .> i];
                        duplicates[rows] .=1
                    end
                end
            end
        end
    end
    sym_combos = sym_combos[duplicates .==0]
    combos = combos[duplicates .==0]

    full_interaction = prod(X[:,combos[1]], dims=2)
    for i ∈ eachindex(combos)[2:end]
        full_interaction = hcat(full_interaction, prod(X[:,combos[i]], dims=2))
    end
    return full_interaction, sym_combos, combos
end


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