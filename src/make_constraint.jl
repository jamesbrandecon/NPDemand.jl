function make_constraint(df::DataFrame, constraints, exchange, combo_vec)
    J = length(combo_vec)
    
    # Find lengths of individual θ_j'ss
    lengths = []
    for j = 1:J
        lengths = vcat(lengths, length(combo_vec[j]))
    end
    # NOTE:: PERMUTATIONS WILL MESS UP SYM COMBOS FOR EVERYTHING BUT FIRST ELEMENT OF EXCHANGEABLE GROUP
    # Calculate order of each share in each polynomial
    order_vec = [];
    for j = 1:J
        sym_combos = combo_vec[j]
        orders = -1 .* ones(size(sym_combos,1), J)
        for i ∈ eachindex(sym_combos)
            inds = getindex.(collect.(findall("_", sym_combos[i])),1) .+1
            share_ind = getindex.(collect.(findall("_", sym_combos[i])),1) .-1
            for j = 1:length(inds) 
                share = parse.(Int,sym_combos[i][share_ind[j]]);
                order = parse.(Int,sym_combos[i][inds[j]]);
                orders[i,share] = order;
            end
        end
        push!(order_vec, orders)
    end
    
    # Initialize constraint matrices 
    Aineq = zeros(1, sum(lengths));
    Aeq = zeros(1, sum(lengths));

    # Find first product in each group of exchangeable products 
    first_in_exchange = []
    for i ∈ eachindex(exchange)
        first_in_exchange = vcat(first_in_exchange, exchange[i][1]);
    end

    if :monotone ∈ constraints
        # Monotonicity in own share
        if :exchangeability ∈ constraints
            for inv_j ∈ first_in_exchange
                if inv_j >1
                    init_ind = sum(lengths[1:inv_j-1])
                else
                    init_ind = 0;
                end
                other_orders = setdiff(collect(1:J), inv_j)
                orders = order_vec[inv_j];
                for i ∈ eachindex(orders[:,1])
                    rows = findall(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& (orders[:,inv_j] .== orders[i,inv_j]+1));
                    rows = getindex.(rows, 1);
                    try
                        rows = rows[1];
                        Aineq = add_constraint(Aineq, i + init_ind, rows + init_ind);
                    catch
                    end
                end
            end
        else
            for inv_j ∈ collect(1:J)
                if inv_j >1
                    init_ind = sum(lengths[1:inv_j-1])
                else
                    init_ind = 0;
                end
                other_orders = setdiff(collect(1:J), inv_j)
                orders = order_vec[inv_j];
                for i ∈ eachindex(orders[:,1])
                    # findall(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& (orders[:,j1] .== orders[i,j2]) .& (orders[:,j2] .== orders[i,j1]));
                    rows = findall(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& (orders[:,inv_j] .== orders[i,inv_j]+1));
                    rows = getindex.(rows, 1);
                    try
                        rows = rows[1];
                        Aineq = add_constraint(Aineq, i + init_ind, rows + init_ind);
                    catch
                    end
                end
            end
        end
    end

    if :all_substitutes ∈ constraints
    #     # Monotonicity in own share
    #     if :exchangeability ∈ constraints
    #         for inv_j ∈ first_in_exchange
    #             if inv_j >1
    #                 init_ind = sum(lengths[1:inv_j-1])
    #             else
    #                 init_ind = 0;
    #             end
    #             other_orders = setdiff(collect(1:J), inv_j)
    #             orders = order_vec[inv_j];
    #             for i ∈ eachindex(orders[:,1])
    #                 rows = findall(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& (orders[:,inv_j] .== orders[i,inv_j]+1));
    #                 rows = getindex.(rows, 1);
    #                 try
    #                     rows = rows[1];
    #                     Aineq = add_constraint(Aineq, i + init_ind, rows + init_ind);
    #                 catch
    #                 end
    #             end
    #         end
    #     else
    #         for inv_j ∈ collect(1:J)
    #             if inv_j >1
    #                 init_ind = sum(lengths[1:inv_j-1])
    #             else
    #                 init_ind = 0;
    #             end
    #             other_orders = setdiff(collect(1:J), inv_j)
    #             orders = order_vec[inv_j];
    #             for i ∈ eachindex(orders[:,1])
    #                 # findall(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& (orders[:,j1] .== orders[i,j2]) .& (orders[:,j2] .== orders[i,j1]));
    #                 rows = findall(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& (orders[:,inv_j] .== orders[i,inv_j]+1));
    #                 rows = getindex.(rows, 1);
    #                 try
    #                     rows = rows[1];
    #                     Aineq = add_constraint(Aineq, i + init_ind, rows + init_ind);
    #                 catch
    #                 end
    #             end
    #         end
    #     end
    end

    if :substitutes_within_group ∈ constraints
       if :all_substitutes ∈ constraints
            println(":all_substitutes and :substitutes_within_group both in constraint vector, ignoring the latter...")
       else
            # Make increasing in all in-group shares

            # Then calculate matrices for derivative of demand constraints
       end
    end

    # diagonal dominance 
        # currently only within group
    if :diagonal_dominance_group ∈ constraints
        for e ∈ eachindex(exchange)
            inv_j = first_in_exchange[e]
            others_in_group = setdiff(exchange[e], inv_j)
            if inv_j >1
                init_ind = sum(lengths[1:inv_j-1])
            else
                init_ind = 0;
            end
            orders = order_vec[inv_j];
            for i = 1:lengths[inv_j]
                for j ∈ setdiff(collect(1:J), inv_j)
                    other_orders = setdiff(collect(1:J), [inv_j, j])
                    rows = findall(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& (orders[:,inv_j] .== orders[i,inv_j]+1) .& (orders[:,j] .== orders[i,j]-1));
                    rows = getindex.(rows, 1);
                    try
                        rows = rows[1];
                        Aineq = add_constraint(Aineq, i + init_ind, rows + init_ind);
                    catch
                    end
                end
            end
        end
    end

    if :diagonal_dominance_all ∈ constraints
        for e ∈ eachindex(exchange)
            inv_j = first_in_exchange[e]
            others = setdiff(1:J, inv_j)
            if inv_j >1
                init_ind = sum(lengths[1:inv_j-1])
            else
                init_ind = 0;
            end
            orders = order_vec[inv_j];
            for i = 1:lengths[inv_j]
                for j ∈ setdiff(collect(1:J), inv_j)
                    other_orders = setdiff(collect(1:J), [inv_j, j])
                    if J >2
                        rows = findall(minimum((orders[i,others]' .== orders[:,others]), dims=2) .& (orders[:,inv_j] .== orders[i,inv_j]+1) .& (orders[:,j] .== orders[i,j]-1));
                    else
                        rows = findall((orders[:,inv_j] .== orders[i,inv_j]+1) .& (orders[:,j] .== orders[i,j]-1));
                    end
                    rows = getindex.(rows, 1);
                    try
                        rows = rows[1];
                        Aineq = add_constraint(Aineq, i + init_ind, rows + init_ind);
                    catch
                    end
                end
            end
        end
    end

    # Exchangeability within groups
    if :exchangeability ∈ constraints
        for e ∈ eachindex(exchange)
            inv_j = first_in_exchange[e]
            others_in_group = setdiff(exchange[e], inv_j)
            if inv_j >1
                init_ind = sum(lengths[1:inv_j-1])
            else
                init_ind = 0;
            end
            for i = 1:lengths[inv_j]
                for j ∈ eachindex(others_in_group)
                    Aeq = add_constraint(Aeq, i + init_ind, init_ind + i + j*lengths[inv_j]);
                end
            end
        end
    end

    Aineq = Aineq[2:end,:];
    Aeq = Aeq[2:end,:];

    mins = dropdims(getindex.(argmin(Aeq, dims=2),2), dims=2);
    order = sortperm(mins, rev=true);
    mins = mins[order];
    maxs = getindex.(argmax(Aeq, dims=2),2);
    maxs = maxs[order];

    return Aineq, Aeq, maxs, mins
end