function make_constraint(df::DataFrame, constraints, exchange, combo_vec)
    J = length(combo_vec)
    try 
        @assert J<20
    catch
        error("J>20 not yet supported - will break constraint ")
    end
    # Find lengths of individual θ_j'ss
    lengths = []
    for j = 1:J
        lengths = vcat(lengths, length(combo_vec[j]))
    end
    # @show combo_vec[1][1:5] combo_vec[3][1:5]
    # NOTE:: PERMUTATIONS WILL MESS UP SYM COMBOS FOR EVERYTHING BUT FIRST ELEMENT OF EXCHANGEABLE GROUP
    # Calculate order of each share in each polynomial
    order_vec = [];
    for j = 1:J
        sym_combos = combo_vec[j]
        orders = -1 .* ones(size(sym_combos,1), J)
        for i ∈ eachindex(sym_combos)
            inds = getindex.(collect.(findall("_", sym_combos[i])),1) .+1
            share_ind = getindex.(collect.(findall("_", sym_combos[i])),1) .-1
            for j2 = 1:J #1:length(inds) 
                share = parse.(Int,sym_combos[i][share_ind[j2]]);
                # if previous digit isn't "e", add 10
                prev_digit = sym_combos[i][share_ind[j2]-1];
                if prev_digit != 's' # if not s, then it will be a number
                    share = share+10 
                end
                order = parse.(Int,sym_combos[i][inds[j2]]);
                orders[i,share] = order;
            end
        end
        push!(order_vec, orders)
    end
    
    # Under exchangeability, we are re-ordering one too many times
    # Need to set orders to the order of the first product in the group
    for e in exchange 
        store_order = order_vec[e[1]];
        for j in e
            order_vec[j] = order_vec[e[1]]
        end
    end
    
    # Initialize constraint matrices 
    Aineq = zeros(1, sum(lengths));
    Aeq = zeros(1, sum(lengths));

    # Find first product in each group of exchangeable products 
    first_in_exchange = getindex.(exchange, 1);

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
                    rows = findall(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& 
                        (orders[:,inv_j] .== orders[i,inv_j]+1));
                    rows = getindex.(rows, 1);
                    if rows !=[]
                        rows = rows[1];
                        # Aineq = add_constraint(Aineq, i + init_ind, rows + init_ind);
                        Aineq = add_constraint(Aineq, init_ind + i, init_ind + rows);
                        # print_index = i + init_ind;
                        # println("inv_j: $inv_j")
                        # @show orders[i,:] orders[rows,:]
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
                    rows = findall(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& 
                        (orders[:,inv_j] .== orders[i,inv_j]+1));
                    rows = getindex.(rows, 1);
                    try
                        rows = rows[1];
                        Aineq = add_constraint(Aineq, init_ind + i, init_ind + rows);
                    catch
                    end
                end
            end
        end
    end

    if :all_substitutes ∈ constraints
    #     # Monotonicity in all shares
        if :exchangeability ∈ constraints
            for inv_j ∈ first_in_exchange
                if inv_j >1
                    init_ind = sum(lengths[1:inv_j-1])
                else
                    init_ind = 0;
                end
                # Loop over all products in demand function
                for j_loop = 1:J
                    other_orders = setdiff(collect(1:J), j_loop)
                    orders = order_vec[j_loop];
                    for i ∈ eachindex(orders[:,1])
                        rows = findall(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& 
                                (orders[:,j_loop] .== orders[i,j_loop]+1));
                        rows = getindex.(rows, 1);
                        if length(rows) >= 1
                            rows = rows[1];
                            Aineq = add_constraint(Aineq, init_ind + i, init_ind + rows);
                        end
                    end
                end
            end
        else
            # print(":all_substitutes currently only implemented for models with some level of exchangeability")
            for inv_j ∈ collect(1:J)
                if inv_j >1
                    init_ind = sum(lengths[1:inv_j-1])
                else
                    init_ind = 0;
                end
                # Loop over all products in demand function
                for j_loop = 1:J
                    other_orders = setdiff(collect(1:J), j_loop)
                    orders = order_vec[j_loop];
                    for i ∈ eachindex(orders[:,1])
                        rows = findall(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& 
                            (orders[:,j_loop] .== orders[i,j_loop]+1));
                        rows = getindex.(rows, 1);
                        try
                            rows = rows[1];
                            Aineq = add_constraint(Aineq, init_ind + i, init_ind + rows);
                        catch
                        end
                    end
                end
            end
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
                # for j ∈ setdiff(collect(1:J), inv_j)
                for j ∈ setdiff(exchange[e], inv_j)
                    other_orders = setdiff(collect(1:J), [inv_j, j])
                    row1 = findfirst(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& 
                    (orders[:,inv_j] .== orders[i,inv_j]+1) .& (orders[:,j] .== orders[i,j]));
                    row2 = findfirst(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& 
                    (orders[:,inv_j] .== orders[i,inv_j]) .& (orders[:,j] .== orders[i,j]-1));
                
                    if (!isnothing(row1)) & (!isnothing(row2))
                        row1 = getindex(row1,1);
                        row2 = getindex(row2,1);
                        Aineq = add_constraint(Aineq, init_ind + row2, init_ind + row1);
                    end
                end
            end
        end
    end

    if :diagonal_dominance_all ∈ constraints
        for inv_j ∈ first_in_exchange
            orders = order_vec[inv_j];
            if inv_j >1
                init_ind = sum(lengths[1:inv_j-1])
            else
                init_ind = 0;
            end
            for i = 1:lengths[inv_j]
                for j ∈ setdiff(collect(1:J), inv_j)
                    other_orders = setdiff(collect(1:J), [inv_j, j])
                    if J > 2
                        row1 = findfirst(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& 
                        (orders[:,inv_j] .== orders[i,inv_j]+1) .& (orders[:,j] .== orders[i,j]));
                        # JMB should orders[i,j] + 1 at the END of this line be plus or minus 1?
                        row2 = findfirst(minimum((orders[i,other_orders]' .== orders[:,other_orders]), dims=2) .& 
                        (orders[:,inv_j] .== orders[i,inv_j]) .& (orders[:,j] .== orders[i,j]-1));
                    else
                        row1 = findfirst((orders[:,inv_j] .== orders[i,inv_j]+1) .& (orders[:,j] .== orders[i,j]));
                        row2 = findfirst((orders[:,inv_j] .== orders[i,inv_j]) .& (orders[:,j] .== orders[i,j]+1));
                    end 

                    if (!isnothing(row1)) & (!isnothing(row2))
                        row1 = getindex(row1,1);
                        row2 = getindex(row2,1);
                        Aineq = add_constraint(Aineq, init_ind + row2, init_ind + row1);
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
    # @show size(Aineq)

    # ------------------------
    # Clean up 
    Aineq = Aineq[2:end,:];
    Aineq = Matrix(hcat(unique(eachrow(Aineq))...)') # Drop redundant inequality constraints
    if size(Aineq,2)==0
        Aineq = [];
    end

    Aeq = Aeq[2:end,:];

    mins = dropdims(getindex.(argmin(Aeq, dims=2),2), dims=2);
    order = sortperm(mins, rev=true);
    mins = mins[order];
    maxs = getindex.(argmax(Aeq, dims=2),2);
    maxs = maxs[order];

    return Aineq, Aeq, maxs, mins
end

function are_constraints_satisfied(npd_problem)
    if npd_problem.Aineq != []
        # Modify theta as in objective function 
        β = npd_problem.results.minimizer;
        θ = β[1:npd_problem.design_width]
        problem_has_linear_constraints = false;
        if npd_problem.Aineq != []
            problem_has_linear_constraints = true;
            linear_constraints_violated = false;
        else 
            linear_constraints_violated = (maximum(npd_problem.Aineq * θ) > npd_problem.constraint_tol);
        end

        # Enforce equality constraints directly
        for i∈eachindex(npd_problem.mins)
            θ[npd_problem.mins[i]] = θ[npd_problem.maxs[i]]
        end
        nonlinear_condition = false;
        if npd_problem.converged!=[] 
            nonlinear_condition = !(npd_problem.converged)
        end
        if linear_constraints_violated | nonlinear_condition 
            if nonlinear_condition
                println("Estimation with nonlinear constraints did not converge")
            end
            if linear_constraints_violated
                println("Linear constraints not satisfied -- maximum deviation is $(maximum(npd_problem.Aineq * θ))")
            end
            return false
        else
            return true
        end
    end
end