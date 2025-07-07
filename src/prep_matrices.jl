function prep_matrices(df::DataFrame, exchange, index_vars, 
    FEmat, product_FEs, order; price_iv = [], inner = false, 
    verbose = true, 
    approximation_details = Dict("sieve_type" => "bernstein", "order" => 2, "max_interaction" => 1), 
    constraints = [])

    sieve_type = approximation_details[:sieve_type]
    basis_function = sieve_type == "bernstein" ? bern : poly
    dbasis_function = sieve_type == "bernstein" ? dbern : dpoly
    
    # Unpack DataFrame df
    s = Matrix(df[:, r"shares"]);
    pt = Matrix(df[:, r"prices"]);
    
    if !inner 
        zt = Matrix(df[:, r"share_iv"]);
        zt = (zt .- minimum(zt, dims=1)) ./ (maximum(zt, dims=1) .- minimum(zt, dims=1));
    end

    J = size(s,2);
    try 
        @assert (sieve_type == "raw_polynomial") | (J<20)
    catch
        error("J>20 not yet supported - will break polynomial and constraint construction")
    end
    bernO = order;
    T = size(df,1);
    IVbernO = order;

    # --------------------------------------------
    # Check normalization
    if !inner
        if maximum(zt)>1 #|| mininum(zt)<0
        throw("Error: Instruments are not normalized to be between 0 and 1 \n")
        end
    end

    find_prices = findall(index_vars .== "prices");
    price_ind = find_prices[1];

    if sieve_type == "raw_polynomial"
        # Unpack approximation details
        order           = approximation_details[:order]
        max_interaction = approximation_details[:max_interaction]
        
        recipes = [ begin
        ex2 = length(exchange)==J ? [] : adjust_exchange(exchange, j1)
        build_poly_recipe(J;
            order           = approximation_details[:order],
            max_interaction = approximation_details[:max_interaction],
            exchange        = ex2)
        end for j1 in minimum.(exchange)]
    end

    # --------------------------------------------
    # Prep for design matrix and constraints
    # Order: index vars, then all FEs other than product FEs, then product FEs
    # --------------------------------------------
    B = []; 
    prod_FE_counter = 1;
    for j = 0:J-1
        index_j = []
        for k ∈ eachindex(index_vars) 
            v = index_vars[k];
            if k ==1
                index_j = df[!, "$(v)$(j)"];
            else
                index_j = hcat(index_j, df[!, "$(v)$(j)"]);
            end
        end
        
        index_j[:,price_ind] = -1 .* index_j[:,price_ind];
        
        # Append vector of FE dummies
        if FEmat!=[]
            index_j = hcat(index_j, FEmat);
        end
        # Append product IDs if specified by user -- treated differently here 
        # because all other FEs must be constant across products
        if product_FEs == true 
            num_FEs = J - length(exchange);
            prodFE = zeros(size(index_j,1),num_FEs);
            which_group = findall(j+1 .∈ exchange)[1];
            first_product_in_group = exchange[which_group][1];

            if j+1 !=first_product_in_group # dropping last product's FE for location normalization
                prodFE[:,prod_FE_counter] .= 1;
                prod_FE_counter +=1;
            end
            index_j = hcat(index_j, prodFE);
        end

        # Append to larger array of index variables
        push!(B, index_j)
    end

    Xvec = []
    Avec = []
    syms = []
    all_combos = [];

    prod_FE_counter = 1;
    for xj = 1:J
        which_group = findall(xj .∈ exchange)[1];
        first_product_in_group = exchange[which_group][1];
        perm = collect(1:J);
        perm[first_product_in_group] = xj; perm[xj] = first_product_in_group;

        if (sieve_type == "bernstein") & (setdiff(constraints, [:exchangeability]) != Symbol[])
            BERN_xj = zeros(T,1);
            
            # Market shares
            for j = 1:1:J
                BERN_xj = [BERN_xj bern(s[:,perm[j]], order)]
            end
            BERN_xj = BERN_xj[:,2:end]
            full_interaction, sym_combos, combos = make_interactions(BERN_xj, exchange, order, xj, perm);
        elseif sieve_type == "raw_polynomial"
            permuted_shares = zeros(T,1);
            for j = 1:1:J
                permuted_shares = [permuted_shares s[:,perm[j]]]
            end
            permuted_shares = permuted_shares[:,2:end]
            # We only want to exchange columns in the current exchange group
            # if it contains more than one product after dropping the current product
            exchange_for_poly = length(exchange) == J ? [] : adjust_exchange(exchange, first_product_in_group);
            full_interaction  = poly_features(
                permuted_shares, 
                order = order, 
                max_interaction = max_interaction, 
                exchange = exchange_for_poly, 
                recipe = recipes[which_group]);
        else # otherwise we are going to use fully generic tensor products
            permuted_shares = zeros(T,1);
            for j = 1:1:J
                permuted_shares = [permuted_shares s[:,perm[j]]]
            end
            permuted_shares = permuted_shares[:,2:end]
            full_interaction = NPDemand.tensor_features(
                permuted_shares,  # <-- Changed from 's' to 'permuted_shares'
                basis_orders = order .* ones(Int, J), 
                exchange = adjust_exchange(exchange, first_product_in_group), 
                basis_function = basis_function,
                dbasis_function = dbasis_function);
        end

        # --------------------------------------------
        # Instruments
        if !inner 
            A_xj = zeros(T,1)
            ztemp = zt;
            if sieve_type == "bernstein"
                for zj = 1:1:size(ztemp, 2)
                    A_xj = [A_xj bern(ztemp[:,perm[zj]], IVbernO)]
                end
                A_xj = A_xj[:, 2:end]
                A_xj, sym_combos, combos = make_interactions(A_xj, exchange, order, xj, perm);
            elseif sieve_type == "raw_polynomial" 
                # A_xj = zeros(T,1);
                # for j = 1:1:J
                #     A_xj = [A_xj ztemp[:,perm[j]]]
                # end
                # A_xj = A_xj[:,2:end]
                exchange_for_poly = length(exchange) == J ? [] : adjust_exchange(exchange, xj);
                A_xj = poly_features(
                    zt, 
                    order = order, 
                    max_interaction = max_interaction, 
                    exchange = exchange_for_poly, 
                    recipe = recipes[which_group]);
            else # otherwise we are going to use fully generic tensor products
                permuted_instruments = zeros(T,1);
                for j = 1:1:J
                    permuted_instruments = [permuted_instruments zt[:,perm[j]]]
                end
                permuted_instruments = permuted_instruments[:,2:end]
                A_xj = NPDemand.tensor_features(
                    permuted_instruments,  # <-- Changed from 'zt' to 'permuted_instruments'
                    basis_orders = order .* ones(Int, J),
                    exchange = adjust_exchange(exchange, xj), 
                    basis_function = basis_function,
                    dbasis_function = dbasis_function);
            end
                
            # Add index vars as IV
            for k ∈ eachindex(index_vars) 
                v = index_vars[k];
                if v!= "prices"
                    A_xj = hcat(A_xj, df[!,"$(v)$(xj-1)"]);
                end
            end

            # add IVs for price
            if (price_iv == []) | (price_iv == ["price_iv"])
                A_xj = hcat(A_xj, df[!,"price_iv$(xj-1)"])
            else
                for p_ivs ∈ price_iv
                    A_xj = hcat(A_xj, df[!,"$(p_ivs)$(xj-1)"]);
                end
            end
            
            # Add FEs/dummies as IVs
            if FEmat !=[]
                A_xj = hcat(A_xj, FEmat);
            end
            if product_FEs == true 
                num_FEs = J - length(exchange);
                prodFE = zeros(size(A_xj,1),num_FEs);

                which_group = findall(xj .∈ exchange)[1];
                first_product_in_group = exchange[which_group][1];

                if xj !=first_product_in_group # dropping last product's FE for location normalization
                    prodFE[:,prod_FE_counter] .= 1;
                    prod_FE_counter +=1;
                end
                A_xj = hcat(A_xj, prodFE);
            end
        end

        if (!inner) & (verbose)
            println("Done with choice $(xj-1)")
        end
        push!(Xvec, full_interaction)
        if !inner
            push!(Avec, A_xj)
        end
        if sieve_type == "bernstein"
            push!(syms, sym_combos)
            push!(all_combos, combos)
        end
    end

    return Xvec, Avec, B, syms, all_combos
end

function adjust_exchange(exchange::Vector{Vector{Int}}, xj::Int)
    new_groups = Vector{Vector{Int}}()
    for g in exchange
        if xj ∈ g
            # remove the focal element but keep the remainder if any
            g2 = filter(i->i != xj, g)
            if !isempty(g2)
                push!(new_groups, g2)
            end
        else
            # non‐focal groups stay intact
            push!(new_groups, copy(g))
        end
    end
    return new_groups
end