function prep_matrices(df::DataFrame, exchange, index_vars, 
    FEmat, product_FEs, bO; price_iv = [], inner = false, verbose = true)

# Unpack DataFrame df
s = Matrix(df[:, r"shares"]);
pt = Matrix(df[:, r"prices"]);
# zt = Matrix(df[:, r"demand_instruments"]);
if !inner 
    zt = Matrix(df[:, r"share_iv"]);
    zt = (zt .- minimum(zt, dims=1)) ./ (maximum(zt, dims=1) .- minimum(zt, dims=1));
end

J = size(s,2);
try 
    @assert J<20
catch
    error("J>20 not yet supported - will break polynomial and constraint construction")
end
bernO = bO;
order = bO;
T = size(df,1);
IVbernO = bO;

# --------------------------------------------
# Check normalization
if !inner
    if maximum(zt)>1 #|| mininum(zt)<0
    throw("Error: Instruments are not normalized to be between 0 and 1 \n")
    end
end

find_prices = findall(index_vars .== "prices");
price_ind = find_prices[1];

# --------------------------------------------
# Prep for design matrix and constraints
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
    BERN_xj = zeros(T,1);
    
    perm = collect(1:J);
    perm[first_product_in_group] = xj; perm[xj] = first_product_in_group;
    # perm_s = copy(s);
    # perm_s[:,first_product_in_group] = s[:,xj]; perm_s[:,xj] = s[:,first_product_in_group];
    
    # Market shares
    for j = 1:1:J
        BERN_xj = [BERN_xj bern(s[:,perm[j]], bO)]
    end
    BERN_xj = BERN_xj[:,2:end]

    # full_interaction, sym_combos, combos = make_interactions(BERN_xj, exchange, bO, first_product_in_group, perm);
    full_interaction, sym_combos, combos = make_interactions(BERN_xj, exchange, bO, xj, perm);
    # temp_combo_size = size(full_interaction,2);
    # println("size of combos: $temp_combo_size")
# --------------------------------------------
# Instruments
if !inner 
    A_xj = zeros(T,1)
    ztemp = zt;
    for zj = 1:1:size(ztemp, 2)
        A_xj = [A_xj bern(ztemp[:,perm[zj]], IVbernO)]
    end
    A_xj = A_xj[:, 2:end]
    A_xj, sym_combos, combos = make_interactions(A_xj, exchange, bO, xj, perm);
    # full_interaction, sym_combos, combos = make_interactions(BERN_xj, exchange, bO, first_product_in_group, perm);
        
    # Add index vars as IV
    for k ∈ eachindex(index_vars) 
        v = index_vars[k];
        if v!= "prices"
            A_xj = hcat(A_xj, df[!,"$(v)$(xj-1)"]);
        end
    end

    # add IVs for price
    if price_iv == []
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
    push!(syms, sym_combos)
    push!(all_combos, combos)
end

return Xvec, Avec, B, syms, all_combos

end