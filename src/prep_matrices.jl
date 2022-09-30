function prep_matrices(df::DataFrame, exchange, index_vars, bO)

# Unpack DataFrame df
s = Matrix(df[:, r"shares"]);
pt = Matrix(df[:, r"prices"]);
zt = Matrix(df[:, r"demand_instruments"]);

J = size(s,2);

bernO = bO;
order = bO;
T = size(df,1);
IVbernO = bO;

# --------------------------------------------
# Check normalization
if maximum(zt)>1 #|| mininum(zt)<0
   throw("Error: Instruments are not normalized to be between 0 and 1 \n")
end

# --------------------------------------------
# Prep for design matrix and constraints
B = [] 
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
    index_j[:,1] = -1 .* index_j[:,1];
    push!(B, index_j)
end

FEmat = []
if FEmat !=[]
    for j = 0:J-1
        B[j] = hcat(B[j], FEmat); # assumes we're already normalizing
        push!(B, index_j)
    end
end

Xvec = []
Avec = []
syms = []

for xj = 1:J
    which_group = findall(xj .∈ exchange)[1];
    first_product_in_group = exchange[which_group][1];
    BERN_xj = zeros(T,1);

    perm = collect(1:J);
    perm[first_product_in_group] = xj; perm[xj] = first_product_in_group;

    perm_s = copy(s);
    perm_s[:,first_product_in_group] = s[:,xj]; perm_s[:,xj] = s[:,first_product_in_group];

   # Market shares
   for j = 1:1:J
        BERN_xj = [BERN_xj bern(perm_s[:,j], bO) ]
   end
   BERN_xj = BERN_xj[:,2:end]
   # --------------------------------------------
   # Instruments
   A_xj = zeros(T,1)
   ztemp = zt;
   for zj = 1:1:size(ztemp, 2)
       A_xj = [A_xj bern(ztemp[:,zj], IVbernO)]
   end
    A_xj = A_xj[:, 2:end]

    A_xj, sym_combos, combos = make_interactions(A_xj, exchange, bO, xj, perm);
    full_interaction, sym_combos, combos = make_interactions(BERN_xj, exchange, bO, first_product_in_group, perm);

    for k ∈ eachindex(index_vars) 
        v = index_vars[k];
        if v!= "prices"
            A_xj = hcat(A_xj, df[!,"$(v)$(xj-1)"]);
        end
    end

   println("Done with choice $(xj-1)")
   push!(Xvec, full_interaction)
   push!(Avec, A_xj)
   push!(syms, sym_combos)
end

return Xvec, Avec, B, syms
end