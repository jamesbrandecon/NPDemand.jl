function solve_s_nested_flexible(sj,inv_sigma,deltas,J,order,included,maxes,marketvars)
# Solve for s as p moves
if size(sj,1)>1
    sj = sj';
end
if marketvars!=nothing
    extra = size(marketvars,2);
else
    extra = 0;
end

s = sj;
bernO = convert.(Integer,order);
BERN = []
# @show J
for xj = 1:J
    if xj>1
        if xj<J
            perm_order = hcat(xj:J, 1:xj-1);
            perm_order = convert.(Integer, perm_order)
        else
            perm_order = convert.(Integer,1:J)
            perm_order = perm_order .- 1
            perm_order[1] = J
        end
    else
        perm_order = convert.(Integer, 1:J)
    end
    s_perm = similar(s)
    for i = 1:J
        s_perm[:,i] = s[:,perm_order[i]];
    end
    # becomes design matrix
    BERN_xj = zeros(size(deltas,1),1)
# Market shares
for j = 1:J
     if included[xj,perm_order[j]]==1
        BERN_xj = [BERN_xj bern(s_perm[:,j], bernO[xj,1] )]
     end
end

# Market characteristics
if marketvars!=nothing
    for j = 1:size(marketvars,2)
        BERN_xj = [BERN_xj bern(s_perm[:,j], bern[xj,1])]
    end
end
BERN_xj = BERN_xj[:,2:end]
push!(BERN, BERN_xj)
end
out = []
for j = 1:J
    numVars = sum(included[j,:]) + extra;
    design = fullInteraction(BERN[j], convert(Integer,numVars), bernO[j,1])
    temp = deltas[1,j] .- design*inv_sigma[j]
    append!(out, temp)
end
return convert.(Float64, out)

end
