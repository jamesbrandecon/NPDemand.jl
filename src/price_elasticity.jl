function price_elasticity_priceIndex(inv_sigma, s, p_points, deltas, bO, whichProducts, included, trueS, maxes)
# --------------------------------------------------------------------
# Code takes in estimates from inverse_demand and returns implied price
# elasticity of first good
# --------------------------------------------------------------------

# inv_sigma     -- estimates returned by inverse_demand.m
# s             -- market shares
# p_points      -- vector of prices corresponding to s
# deltas        -- index (excluding xi) at which to evaluate demand
# b0            -- order of Bernstein polynomials
# whichProducts -- [derivative of s_j, with respect to k]
# included      -- matrix indicating which products are substitutes
# trueS         -- evaluate derivatives at realized shares or calculate
#                   counterfactual shares
# maxes         -- object with maxes.maxs = max(s), maxes.mins = min(s),
#                   used if s is small enough to cause numerical issues and needs to be rescaled


J = size(s,2);
numBadMarkets = 0;
if size(bO,1) !=J
    if size(bO,2)==J
        bO=bO';
    elseif length(bO) ==1
        bO = bO*ones(J,1);
    else
        throw("Error: Matrix of polynomial orders has wrong number of elements")
    end
end
bernO = convert.(Integer, bO);
order = bernO;

sig = [];
for j = 1:1:J
    push!(sig, inv_sigma[j])
end
if !isempty(maxes)
    maxs = maxes[1];
    mins = maxes[2];
end


## Declare prices and shares to evaluate derivatives
svec = similar(deltas)
if trueS == 0
    for p_i = 1:size(deltas,1)
        s!(sj) = solve_s_nested_flexible(sj, inv_sigma, deltas[p_i,:]', J, bernO, included, maxes,nothing);
        ans = nlsolve(s!, 1/(2*J) .* ones(J))
        svec[p_i,:] =  ans.zero;
    end
else
    svec = s;
end

# Share Jacobian
dsids = zeros(J,J,size(deltas,1)) # initialize matrix of \partial s^{-1} /\partial s
for j1 = 1:J
    if j1>1
        if j1<J
            perm_order = hcat(convert.(Integer,j1:J)', convert.(Integer, 1:j1-1)');
            perm_order = convert.(Integer, perm_order)
        else
            perm_order = convert.(Integer,1:J)
            perm_order = perm_order .- 1
            perm_order[1] = J
        end
    else
        perm_order = convert.(Integer, 1:J)
    end
    s_perm = similar(svec)
    for i = 1:J
        s_perm[:,i] = svec[:,perm_order[i]];
    end
    for j2 = 1:J
        if included[j1,j2]==1 # cross-derivatives are zero if not included in j1 demand
            tempmat_s = zeros(size(deltas,1),1)
            for j3 = 1:1:J
                stemp = s_perm[:,j3];
                if included[j1,perm_order[j3]]==1 # design matrix only uses included products
                    if j2==perm_order[j3]
                        tempmat_s = [tempmat_s dbern(stemp, bernO[j1,1])];
                    else
                        tempmat_s = [tempmat_s bern(stemp, bernO[j1,1])];
                    end
                end
            end

            # Market characteristics
            marketvars = nothing
            if marketvars!=nothing
                for i = 1:size(marketvars,2)
                    tempmat_s = [tempmat_s bern(marketvars[:,i], bernO[j1,1])];
                end
            end
            tempmat_s = tempmat_s[:,2:end]
            tempmat_s = fullInteraction(tempmat_s, convert.(Integer, sum(included[j1,:])), convert.(Integer, bernO[j1,1]));
            if !isempty(maxes)
                dsids[j1,j2,:] = tempmat_s*inv_sigma[j1] ./ (maxs[j2] - mins[j2])  ;
            else
                dsids[j1,j2,:] = tempmat_s*inv_sigma[j1];
            end
        else
            zerotemp = zeros(size(svec,1),1);
            dsids[j1,j2] = zerotemp;
        end
    end
end

Jmat = [];
J_sp = zeros(size(svec[:,1]));
all_own = zeros(size(svec,1),J);
if !isempty(maxes)
    svec2 = svec.*(maxs - mins) + mins;
else
    svec2 = svec;
end
for ii = 1:length(dsids[1,1,:])
    J_s = [];
    for j1 = 1:J
        J_sj1 = [];
        for j2 = 1:J
            if j2==1
                J_sj1 = dsids[j1,j2,ii]
            else
                J_sj1 = [J_sj1 dsids[j1,j2,ii]]
            end
        end
        if j1==1
            J_s = J_sj1;
        else
            J_s = vcat(J_s, J_sj1);
        end
    end
    temp = -1*inv(J_s);
    push!(Jmat, temp)
    ps = p_points[ii,:]./svec2[ii,:]
    all_own[ii,:] = -1*inv(J_s)*ps;
    J_sp[ii,1] = temp[whichProducts[1],whichProducts[2]];
    #print("Market $ii \n")
end

esep = J_sp.*p_points[:,1]./svec2[:,1]; # own-price varying
numBadMarkets = 0
# if trueS==1
#     print("There were $numBadMarkets bad markets")
# end
return esep, Jmat, svec, all_own
end
