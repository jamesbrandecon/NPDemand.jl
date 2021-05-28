function inverse_demand(df ; included = ones(size(eachcol(df[:, r"shares"]),1), size(eachcol(df[:, r"shares"]),1)),
    bO = 2 .*ones(size(included,2),1), iv::Integer = 0, marketvars = nothing, constrained::Bool = false)
# Function to estimate inverse demand function as in Compiani (2018)
    # bO should be a (J x 1) matrix where each entry is
    # the order of the polynomials for the demand function for one product.
# include("fullInteraction.jl")
# include("bern.jl")
# include("dbern.jl")
# include("b.jl")
# include("db.jl")

 if size(included,2) > 4
     bigProblem = 1
 else
     bigProblem = 0
 end
J = size(included,2);

# Unpack DataFrame df
s = Matrix(df[:, r"shares"]);
pt = Matrix(df[:, r"prices"]);
xt = Matrix(df[:, r"x"]);
zt = Matrix(df[:, r"demand_instruments"]);

bernO = bO;
order = bO;
T = size(s,1);
IVbernO = bO.+iv;
if size(bO,1) !=J
    if size(bO,2)==J
        bO=bO';
    else
        throw("Error: Matrix of polynomial orders has wrong number of elements \n")
    end
end

# --------------------------------------------
# Check normalization
if maximum(zt)>1 #|| mininum(zt)<0
    throw("Error: Instruments are not normalized to be between 0 and 1 \n")
end

if marketvars != nothing
    if maximum(marketvars)>1 || minimum(zt)<0
        throw("Error: Market-level variables are not normalized to be between 0 and 1 \n")
    end
end

# --------------------------------------------
# Prep for design matrix and constraints
delta = zeros(T,J)
for j = 1:J
    delta[:,j] = -1*pt[:,j]
end

BERN = []
A = []
design = []

for xj = 1:J
    BERN_xj = zeros(T,1)
    if xj>1
        if xj<J
            perm_order = hcat(convert.(Integer, xj:J)', convert.(Integer, 1:xj-1)');
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
    # Market shares
    for j = 1:1:J
         if included[xj,perm_order[j]]==1
            BERN_xj = [BERN_xj bern(s_perm[:,j],convert(Integer, bernO[xj,1])) ]
         end
    end

    # Market characteristics
    if marketvars!=nothing
        for j = 1:1:size(marketvars,2)
            BERN_xj = [BERN_xj bern(marketvars[:,j], bernO[xj,1])]
        end
    end

    if bigProblem == 1
        print("Done Constructing Design Matrix \n")
    end
    # Define instruments
    cols = convert.(Bool, included[xj,:]); # only need instruments for included products
    z_xj = zt[:,cols];
    if marketvars!=nothing
        z_xj = [zt[:,cols] marketvars];
    end

    # --------------------------------------------
    # Instruments
    A_xj = zeros(T,1)
    ztemp = z_xj;
    for zj = 1:1:size(ztemp, 2)
        A_xj = [A_xj bern(ztemp[:,zj], convert(Integer,IVbernO[xj,1]) )]
    end
    A_xj = A_xj[:, 2:end]

    numVars = convert(Integer, size(A_xj, 2)./(IVbernO[xj,1]+1) )
    A_xj = fullInteraction(A_xj, numVars, convert(Integer, IVbernO[xj,1]))
    if bigProblem == 1
        print("Done with instruments \n")
    end
    # --------------------------------------------
    # Prep for estimation
    if marketvars==nothing
        numVars = sum(included[xj,:]);
    else
        numVars = sum(included[xj,:]) + size(marketvars,2);
    end
    BERN_xj = BERN_xj[:,2:end]
    design_xj = fullInteraction(BERN_xj, convert(Integer, numVars), convert(Integer,bernO[xj,1]) )
    if bigProblem == 1
        print("Done with estimation prep \n")
    end
    push!(BERN, BERN_xj)
    push!(design, design_xj)
    push!(A, A_xj)
end

# --------------------------------------------
# Estimation

if constrained==false
    sig = []
    designs = []
    for xj = 1:J
        # Unconstrained Estimator
        if bigProblem == 1
            print("UNCONSTRAINED ESTIMATIION BEGINNING \n")
        end
        deltatemp = delta[:,xj]
        Atemp = A[xj]
        designtemp = design[xj]
        sdt = size(designtemp,2)
        if bigProblem == 1
            print("Number of parameters to estimate is: $sdt \n")
        end
        TOL = sqrt(eps(real(float(one(eltype(Atemp))))))
        sigtemp = pinv((designtemp'*Atemp*pinv(Atemp'*Atemp, rtol = TOL)*Atemp'*designtemp), rtol=TOL)*designtemp'*Atemp*pinv(Atemp'*Atemp, rtol=TOL)*Atemp'deltatemp;
        push!(sig, sigtemp)
        push!(designs, designtemp)
        if bigProblem == 1
            print("Estimation for product $xj is done \n")
        end
    end
else
    sig = []
    for xj = 1:1:J
        # Constrained Estimator
        if bigProblem == 1
            print("CONSTRAINED ESTIMATIION BEGINNING \n")
        end
        deltatemp = delta[:,xj]
        Atemp = A[xj]
        designtemp = design[xj]
        ApA = Atemp'*Atemp
        #eval(['ApA = transpose(A', num2str(xj), ')*A',num2str(xj),';']);
        if marketvars!=nothing
            numVars = sum(included(xj,:)) + size(marketvars,2);
        else
            numVars = sum(included[xj,:]);
        end
        #eval(['constMat = makeConstraint(BERN_x', num2str(xj), ', numVars, bernO(xj,1),1,1,1);']);
        numVars = convert(Integer, numVars)
        bernO = convert.(Integer, bernO)
        constMat = makeConstraint(BERN[xj], numVars, bernO[xj,1],1,1,1);
        ineqMat = 0*ones(size(constMat[1],1),1);
        sdt =size(designtemp,2)
        if bigProblem == 1
            print("Number of parameters to estimate is: $sdt \n")
        end
        ApA = pinv(ApA);
        constMat = constMat[1];
        bigA = Atemp*ApA*Atemp';
        #obj(x) = objective_priceIndex(x, designtemp ,deltatemp , Atemp, ApA);
        function obj1(x::Vector)
            xi = (deltatemp - designtemp*x);
            out = xi'*bigA*xi;
            return out
        end
        function con1(x::Vector)
            out = constMat*x
            return out
        end
        g = x -> ForwardDiff.gradient(obj1, x)
        g2 = x-> ForwardDiff.gradient(con1, x)
        function obj(x::Vector, grad::Vector)
            if length(grad) > 0
                grad = g(x);
            end
            xi = (deltatemp - designtemp*x);
            out = xi'*bigA*xi;
            return out
        end
        function constraint(result::Vector, x::Vector, grad::Matrix, constMat)
            result = constMat*x
            if length(grad) > 0
                grad = g2(x);
            end
            return result
        end
        opt = Opt(:LN_COBYLA, size(designtemp,2))
        opt.xtol_rel = 1e-4
        opt.min_objective = obj
        inequality_constraint!(opt, (result,x,g) -> constraint(result, x, g, constMat), dropdims(1e-8.*ones(1,size(designtemp,2)),dims=1) )
        (minf,minx,ret) = NLopt.optimize(opt, dropdims(ones(size(designtemp,2),1),dims = 2) )
        sigtemp = minx
        push!(sig, sigtemp)
        designs = [];
    end
end
inv_sigma = []
for j = 1:1:J
    push!(inv_sigma, sig[j])
end

return inv_sigma, designs
end
