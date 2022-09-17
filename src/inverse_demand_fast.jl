function inverse_demand_fast(df ; included = ones(size(eachcol(df[:, r"shares"]),1), size(eachcol(df[:, r"shares"]),1)),
    bO = 2 .*ones(size(included,2),1), iv::Integer = 0, marketvars = nothing, constrained::Bool = false)
# Function to estimate inverse demand function as in Compiani (2018)
    # bO should be a (J x 1) matrix where each entry is
    # the order of the polynomials for the demand function for one product.

# J = size(included,2);
if index_vars[1]!="prices"
    error("Prices has to be the first variable in the index")
end

using ForwardDiff, Statistics
using Optim, LineSearches

J=2; # of products
T =5000; # # of markets
beta = -0.4; # price coefficient
sdxi = 0.15; # standard deviation of xi
bO = 2;
s, p, z, x, xi  = simulate_logit(J, T, beta, sdxi);
df = toDataFrame(s,p,z,x);
# exchange = [[1 2 3 4], [5 6]]
exchange = []
index_vars = ["prices", "x"]
normalization = [];
tol = 1e-5
constraints = [:monotone]
# , :exchangeability, :diagonal_dominance
Xvec, Avec, Bvec, syms = prep_matrices(df, exchange, index_vars, bO);
Aineq, Aeq, maxs, mins = make_constraint(constraints, exchange, syms);
matrices = prep_inner_matrices(Xvec, Avec, Bvec)

design_width = sum(size.(Xvec,2));
lambda1 = 10000
# Define objective and gradients
obj_func(β::Vector) = md_obj(β;X = Xvec, B = Bvec, A = Avec,
        m1=matrices.m1, 
        m2=matrices.m2, 
        m3=matrices.m3, 
        m4=matrices.m4, 
        m5=matrices.m5, 
        m6=matrices.m6, 
        m7=matrices.m7, 
        m8=matrices.m8, 
        m9=matrices.m9,
        DWD = matrices.DWD,
        WX = matrices.WX, 
        WB = matrices.WB,
        Aineq = Aineq, Aeq = Aeq, design_width = design_width, mins = mins, maxs = maxs, normalization = normalization, price_index = 1, lambda1 = lambda1);

grad_func!(grad::Vector, β::Vector) = md_grad!(grad, β; X = Xvec, B = Bvec, A = Avec,
        m1=matrices.m1, 
        m2=matrices.m2, 
        m3=matrices.m3, 
        m4=matrices.m4, 
        m5=matrices.m5, 
        m6=matrices.m6, 
        m7=matrices.m7, 
        m8=matrices.m8, 
        m9=matrices.m9,
        DWD = matrices.DWD,
        WX = matrices.WX, 
        WB = matrices.WB,
        Aineq = Aineq, Aeq = Aeq, design_width = design_width, mins = mins, maxs = maxs, normalization = normalization, price_index = 1, lambda1 = lambda1);

# Confirm that autodifferentiation agrees with analytic gradient
fdiff = ForwardDiff.gradient(obj_func, β_init)
G = zeros(length(β_init))
adiff = grad_func!(G, β_init)
@assert cor(fdiff, adiff) > 0.99

# Estimation 
β_length = design_width + sum(size(Bvec[1],2))
β_init = -1 .* rand(β_length)

results =  Optim.optimize(obj_func, grad_func!, rand(length(β_init)),
    LBFGS(), Optim.Options(show_trace = true, iterations = 10000));
results =  Optim.optimize(obj_func, grad_func!, results.minimizer,
    LBFGS(;linesearch = BackTracking(order=3)), Optim.Options(show_trace = true, iterations = 10000));

# Unpack estimates and verify that constraints are satisfied
β = results.minimizer
θ = β[1:design_width]
γ = β[length(θ)+1:end]
γ[1] = 1;
for i ∈ normalization
    γ[i] =0; 
end
for i∈eachindex(mins)
    θ[mins[i]] = θ[maxs[i]]
end
@assert maximum(Aineq*θ) < 1e-3

estimates = inner_estimation(obj_func, grad_func!, β_init)

monotonicity is messed up for second good

at = Matrix(df[!,r"prices"]);
esep, Jmat, svec, all_own = price_elasticity2(vcat(θ, γ), df, bO; X =Xvec, B = Bvec, at=at, whichProducts=[2,2]);
trueEsep = beta * df.prices0 .* (1 .- df.shares0);
trueEsep = beta * df.prices1 .* (1 .- df.shares1);
scatter(trueEsep, esep)
return inv_sigma
end


using CSV
AGC = Matrix(CSV.read("/Users/jamesbrand/Dropbox/NPD Pricing/Matrices/J6_subst/Aineq.csv", DataFrame))
matched = zeros(size(Aineq,1))
for i∈ eachindex(Aineq[:,1])
    matched[i] = length(findall(minimum(Aineq[i,:]' .== AGC)))
end