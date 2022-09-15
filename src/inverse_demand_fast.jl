function inverse_demand_fast(df ; included = ones(size(eachcol(df[:, r"shares"]),1), size(eachcol(df[:, r"shares"]),1)),
    bO = 2 .*ones(size(included,2),1), iv::Integer = 0, marketvars = nothing, constrained::Bool = false)
# Function to estimate inverse demand function as in Compiani (2018)
    # bO should be a (J x 1) matrix where each entry is
    # the order of the polynomials for the demand function for one product.

# J = size(included,2);
if index_vars[1]!="prices"
    error("Prices has to be the first variable in the index")
end

exchange = [[1 2 3 4], [5 6]]
# exchange = [[1 2 3],[4 5 6], [7 8 9 10]]
index_vars = ["prices", "x"]
normalization = [];
constraints = [:monotone, :exchangeability, :diagonal_dominance]

Xvec, Avec, Bvec, syms = prep_matrices(df, exchange, index_vars, bO);
Aineq, Aeq = make_constraint(constraints, exchange, combo_vec);

mins = dropdims(getindex.(argmin(Aeq, dims=2),2), dims=2);
order = sortperm(mins, rev=true);
mins = mins[order];
maxs = getindex.(argmax(Aeq, dims=2),2);
maxs = maxs[order]

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

# Estimation 
β_length = design_width + sum(size(B[1],2))
β_init = -1 .* rand(β_length)
using Optim, LineSearches
results =  Optim.optimize(obj_func, grad_func!, β_init,
    LBFGS(;linesearch = BackTracking(order=3)), Optim.Options(show_trace = true, iterations = 10000));
results =  Optim.optimize(obj_func, grad_func!, results.minimizer,
    LBFGS(;linesearch = BackTracking(order=3)), Optim.Options(show_trace = true, iterations = 10000));
β = results.minimizer
θ = β[1:design_width]
γ = β[length(θ)+1:end]
γ[price_index] = 1;
for i ∈ normalization
    γ[i] =0; 
end
for i∈eachindex(mins)
    θ[mins[i]] = θ[maxs[i]]
end
estimates = inner_estimation(obj_func, grad_func!, β_init)

price_elasticity2(vcat(θ, γ), df, bO; X =Xvec, B = Bvec)

return inv_sigma
end
