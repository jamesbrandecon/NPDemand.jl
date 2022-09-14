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
constraints = [:monotone, :exchangeability, :diagonal_dominance]

Xvec, Avec, Bvec, syms = prep_matrices(df, exchange, index_vars, bO);
Aineq, Aeq = make_constraint(constraints, exchange, combo_vec);

matrices = prep_inner_matrices(Xvec, Avec, Bvec)

design_width = sum(size.(Xvec,2));

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
Aineq = Aineq, Aeq = Aeq, design_width = design_width, price_index = 1);

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
Aineq = Aineq, Aeq = Aeq, design_width = design_width, price_index = 1);

# Estimation 
β_length = design_width + sum(size.(B,2))
β_init = -1 .* ones(β_length)
using Optim
results =  Optim.optimize(obj_func, grad_func!, β_init,
    LBFGS(), Optim.Options(show_trace = true, iterations = 10000));

estimates = inner_estimation(obj_func, grad_func!, β_init)


return inv_sigma
end
