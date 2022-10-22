"""
    define_problem(df::DataFrame; exchange = [], index_vars = ["prices"], FE = [], constraints = [], bO = 2, tol = 1e-5)

Constructs a `problem`::NPDProblem using the provided problem characteristics. Inputs: 

- `exchange`::Vector{Matrix{Int64}}: A vector of groups of products which are exchangeable. E.g., with 4 goods, if the first
and second are exchangeable and so are the third and fourth, set `exchange` = [[1 2], [3 4]].
- `index_vars`: String array listing column names in `df` which represent variables that enter the inverted index.
- `FE`: String array listing column names in `df` which should be included as fixed effects.
- `constraint_tol`: Tolerance specifying tightness of constraints
- `obj_tol`: Tolerance specifying g_abstol in Optim.Options()
    - Note: All fixed effects are estimated as parameters by the minimizer, so be careful adding fixed effects for variables that take 
    many values.
- `constraints`: A list of symbols of accepted constraints. Currently supported constraints are: 
    - :monotone  
    - :all_substitutes 
    - :diagonal\\_dominance\\_group 
    - :diagonal\\_dominance\\_all 
    - :subs\\_in\\_group (Note: this constraint is the only available nonlinear constraint and will slow down estimation considerably)
"""
function define_problem(df::DataFrame; exchange::Vector = [], index_vars = ["prices"], FE = [], 
    constraints = [], bO = 2, obj_tol = 1e-5, constraint_tol = 1e-5, normalization=[])
    
    find_prices = findall(index_vars .== "prices")[1];

    if (find_prices !=1 ) | !(typeof(index_vars)<:Vector) #index_vars[1] !="prices"
        error("Variable index_vars must be a Vector, and `prices` must be the first element")
    end

    # Reshape FEs into matrix of dummy variables
    FEmat = [];
    if FE!=[]
        FEvec = [];
        FEmat = [];
        println("Reshaping fixed-effects into dummy variables....")
        for f ∈ FE
            if f != "product"
                unique_vals = unique(df[!,f]);
                unique_vals = unique_vals[1:end-1]; # Drop one category per FE dimension
                for fi ∈ unique_vals
                    if (f==FE[1]) & (fi==unique_vals[1])
                        FEmat = reshape((df[!,f] .== fi), size(df,1),1)
                    else
                        FEmat = hcat(FEmat, reshape((df[!,f] .== fi), size(df,1),1))
                    end
                end
            end
        end
    end

    product_FEs = false;
    if "product" ∈ FE
        product_FEs = true;
    end

    println("Making Bernstein polynomials....")
    Xvec, Avec, Bvec, syms = prep_matrices(df, exchange, index_vars, FEmat, product_FEs, bO);
    
    # @show size(syms)
    if constraints !=[]
        println("Making linear constraint matrices....")
        Aineq, Aeq, maxs, mins = make_constraint(df, constraints, exchange, syms);
    end
    
    println("Reformulating problem....")
    matrices = prep_inner_matrices(Xvec, Avec, Bvec);

    design_width = sum(size.(Xvec,2));
    elast_mats = [];
    elast_prices = [];

    problem = NPDProblem(df,
                        matrices, 
                        Xvec, 
                        Bvec, 
                        Avec,
                        index_vars,
                        constraints,
                        syms,
                        Aineq, 
                        Aeq,
                        mins,
                        maxs, 
                        FE,
                        normalization,
                        exchange,
                        design_width,
                        obj_tol,
                        constraint_tol,
                        bO,
                        [],
                        elast_mats,
                        elast_prices)

    if :subs_in_group ∈ constraints
        println("Preparing inputs for nonlinear constraints....")
        subset = subset_for_elast_const(problem, df; grid_size=2);
        elast_mats, elast_prices = make_elasticity_mat(problem, subset);
        problem.elast_mats = elast_mats;
        problem.elast_prices = elast_prices;
    end

    return problem
end

mutable struct NPDProblem
    data 
    matrices 
    Xvec 
    Bvec 
    Avec
    index_vars
    constraints 
    syms
    Aineq 
    Aeq 
    mins 
    maxs
    FE 
    normalization
    exchange 
    design_width 
    obj_tol
    constraint_tol 
    bO
    results
    elast_mats
    elast_prices
end

import Base.+
"""
    update_constraints!(problem::NPDProblem, new_constraints::Vector{Symbol})

Re-calculates constraint matrices under `new_constraints`, ignoring previous constraints used to define the problem.
"""
function update_constraints!(problem::NPDProblem, new_constraints::Vector{Symbol})
    case1  = ((:exchangeability ∈ new_constraints) & (:exchangeability ∉ problem.constraints));
    case2 = ((:exchangeability ∉ new_constraints) & (:exchangeability ∈ problem.constraints));
    if case1 | case2
        error("Constraint :exchangeability should be included in updated problem (only) if in original problem.")
    else
        println("Updating linear constraint matrices....")
        Aineq, Aeq, maxs, mins = make_constraint(problem.data, new_constraints, 
                                                        problem.exchange, problem.syms);

        problem.Aineq = Aineq;
        # problem.Aeq = Aeq; # don't update equality matrix because exchangeability unchanged
        problem.maxs = maxs;
        problem.mins = mins;

        if :subs_in_group ∈ new_constraints
            println("Preparing inputs for nonlinear constraints....")
            subset = subset_for_elast_const(problem, problem.data; grid_size=2);
            elast_mats, elast_prices = make_elasticity_mat(problem, subset);
            problem.elast_mats = elast_mats;
            problem.elast_prices = elast_prices;
        end
        problem.constraints = new_constraints;
    end
end

function Base.show(io::IO, problem::NPDProblem)
    J = length(problem.Xvec);
    constraints = problem.constraints;
    bO = problem.bO;
    index_vars = problem.index_vars;
    exchange = problem.exchange;
    FE = problem.FE;
    
    println(io, "NPD Problem:")
    println(io, "- Number of choices: $(J)")
    println(io, "- Exchangeable groups of choices: $exchange")
    println(io, "- Constraints: $constraints")
    println(io, "- Fixed Effects: $FE")
    println(io, "- Index Variables: $index_vars")
    println(io, "- Bernstein polynomials of order: $bO")

end