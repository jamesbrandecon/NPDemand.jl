"""
    define_problem(df::DataFrame; exchange = [], index_vars = ["prices"], FE = [], constraints = [], bO = 2, tol = 1e-5)

Constructs a `problem`::NPDProblem using the provided problem characteristics. Inputs: 

- `exchange`: A vector of groups of products which are exchangeable. E.g., with 4 goods, if the first
and second are exchangeable and so are the third and fourth, set `exchange` = [[1 2], [3 4]].
- `index_vars`: String array listing column names in `df` which represent variables that enter the inverted index.
    - "prices" must be the first element of `index_vars`
- `FE`: String array listing column names in `df` which should be included as fixed effects.
    - Note: All fixed effects are estimated as parameters by the minimizer, so be careful adding fixed effects for variables that take many values.
- `bO`: Order of the univariate Bernstein polynomials in market shares. Default is 2.
- `constraint_tol`: Tolerance specifying tightness of constraints
- `obj_xtol`: Tolerance specifying x_tol in Optim.Options()
- `obj_ftol`: Tolerance specifying f_tol in Optim.Options()
- `chunk_size`: Controls chunk size in ForwardDiff.Gradient autodiff calculation for nonlinear constraints. Only used if :subs\\_in\\_group specified. 
- `constraints`: A list of symbols of accepted constraints. Currently supported constraints are: 
    - :monotone  
    - :all_substitutes 
    - :diagonal\\_dominance\\_group 
    - :diagonal\\_dominance\\_all 
    - :subs\\_in\\_group (Note: this constraint is the only available nonlinear constraint and will slow down estimation considerably)
"""
function define_problem(df::DataFrame; exchange::Vector = [], index_vars = ["prices"], FE = [], 
    constraints = [], bO = 2, obj_xtol = 1e-5, obj_ftol = 1e-5, 
    constraint_tol = 1e-5, normalization=[], chunk_size = [])
    
    find_prices = findall(index_vars .== "prices")[1];

    # Checking structure of index
    if (find_prices !=1 ) | !(typeof(index_vars)<:Vector) #index_vars[1] !="prices"
        error("Variable index_vars must be a Vector, and `prices` must be the first element")
    end

    # Checking constraints
    if (:diagonal_dominance_all ∈ constraints) | (:diagonal_dominance_group ∈ constraints)
        if :monotone ∉ constraints 
            error("Diagonal dominance only implemented in conjunction with monotonicity-- add :monotone to constraints")
        end
        if (exchange != []) & (:exchangeability ∉ constraints)
            error("Vector exchange is nonempty but :exchangeability is listed in constraints")
        end
    end

    E1 = (:exchangeability ∈ constraints);
    E2 = (exchange!=[])
    if (E1!=E2)
        error("Keyword `exchange` should be specified (only) if :exchangeability is in `constraints` vector")
    end

    # Confirm that shares are numbered as expected: 
    J = size(df[!,r"shares"],2);
    missingshares = 0;
    missingprices = 0;
    for i = 0:(J-1)
        if string("shares", string(i)) ∉ names(df)
            missingshares +=1;
        end
        if missingshares > 0
            error("$J Products detected: shares should be numbered 0 to $J-1")
        end
        if string("prices", string(i)) ∉ names(df)
            missingprices +=1;
        end
        if missingprices > 0
            error("$J Products detected: prices should be numbered 0 to $J-1")
        end
    end

    if exchange ==[] 
        for j = 1:J 
            push!(exchange, [j])
        end
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
                        obj_xtol,
                        obj_ftol,
                        constraint_tol,
                        bO,
                        [],
                        elast_mats,
                        elast_prices,
                        [])

    if :subs_in_group ∈ constraints
        println("Preparing inputs for nonlinear constraints....")
        subset = subset_for_elast_const(problem, df; grid_size=2);
        elast_mats, elast_prices = make_elasticity_mat(problem, subset);
        problem.elast_mats = elast_mats;
        problem.elast_prices = elast_prices;
        θ_packed =  pack_parameters(zeros(Float64, sum(size.(Xvec,2))), exchange, size.(Xvec,2))
        if chunk_size ==[]
            cfg = GradientConfig(nothing, θ_packed);
        else
            cfg = GradientConfig(nothing, θ_packed, Chunk{chunk_size}());
        end
        problem.cfg = cfg;
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
    obj_xtol
    obj_ftol
    constraint_tol 
    bO
    results
    elast_mats
    elast_prices
    cfg
end

import Base.+
"""
    update_constraints!(problem::NPDProblem, new_constraints::Vector{Symbol})

Re-calculates constraint matrices under `new_constraints`, ignoring previous constraints used to define the problem.
The `exchangeability` constraint is an exception. To change anything about the structure of exchangeability for the problem changes, define a new problem.
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
    obj_xtol = problem.obj_xtol;
    obj_ftol = problem.obj_ftol;
    
    println(io, "NPD Problem:")
    println(io, "- Number of choices: $(J)")
    println(io, "- Exchangeable groups of choices: $exchange")
    println(io, "- Constraints: $constraints")
    println(io, "- Fixed Effects: $FE")
    println(io, "- Index Variables: $index_vars")
    println(io, "- Bernstein polynomials of order: $bO")
    println(io, "- (x_tol, f_tol): ($obj_xtol, $obj_ftol)")

end