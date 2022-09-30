function define_problem(df::DataFrame; exchange = exchange, index_vars = ["prices"], FE = [], constraints = [], bO = 2, tol = 1e-5)
    if index_vars[1]!="prices"
        error("Variable index_vars must be a Vector, and `prices` must be the first element")
    end

    # Reshape FEs into matrix of dummy variables
    if FE!=[]
        println("Reshaping fixed-effects into dummy variables....")
        for f = 1:length(FE)
            unique_vals = unique(df[!,f]);
            for fi ∈ unique_vals
                if (f==1) & (fi==unique_vals[1])
                    FEmat = (df[!,f] .== fi)
                else
                    FEmat = hcat(FEmat, (df[!,f] .== fi))
                end
            end
        end
    end
    println("Making Bernstein polynomials....")
    Xvec, Avec, Bvec, syms = prep_matrices(df, exchange, index_vars, bO);
    
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
                        normalization,
                        exchange,
                        design_width,
                        tol,
                        bO,
                        [],
                        elast_mats,
                        elast_prices)

    if :subs_in_group ∈ constraints
        println("Preparing inputs for nonlinear constraints....")
        subset = subset_for_elast_const(problem, df; grid_size=2);
        elast_mats, elast_prices = make_elasticity_mat(problem, df::DataFrame);
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
    normalization
    exchange 
    design_width 
    tol 
    bO
    results
    elast_mats
    elast_prices
end

import Base.+
function +(problem::NPDProblem, new_constraint::Symbol)
    println("Redefining problem with new constraint...")
    if new_constraint != :exchangeability
        new_problem = define_problem(problem.data; 
                    exchange = problem.exchange, 
                    index_vars = problem.index_vars, 
                    constraints = hcat(problem.constraints, new_constraint), 
                    bO = problem.bO, tol = problem.tol);
    else
        error("If adding :exchangeability, must provide vector of groupings")
    end
    return new_problem    
end

function +(problem::NPDProblem, new_constraint::Symbol, exchange = [])
    println("Redefining problem with new constraint...")
    if new_constraint ==:exchangeability
        new_problem = define_problem(problem.data; 
                    exchange = exchange, 
                    index_vars = problem.index_vars, 
                    constraints = hcat(problem.constraints, new_constraint), 
                    bO = problem.bO, tol = problem.tol);
    else
        error("If new constraint is not :exchangeability, do not provide vector of exchangeable groups")
    end
    return new_problem    
end

function Base.show(io::IO, problem::NPDProblem)
    J = length(problem.Xvec);
    constraints = problem.constraints;
    bO = problem.bO;
    index_vars = problem.index_vars;
    exchange = problem.exchange;

    println(io, "NPD Problem:")
    println(io, "- Number of choices: $(J)")
    println(io, "- Exchangeable groups of choices: $exchange")
    println(io, "- Constraints: $constraints")
    println(io, "- Index Variables: $index_vars")
    println(io, "- Bernstein polynomials of order: $bO")
end