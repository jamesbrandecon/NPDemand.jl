"""
    list_constraints()

Returns a dictionary of constraints supported by the `define_problem` function in the NPDemand package. Each constraint is represented by a symbol and its corresponding description.
"""
function list_constraints()
   result = Dict(
    :monotone => "Impose downward sloping demand curves",
    :diagonal_dominance_all => "Impose diagonal dominance Jacobian of demand -- limits the strength of cross-price effects relative to own-price effects",
    :diagonal_dominance_group => "Same as diagonal_dominance_all, but only within exchangeable groups",
    :all_substitutes => "Impose that all products are substitutes. This constraint is imposed linearly",
    :exchangeability => "Impose that all products within provided groups (in `exchange`) are exchangeable",
    :subs_in_group => "Impose that all products within provided groups (in `exchange`) are substitutes. This constraint is imposed only via quasi-bayes.",
    :complements_in_group => "Impose that all products within provided groups (in `exchange`) are complements. This constraint is only imposed via quasi-bayes.",
    :subs_across_group => "Impose that all products *across* provided groups (in `exchange`) are substitutes. No constraints are imposed within group. This constraint is only imposed via quasi-bayes.", 
    :complements_across_group => "Impose that all products *across* provided groups (in `exchange`) are complements. No constraints are imposed within group. This constraint is only imposed via quasi-bayes."
   ) 
   return result
end

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
- `chunk_size`: Controls chunk size in ForwardDiff.Gradient autodiff calculation for nonlinear constraints. Only used if :subs\\_in\\_group specified. 
- `constraints`: A list of symbols of accepted constraints. Currently supported constraints are: 
    - :monotone  
    - :all_substitutes 
    - :diagonal\\_dominance\\_group 
    - :diagonal\\_dominance\\_all 
    - :subs\\_in\\_group (Note: this constraint is the only available nonlinear constraint and will slow down estimation considerably)
- `verbose`: if `false`, will not print updates as problem is generated
"""
function define_problem(df::DataFrame; exchange::Vector = [], 
    index_vars = ["prices"], price_iv = [], FE = [], 
    constraints = [], bO = 2, 
    obj_xtol = 1e-5, obj_ftol = 1e-5, 
    constraint_tol = 1e-5, # these inputs are no longer used 
    normalization=[], 
    verbose = false, 
    approximation_details = Dict())
    
    if approximation_details != Dict()
        bO    = approximation_details[:order]
        order = approximation_details[:order]
         
        max_interaction = approximation_details[:max_interaction]
        sieve_type = approximation_details[:sieve_type]
    else 
        error("No approximation details provided. Please provide a Dict with keys :order, :max_interaction, and :sieve_type.")
    end
    
    find_prices = findall(index_vars .== "prices")[1];

    # Set default value of price_iv 
    if price_iv == []
        price_iv = ["price_iv"];
    end

    # Check to make all constraints are valid
    for con ∈ constraints
        if con ∉ [:monotone, :all_substitutes, 
            :diagonal_dominance_group, :diagonal_dominance_all, 
            :exchangeability, :subs_in_group,
            :complements_across_group, :subs_across_group, 
            :complements_in_group]
            error("Constraint $con not recognized. Valid constraints include: 
            :monotone, :all_substitutes, :diagonal_dominance_group, :diagonal_dominance_all, 
            :exchangeability, :subs_in_group, :complements_across_group, :subs_across_group,
            :complements_in_group")
        end
    end

    problem_has_nonlinear_constraints = false;
    if (:subs_in_group ∈ constraints) | (:complements_in_group ∈ constraints) | (:all_substitutes_nonlinear ∈ constraints) | (:subs_across_group ∈ constraints) | (:complements_across_group ∈ constraints) |
        (:monotone_nonlinear ∈ constraints)
        problem_has_nonlinear_constraints = true;
    end

    # Make sure that we're not trying to constrain subs in group and complements in group, or the same across groups 
    if (:subs_in_group ∈ constraints) & (:complements_in_group ∈ constraints)
        error("Cannot constrain both :subs_in_group and :complements_in_group")
    end
    if (:subs_across_group ∈ constraints) & (:complements_across_group ∈ constraints)
        error("Cannot constrain both :subs_in_group and :complements_across_group")
    end

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

    # Re-sort columns so that they are in numeric order 
    for column_name in union(["shares", "share_iv"], index_vars, price_iv)
        stub_cols = filter(col -> occursin(Regex("($column_name)\\d+"), col), names(df))
        sorted_stub_cols = sort(stub_cols, by = col -> parse(Int, match(Regex("$column_name(\\d+)"), col).captures[1]))
        df = df[:, vcat(setdiff(names(df), stub_cols), sorted_stub_cols)]
    end

    try 
        @assert (length(exchange) <=2) | (sieve_type == "polynomial")
    catch 
        error("NPDemand currently only supports models with two or fewer exchangeable groups in `exchange`")
    end

    # Check that exchange is only specified if exchangeability is in constraints
    E1 = (:exchangeability ∈ constraints);
    E2 = (exchange!=[])
    if (E1!=E2)
        error("Keyword `exchange` should be specified (only) if :exchangeability is in `constraints` vector")
    end

    # If user provides a 0-indexed exchange vector, convert to 1-indexed
    if (exchange !=[]) && (0 ∈ union(exchange...))
        for i in eachindex(exchange)
            exchange[i] = exchange[i] .+ 1;
        end
    end

    # Check that price_iv is a vector of strings or symbols
    if (price_iv !=[]) & !(typeof(price_iv) <: Vector)
        error("Keyword `price_iv` should be a Vector of Symbols or Strings indicating column names in `df`")
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
        if "product" ∈ FE
            error("Product FEs are redundant/not identified if no products are exchangeable")
        end

        for j = 1:J 
            push!(exchange, [j])
        end
    end

    # Reshape FEs into matrix of dummy variables
    FEmat = [];
    fe_param_mapping = Dict{Int, NamedTuple}(); # Map the index in the FE mat which corresponds to a given FE name and value
    if FE!=[]
        FEmat = [];
        column_counter = 1;
        verbose && println("Reshaping fixed-effects into dummy variables....")
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
                    fe_param_mapping[column_counter] = (name = f, value = fi)
                    column_counter += 1
                end
            end
        end
    end

    product_FEs = false;
    if "product" ∈ FE
        product_FEs = true        
        for j = 1:J
            which_group = findall(j .∈ exchange)[1];
            first_product_in_group = exchange[which_group][1];

            if j !=first_product_in_group # dropping last product's FE for location normalization
                fe_param_mapping[column_counter] = (name = "product", value = j)    
                column_counter += 1
            end
        end
    end
    
    verbose && println("Making polynomial approximations....")
    Xvec, Avec, Bvec, syms, combos = prep_matrices(
        df, exchange, index_vars, FEmat, product_FEs, bO; 
        price_iv = price_iv, verbose = verbose, 
        approximation_details = approximation_details);
    
    if constraints !=[] && sieve_type == "bernstein"
        verbose && println("Making linear constraint matrices....")
        Aineq, Aeq, maxs, mins = make_constraint(df, constraints, exchange, syms);
    # elseif constraints !=[] && sieve_type == "polynomial"
    #     # verbose && println("Making linear constraint matrices....")
    #     Aeq = [];
    #     # for exchange_ind = 1:length(exchange)
    #     #     for prod_ind in exchange[exchange_ind]
                
    #     #     end
    #     # end
    #     # vcat!(
    #     #     Aeq, 
    else 
        Aineq = [];
        Aeq = [];
        mins = []; 
        maxs = [];
    end
    
    verbose && println("Reformulating problem....")

    design_width = sum(size.(Xvec,2));
    elast_mats = Matrix[];
    elast_prices = Matrix[];

    weight_matrices = [Avec[i]*pinv(Avec[i]'*Avec[i])*Avec[i]' for i in 1:length(Xvec)]; 

    problem = NPDProblem(df,
                        [], 
                        Xvec, 
                        Bvec, 
                        Avec,
                        weight_matrices,
                        index_vars,
                        constraints,
                        syms,
                        combos,
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
                        [],
                        [],
                        [], 
                        [], 
                        [], 
                        [], 
                        [], 
                        [], 
                        approximation_details)

    verbose && println("Constructing helper matrices for elasticities...")
    problem.tempmats = calc_tempmats(
        problem; 
        approximation_details = approximation_details);
    verbose && println("Done constructing problem.")
    return problem
end

"""
    NPD_parameters

    Custom struct to store estimated parameters specifically. This can be used to replace the candidate parameters in an NPDProblem struct. The two key fields 
    are `minimizer` and `filtered_chain`. The `minimizer` field stores the estimated parameters, while the `filtered_chain` field stores the Markov chain for quasi-Bayes
    methods, after filtering out burn-in and thinning but before reformatting into the full parameter sieve.
"""
mutable struct NPD_parameters 
    minimizer
    filtered_chain
end

mutable struct NPDProblem
    data 
    matrices 
    Xvec 
    Bvec 
    Avec
    weight_matrices
    index_vars
    constraints 
    syms 
    combos
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
    all_elasticities
    all_jacobians
    converged
    chain
    tempmats
    smc_results
    sampling_details
    approximation_details
end

import Base.+
"""
    update_constraints!(problem::NPDProblem, new_constraints::Vector{Symbol})

Re-calculates constraint matrices under `new_constraints`, ignoring previous constraints used to define the problem.
The `exchangeability` constraint is an exception. To change anything about the structure of exchangeability for the problem changes, define a new problem.
"""
function update_constraints!(problem::NPDProblem, new_constraints::Vector{Symbol}; verbose = true)
    case1  = ((:exchangeability ∈ new_constraints) & (:exchangeability ∉ problem.constraints));
    case2 = ((:exchangeability ∉ new_constraints) & (:exchangeability ∈ problem.constraints));
    if case1 | case2
        error("Constraint :exchangeability should be included in updated problem (only) if in original problem.")
    else
        verbose && println("Updating linear constraint matrices....")
        Aineq, ~, maxs, mins = make_constraint(problem.data, new_constraints, 
                                                        problem.exchange, problem.syms);

        problem.Aineq = Aineq;
        # problem.Aeq = Aeq; # don't update equality matrix because exchangeability unchanged
        problem.maxs = maxs;
        problem.mins = mins;
        problem.constraints = new_constraints;
    end
end

function Base.show(io::IO, problem::NPDProblem)
    J = length(problem.Xvec);
    T = size(problem.Xvec[1],1);

    constraints = problem.constraints;
    bO = problem.bO;
    index_vars = problem.index_vars;
    exchange = problem.exchange;
    FE = problem.FE;
    obj_xtol = problem.obj_xtol;
    obj_ftol = problem.obj_ftol;
    estimated_TF = ((problem.results !=[]) | (problem.chain != []));
    
    println(io, "NPD Problem:")
    println(io, "- Number of choices: $(J)")
    println(io, "- Number of markets: $(T)")
    println(io, "- Exchangeable groups of choices: $exchange")
    println(io, "- Constraints: $constraints")
    println(io, "- Fixed Effects: $FE")
    println(io, "- Index Variables: $index_vars")
    println(io, "- Bernstein polynomials of order: $bO")
    println(io, "- (x_tol, f_tol): ($obj_xtol, $obj_ftol)")
    println(io, "- Estimated: $estimated_TF")
end

function make_param_mapping(problem::NPDProblem)
    FEmat = [];
    fe_param_mapping = Dict{Int, NamedTuple}(); # Map the index in the FE mat which corresponds to a given FE name and value
    FE = problem.FE;
    df = problem.data;
    exchange = problem.exchange;
    J = length(problem.Xvec);

    if FE!=[]
        FEmat = [];
        column_counter = 1;
        for f ∈ FE
            if f != "product"
                unique_vals = sort(unique(df[!,f]));
                unique_vals = unique_vals[1:end-1]; # Drop one category per FE dimension
                for fi ∈ unique_vals
                    if (f==FE[1]) & (fi==unique_vals[1])
                        FEmat = reshape((df[!,f] .== fi), size(df,1),1)
                    else
                        FEmat = hcat(FEmat, reshape((df[!,f] .== fi), size(df,1),1))
                    end
                    fe_param_mapping[column_counter] = (name = f, value = fi)
                    column_counter += 1
                end
            end
        end
    end

    product_FEs = false;
    if "product" ∈ FE
        product_FEs = true        
        for j = 1:J  
            which_group = findall(j .∈ exchange)[1];  
            first_product_in_group = exchange[which_group][1]; 

            if j !=first_product_in_group # dropping last product's FE for location normalization
                fe_param_mapping[column_counter] = (name = "product", value = j)    
                column_counter += 1
            end
        end
    end

    # Now re-sort the Dict to make it be in the desired order 
    # sorted_keys = sort(collect(keys(fe_param_mapping)));
    # fe_param_mapping_sorted = Dict{Int, NamedTuple}();
    # for k in sorted_keys
    #     fe_param_mapping_sorted[k] = fe_param_mapping[k];
    # end

    return sort(fe_param_mapping)
end