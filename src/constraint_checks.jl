function report_constraint_violations(problem;
    verbose = true,
    params = [], 
    output = "dict")

    J = length(problem.Xvec)

    if params == []
        param_vec = problem.results.minimizer;
    else 
        param_vec = params;
    end
    jacobians = elast_mat_zygote(param_vec, problem, problem.tempmats; at = Matrix(problem.data[!,r"prices"]), s = Matrix(problem.data[!,r"shares"]));
    reshaped_jacobians = [jacobians[i][j1,j2] for j1=1:J, j2 = 1:J, i = 1:size(problem.data,1)];

    violations = Dict();
    all_satisfied = ones(Bool, size(problem.data,1));
    num_violated_per_market = zeros(Int64, size(problem.data,1));

    # Monotonicity
    if :monotone in problem.constraints
        monotone_satisfied = [all(diag(reshaped_jacobians[:,:,i]) .<=0) for i in axes(reshaped_jacobians,3)];
        all_satisfied = all_satisfied .& monotone_satisfied;
        num_violated_per_market .+= (1 .- monotone_satisfied);
        frac_monotone_violations = round(1 - mean(monotone_satisfied), digits = 2);
        verbose && println("Fraction of violations of :monotonicity: $frac_monotone_violations")
        push!(violations, :monotone => frac_monotone_violations)
    end
    
    # All substitutes
    if :all_substitutes in problem.constraints
        all_subs_satisfied = [check_all_subs(reshaped_jacobians[:,:,i]) for i in axes(reshaped_jacobians,3)];
        all_satisfied = all_satisfied .& all_subs_satisfied;
        num_violated_per_market .+= (1 .- all_subs_satisfied);
        frac_all_subs_violations = round(1 - mean(all_subs_satisfied), digits = 2);
        verbose && println("Fraction of violations of :all_substitutes: $frac_all_subs_violations")
        push!(violations, :all_substitutes => frac_all_subs_violations)
    end

    # Diagonal dominance
    if :diagonal_dominance_all in problem.constraints
        diag_dom_satisfied = [check_diagonal_dominance(reshaped_jacobians[:,:,i]) for i in axes(reshaped_jacobians,3)];
        all_satisfied = all_satisfied .& diag_dom_satisfied;
        num_violated_per_market .+= (1 .- diag_dom_satisfied);
        frac_diag_dom_violations = round(1 - mean(diag_dom_satisfied), digits = 2);
        verbose && println("Fraction of violations of :diagonal_dominance_all: $frac_diag_dom_violations")
        push!(violations, :diagonal_dominance_all => frac_diag_dom_violations)
    end

    # Substitutes within group
    if :subs_in_group in problem.constraints
        subs_in_group_satisfied = [check_subs_in_group(reshaped_jacobians[:,:,i], problem.exchange) for i in axes(reshaped_jacobians,3)];
        all_satisfied = all_satisfied .& subs_in_group_satisfied;
        num_violated_per_market .+= (1 .- subs_in_group_satisfied)
        frac_subs_in_group_violations = round(1 - mean(subs_in_group_satisfied), digits = 2);
        verbose && println("Fraction of violations of :subs_in_group: $frac_subs_in_group_violations")
        push!(violations, :subs_in_group => frac_subs_in_group_violations)
    end

    # Substitutes across group
    if :subs_across_group in problem.constraints
        subs_across_group_satisfied = [check_subs_across_group(reshaped_jacobians[:,:,i], problem.exchange) for i in axes(reshaped_jacobians,3)];
        all_satisfied = all_satisfied .& subs_across_group_satisfied;
        num_violated_per_market .+= (1 .- subs_across_group_satisfied)
        frac_subs_across_group_violations = round(1 - mean(subs_across_group_satisfied), digits = 2);
        verbose && println("Fraction of violations of :subs_across_group: $frac_subs_across_group_violations")
        push!(violations, :subs_across_group => frac_subs_across_group_violations)
    end

    # All complements
    if :all_complements in problem.constraints
        all_comps_satisfied = [check_all_complements(reshaped_jacobians[:,:,i]) for i in axes(reshaped_jacobians,3)];
        all_satisfied = all_satisfied .& all_comps_satisfied;
        num_violated_per_market .+= (1 .- all_comps_satisfied)
        frac_all_comps_violations = round(1 - mean(all_comps_satisfied), digits = 2);
        verbose && println("Fraction of violations of :all_complements: $frac_all_comps_violations")
        push!(violations, :all_complements => frac_all_comps_violations)
    end

    # Complements within group
    if :complements_in_group in problem.constraints
        complements_in_group_satisfied = [check_comps_in_group(reshaped_jacobians[:,:,i], problem.exchange) for i in axes(reshaped_jacobians,3)];
        all_satisfied = all_satisfied .& complements_in_group_satisfied;
        num_violated_per_market .+= (1 .- complements_in_group_satisfied)
        frac_complements_in_group_violations = round(1 - mean(complements_in_group_satisfied), digits = 2);
        verbose && println("Fraction of violations of :complements_in_group: $frac_complements_in_group_violations")
        push!(violations, :complements_in_group => frac_complements_in_group_violations)
    end

    # Complements across group
    if :complements_across_group in problem.constraints
        complements_across_group_satisfied = [check_comps_across_group(reshaped_jacobians[:,:,i], problem.exchange) for i in axes(reshaped_jacobians,3)];
        all_satisfied = all_satisfied .& complements_across_group_satisfied;
        num_violated_per_market .+= (1 .- complements_across_group_satisfied)
        frac_complements_across_group_violations = round(1 - mean(complements_across_group_satisfied), digits = 2);
        verbose && println("Fraction of violations of :complements_across_group: $frac_complements_across_group_violations")
        push!(violations, :complements_across_group => frac_complements_across_group_violations)
    end

    any_violations = 1 - mean(all_satisfied);
    push!(violations, :any => round(any_violations, digits = 2))

    if output == "dict"
        return violations
    elseif output == "count"
        return Int.(num_violated_per_market)
    else
        return Float64::violations[:any]
    end
end

function run_elasticity_check(elasts, constraints, exchange)
    elasticity_check = true;

    @views begin
        if :monotone in constraints
            elasticity_check = all([all(elasts[i,i,:] .<=0) for i in axes(elasts,1)]);
        end
        if :all_substitutes in constraints
            elasticity_check = elasticity_check & (all([check_all_subs(elasts[:,:,i]) for i in axes(elasts,3)]));
        end
        if :diagonal_dominance_all in constraints
            elasticity_check = elasticity_check & (all([check_diagonal_dominance(elasts[:,:,i]) for i in axes(elasts,3)]));
        end
        if :subs_in_group in constraints 
            elasticity_check = elasticity_check & all([check_subs_in_group(elasts[:,:,i], exchange) for i in axes(elasts,3)]);
        end
        if :complements_across_group in constraints 
            elasticity_check = elasticity_check & all([check_comps_across_group(elasts[:,:,i], exchange) for i in axes(elasts,3)]);
        end
    end

    return elasticity_check
end

function check_diagonal_dominance(elast_one_market)
    J = size(elast_one_market,1);
    
    # Check columnwise sums
    col_sums_minus_diag = [sum(elast_one_market[setdiff(1:J,i),i]) for i in 1:J]; #sum([elast_one_market[i,j] for i in 1:J for j in setdiff(1:J,i), for i in 1:J]);
    if all(abs.(col_sums_minus_diag) .<= abs.(diag(elast_one_market)))
        return true
    else 
        return false 
    end
end

function check_all_subs(elast_one_market)
    J = size(elast_one_market,1);
    if all([elast_one_market[i,j] for i in 1:J for j in setdiff(1:J,i)] .>=0)
        return true
    else 
        return false 
    end
end

function check_subs_in_group(elast_one_market, exchange)
    if length(exchange) ==1
        return check_all_subs(elast_one_market)
    else # two elements in exchange
        elast_one_market_group1 = elast_one_market[exchange[1][1]:exchange[1][2], exchange[1][1]:exchange[1][2]];
        elast_one_market_group2 = elast_one_market[exchange[2][1]:exchange[2][2], exchange[2][1]:exchange[2][2]];
        return check_all_subs(elast_one_market_group1) .& 
            check_all_subs(elast_one_market_group2)
    end
end

function check_subs_across_group(elast_one_market, exchange)
    try 
        @assert length(exchange)==2
    catch
        error("Cannot use `across_group` constraints with only one exchangeable group")
    end 
    J = maximum(vcat(exchange[1], exchange[2]))
    elast_one_market_group1_2 = dropdims(elast_one_market[exchange[1], setdiff(1:J, exchange[1])], dims=1);
    elast_one_market_group2_1 = dropdims(elast_one_market[exchange[2], setdiff(1:J, exchange[2])], dims=1);

    return check_all_subs(elast_one_market_group1_2) .& 
        check_all_subs(elast_one_market_group2_1)
end

function check_all_complements(elast_one_market)
    J = size(elast_one_market,1);
    if all([elast_one_market[i,j] for i in 1:J for j in setdiff(1:J,i)] .<=0)
        return true
    else 
        return false 
    end
end

function check_comps_in_group(elast_one_market, exchange)
    if length(exchange) ==1
        return check_all_complements(elast_one_market)
    else # two elements in exchange
        elast_one_market_group1 = elast_one_market[exchange[1][1]:exchange[1][2], exchange[1][1]:exchange[1][2]];
        elast_one_market_group2 = elast_one_market[exchange[2][1]:exchange[2][2], exchange[2][1]:exchange[2][2]];
        return check_all_complements(elast_one_market_group1) .& 
               check_all_complements(elast_one_market_group2)
    end
end

function check_comps_across_group(elast_one_market, exchange)
    try 
        @assert length(exchange)==2
    catch
        error("Cannot use `across_group` constraints with only one exchangeable group")
    end 
    J = maximum(vcat(exchange[1], exchange[2]))
    elast_one_market_group1_2 = dropdims(elast_one_market[exchange[1], setdiff(1:J, exchange[1])], dims=1);
    elast_one_market_group2_1 = dropdims(elast_one_market[exchange[2], setdiff(1:J, exchange[2])], dims=1);

    return all(elast_one_market_group1_2 .<=0) .& 
           all(elast_one_market_group2_1 .<=0)
end


function check_linear_constraints(npd_problem)
    if npd_problem.Aineq != []
        # Modify theta as in objective function 
        β = npd_problem.results.minimizer;
        θ = β[1:npd_problem.design_width]
        problem_has_linear_constraints = false;
        
        # Enforce equality constraints directly
        for i∈eachindex(npd_problem.mins)
            θ[npd_problem.mins[i]] = θ[npd_problem.maxs[i]]
        end

        if npd_problem.Aineq != []
            problem_has_linear_constraints = true;
            linear_constraints_violated = false;
        else 
            linear_constraints_violated = (maximum(npd_problem.Aineq * θ) >= npd_problem.constraint_tol);
        end

        if linear_constraints_violated
            println("Linear constraints not satisfied -- maximum deviation is $(maximum(npd_problem.Aineq * θ))")
        end
        return false
    end
end