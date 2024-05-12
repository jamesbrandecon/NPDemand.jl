function report_constraint_violations(problem;
    verbose = true)
    tempmats = calc_tempmats(problem)
    J = length(problem.Xvec)
    jacobians = elast_mat_zygote(problem.results.minimizer, problem, tempmats; at = Matrix(problem.data[!,r"prices"]), s = Matrix(problem.data[!,r"shares"]));
    reshaped_jacobians = [jacobians[i][j1,j2] for j1=1:J, j2 = 1:J, i = 1:size(problem.data,1)];

    violations = Dict();
    # Monotonicity
    if :monotone in problem.constraints
        frac_monotone_violations = round(1 - mean([all(diag(reshaped_jacobians[:,:,i]) .<0) for i in 1:size(reshaped_jacobians,3)]), digits = 2);
        verbose && println("Fraction of violations of :monotonicity: $frac_monotone_violations")
        push!(violations, :monotone => frac_monotone_violations)
    end

    # All substitutes
    if :all_substitutes in problem.constraints
        frac_all_subs_violations = round(1 - mean([check_all_subs(reshaped_jacobians[:,:,i]) for i in 1:size(reshaped_jacobians,3)]), digits = 2);
        verbose && println("Fraction of violations of :all_substitutes: $frac_all_subs_violations")
        push!(violations, :all_substitutes => frac_all_subs_violations)
    end

    # Diagonal dominance
    if :diagonal_dominance_all in problem.constraints
        frac_diag_dom_violations = round(1 - mean([check_diagonal_dominance(reshaped_jacobians[:,:,i]) for i in 1:size(reshaped_jacobians,3)]), digits = 2);
        verbose && println("Fraction of violations of :diagonal_dominance_all: $frac_diag_dom_violations")
        push!(violations, :diagonal_dominance_all => frac_diag_dom_violations)
    end

    return violations
end

function run_elasticity_check(elasts, constraints)
    elasticity_check = true;

    if :monotone in constraints
        elasticity_check = all([all(elasts[i,i,:] .<0) for i in 1:size(elasts,1)]);
    end
    if :all_substitutes in constraints
        elasticity_check = elasticity_check & (all([check_all_subs(elasts[:,:,i]) for i in 1:size(elasts,3)]));
    end
    if :diagonal_dominance_all in constraints
        elasticity_check = elasticity_check & (all([check_diagonal_dominance(elasts[:,:,i]) for i in 1:size(elasts,3)]));
    end

    return elasticity_check
end

function check_diagonal_dominance(elast_one_market)
    J = size(elast_one_market,1);
    
    # Check columnwise sums
    col_sums_minus_diag = [sum(elast_one_market[setdiff(1:J,i),i]) for i in 1:J]; #sum([elast_one_market[i,j] for i in 1:J for j in setdiff(1:J,i), for i in 1:J]);
    if all(abs.(col_sums_minus_diag) .< abs.(diag(elast_one_market)))
        return true
    else 
        return false 
    end
end

function check_all_subs(elast_one_market)
    J = size(elast_one_market,1);
    if all([elast_one_market[i,j] for i in 1:J for j in setdiff(1:J,i)] .>0)
        return true
    else 
        return false 
    end
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
            linear_constraints_violated = (maximum(npd_problem.Aineq * θ) > npd_problem.constraint_tol);
        end

        if linear_constraints_violated
            println("Linear constraints not satisfied -- maximum deviation is $(maximum(npd_problem.Aineq * θ))")
        end
        return false
    end
end