function estimate_fast!(problem::NPDProblem; linear_solver = "Ipopt", verbose = true)
    β, γ = jmp_obj(problem, linear_solver = linear_solver, verbose = verbose);
    problem.results = NPD_JuMP_results([β;γ]);
end

function jmp_obj(npd_problem::NPDProblem; linear_solver = "Ipopt", verbose = true)
    # Unpack tolerances, even if not using
    constraint_tol = npd_problem.constraint_tol;
    obj_xtol = npd_problem.obj_xtol;
    obj_ftol = npd_problem.obj_ftol;

    # Unpack data
    Avec = npd_problem.Avec;
    Xvec = npd_problem.Xvec;
    Bvec = npd_problem.Bvec;
    indexes = vcat(0,cumsum(size.(Xvec,2)));
    J = length(Xvec);

    # Define JuMP problem 
    if linear_solver =="Ipopt"
        verbose_int = 0;
        if verbose ==true
            verbose_int = 5;
        end
        model = Model(optimizer_with_attributes(Ipopt.Optimizer, 
            "constr_viol_tol" => constraint_tol,
            "print_level" => verbose_int));
    elseif linear_solver =="OSQP"
        model = Model(optimizer_with_attributes(OSQP.Optimizer, 
            "check_termination" => 20000,
            "max_iter" => 20000));
    end
    # set_optimizer_attribute(model, "check_termination", min_iter);
    # set_optimizer_attribute(model, "eps_abs", 1e-12);
    # set_optimizer_attribute(model, "eps_prim_inf", 1e-12);
    # set_optimizer_attribute(model, "eps_rel", 1e-12);
    # set_optimizer_attribute(model, "max_iter", max_iter);
    
    @variable(model, γ[1:size(npd_problem.Bvec[1],2)]);
    @variable(model, β[1:npd_problem.design_width]);
    verbose && println("Setting up problem in JuMP ....")
    @objective(model, Min,
    sum((Bvec[i]*γ - Xvec[i] * β[(indexes[i]+1:indexes[i+1])])'*Avec[i]*pinv(Avec[i]'*Avec[i])*Avec[i]'*(Bvec[i] * γ - Xvec[i] * β[indexes[i]+1:indexes[i+1]]) for i ∈ 1:J));

    # Add constraints
    @constraint(model, γ[1]==1); # Price coefficient normalized

    # @constraint(model, [i = 1:size(npd_problem.mins,1)], 
    #     β[npd_problem.mins[i]] == β[npd_problem.maxs[i]]) # Enforcing exchangeability

    @constraint(model, [i = 1:size(npd_problem.Aineq,1)], # Enforcing inequality constraints
        sum(npd_problem.Aineq[i,:] .* β) <= 0)
    
    @constraint(model, [i = 1:size(npd_problem.Aeq,1)], # Enforcing exchangeability
        sum(npd_problem.Aeq[i,:] .* β) == 0)
    
    verbose && println("Solving problem in JuMP ....")

    # Solve problem and store results
    JuMP.optimize!(model);
    β_solved = value.(β);
    γ_solved = value.(γ);
    return β_solved, γ_solved
end

"""
    estimate!(problem::NPDProblem; max_iterations = 10000, show_trace = false, chunk_size = Int[])

`estimate!` solves `problem` subject to provided constraints, and replaces `problem.results` with the resulting parameter vector.
To enforce constraints, we iteratively minimize the sum of the objective function and a penalty term capturing violations of the constraint. 
For each outer iteration, we increase the size of the penalty term by an order of magnitude.  

Options: 
- `max_inner_iterations`: controls the number of inner iterations for each call to the optimizer via Optim.jl
- `max_outer_iterations`: controls the number of total calls to Optim, i.e., the number of times the penalty enforcing the constraints will increase before exiting
- `show_trace`: if `true`, Optim will print the trace for each outer iteration. 
- `verbose`: if `false`, will not print outer iteration updates within optimization
"""
function estimate!(problem::NPDProblem; max_inner_iterations = 10000, 
    max_outer_iterations = 100, 
    show_trace = false, 
    verbose = true,
    linear_solver = "Ipopt")

    # Check that linear solver is Ipopt or OSQP 
    if linear_solver ∉ ["Ipopt", "OSQP"]
        error("Linear solver must be Ipopt or OSQP")
    end

    # Unpack problem 
    matrices = problem.matrices;
    Xvec = problem.Xvec;
    Bvec = problem.Bvec;
    Avec = problem.Avec;
    Aineq = problem.Aineq; 
    Aeq = problem.Aeq;
    mins = problem.mins;  
    maxs = problem.maxs;
    normalization = problem.normalization;
    design_width = problem.design_width;
    elast_mats = problem.elast_mats; 
    elast_prices = problem.elast_prices;
    constraint_tol = problem.constraint_tol;
    obj_xtol = problem.obj_xtol;
    obj_ftol = problem.obj_ftol;
    exchange = problem.exchange;
    cfg = problem.cfg;

    find_prices = findall(problem.index_vars .== "prices");
    price_index = find_prices[1];

    # First, estimate the problem only with linear constraints 
    verbose && println("Estimating problem in JuMP without nonlinear constraints....")
    estimate_fast!(problem, linear_solver = linear_solver, verbose = verbose); 
    
    # Only move onto next steps if nonlinear constraints are included. 
    # Current nonlinear constraints: [:subs_in_group, :all_substitutes_nonlinear]
    problem_has_nonlinear_constraints = false;
    if maximum(x ∈ [:subs_in_group, :all_substitutes_nonlinear] for x ∈ problem.constraints)
        problem_has_nonlinear_constraints = true;
    end
    if !problem_has_nonlinear_constraints
        return
    else
        # Constraint matrix for nonlinear constraints 
        J = length(Xvec);
        # Start with two constraint matrix that has all zeros, which means (respectively) 
        # all goods are substitutes and all are complements
            # NOTE: already asserted previously that these constraints should not conflict
            if maximum(x ∈[:subs_in_group, :all_substitutes_nonlinear, :subs_across_groups] for x ∈ problem.constraints)
                conmat_subs = zeros(Float64,J,J);
            else
                conmat_subs=[];
            end
            if maximum(x∈[:complements_in_group, :complements_across_group] for x ∈ problem.constraints)
                conmat_complements = zeros(Float64,J,J);
            else
                conmat_complements = [];
            end
            if (:subs_in_group ∈ problem.constraints) | (:all_substitutes_nonlinear ∈ problem.constraints)
                # If subs_in_group, then only need to constrain within groups
                # All 
                for j1 = 1:J
                    ej = getindex.(findall(j1 .∈ exchange),1)[1];
                    for j2 = 1:J
                        if (j2==j1) | (j2 ∉ exchange[ej])
                            conmat_subs[j1,j2] = -Inf;
                        end                
                    end
                end
            end
            if (:subs_across_group ∈ problem.constraints) | (:all_substitutes_nonlinear ∈ problem.constraints)
                # All products in different groups should have conmat[j1,j2] = 0
                for j1 = 1:J
                    ej = getindex.(findall(j1 .∈ exchange),1)[1];
                    for j2 = 1:J
                        if (j2 ∉ exchange[ej])
                            conmat_subs[j1,j2] = 0;
                        end                
                    end
                end
            end
            
            # For complements, sign of infinities are reversed
            if :complements_in_group ∈ problem.constraints
            # If subs_in_group, then only need to constrain within groups
            # All 
                for j1 = 1:J
                    ej = getindex.(findall(j1 .∈ exchange),1)[1];
                    for j2 = 1:J
                        if (j2==j1) | (j2 ∉ exchange[ej])
                            conmat_complements[j1,j2] = Inf;
                        end                
                    end
                end
            end
            if :complements_across_groups ∈ problem.constraints
                # All products in different groups should have conmat[j1,j2] = 0
                for j1 = 1:J
                    ej = getindex.(findall(j1 .∈ exchange),1)[1];
                    for j2 = 1:J
                        if (j2 ∉ exchange[ej])
                            conmat_complements[j1,j2] = 0;
                        end                
                    end
                end
            end   
            
            conmat = Dict(
                :subs => conmat_subs,
                :complements => conmat_complements
            )

        obj_func(β::SizedArray, lambda::Real) = md_obj(β;exchange =exchange,X = Xvec, B = Bvec, A = Avec,
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
                Aineq = Aineq, Aeq = Aeq, design_width = design_width, 
                mins = mins, maxs = maxs, normalization = normalization, price_index = price_index, 
                lambda1 = lambda, elast_mats = elast_mats, elast_prices = elast_prices, conmat = conmat);

        grad_func!(grad, β, lambda::Real, g) = md_grad!(grad, β; exchange =exchange, X = Xvec, B = Bvec, A = Avec,
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
                Aineq = Aineq, Aeq = Aeq, design_width = design_width, 
                mins = mins, maxs = maxs, normalization = normalization, price_index = price_index, 
                lambda1 = lambda, elast_mats = elast_mats, elast_prices = elast_prices, 
                chunk_size = [], cfg = cfg, g = g, conmat = conmat);

        # Minimizer method 
        method = LBFGS();

        # Estimation 
        β_length = design_width + sum(size(Bvec[1],2))
        if problem.results == []
            # Random.seed!(12345)
            β_init = -0.5 .* ones(β_length)
        else
            verbose && println("Problem already has result vector -- using warm start")
            β_init = problem.results.minimizer;
        end

        obj_uncon(x::SizedVector) = obj_func(x,0);
        grad_uncon!(G::SizedVector,x::SizedVector) = grad_func!(G,x,0, x -> 0);

        penalty_violated = false;

        if isempty(Aineq) & (:subs_in_group ∉ problem.constraints);
            verbose && println("Problem only has equality constraints. Solving...")
            # c = @MArray zeros(length(β_init))
            c = SizedVector{length(β_init)}(β_init);
            c .= β_init;

            results =  Optim.optimize(obj_uncon, grad_uncon!, c,
            method, Optim.Options(show_trace = show_trace, iterations = max_inner_iterations));
            penalty_violated = false;
        else
            verbose && println("Solving problem without inequality constraints....")
            # c = @SizedVector zeros(length(β_init))
            c = SizedVector{length(β_init)}(β_init);
            c .= β_init;
            
            # @time obj_uncon(c)
            # @time grad_uncon!(c,c)

            results =  Optim.optimize(obj_uncon, grad_uncon!, c,
            method, Optim.Options(show_trace = show_trace, iterations = max_inner_iterations, x_tol = obj_xtol, f_tol = obj_ftol));
            
            L = 0.1;
            θ = results.minimizer[1:design_width];
            
            iter = 0;
            penalty_violated = true
            
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

            if !isempty(Aineq)
                penalty_violated = (maximum(Aineq * θ) > constraint_tol);
            end

            if elast_mats!=[]
                J = length(Xvec);
                c = results.minimizer[1:design_width];
                penalty_violated = (penalty_violated) | (elast_penaltyrev(c, exchange, elast_mats, elast_prices, L, conmat)/L >constraint_tol);
            end

            if penalty_violated 
                verbose && println("Iteratively increasing penalties on inequality constraints....")
            end

            while (penalty_violated) & (iter <max_outer_iterations)
                verbose && println("Iteration $(iter)...")
                L *= 100;

                g_d(x) = elast_penaltyrev(x, exchange, elast_mats, 
                                            elast_prices, L/200, conmat);
                g(x) = ForwardDiff.gradient(g_d, x);

                obj(x::SizedVector) = obj_func(x,L);
                grad!(G::SizedVector,x::SizedVector) = grad_func!(G,x,L, g);

                if iter ==0
                    R = results.minimizer;
                    c = SizedVector{length(R)}(R);
                    c .= R;

                    results =  Optim.optimize(obj, grad!, c,
                    method, Optim.Options(show_trace = show_trace, iterations = max_inner_iterations, 
                        x_tol = obj_xtol, f_tol = obj_ftol));
                else
                    R = results.minimizer; 
                    c = R;
                    results =  Optim.optimize(obj, grad!, c,
                    method, Optim.Options(show_trace = show_trace, iterations = max_inner_iterations, 
                        x_tol = obj_xtol, f_tol = obj_ftol));
                end

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

                if !isempty(Aineq)
                    penalty_violated = (maximum(Aineq * θ) > constraint_tol);
                end
                if (elast_mats!=[]) & problem_has_nonlinear_constraints # Added second check just to be sure
                    c = θ;
                    penalty_violated = (penalty_violated) | (elast_penaltyrev(c, exchange, elast_mats, elast_prices, L, conmat)/L > constraint_tol);
                end

                iter+=1;
            end
        end
        if penalty_violated 
            problem.converged = false;
        else
            problem.converged = true;
        end
        problem.results = results;
        return 
    end
end