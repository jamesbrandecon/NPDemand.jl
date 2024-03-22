function estimate_fast!(problem::NPDProblem; linear_solver = "Ipopt", 
    verbose = true, nonlinear_method = "grid", conmat = [])

    β, γ = JMP_obj_constrained(problem, 
        linear_solver = linear_solver, 
        verbose = verbose, 
        nonlinear_method = nonlinear_method, 
        conmat = conmat);
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

function JMP_obj_constrained(npd_problem::NPDProblem; 
    linear_solver = "Ipopt", 
    verbose = true, 
    nonlinear_method = "grid", 
    conmat = [])

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
            "print_level" => verbose_int, 
            "max_iter" => 100000));
    elseif linear_solver =="OSQP"
        model = Model(optimizer_with_attributes(OSQP.Optimizer, 
            "check_termination" => 20000,
            "max_iter" => 20000));
    end
    @variable(model, γ[1:size(npd_problem.Bvec[1],2)]);
    @variable(model, β[1:npd_problem.design_width]);
    verbose && println("Setting up problem in JuMP ....")
    tempmat_storage = NPDemand.calc_tempmats(npd_problem);
    jump_penalty(x) = elast_penalty_JUMP(x, npd_problem.exchange, 
        npd_problem.elast_mats, npd_problem.elast_prices, 100.0, conmat; J = J,
            during_obj = false,
            quantile_vec = [0.5], 
            problem_details_dict = Dict(
            "data" => npd_problem.data,
            "β" => x,
            "design_width" => npd_problem.design_width,
            "J" => J,
            "exchange" => npd_problem.exchange,
            "normalization" => npd_problem.normalization,
            "tempmat_storage" => tempmat_storage,
            "mins" => npd_problem.mins,
            "maxs" => npd_problem.maxs,
            "Xvec" => Xvec,
            "Bvec" => Bvec,
            "bO" => npd_problem.bO
        ),
        tempmat_storage = tempmat_storage)
    jump_wrap(x...) = jump_penalty(x)
    # @show jump_penalty(rand(npd_problem.design_width))
    register(model, :pinv, 1, pinv; autodiff = true)
    # register(model, :jump_penalty, 1, jump_wrap; autodiff = true)
    @objective(model, Min, sum((Bvec[i]*γ - Xvec[i] * β[(indexes[i]+1:indexes[i+1])])'*Avec[i]*pinv(Avec[i]'*Avec[i])*Avec[i]'*(Bvec[i] * γ - Xvec[i] * β[indexes[i]+1:indexes[i+1]]) for i ∈ 1:J));
 
    # Add constraints
    @constraint(model, γ[1]==1); # Price coefficient normalized

    @constraint(model, [i = 1:size(npd_problem.Aineq,1)], # Enforcing inequality constraints
        sum(npd_problem.Aineq[i,:] .* β) <= 0)
    
    @constraint(model, [i = 1:size(npd_problem.Aeq,1)], # Enforcing exchangeability
        sum(npd_problem.Aeq[i,:] .* β) == 0)
    
    # if nonlinear_method == "jump"
    #     print("Adding nonlinear constraints in JuMP....")
    #     tempmat_storage = NPDemand.calc_tempmats(npd_problem);
    #     temp_storage_mat = Array{Matrix{Float64},2}(undef, J,J);
    #     at = npd_problem.data[!,r"prices"];
    #     svec2 = npd_problem.data[!,r"shares"];
    #     ps_mat = zeros(J,J,size(at,1))
    #     for j1 = 1:J, j2 = 1:J, ii = 1:size(at,1)
    #         ps_mat[j1,j2,ii] = at[ii,j2]/svec2[ii,j1];
    #     end
    #     counter = 1
    #     for j1 = 1:J
    #         for j2 = 1:J
    #             temp_storage_mat[j1,j2] = tempmat_storage[counter];
    #             counter +=1;
    #         end
    #     end
    #     # temp_storage_mat = npd_problem.elast_mats;
    #     beta_indexes = vcat([1:size(Xvec[1],2)], [sum(size.(Xvec[1:j1-1],2))+1:sum(size.(Xvec[1:j1-1],2))+size(Xvec[j1],2) for j1 in 2:J])
    #     dsids_expr = @expression(model,[j1=1:J, j2 = 1:J], temp_storage_mat[j1,j2] *  β[beta_indexes[j1]])
    #     #         # all_elast_expr = @expression(model, [j1=1:J, j2=1:J], -1 .* dsids_expr[j1,j2] .* ps_mat[j1,j2,:])
    #     #         # dsids_expr = @expression(model,[j1=1:J, j2 = 1:J], temp_storage_mat[j1,j2] *  β[beta_indexes[j1]])
    #     @variable(model, all_elast_mat[j1=1:J, j2=1:J, i = 1:size(npd_problem.elast_mats[1],1)]);
    #     @constraint(model, [i = 1:size(npd_problem.elast_mats[1],1)], -1 .* getindex.(dsids_expr,i) * all_elast_mat[:,:,i] .== ps_mat[:,:,i]);
        
    #     # my_conmat = [-100 0; 0 -100]
    #     # @constraint(model, mean(all_elast_mat, dims=3) .>= my_conmat);
    #     conmat_subs = conmat[:subs]
    #     conmat_subs[conmat_subs .== -Inf] .= -100;
    #     conmat_complements = conmat[:complements] 
    #     conmat_complements[conmat_complements .== Inf] .= 100;

    #     if conmat_subs != []
    #         @constraint(model, [i = 1:size(npd_problem.elast_mats[1],1)], all_elast_mat[:,:,i] .>= conmat_subs);
    #     end
    #     if conmat_complements != []
    #         @constraint(model, [i = 1:size(npd_problem.elast_mats[1],1)], all_elast_mat[:,:,i] .<= conmat_complements);
    #     end
    #     # @constraint(model, [i = 1:size(npd_problem.elast_mats[1],1)], all_elast_mat[:,:,i] .>= conmat_subs);
    #     # @constraint(model, [i = 1:size(npd_problem.elast_mats[1],1)], all_elast_mat[:,:,i] .<= conmat_complements);
    #     # @constraint(model, mean.(all_elast_expr) .>= my_conmat);
    # end
    
    verbose && println("Solving problem in JuMP ....")

    # Solve problem and store results
    JuMP.optimize!(model);
    β_solved = value.(β);
    γ_solved = value.(γ);
    return β_solved, γ_solved
end

function make_conmat(problem)
    exchange = problem.exchange;
    J = length(problem.Xvec);

    conmat_monotone = [];
    if :monotone_nonlinear in problem.constraints 
        conmat_monotone = zeros(Float64,J,J);
        conmat_monotone .= Inf;
        for j = 1:J    
            conmat_monotone[j,j] = 0.0;
        end
    end

    if maximum(x ∈[:subs_in_group, :all_substitutes_nonlinear, :subs_across_group] for x ∈ problem.constraints)
        conmat_subs = zeros(Float64,J,J);
        conmat_subs .= -Inf;
    else
        conmat_subs=[];
    end
    if maximum(x∈[:complements_in_group, :complements_across_group] for x ∈ problem.constraints)
        conmat_complements = zeros(Float64,J,J);
        conmat_complements .= Inf;
    else
        conmat_complements = [];
    end
    if (:subs_in_group ∈ problem.constraints) | (:all_substitutes_nonlinear ∈ problem.constraints)
        # If subs_in_group, then only need to constrain within groups
        # All 
        for j1 = 1:J
            ej = getindex.(findall(j1 .∈ exchange),1)[1];
            for j2 = 1:J
                if (j2!=j1) | (j2 ∈ exchange[ej])
                    conmat_subs[j1,j2] = 0;
                end                
            end
        end
    end
    if (:subs_across_group ∈ problem.constraints) | (:all_substitutes_nonlinear ∈ problem.constraints)
        # All products in different groups should have conmat[j1,j2] = 0
        for j1 = 1:J
            ej = getindex.(findall(j1 .∈ exchange),1)[1];
            for j2 = 1:J
                if (j2 ∉ exchange[ej]) & (j1!=j2)
                    conmat_subs[j1,j2] = 0;
                end                
            end
        end
    end

    # For complements, sign of infinities are reversed
    if :complements_in_group ∈ problem.constraints
        for j1 = 1:J
            ej = getindex.(findall(j1 .∈ exchange),1)[1];
            for j2 = 1:J
                if (j2 ∈ exchange[ej])
                    conmat_complements[j1,j2] = 0;
                end                
            end
        end
    end
    if :complements_across_group ∈ problem.constraints
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

    # conmat_subs = [-Inf 0.0; 0.0 -Inf];
    conmat = Dict(
        :subs => conmat_subs,
        :complements => conmat_complements, 
        :monotone => conmat_monotone
    )
    return conmat
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
    linear_solver = "Ipopt", 
    quantiles = [0.5],
    nonlinear_method = "grid", 
    penalized::Bool = true,
    lambda::Float64 = 1.0)

    # Check that linear solver is Ipopt or OSQP 
    if linear_solver ∉ ["Ipopt", "OSQP"]
        error("Linear solver must be Ipopt or OSQP")
    end

    # Unpack problem 
    df = problem.data;
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
    bO = problem.bO;
    exchange = problem.exchange;
    cfg = problem.cfg;

    JUMP = (nonlinear_method == "jump");
    find_prices = findall(problem.index_vars .== "prices");
    price_index = find_prices[1];

    conmat = make_conmat(problem)

    # First, estimate the problem only with linear constraints 
    verbose && println("Estimating problem in JuMP without nonlinear constraints....")
    estimate_fast!(problem, 
        linear_solver = linear_solver, 
        verbose = verbose, 
        nonlinear_method = nonlinear_method, 
        conmat = conmat); 

    # Only move onto next steps if nonlinear constraints are included. 
    # Current nonlinear constraints: [:subs_in_group, :all_substitutes_nonlinear]
    problem_has_nonlinear_constraints = false;
    if !JUMP && maximum(x ∈ [:subs_in_group, :all_substitutes_nonlinear, :subs_across_group, 
        :complements_across_group, :complements_in_group, :monotone_nonlinear] for x ∈ problem.constraints)
        problem_has_nonlinear_constraints = true;
    end

    if JUMP | (!problem_has_nonlinear_constraints)
        return
    else
        # Constraint matrix for nonlinear constraints 
        J = length(Xvec);
        # Start with two constraint matrix that has all zeros, which means (respectively) 
        # all goods are substitutes and all are complements
            # NOTE: already asserted previously that these constraints should not conflict

            if problem_has_nonlinear_constraints
                tempmat_storage = calc_tempmats(problem);
            else
                tempmat_storage = [];
            end
        obj_func(β, lambda::Real,w) = md_obj(β;df = df, exchange =exchange,X = Xvec, B = Bvec, A = Avec,
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
                lambda1 = lambda, elast_mats = elast_mats, elast_prices = elast_prices, conmat = conmat, 
                nonlinear_method = nonlinear_method, quantiles = quantiles, bO = bO, 
                tempmat_storage = tempmat_storage, weights = [1.0,1.0]);

        grad_func!(grad, β, lambda::Real, g,w) = md_grad!(grad, β; df = df, exchange =exchange, X = Xvec, B = Bvec, A = Avec,
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
                chunk_size = [], cfg = cfg, g = g, conmat = conmat,
                weights = w);

        # Minimizer method 
        method = LBFGS(); #LBFGS();

        # Estimation 
        β_length = design_width + sum(size(Bvec[1],2))
        if problem.results == []
            # Random.seed!(12345)
            β_init = -0.5 .* ones(β_length)
        else
            verbose && println("Problem already has result vector -- using warm start")
            β_init = problem.results.minimizer;
        end

        obj_uncon(x::Vector) = obj_func(x,0, [1.0, 0.0]);
        grad_uncon!(G::Vector,x::Vector) = grad_func!(G,x,0, x -> 0, [1.0, 0.0]);

        penalty_violated = false;

        if isempty(Aineq) & (!problem_has_nonlinear_constraints);
            verbose && println("Problem only has equality constraints. Solving...")
            # c = @MArray zeros(length(β_init))
            # c = SizedVector{length(β_init)}(β_init);
            c = Vector{Float64}(undef, length(β_init));
            c .= β_init;

            results =  Optim.optimize(obj_uncon, grad_uncon!, c,
            method, Optim.Options(show_trace = show_trace, iterations = max_inner_iterations));
            penalty_violated = false;
        else
            verbose && println("Solving problem without inequality constraints....")
            # c = @SizedVector zeros(length(β_init))
            # c = SizedVector{length(β_init)}(β_init);
            c = Vector{Float64}(undef, length(β_init));
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

            not_penalized_or_first_run = true;
            if penalized 
                L = lambda/100;
            end

            while (not_penalized_or_first_run) & (penalty_violated) & (iter <max_outer_iterations)
                verbose && println("Iteration $(iter)...")
                L *= 100;
                
                g_d(x) = elast_penalty_all(x, exchange, elast_mats, 
                    elast_prices, L, conmat; 
                    quantile_vec = quantiles,
                    nonlinear_method = nonlinear_method,
                    problem_details_dict = Dict(
                        "data" => df,
                        "β" => β,
                        "design_width" => design_width,
                        "J" => J,
                        "exchange" => exchange,
                        "normalization" => normalization,
                        "tempmat_storage" => tempmat_storage,
                        "mins" => mins,
                        "maxs" => maxs,
                        "Xvec" => Xvec,
                        "Bvec" => Bvec,
                        "bO" => bO
                    ),
                    during_obj = false);

                g(x) = ForwardDiff.gradient(g_d, x);
                # g(x) = Zygote.gradient(g_d, x)
                if L == 10
                    w = [0.0, 1e9];
                else
                    w = [1.0, 1e6];
                end
                obj(x::Vector) = obj_func(x,L, w);
                grad!(G::Vector,x::Vector) = grad_func!(G,x,L,g, w);

                if iter ==0
                    R = results.minimizer;
                    # c = SizedVector{length(R)}(R);
                    c = Vector{Float64}(undef, length(R));
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
                    # penalty_violated = (penalty_violated) | (elast_penaltyrev(c, exchange, elast_mats, elast_prices, L, conmat)/L > constraint_tol);
                    nonlinear_penalty = elast_penalty_all(θ, exchange, elast_mats, 
                        elast_prices, L, conmat; 
                        quantile_vec = quantiles,
                        nonlinear_method = nonlinear_method,
                        problem_details_dict = Dict(
                            "data" => df,
                            "β" => β,
                            "design_width" => design_width,
                            "J" => J,
                            "exchange" => exchange,
                            "normalization" => normalization,
                            "tempmat_storage" => tempmat_storage,
                            "mins" => mins,
                            "maxs" => maxs,
                            "Xvec" => Xvec,
                            "Bvec" => Bvec,
                            "bO" => bO
                        ),
                        during_obj = false)
                        println("Current nonlinear penalty: $nonlinear_penalty")
                    penalty_violated = (penalty_violated) | (nonlinear_penalty > constraint_tol);
                    
                end
                if penalized 
                    not_penalized_or_first_run = false;
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
