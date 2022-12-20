"""
    estimate!(problem::NPDProblem; max_iterations = 10000, show_trace = false, chunk_size = Int[])

`estimate!` solves `problem` subject to provided constraints, and replaces `problem.results` with the resulting parameter vector.
To enforce constraints, we iteratively minimize the sum of the objective function and a penalty term capturing violations of the constraint. 
For each outer iteration, we increase the size of the penalty term by an order of magnitude.  

Options: 
- `max_iterations`: controls the number of inner iterations for each outer iteration (i.e., for each value of the penalty term, the number of iterations used by Optim)
- `show_trace`: if `true`, Optim will print the trace for each outer iteration. 
"""
function estimate!(problem::NPDProblem; max_iterations = 10000, show_trace = false)
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

obj_func(β::Vector, lambda::Int) = md_obj(β;exchange =exchange,X = Xvec, B = Bvec, A = Avec,
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
        lambda1 = lambda, elast_mats = elast_mats, elast_prices = elast_prices);

grad_func!(grad::Vector, β::Vector, lambda::Int) = md_grad!(grad, β; exchange =exchange, X = Xvec, B = Bvec, A = Avec,
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
        chunk_size = [], cfg = cfg);

    # Estimation 
    β_length = design_width + sum(size(Bvec[1],2))
    if problem.results == []
        # Random.seed!(12345)
        β_init = -0.5 .* ones(β_length)
    else
        println("Problem already has result vector -- Assuming warm start")
        β_init = problem.results.minimizer;
    end

    obj_uncon(x::Vector) = obj_func(x,0);
    grad_uncon!(G::Vector,x::Vector) = grad_func!(G,x,0);

    if isempty(Aineq) & (:subs_in_group ∉ problem.constraints);
        println("Problem only has equality constraints. Solving...")
        results =  Optim.optimize(obj_uncon, grad_uncon!, β_init,
        LBFGS(), Optim.Options(show_trace = show_trace, iterations = max_iterations));
    else
        println("Solving problem without inequality constraints....")
        results =  Optim.optimize(obj_uncon, grad_uncon!, β_init,
        LBFGS(), Optim.Options(show_trace = show_trace, iterations = max_iterations, x_tol = obj_xtol, f_tol = obj_ftol));
        L = 1;
        θ = results.minimizer[1:design_width];
        iter = 0;

        println("Iteratively increasing penalties on inequality constraints....")
        penalty_violated = true
        if !isempty(Aineq)
            penalty_violated = (maximum(Aineq * θ) > constraint_tol);
        end
        if elast_mats!=[]
            J = length(Xvec);
            conmat = zeros(eltype(θ),J,J);
            for j1 = 1:J
                ej = getindex.(findall(j1 .∈ exchange),1)[1];
                for j2 = 1:J
                    if (j2==j1) | (j2 ∉ exchange[ej])
                        conmat[j1,j2] = -Inf;
                    end                
                end
            end
            penalty_violated = (penalty_violated) & (elast_penalty(θ, exchange, elast_mats, elast_prices, L, conmat) >constraint_tol);
        end
        while penalty_violated
            println("Iteration $(iter)...")
            L *= 10;
            obj(x::Vector) = obj_func(x,L);
            grad!(G::Vector,x::Vector) = grad_func!(G,x,L);
            if iter ==0
                results =  Optim.optimize(obj, grad!, results.minimizer,
                    LBFGS(), Optim.Options(show_trace = show_trace, iterations = max_iterations, x_tol = obj_xtol, f_tol = obj_ftol));
            else
                results =  Optim.optimize(obj, grad!, results.minimizer,
                    LBFGS(), Optim.Options(show_trace = show_trace, iterations = max_iterations, x_tol = obj_xtol, f_tol = obj_ftol));
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
            if elast_mats!=[]
                conmat = zeros(eltype(θ),J,J);
                for j1 = 1:J
                    ej = getindex.(findall(j1 .∈ exchange),1)[1];
                    for j2 = 1:J
                        if (j2==j1) | (j2 ∉ exchange[ej])
                            conmat[j1,j2] = -Inf;
                        end                
                    end
                end
                penalty_violated = (penalty_violated) & (elast_penalty(θ, exchange, elast_mats, elast_prices, L, conmat) > constraint_tol);
            end

            iter+=1;
        end
    end

    problem.results = results;
    return 
end