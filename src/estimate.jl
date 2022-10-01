"""
    estimate!(problem::NPDProblem)

`estimate!` solves `problem` subject to provided constraints, and replaces `problem.results` with the resulting parameter vector
"""
function estimate!(problem::NPDProblem)
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
    tol = problem.tol;

obj_func(β::Vector, lambda::Int) = md_obj(β;X = Xvec, B = Bvec, A = Avec,
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
        mins = mins, maxs = maxs, normalization = normalization, price_index = 1, lambda1 = lambda, elast_mats = elast_mats, elast_prices = elast_prices);

grad_func!(grad::Vector, β::Vector, lambda::Int) = md_grad!(grad, β; X = Xvec, B = Bvec, A = Avec,
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
        mins = mins, maxs = maxs, normalization = normalization, price_index = 1, lambda1 = lambda, elast_mats = elast_mats, elast_prices = elast_prices);

    # Estimation 
    β_length = design_width + sum(size(Bvec[1],2))
    β_init = -1 .* rand(β_length)

    obj_uncon(x::Vector) = obj_func(x,0);
    grad_uncon!(G::Vector,x::Vector) = grad_func!(G,x,0);

    if isempty(Aineq) & (:subs_in_group ∉ problem.constraints);
        println("Problem only has equality constraints. Solving...")
        results =  Optim.optimize(obj_uncon, grad_uncon!, β_init,
        LBFGS(), Optim.Options(show_trace = true, iterations = 10000));
    else
        println("Solving problem without inequality constraints....")
        results =  Optim.optimize(obj_uncon, grad_uncon!, β_init,
        LBFGS(), Optim.Options(show_trace = false, iterations = 10000));
        L = 1;
        θ = results.minimizer[1:design_width];
        iter = 0;

        println("Iteratively increasing penalties on inequality constraints....")
        penalty_violated = true
        if !isempty(Aineq)
            penalty_violated = (maximum(Aineq * θ) > tol);
        end
        if elast_mats!=[]
            penalty_violated = (penalty_violated) & (elast_penalty(θ, exchange, elast_mats, elast_prices, L) >tol);
        end
        while penalty_violated
            println("Iteration $(iter)...")
            L *= 10;
            obj(x::Vector) = obj_func(x,L);
            grad!(G::Vector,x::Vector) = grad_func!(G,x,L);
            if iter ==1
                results =  Optim.optimize(obj, grad!, results.minimizer,
                    LBFGS(), Optim.Options(show_trace = true, iterations = 10000));
            else
                results =  Optim.optimize(obj, grad!, results.minimizer,
                    LBFGS(), Optim.Options(show_trace = true, iterations = 10000));
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
                penalty_violated = (maximum(Aineq * θ) > tol);
            end
            if elast_mats!=[]
                penalty_violated = (penalty_violated) & (elast_penalty(θ, exchange, elast_mats, elast_prices, L) > tol);
            end

            iter+=1;
        end
    end

    problem.results = results;
    return 
end