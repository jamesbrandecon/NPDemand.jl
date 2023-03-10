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

    # Constraint matrix for nonlinear constraints 
    J = length(Xvec);
    conmat = zeros(Float64,J,J);
    for j1 = 1:J
        ej = getindex.(findall(j1 .∈ exchange),1)[1];
        for j2 = 1:J
            if (j2==j1) | (j2 ∉ exchange[ej])
                conmat[j1,j2] = -Inf;
            end                
        end
    end
    # @show conmat 
    # error("end")

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
        println("Problem already has result vector -- Assuming warm start")
        β_init = problem.results.minimizer;
    end

    obj_uncon(x::SizedVector) = obj_func(x,0);
    grad_uncon!(G::SizedVector,x::SizedVector) = grad_func!(G,x,0, x -> 0);

    if isempty(Aineq) & (:subs_in_group ∉ problem.constraints);
        println("Problem only has equality constraints. Solving...")
        # c = @MArray zeros(length(β_init))
        c = SizedVector{length(β_init)}(β_init);
        c .= β_init;

        results =  Optim.optimize(obj_uncon, grad_uncon!, c,
        method, Optim.Options(show_trace = show_trace, iterations = max_iterations));
    else
        println("Solving problem without inequality constraints....")
        # c = @SizedVector zeros(length(β_init))
        c = SizedVector{length(β_init)}(β_init);
        c .= β_init;
        
        # @time obj_uncon(c)
        # @time grad_uncon!(c,c)

        results =  Optim.optimize(obj_uncon, grad_uncon!, c,
        method, Optim.Options(show_trace = show_trace, iterations = max_iterations, x_tol = obj_xtol, f_tol = obj_ftol));
        
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
            penalty_violated = (penalty_violated) & (elast_penaltyrev(c, exchange, elast_mats, elast_prices, L, conmat) >constraint_tol);
        end

        if penalty_violated 
            println("Iteratively increasing penalties on inequality constraints....")
        end

        while penalty_violated
            println("Iteration $(iter)...")
            L *= 100;

            # Pre-allocating 
            # dsids = @MArray zeros(ForwardDiff.Dual{Float64}, J, J, size(elast_mats[1,1],1));
            # temp = @MArray zeros(ForwardDiff.Dual{Float64}, J,J);
            # J_s = @MArray zeros(ForwardDiff.Dual{Float64},J,J)

            g_d(x) = elast_penaltyrev(x, exchange, elast_mats, 
                                        elast_prices, L, conmat);
            g(x) = ForwardDiff.gradient(g_d, x);

            obj(x::SizedVector) = obj_func(x,L);
            grad!(G::SizedVector,x::SizedVector) = grad_func!(G,x,L, g);

            if iter ==0
                R = results.minimizer;
                c = SizedVector{length(R)}(R);
                c .= R;

                results =  Optim.optimize(obj, grad!, c,
                method, Optim.Options(show_trace = show_trace, iterations = max_iterations, 
                    x_tol = obj_xtol, f_tol = obj_ftol));
            else
                R = results.minimizer; 
                # c = @MArray zeros(length(R))
                # c = SizedVector{length(R)}(R);
                c = R;
                results =  Optim.optimize(obj, grad!, c,
                method, Optim.Options(show_trace = show_trace, iterations = max_iterations, 
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
            if elast_mats!=[]
                # c = @MArray zeros(length(θ))
                # c = SizedVector{length(θ)}(θ);
                c = θ;
                penalty_violated = (penalty_violated) | (elast_penaltyrev(c, exchange, elast_mats, elast_prices, L, conmat) > constraint_tol);
            end

            iter+=1;
        end
    end

    problem.results = results;
    # problem.deltas = deltas;
    return 
end