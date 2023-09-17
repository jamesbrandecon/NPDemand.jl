"""
    simulate_logit(J,T, beta, v)

Simulates logit demand for `J` products in `T` markets, with price preference parameter `beta` and market shocks with standard deviation `v`.
"""
function simulate_logit(J,T, beta, v; with_product_FEs = false)
    # --------------------------------------------------%
    # Simulate logit with outside option
    # --------------------------------------------------%
    # J = number of products
    # T = number of markets
    # beta = coefficient on delta
    # v = standard deviation of market-product demand shock

    zt = 0.9 .* rand(T,J) .+ 0.05;
    xit = randn(T,J).*v;
    pt = 2 .*(zt .+ rand(T,J).*0.1).+xit;
    xt = rand(T,J);
    if with_product_FEs
        product_FEs = repeat(reshape(collect(1:J)./J,1,J), T,1);
    else
        product_FEs = repeat(zeros(1,J), T,1);
    end
    denominator = 1;
    s = zeros(T,J)
    for j = 1:1:J
        denominator = denominator .+ exp.(beta*pt[:,j] .+ xt[:,j] .+ product_FEs[:,j] .+ xit[:,j]);
    end
    for j = 1:1:J
        s[:,j] = exp.(beta*pt[:,j] .+ xt[:,j] .+ product_FEs[:,j] .+ xit[:,j])./denominator;
    end
    if with_product_FEs
        return s, pt, zt, xt, xit, product_FEs
    else
        return s, pt, zt, xt, xit
    end
end
