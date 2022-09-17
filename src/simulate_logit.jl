function simulate_logit(J,T, beta, v)
# --------------------------------------------------%
# Simulate logit with outside option
# --------------------------------------------------%
# J = number of products
# T = number of markets
# beta = coefficient on delta
# v = standard deviation of market-product demand shock

zt = 0.9 .* rand(T,J) .+ 0.05;
xit = randn(T,J).*v;
pt = 2 .*(zt .+ rand(T,J)*0.1).+xit;
xt = rand(T,J);

denominator = 1;
s = zeros(T,J)
for j = 1:1:J
    denominator = denominator .+ exp.(beta*pt[:,j] .+ xt[:,j] .+ xit[:,j]);
end
for j = 1:1:J
    s[:,j] = exp.(beta*pt[:,j] .+ xt[:,j] .+ xit[:,j])./denominator;
end
s, pt, zt, xt, xit
end
