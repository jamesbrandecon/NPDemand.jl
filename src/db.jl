function db(t,n,k)
    out = binomial(n,k).*(k.*t.^(k.-1).*(1 .-t).^(n-k) .- t.^k.*(n-k).*(1 .-t).^(n-k-1));
    return out
end
