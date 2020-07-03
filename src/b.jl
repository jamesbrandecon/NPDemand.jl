function b(t,n,k)
    out = binomial(n,k).*(t.^k).*((1 .- t).^(n-k))
    return out
end
