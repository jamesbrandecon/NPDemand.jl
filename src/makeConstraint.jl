using Primes
function makeConstraint(data, numVars, bernO, priceIndex, whichShare, own)
# Data: matrix of non-interacted bernstein polynomials
# numVars: number of variables when priceIndex==1, number of variables per
# product otherwise
# priceIndex: whether price is in index or not
# which market share we care abou
p = primes(10000);
pdata = zeros(1,size(data,2));
for i = 1:size(data,2)
    pdata[1,i] = p[i]; # list of primes the size of the data
end

pdata2 = fullInteraction(convert.(Integer,pdata), convert.(Integer, numVars), convert.(Integer,bernO)); # interact primes into design matrix

s1p = pdata[(bernO+1)*(whichShare-1)+1:(bernO+1)*whichShare]; # locations of univariate bernstein polynomial in s1

# monotonicity
if numVars>1
    constMat = [];
    for i = 1:length(s1p)-1
        t1 = pdata2./s1p[i]; # divide data by primes corresponding to terms in polynomial of s
        t2 = pdata2./s1p[i+1];
        for j = 1:1:size(t1,2)
            if ~isempty(findall(x->x==t1[j], t2)) # & (t1[j]==floor(t1[j])))
                z = zeros(1,size(pdata2,2));
                    # assumes s increasing in delta
                    z[j] = 1; # add a 1 at the current location
                    z[findall(x->x==t1[j], t2)] .= -1; # add a -1 where the order of s is one higher
                if i==1
                    constMat = z;
                else
                    constMat = [constMat; z];
                end
            end
        end
    end
else
    constMat = [];
    i1 = 1;
    i2 = 2;
    for i = 1:size(data,2)-1
        z = zeros(1,size(data,2));
        z[i1] = 1;
        z[i2] = -1;
        if i==1
            constMat = z;
        else
            constMat = [constMat;z];
        end
        i1 = i1+1;
        i2 = i2+1;
    end
end

return constMat
end
