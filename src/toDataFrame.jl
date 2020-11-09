function toDataFrame(s::Matrix, p::Matrix, z::Matrix, x::Matrix = zeros(size(s)) )
    df = DataFrame();
    J = size(s,2);
    for j = 1:J
        df[!, "s$j"] =  s[:,j];
        df[!, "p$j"] =  p[:,j];
        df[!, "z$j"] =  z[:,j];
        df[!, "x$j"] =  x[:,j];
    end
    return df;
end
