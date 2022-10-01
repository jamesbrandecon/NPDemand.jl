"""
    toDataFrame(s::Matrix, p::Matrix, z::Matrix, x::Matrix = zeros(size(s)))

Converts the results of `simulate_logit` to a DataFrame with column names that can be processed by NPDemand.jl
"""
function toDataFrame(s::Matrix, p::Matrix, z::Matrix, x::Matrix = zeros(size(s)) )
    df = DataFrame();
    J = size(s,2);
    for j = 0:J-1
        df[!, "shares$j"] =  s[:,j+1];
        df[!, "prices$j"] =  p[:,j+1];
        df[!, "demand_instruments$j"] =  z[:,j+1];
        df[!, "x$j"] =  x[:,j+1];
    end
    # for j = 0:J-1
    #     df[!, "demand_instruments$(j+J-1)"] =  x[:,j+1];
    # end
    return df;
end
