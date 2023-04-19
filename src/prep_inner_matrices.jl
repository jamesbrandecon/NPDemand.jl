function prep2(B::Matrix, X::Matrix, W::Matrix)
    price = B[:,1];
    Btemp = B[:,2:end];
    mat1 = price'*W*price; mat2 = price'*W*Btemp;
    mat3 = price' * W * X; mat4 = Btemp'*W*price;
    mat5 = Btemp'*W*Btemp; mat6 = Btemp'*W*X;
    mat7 = X'*W*price;     mat8 = X'*W*Btemp;
    mat9 = X'*W*X;

    return mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8, mat9
end

function prep_inner_matrices(bigX, bigA, bigB; verbose = true)
    m1 = []; m2 = []; m3 = []; m4 = []; m5 = []; m6 = []; m7 = []; m8 = []; m9 = [];
    bigW = [];
    GC.gc()
    WX = [];
    WB = [];
    W = []; X = []; B = [];
    DWD = [];
    J = length(bigX);

    for j = 1:J
        A = bigA[j];
        # println("loaded X, A, B")

        W = A*pinv(A'A)*A'
        # push!(bigW, W); # JMB run this if using ultra = false

        push!(WX, W*bigX[j]);
        push!(WB, W*bigB[j]);
        # println("beginning to prep mini matrices")

        mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8, mat9 = prep2(bigB[j],bigX[j], W)
        
        push!(m1, mat1)
        push!(m2, Array(mat2))
        push!(m3, Array(mat3))
        push!(m4, mat4)
        push!(m5, mat5)
        push!(m6, mat6)
        push!(m7, mat7)
        push!(m8, mat8)
        push!(m9, mat9)

        D = hcat(bigB[j], -1 .* bigX[j]);
        push!(DWD, D'*W*D);

        GC.gc()
        if verbose
            println("Done with choice $(j-1)")
        end
    end

    matrices = PreppedMatrices(m1,
    m2,
    m3,
    m4, 
    m5, 
    m6, 
    m7, 
    m8, 
    m9, 
    DWD, 
    WX, 
    WB,
    bigW)
    return matrices
end

mutable struct PreppedMatrices
    m1
    m2
    m3
    m4 
    m5 
    m6 
    m7 
    m8 
    m9 
    DWD 
    WX 
    WB
    bigW 
end