using Test
using NPDemand

@testset "Tensor features without exchange" begin
    X = [0.1 0.2; 0.3 0.4; 0.5 0.6]
    basis_orders = [1,1]
    n,p = size(X)

    # precompute univariate bases
    Bs = [NPDemand.basis(X[:,j], basis_orders[j]) for j in 1:p]

    Z = NPDemand.tensor_features(X, basis_orders = basis_orders, exchange = [])
    @test size(Z) == (3,4)
    @test Z[:,1] == NPDemand.basis(X[:,1], basis_orders[1])[:,1] .* NPDemand.basis(X[:,2], basis_orders[2])[:,1]
end

@testset "Tensor features with exchange" begin
    X = [0.1 0.2; 0.3 0.4; 0.5 0.6]
    basis_orders = [1,1]
    Z = NPDemand.tensor_features(X, basis_orders = basis_orders, exchange=[[1,2]])
    @test size(Z) == (3,3)
end

@testset "Tensor derivative without exchange" begin
    X = [0.1 0.2; 0.3 0.4; 0.5 0.6]
    basis_orders = [1,1]
    dZ = NPDemand.tensor_features_derivative(X, var_index = 1, basis_orders = basis_orders, exchange = [])
    @test size(dZ) == (3,4)
end

@testset "Tensor derivative with exchange" begin
    X = [0.1 0.2; 0.3 0.4; 0.5 0.6]
    basis_orders = [1,1]
    dZ = NPDemand.tensor_features_derivative(X, var_index = 2, basis_orders = basis_orders, exchange=[[1,2]])
    @test size(dZ) == (3,3)
end
