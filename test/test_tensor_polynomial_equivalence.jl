using NPDemand
using Test
using ForwardDiff
using LinearAlgebra
using Random

@testset "tensor polynomial derivative correctness" begin
    Random.seed!(1234)

    @testset "With exchange constraints" begin
        # Generate test data
        n = 50  # Number of observations
        p = 4   # Number of variables
        X = rand(n, p)  # Random data
        
        # Set parameters
        basis_orders = [2, 2, 2, 2]  # Same order for simplicity
        exchange = [[1, 2], [3, 4]]  # Exchange groups
        
        # Get tensor features
        Z = NPDemand.tensor_features(X, basis_orders=basis_orders, exchange=exchange)
        
        # Create a proper deep copy for the shifted version
        X_shifted = copy(X)  # Create an independent copy
        
        # Perform finite difference for column 2 (var_idx = 2)
        var_idx_test = 3
        h = 1e-6  # Use a smaller step for better accuracy
        X_shifted[:,var_idx_test] .+= h
        
        Z_shifted = NPDemand.tensor_features(X_shifted, basis_orders=basis_orders, exchange=exchange)
        
        # Manual finite difference approximation of derivative
        manual_dZ = (Z_shifted .- Z) ./ h
        
        # Compare with analytical derivative for this variable
        analytical_dZ = NPDemand.tensor_features_derivative(X, var_index=var_idx_test, basis_orders=basis_orders, exchange=exchange)
        
        # Check if they're approximately equal
        @test isapprox(manual_dZ, analytical_dZ, rtol=1e-4)
    end
    
end
