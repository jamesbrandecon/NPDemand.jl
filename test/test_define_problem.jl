
## -------------------
# Prep
## -------------------
# Simulate data
J = 4; # of products
T = 10000; # # of markets
beta = -1; # price coefficient
sdxi = 0.01; # standard deviation of xi
s, p, z, x, xi  = simulate_logit(J, T, beta, sdxi);
df = toDataFrame(s,p,z,x);

exchange = [[1;2;3;4]];  
index_vars = ["prices", "x"]

@testset "defining problems with different approximations" begin
    for constraints ∈ [[:exchangeability], 
                        [:exchangeability, :monotone, :diagonal_dominance_all]]
        for sieve_type ∈ ["bernstein", "polynomial"]
            for tensor ∈ [false, true]
            approximation_details = Dict(
                                :order => 2, 
                                :max_interaction => 1, 
                                :sieve_type => sieve_type, 
                                :tensor => false
                            )

            npd_problem = define_problem(df; 
                                    exchange = exchange, 
                                    index_vars = index_vars, 
                                    constraints = constraints,
                                    FE = [], 
                                    approximation_details = approximation_details, 
                                    verbose = true
                                );

            @test npd_problem isa NPDemand.NPDProblem
            end
        end
    end
end