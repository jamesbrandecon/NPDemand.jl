ultra = true
function md_obj(β::Vector; exchange = [], X = [], B = [], A = [],
    m1=[], m2=[], m3=[], m4=[], m5=[], m6=[], m7=[], m8=[], m9=[], DWD=[], WX = [], WB = [],
    Aineq = [], Aeq = [], design_width = 1, mins = [], maxs = [], normalization = [], 
    price_index = 1, lambda1=0, elast_mats=[], elast_prices = [])

    J = length(X);

    # Initialize objective function
    obj = zero(eltype(β));
    y = zeros(eltype(β),size(X[1],1))

    # Unpack: γ and θ
    θ = β[1:design_width]
    γ = β[length(θ)+1:end]

    # Normalize own-price coefficient
    γ[price_index] = 1;

    # Fixed-effect normalizations
    for i ∈ normalization
        γ[i] =0; 
    end

    # # Sign constraints
    # γ[2:5] = abs.(γ[2:5]); 

    # Enforce equality constraints directly
    for i∈eachindex(mins)
        θ[mins[i]] = θ[maxs[i]]
    end

    # Unconstrained objective function value
    theta_count = 0;
    for j = 1:J
        design_width_j = size(X[j],2);
        j_ind = (theta_count + 1):(theta_count + design_width_j);
        θ_j = θ[j_ind]
        if ultra == false 
            vecmat!(y, B[j],γ)
            obj += wsse_avx(y, X[j], θ_j, W[j]);
        else
            γtemp = γ[2:end];
            temp = m1[j] .+ Array(m2[j]*γtemp) .- Array(m3[j]*θ_j) .+ γtemp'*m4[j] .+ γtemp'* m5[j] * γtemp .- γtemp'*m6[j]*θ_j .- θ_j'*m7[j] .- Array(θ_j'*m8[j])*γtemp .+ θ_j'*m9[j]*θ_j;
            obj += temp[1];
        end
        theta_count += design_width_j;
    end

    if Aineq !=[]
        Atheta = Aineq*θ;
        temp_ineq = sum(lambda1 .* (Atheta[Atheta .>0]).^2 );
        obj +=  temp_ineq[1];
    end

    # Nonlinear constraints
    if (elast_mats != []) & (lambda1!=0)
        obj += elast_penalty(θ, exchange, elast_mats, elast_prices, lambda1);
    end

    obj
end

function md_grad!(grad::Vector, β::Vector; exchange = [], X = [], B = [], A = [],
    m1=[], m2=[], m3=[], m4=[], m5=[], m6=[], m7=[], m8=[], m9=[], DWD=[], WX = [], WB = [],
    Aineq = [], Aeq = [], design_width = 1, mins = [], maxs = [], normalization = [], price_index = 1, 
    lambda1=0, elast_mats=[], elast_prices = [])

    grad0 = zeros(eltype(β), size(β));
    ineq_con = zeros(eltype(β), size(Aineq,1));
    eq_con = zeros(eltype(β), size(Aeq,1))
    y = zeros(eltype(β), size(X[1],1));
    e = zeros(eltype(β), size(y));

    J = length(X);
    
    # Unpack: γ and θ
    θ = β[1:design_width]
    γ = β[length(θ)+1:end]

    # Normalize own-price coefficient
    γ[price_index] = 1;
    
    # Fixed-effect normalizations
    for i ∈ normalization
        γ[i] = 0; 
    end

    # # Sign constraints
    # γ[2:5] = abs.(γ[2:5]); 

    # Enforce equality constraints directly
    for i∈eachindex(mins)
        θ[mins[i]] = θ[maxs[i]]
    end

    # Gradient of unconstrained function
    grad_temp = zeros(length(grad0[length(θ)+1:end]));
    y = copy(e)
    e = zeros(eltype(β), size(X[1],1));

    theta_count = 0;
    for j = 1:J    
        design_width_j = size(X[j],2);
        # j_ind = ((j-1)*design_width_j + 1):(j*design_width_j);
        j_ind = (theta_count + 1):(theta_count + design_width_j);
        # @show j_ind
        θ_j = θ[j_ind];
        β_j = vcat(γ, θ_j)
        tempgrad = 2 .* DWD[j] * β_j;
        γtemp = γ[2:end];
        temp = -1 .* dropdims(m3[j], dims=1) .- dropdims(γtemp'*m6[j], dims=1) .- m7[j] .- Array(m8[j])*γtemp .+ 2 .* m9[j]*θ_j
        grad0[j_ind] = temp; 
        grad0[length(θ)+2:end] += tempgrad[2:length(γ)]
        theta_count += design_width_j;
    end 
    if Aineq !=[]
        # Gradient of inequality constraint 
        # vecmat!(ineq_con, Aineq, θ)
        ineq_con = Aineq * θ;
        
        pen1::Matrix = (lambda1 .* (ineq_con .> 0) .* 2 .* (ineq_con))';
        temp_ineq = dropdims(pen1 * Aineq, dims=1)
        grad0[1:length(θ)] += temp_ineq; # check matrix sizes, maybe need a '
    end 

    # Enforce normalization in gradient too
    grad0[length(θ)+1] = 0; 
    for i ∈ normalization
        grad0[length(θ)+i] = 0;
        grad0[length(θ)+i] = 0;
    end
    
    for i∈eachindex(mins)
        grad0[maxs[i]] += grad0[mins[i]];
    end
    for i∈eachindex(mins)
        grad0[mins[i]] = 0;
    end
    
    # Nonlinear constraints
    if (elast_mats != []) & (lambda1!=0)
        grad0[1:length(θ)] += ForwardDiff.gradient(x -> elast_penalty(x, exchange, elast_mats, elast_prices, lambda1), θ);
    end

    try 
        grad .= grad0
    catch
        grad .= dropdims(grad0, dims=1);
    end
    
    grad
end  

# function vecmat!(y, 𝐀, 𝐱)
#     @tturbo for i ∈ eachindex(y)
#         yi = zero(eltype(y))
#         for j ∈ eachindex(𝐱)
#             yi += 𝐀[i,j] * 𝐱[j]
#         end
#         y[i] = yi
#     end
# end

# function wsse_avx(y::Vector{Float64}, X::Matrix, β::Vector{<:Real}, W::Matrix)
#     esum = zero(eltype(β))
#     a = zeros(eltype(β), size(y))
#     vecmat!(a, X, β)
#     @tturbo e = y .- a    
#     @tturbo for i ∈ eachindex(y)
#         for j ∈ eachindex(y)
#             esum += e[i] * e[j] * W[i,j];
#         end
#     end
#     esum::eltype(β)
# end
