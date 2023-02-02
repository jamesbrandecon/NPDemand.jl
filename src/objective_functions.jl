ultra = true
function md_obj(β::AbstractVector; exchange = [], X = [], B = [], A = [],
    m1=[], m2=[], m3=[], m4=[], m5=[], m6=[], m7=[], m8=[], m9=[], DWD=[], WX = [], WB = [],
    Aineq = [], Aeq = [], design_width = 1, mins = [], maxs = [], normalization = [], 
    price_index = 1, lambda1=0.0, elast_mats=[], elast_prices = [], conmat = [])

    J = length(X);

    # Initialize objective function
    obj = zero(eltype(β));
    # y = zeros(eltype(β),size(X[1],1))

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
    temp = zeros(1);
    for j = 1:J
        design_width_j = size(X[j],2);
        j_ind = (theta_count + 1):(theta_count + design_width_j);
        @views θ_j = θ[j_ind]
        if ultra == false 
            vecmat!(y, B[j],γ)
            obj += wsse_avx(y, X[j], θ_j, W[j]);
        else
            @views γtemp = γ[2:end];
            @views temp = m1[j] .+ Array(m2[j]*γtemp) .- Array(m3[j]*θ_j) .+ γtemp'*m4[j] .+ γtemp'* m5[j] * γtemp .- γtemp'*m6[j]*θ_j .- θ_j'*m7[j] .- Array(θ_j'*m8[j])*γtemp .+ θ_j'*m9[j]*θ_j;
            obj += temp[1];
        end
        theta_count += design_width_j;
    end

    if Aineq !=[]
        Atheta = Aineq*θ;
        @views temp_ineq = sum(lambda1 .* (Atheta[Atheta .>0]).^2 );
        obj +=  temp_ineq[1];
    end

    # Nonlinear constraints
    if (elast_mats != []) & (lambda1!=0)
        # conmat = zeros(eltype(θ),J,J);
        # Threads.@threads for j1 = 1:J
        #     ej = getindex.(findall(j1 .∈ exchange),1)[1];
        #     for j2 = 1:J
        #         if (j2==j1) | (j2 ∉ exchange[ej])
        #             conmat[j1,j2] = -Inf;
        #         end                
        #     end
        # end
        # c = @MArray zeros(length(θ))
        # c = SizedVector{length(θ)}(θ);
        c = θ;
        obj += elast_penaltyrev(c, exchange, elast_mats, elast_prices, lambda1, conmat; during_obj=true);
    end

    obj
end

function md_grad!(grad::AbstractVector, β::AbstractVector; exchange::Array = [], X = [], B = [], A = [],
    m1=[], m2=[], m3=[], m4=[], m5=[], m6=[], m7=[], m8=[], m9=[], DWD=[], WX = [], WB = [],
    Aineq = [], Aeq = [], design_width = 1, mins = [], maxs = [], normalization = [], price_index = 1, 
    lambda1 = 0.0, g = x -> x , elast_mats =[], elast_prices = [], chunk_size = [], cfg = [],
    ineq_con = zeros(eltype(β), size(Aineq,1)),
    eq_con = zeros(eltype(β), size(Aeq,1)), conmat = [], grad0 = zeros(eltype(β), size(grad)))

    # y = zeros(eltype(β), size(X[1],1));
    # e = zeros(eltype(β), size(y));
    # 

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
    # grad_temp = zeros(length(grad0[length(θ)+1:end]));
    # y = copy(e)
    # e = zeros(eltype(β), size(X[1],1));

    theta_count = 0; # had to be int
    temp = zero(eltype(θ));
    for j = 1:J    
        design_width_j = size(X[j],2);
        # j_ind = ((j-1)*design_width_j + 1):(j*design_width_j);
        j_ind = (theta_count + 1):(theta_count + design_width_j);
        # @show j_ind
        θ_j = θ[j_ind];
        β_j = vcat(γ, θ_j)
        @views tempgrad = 2 .* DWD[j] * β_j;
        γtemp = γ[2:end];
        @views temp = -1 .* dropdims(m3[j], dims=1) .- dropdims(γtemp'*m6[j], dims=1) .- m7[j] .- Array(m8[j])*γtemp .+ 2 .* m9[j]*θ_j
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

    # Nonlinear constraints
    if (elast_mats != []) & (lambda1!=0)
        # Convert jacobian_vec to Penalty
        # Have to calculate sign matrix from exchange
        # g(x) = elast_penaltyrev(x, exchange, elast_mats, elast_prices, lambda1, conmat);

        # g(x::Vector) = sum(x .^2)
        # θ_packed =  pack_parameters(θ, exchange, size.(X,2))
        # if chunk_size ==[]
        #     # cfg = GradientConfig(g, θ_packed);
        #     cfg = GradientConfig(nothing, θ_packed);
        # else
        #     # cfg = GradientConfig(g, θ_packed, Chunk{chunk_size}());
        #     cfg = GradientConfig(nothing, θ_packed, Chunk{chunk_size}());
        # end
        # @show g(θ_packed)
        # print(reshape(unpack(θ_packed, exchange, size.(X,2), grad=true), 140, 5))
        # c = @MVector ones(length(θ_packed))
        # c = SizedVector{length(θ_packed)}(θ_packed);
        # c = θ_packed;
        # c  = @MArray [i for i in pack_parameters(θ, exchange, size.(X,2))]
        # packed_grad = ForwardDiff.gradient(g,
            # c, GradientConfig(g, c, Chunk{1}())); 
        # packed_grad = FiniteDiff.finite_difference_gradient(g, c);
        # packed_grad = ForwardDiff.gradient(g, c, GradientConfig(g, c, Chunk{1}())); #MArray{Tuple(length(θ_packed))}(θ_packed)
        
        # grad0[1:length(θ)] += unpack(packed_grad, exchange, size.(X,2), grad=true);
        # grad0[1:length(θ)] += ForwardDiff.gradient(g, θ);
        grad0[1:length(θ)] += g(θ);
    end

    # Enforce normalization in gradient too
    grad0[length(θ)+1] = zero(eltype(θ)); 
    for i ∈ normalization
        grad0[length(θ)+i] = zero(eltype(θ));
        grad0[length(θ)+i] = zero(eltype(θ));
    end
    
    for i∈eachindex(mins)
        grad0[maxs[i]] += grad0[mins[i]];
    end
    for i∈eachindex(mins)
        grad0[mins[i]] = zero(eltype(θ));
    end

    try 
        grad .= grad0
    catch
        grad .= dropdims(grad0, dims=1);
    end
    # @show typeof(θ)
    grad
end  

function pack_parameters(θ, exchange::Array, lengths)
    θ_packed = eltype(θ)[];
    index = 1;
    for i ∈ eachindex(exchange)
        params_per_prod_in_group = lengths[exchange[i][1]];
        θ_packed = vcat(θ_packed, θ[index:(index + params_per_prod_in_group -1)]);
        index += sum(lengths[exchange[i]]);
    end
    return θ_packed
end

function unpack(θ_packed, exchange::Array, lengths; grad=true)
    θ = eltype(θ_packed)[];
    index = 1;
    for i ∈ eachindex(exchange)
        params_per_prod_in_group = lengths[exchange[i][1]];
        add_θ = repeat(θ_packed[index:(index + params_per_prod_in_group -1)], length(exchange[i]));
        # if grad==true
        #     add_θ[params_per_prod_in_group+1:end] .=0;
        # end
        θ = vcat(θ, add_θ);
        index += params_per_prod_in_group;
    end
    return θ
end