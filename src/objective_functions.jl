ultra = true
function md_obj(β::Vector; exchange = [], X = [], B = [], A = [],
    m1=[], m2=[], m3=[], m4=[], m5=[], m6=[], m7=[], m8=[], m9=[], DWD=[], WX = [], WB = [],
    Aineq = [], Aeq = [], design_width = 1, mins = [], maxs = [], normalization = [], 
    price_index = 1, lambda1=0, elast_mats=[], elast_prices = [])

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
        θ_j = θ[j_ind]
        if ultra == false 
            vecmat!(y, B[j],γ)
            obj += wsse_avx(y, X[j], θ_j, W[j]);
        else
            γtemp = γ[2:end];
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
        conmat = zeros(eltype(θ),J,J);
        Threads.@threads for j1 = 1:J
            ej = getindex.(findall(j1 .∈ exchange),1)[1];
            for j2 = 1:J
                if (j2==j1) | (j2 ∉ exchange[ej])
                    conmat[j1,j2] = -Inf;
                end                
            end
        end
        obj += elast_penalty(θ, exchange, elast_mats, elast_prices, lambda1, conmat);
    end

    obj
end

function md_grad!(grad::Vector, β::Vector; exchange = [], X = [], B = [], A = [],
    m1=[], m2=[], m3=[], m4=[], m5=[], m6=[], m7=[], m8=[], m9=[], DWD=[], WX = [], WB = [],
    Aineq = [], Aeq = [], design_width = 1, mins = [], maxs = [], normalization = [], price_index = 1, 
    lambda1=0, elast_mats=[], elast_prices = [], chunk_size = [])

    grad0 = zeros(eltype(β), size(β));
    ineq_con = zeros(eltype(β), size(Aineq,1));
    eq_con = zeros(eltype(β), size(Aeq,1))
    # y = zeros(eltype(β), size(X[1],1));
    # e = zeros(eltype(β), size(y));

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
    # y = copy(e)
    # e = zeros(eltype(β), size(X[1],1));

    theta_count = 0;
    temp = zeros(1);
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
        # Convert jacobian_vec to Penalty
        # Have to calculate sign matrix from exchange
        conmat = zeros(eltype(θ),J,J);
        for j1 = 1:J
            ej = getindex.(findall(j1 .∈ exchange),1)[1];
            for j2 = 1:J
                if (j2==j1) | (j2 ∉ exchange[ej])
                    conmat[j1,j2] = -Inf;
                end                
            end
        end
        g(x) = elast_penalty(x, exchange, elast_mats, elast_prices, lambda1, conmat);
        θ_packed =  pack_parameters(θ, exchange, size.(X,2))
        if chunk_size ==[]
            cfg = GradientConfig(g, θ_packed);
        else
            cfg = GradientConfig(g, θ_packed, Chunk{chunk_size}());
        end
        # @show g(θ_packed)
        # print(reshape(unpack(θ_packed, exchange, size.(X,2), grad=true), 140, 5))
        packed_grad = ForwardDiff.gradient(g, θ_packed, cfg);
        
        grad0[1:length(θ)] += unpack(packed_grad, exchange, size.(X,2), grad=true);
        # grad0[1:length(θ)] += ForwardDiff.gradient(g, θ, cfg);
    end

    try 
        grad .= grad0
    catch
        grad .= dropdims(grad0, dims=1);
    end
    
    grad
end  

function pack_parameters(θ, exchange, lengths)
    θ_packed = eltype(θ)[];
    index = 1;
    for i ∈ eachindex(exchange)
        params_per_prod_in_group = lengths[exchange[i][1]];
        θ_packed = vcat(θ_packed, θ[index:(index + params_per_prod_in_group -1)]);
        index += sum(lengths[exchange[i]]);
    end
    return θ_packed
end

function unpack(θ_packed, exchange, lengths; grad=true)
    θ = eltype(θ_packed)[];
    index = 1;
    for i ∈ eachindex(exchange)
        params_per_prod_in_group = lengths[exchange[i][1]];
        add_θ = repeat(θ_packed[index:(index + params_per_prod_in_group -1)], length(exchange[i]));
        if grad==true
            add_θ[params_per_prod_in_group+1:end] .=0;
        end
        θ = vcat(θ, add_θ);
        index += params_per_prod_in_group;
    end
    return θ
end