function calc_tempmats(problem::NPDProblem;
    recipe = nothing)

    J = length(problem.Xvec);

    s        = Matrix(problem.data[:, r"shares"]);
    exchange = problem.exchange;
    bO       = problem.approximation_details[:order];
    tensor   = haskey(problem.approximation_details, :tensor) ? problem.approximation_details[:tensor] : true;
    bernO    = convert.(Integer, bO);
    
    tempmats = Matrix{Float64}[]
    perm_s   = zeros(size(s));
    nbetas   = size.(problem.Xvec,2);
    approximation_details = problem.approximation_details;

    if approximation_details[:sieve_type] == "polynomial" && isnothing(recipe)
        recipes = [ begin
            ex2 = length(exchange)==J ? [] : adjust_exchange(exchange, j1)
            build_poly_recipe(J;
                order           = approximation_details[:order],
                max_interaction = approximation_details[:max_interaction],
                exchange        = ex2)
          end for j1 in minimum.(exchange)]
    end

    if isempty(exchange) exchange = []; end # if exchange is empty, set it to empty vector
    for j1 = 1:J
        which_group = findfirst(j1 .∈  exchange); # find the group corresponding to this product
        first_product_in_group = minimum(exchange[which_group]);
        _, permuted_shares, permutations = get_params_one_equation(j1; 
            exchange = exchange, 
            s = s, 
            θ = 1:problem.design_width, # not needed --providing a dummy vector
            nbetas = nbetas)
        
        for j2 = 1:J 
            # println("Calculating tempmat for j1 = ", j1, " and j2 = ", j2)
            tempmat_s = calc_derivative_sieve(permutations[j1], permutations[j2];
                exchange          = ((exchange==[]) || ((approximation_details[:sieve_type] == "bernstein") && (tensor == true))) ? 
                    exchange : adjust_exchange(exchange, first_product_in_group),
                shares            = s,             
                permuted_shares   = permuted_shares,
                perm              = permutations,
                bernO             = bernO,
                sieve_type        = approximation_details[:sieve_type],
                recipe            = approximation_details[:sieve_type] == "polynomial" ? recipes[which_group] : nothing, 
                max_interaction   = approximation_details[:max_interaction],
                constraints       = problem.constraints, 
                tensor            = haskey(approximation_details, :tensor) ? approximation_details[:tensor] : true
                )
            push!(tempmats, tempmat_s)
        end
    end
    # Take transpose of temp_storage so that it's correct for future use
    temp_storage_mat = reshape(tempmats, J,J);
    temp_elast_mats  = deepcopy(temp_storage_mat);
    for j1 = 1:J
        for j2 = 1:J
            temp_elast_mats[j1,j2] = temp_storage_mat[j2,j1];
        end
    end
    temp_storage_mat = temp_elast_mats;
    return temp_storage_mat
end

function inner_elast_loop(dsids_i::AbstractMatrix{T}, J::Int, at::AbstractVector{Float64}, svec::AbstractVector{Float64}; type::String = "jacobian") where T
    # J_s = [dsids_i[j1,j2] for j1 in 1:J, j2 in 1:J]
    temp = try
        -inv(dsids_i)
    catch
        -pinv(dsids_i)
    end

    if type == "jacobian"
        return temp 
    else
        return temp .* [at[j2]/svec[j1] for j1 in 1:J, j2 in 1:J]
    end
end

function elast_mat_zygote(θ::AbstractArray{T}, 
    problem::NPDemand.NPDProblem,
    tempmat_storage::Matrix{Matrix{Float64}} = []; 
    at::Matrix = [], s::Matrix = [], 
    type::String = "jacobian") where T <:Real

    J           = length(problem.Xvec);
    indexes     = [0;cumsum(size.(problem.Xvec,2))];
    temp_length = size(problem.data,1); #length(dsids[1,1,:]);

    dsids_raw     = [tempmat_storage[j1, j2] * θ[indexes[j1]+1:indexes[j1+1]] for j1 in 1:J, j2 in 1:J]
    # dsids         = [dsids_raw[j1, j2][i] for j1 in 1:J, j2 in 1:J, i in 1:length(dsids_raw[1, 1])]
    # all_elast_mat = [inner_elast_loop(dsids[:, :, ii], J, at[ii, :], s[ii, :]; type = type) for ii in 1:temp_length]
    all_elast_mat = [inner_elast_loop([dsids_raw[j1,j2][i] for j1 in 1:J, j2 in 1:J], J, view(at, i, :), view(s, i, :); type = type) for i in 1:temp_length]

    return all_elast_mat
end

function sieve_to_betas_index(problem)
    starts = [1;cumsum(size.(problem.Xvec,2))[1:end-1] .+ 1]
    ends = cumsum(size.(problem.Xvec,2))
    cols = []
    for i in 1:length(problem.exchange)
        push!(cols, starts[minimum.(problem.exchange)[i]]:ends[minimum.(problem.exchange)[i]])
    end
    return reduce(vcat, cols)
end

function get_nbetas(problem::NPDemand.NPDProblem)
    sieve_widths = size.(problem.Xvec,2);
    first_products = first.(problem.exchange);
    nbetas = [getindex(sieve_widths, i) for i in first_products]
    return nbetas
end

function get_lower_bounds(problem)
    if problem.Aineq != []
        A = problem.Aineq[:,sieve_to_betas_index(problem)]
        lbs = []
        for j in axes(A,2)
            if findall(A[:,j] .== -1) == []
                push!(lbs, nothing)
            else
                push!(lbs, findall(vec(sum(A[findall(A[:,j] .== -1),:], dims=1)) .== 1))
            end
        end
    elseif problem.Aeq != []
        lbs = 5000 .* ones(Int, size(problem.Aeq[:,sieve_to_betas_index(problem)], 2))
    else
        lbs = 5000 .* ones(Int, sum(size.(problem.Xvec,2)))
    end
    return lbs
end

function get_parameter_order(lbs)
    if !(all(lbs .== 5000))
        assigned = []
        unassigned = [1:length(lbs);]
        while length(unassigned) > 0
            revisit = []
            for i in unassigned
                if isnothing(lbs[i]) || count(x->x in lbs[i], assigned) == length(lbs[i])
                    push!(assigned, i)
                else 
                    if any(x->i<x, lbs[i]) || any(x->x in lbs[i], revisit)
                        push!(revisit, i)
                    else 
                        push!(assigned, i)
                    end        
                end
            end
            unassigned = revisit
        end
    else 
        assigned = collect(1:length(lbs));
    end
    return assigned
end

function reparameterization(betastar::AbstractVector{T}, lbs::AbstractVector, parameter_order::AbstractVector; buffer_beta = similar(betastar)) where T<:Real
    if all(lbs .== 5000)
        return betastar
    else
        for i in parameter_order
            if isnothing(lbs[i])
                buffer_beta[i] = betastar[i]
            else
                buffer_beta[i] = maximum(buffer_beta[lbs[i]]) + exp(betastar[i])
            end
        end
        return copy(buffer_beta)
    end
end

function ChainRulesCore.rrule(::typeof(reparameterization), betastar::AbstractVector, lbs::AbstractVector, parameter_order::AbstractVector)
    if all(lbs .== 5000)
        trivial_pullback(ȳ) = ChainRulesCore.NoTangent(), ChainRulesCore.unthunk(ȳ), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
        return betastar, trivial_pullback
    else
        beta = similar(betastar)
        jmax = zeros(Int, length(betastar))
        for i in parameter_order
            if isnothing(lbs[i])
                beta[i] = betastar[i]
            else
                am    = argmax(beta[lbs[i]])
                jmax[i] = lbs[i][am]
                beta[i] = beta[jmax[i]] + exp(betastar[i])
            end
        end
        beta_out = copy(beta)
        function ordering_pullback(ȳ)
            ȳ_work    = copy(ChainRulesCore.unthunk(ȳ))
            ∂betastar = zeros(eltype(betastar), length(betastar))
            for i in Iterators.reverse(parameter_order)
                if isnothing(lbs[i])
                    ∂betastar[i] = ȳ_work[i]
                else
                    ∂betastar[i]   = ȳ_work[i] * exp(betastar[i])
                    ȳ_work[jmax[i]] += ȳ_work[i]
                end
            end
            return ChainRulesCore.NoTangent(), ∂betastar, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
        end
        return beta_out, ordering_pullback
    end
end

function reparameterization_draws(betastar_draws, lbs, parameter_order)
    nbeta = size(betastar_draws, 2)
    ndraws = size(betastar_draws, 1)
    beta_draws = zeros(eltype(betastar_draws), ndraws,nbeta)
    if all(lbs .== 5000) || (lbs == [])
        beta_draws .= betastar_draws;
    else
        for r in 1:ndraws
            for i in parameter_order
                if isnothing(lbs[i])
                    beta_draws[r,i] = betastar_draws[r,i]
                else
                    beta_draws[r,i] = findmax(beta_draws[r, lbs[i]])[1] + exp(betastar_draws[r,i])
                end
            end
        end
    end
    return beta_draws
end

function map_to_sieve(beta::AbstractArray{T}, gamma::AbstractArray{T}, exchange::Vector, 
    nbetas::Vector{Int64}, problem::NPDemand.NPDProblem; sieve_type = "bernstein") where T
    
    if sieve_type == "polynomial"
        # –– build product‐wise index (one block of size=size(Xvec[j],2))
        J            = length(problem.Xvec)
        counts       = size.(problem.Xvec,2)
        starts_prod  = [1; cumsum(counts)[1:end-1] .+ 1]
        ends_prod    = cumsum(counts)
        # –– build group‐wise index in β
        starts_grp   = [1; cumsum(nbetas)[1:end-1] .+ 1]
        ends_grp     = cumsum(nbetas)
        # –– fill each product j with its group’s β‐block
        sieve_params = zeros(T, sum(counts))
        for j in 1:J
            g = findfirst(x->j in x, exchange)
            sieve_params[starts_prod[j]:ends_prod[j]] .=
                beta[starts_grp[g]:ends_grp[g]]
        end
        # –– append constant and γ, reshape as before
        allp = [sieve_params; 1.0; gamma]
        return reshape(allp, 1, length(allp))
    else
        nbeta = sum(nbetas); # number of parameters in each unique sieve
        J = length(problem.Xvec);

        # indexes in sieve space 
        starts_sieve  = [1;cumsum(size.(problem.Xvec,2))[1:end-1] .+ 1]; 
        ends_sieve    = cumsum(size.(problem.Xvec,2))

        # indexes in base parameter space
        starts_params = [1;cumsum(nbetas)[1:end-1] .+ 1];
        ends_params   = cumsum(nbetas);

        # Transform 
        # sieve_params = Zygote.Buffer(zeros(T, problem.design_width));     
        sieve_params = zeros(T, problem.design_width);
        for j in 1:J
            which_group = findfirst(j .∈  exchange); # find the group corresponding to this product
            sieve_params[starts_sieve[j]:ends_sieve[j]] = beta[starts_params[which_group]:ends_params[which_group]]
        end

        # all_params = [copy(sieve_params); 1.0; gamma];
        all_params = [sieve_params; 1.0; gamma];
        all_params = reshape(all_params, 1, length(all_params));

        return all_params
    end
end

function gmm_fast_blocks(problem::NPDemand.NPDProblem, nbetas)
    J = length(problem.Xvec)
    Xvec, Bvec, df = problem.Xvec, problem.Bvec, problem.data
    augmented_X = [hcat(Xvec[i], -1 .* Bvec[i][:,2:end]) for i in 1:J]
    yZX = [(-1 .* df[!,"prices$i"])' * problem.weight_matrices[i+1] * augmented_X[i+1] for i in 0:J-1]
    XX  = [augmented_X[i+1]' * problem.weight_matrices[i+1] * augmented_X[i+1] for i in 0:J-1]
    XZy = [augmented_X[i+1]' * problem.weight_matrices[i+1]' * (-1 .* df[!,"prices$i"]) for i in 0:J-1]
    nθ  = size.(Xvec, 2)
    yZX_γ = [copy(parent(yZX[i])[nθ[i]+1:end]) for i in 1:J]
    XZy_γ = [XZy[i][nθ[i]+1:end] for i in 1:J]
    XX_γγ = [XX[i][nθ[i]+1:end, nθ[i]+1:end] for i in 1:J]
    return Dict(
        "yZX_β" => [copy(parent(yZX[i])[1:nθ[i]]) for i in 1:J],
        "XZy_β" => [XZy[i][1:nθ[i]] for i in 1:J],
        "XX_ββ" => [XX[i][1:nθ[i], 1:nθ[i]] for i in 1:J],
        "XX_βγ" => [XX[i][1:nθ[i], nθ[i]+1:end] for i in 1:J],
        "yZX_γ_sum" => sum(yZX_γ),
        "XZy_γ_sum" => sum(XZy_γ),
        "XX_γγ_sum" => sum(XX_γγ),
        "starts_params_v2" => [1; cumsum(nbetas)[1:end-1] .+ 1],
        "ends_params_v2" => Vector(cumsum(nbetas)),
        "group_for_product" => [findfirst(j .∈ problem.exchange) for j in 1:J])
end

function pick_step_size(problem, prior, tempmats, bigA; target = 0.2, n_samples = 100)
    step_grid = collect(range(0.001, 3, length = 100)) ./ sqrt(problem.design_width)
    accept = [];
    for x in step_grid
        step = x;
        _nbetas = prior["nbetas"];
        _nγ     = size(problem.Bvec[1],2) - 1;
        chain = Turing.sample(sample_quasibayes(problem, prior, tempmats, bigA), MH(
            :z_gamma => AdvancedMH.RandomWalkProposal(MvNormal(zeros(_nγ),          Diagonal(fill(step, _nγ)))),
            :z_beta  => AdvancedMH.RandomWalkProposal(MvNormal(zeros(sum(_nbetas)), Diagonal(fill(step, sum(_nbetas)))))
            ), n_samples, chain_type = MCMCChains.Chains, discard_initial = 1);
        push!(accept, mean(chain["z_beta[1]"][2:end,:] - chain["z_beta[1]"][1:(end-1),:] .!= 0))
    end
    return step_grid[findmin(abs.(accept .- target))[2]], step_grid, accept
end

@model function sample_quasibayes(problem::NPDemand.NPDProblem, 
    prior::Dict, 
    tempmats::Matrix{Matrix{T}}=[], 
    weight_matrices::Vector{Matrix{T}}=[],
    prices::Matrix{T} = Matrix(problem.data[!,r"prices"]), 
    shares::Matrix{T} = Matrix(problem.data[:, r"shares"]); 
    penalty = 0,
    matrix_storage_dict = Dict(), 
    sieve_type = "bernstein") where T <: Real
    
    # prior
    betabar         = prior["betabar"]
    gammabar        = prior["gammabar"]
    vbeta           = prior["vbeta"]
    vgamma          = prior["vgamma"]    
    lbs             = prior["lbs"]
    parameter_order = prior["parameter_order"]
    nbetas          = prior["nbetas"]
    
    gamma_length::Int = size(problem.Bvec[1],2);

    # Non-centered parameterization: sample standardized noise, reconstruct parameters.
    # All latent variables have unit prior variance — better HMC geometry when
    # polynomial coefficients have very different scales.
    z_beta  ~ filldist(Normal(0, 1), length(betabar))
    z_gamma ~ filldist(Normal(0, 1), gamma_length-1)
    betastar = betabar .+ sqrt.(vbeta) .* z_beta
    gamma    = gammabar .+ sqrt(vgamma) .* z_gamma

    # Apply reparameterization (uses the kwarg sieve_type, then overwrite from prior for map_to_sieve)
    beta = sieve_type == "bernstein" ? reparameterization(betastar, lbs, parameter_order) : betastar;
    sieve_type = get(prior, "sieve_type", "bernstein")

    use_v2 = matrix_storage_dict != Dict()

    if !((penalty == 0) | (problem.constraints == [:exchangeability]))
        # Penalty branch: map_to_sieve still needed for elast_mat_zygote
        all_params = map_to_sieve(beta, gamma, problem.exchange, nbetas, problem; sieve_type = sieve_type)
        J = length(problem.Xvec);
        elasts = elast_mat_zygote(all_params, problem, tempmats; at = prices, s = shares);
        reshaped_elasts = [elasts[i][j1,j2] for j1=1:J, j2=1:J, i=1:size(problem.data,1)];
        elasticity_check = run_elasticity_check(reshaped_elasts, problem.constraints, problem.exchange)
        loglik = use_v2 ?
            -0.5 * gmm_fast_v2(beta, gamma,
                matrix_storage_dict["yZX_β"], matrix_storage_dict["XZy_β"],
                matrix_storage_dict["XX_ββ"], matrix_storage_dict["XX_βγ"],
                matrix_storage_dict["yZX_γ_sum"], matrix_storage_dict["XZy_γ_sum"],
                matrix_storage_dict["XX_γγ_sum"],
                matrix_storage_dict["starts_params_v2"], matrix_storage_dict["ends_params_v2"],
                matrix_storage_dict["group_for_product"], length(problem.Avec)) :
            -0.5 * gmm(all_params, problem, weight_matrices)
        Turing.@addlogprob! (elasticity_check[1] ? loglik : loglik - penalty)
        return
    else
        # No-penalty branch: skip map_to_sieve entirely when partitioned matrices are available
        if use_v2
            Turing.@addlogprob! -0.5 * gmm_fast_v2(beta, gamma,
                matrix_storage_dict["yZX_β"], matrix_storage_dict["XZy_β"],
                matrix_storage_dict["XX_ββ"], matrix_storage_dict["XX_βγ"],
                matrix_storage_dict["yZX_γ_sum"], matrix_storage_dict["XZy_γ_sum"],
                matrix_storage_dict["XX_γγ_sum"],
                matrix_storage_dict["starts_params_v2"], matrix_storage_dict["ends_params_v2"],
                matrix_storage_dict["group_for_product"], length(problem.Avec))
        else
            all_params = map_to_sieve(beta, gamma, problem.exchange, nbetas, problem; sieve_type = sieve_type)
            Turing.@addlogprob! -0.5 * gmm(all_params, problem, weight_matrices)
        end
        return
    end
end

function posterior_elasticities(j, k, betadraws, gammadraws, tempmats, problem)
    ndraws = min(size(betadraws,1), 1_000);
    tmpout = zeros(eltype(betadraws), size(problem.data,1), ndraws)
    for i in 1:ndraws
        st       = problem.approximation_details[:sieve_type]
        params_i = map_to_sieve(betadraws[i,:], gammadraws[i,:],
                            problem.exchange, nbetas, problem;
                            sieve_type=st)
        tmpout[:,i] = getindex.(elast_mat_zygote(params_i, problem, tempmats; 
            at = Matrix(problem.data[!,r"prices"]), s = Matrix(problem.data[!,r"shares"])), j,k); 
    end
    return tmpout
end


function find_starting_point(problem, prior, 
    tempmats, weight_matrices; 
    n_attempts = 1000)
    
    prior_chain = Turing.sample(
        sample_quasibayes(
            problem, 
            prior, 
            tempmats, 
            weight_matrices), 
        Prior(), 
        n_attempts,
        chain_type = MCMCChains.Chains,
        discard_initial = 1)

    nbetas          = prior["nbetas"];
    lbs             = prior["lbs"];
    parameter_order = prior["parameter_order"];
    gamma_length    = size(problem.Bvec[1],2);

    betabar         = prior["betabar"]
    gammabar        = prior["gammabar"]
    vbeta           = prior["vbeta"]
    vgamma          = prior["vgamma"]
    z_betadraws     = hcat([prior_chain["z_beta[$i]"]  for i in 1:sum(nbetas)]...)
    z_gammadraws    = hcat([prior_chain["z_gamma[$i]"] for i in 1:gamma_length-1]...)
    betastardraws   = betabar' .+ sqrt.(vbeta') .* z_betadraws
    gammadraws      = gammabar' .+ sqrt(vgamma) .* z_gammadraws
    betadraws       = reparameterization_draws(betastardraws, lbs, parameter_order)
    J               = length(problem.Xvec);

    # Initialize output
    param_out       = (z_beta = zeros(sum(nbetas)), z_gamma = zeros(gamma_length-1))

    # Start looping over samples parameters
    i = 1;
    constraints_satisfied = false
    while (i < n_attempts) && !(constraints_satisfied)
        st           = problem.approximation_details[:sieve_type]
        sieve_params = map_to_sieve(betadraws[i,:], gammadraws[i,:],
                               problem.exchange, prior["nbetas"], problem;
                               sieve_type=st)
        
        # Calculate elasticities at these parameters 
        elasts_i = elast_mat_zygote(sieve_params, problem, tempmats; 
            at = Matrix(problem.data[!,r"prices"]), s = Matrix(problem.data[!,r"shares"]));
        elasts_i = [elasts_i[i][j1,j2] for j1=1:J, j2 = 1:J, i = 1:size(problem.data,1)];
        
        # Then check if constraints are satisfied
        constraints_satisfied = run_elasticity_check(elasts_i, problem.constraints, problem.exchange);

        i+=1;
    end

    if constraints_satisfied
        exit_flag = "success";
        param_out = (z_beta = z_betadraws[i-1,:], z_gamma = z_gammadraws[i-1,:])
    else
        exit_flag = "failed";
    end
    
    return param_out, exit_flag
end

# Leapfrog HMC using fully analytical gradients — no AD overhead.
# For the quasi-Bayes GMM log-posterior (quadratic in parameters, Gaussian NCP prior)
# the gradient is a closed-form linear expression; no Turing model machinery needed.
function analytical_hmc(prior::Dict, msd::Dict, J::Int;
    n_samples::Int      = 1000,
    step_size::Real     = 0.01,
    n_leapfrog::Int     = 10,
    n_adapt::Int        = 0,        # warm-up steps for dual-averaging (0 = off)
    target_accept::Real = 0.8,      # target acceptance rate during adaptation
    adapt_L::Bool       = true,     # also adjust L to keep trajectory length ~ 1
    z_init              = nothing,
    burn_in::Int        = 0,
    seed::Union{Int,Nothing} = nothing,
    verbose::Bool       = true)

    betabar  = prior["betabar"];  vbeta    = prior["vbeta"]
    gammabar = prior["gammabar"]; vgamma   = prior["vgamma"]
    lbs      = prior["lbs"];      parameter_order = prior["parameter_order"]
    lbs_trivial = all(lbs .== 5000)

    nbeta  = length(betabar);   ngamma = length(gammabar)
    n      = nbeta + ngamma

    sqrt_vbeta  = sqrt.(vbeta);  sqrt_vgamma = sqrt(vgamma)

    yZX_β     = msd["yZX_β"];    XZy_β     = msd["XZy_β"]
    XX_ββ     = msd["XX_ββ"];    XX_βγ     = msd["XX_βγ"]
    yZX_γ_sum = msd["yZX_γ_sum"]; XZy_γ_sum = msd["XZy_γ_sum"]
    XX_γγ_sum = msd["XX_γγ_sum"]
    starts    = msd["starts_params_v2"]; ends = msd["ends_params_v2"]
    gfp       = msd["group_for_product"]

    # Per-group beta sizes; pre-allocate inner-loop buffers once
    nbeta_per_g = ends .- starts .+ 1
    nb_max      = maximum(nbeta_per_g)
    buf_ββ  = zeros(nb_max)  # XX_ββ[i] * β_i (size nbeta_i)
    buf_βγ  = zeros(nb_max)  # XX_βγ[i] * gamma (size nbeta_i)
    buf_γ   = zeros(ngamma)  # XX_βγ[i]' * β_i and XX_γγ_sum * gamma (size ngamma)
    beta    = zeros(nbeta);   gamma  = zeros(ngamma)
    ∂beta   = zeros(nbeta);   ∂gamma = zeros(ngamma)

    # Computes log-posterior and fills grad in-place.  Zero heap allocations for lbs_trivial.
    function logpost_grad!(grad, z)
        z_β = @view z[1:nbeta];  z_γ = @view z[nbeta+1:end]

        @. beta  = betabar + sqrt_vbeta * z_β
        @. gamma = gammabar + sqrt_vgamma * z_γ

        repar_pb = nothing
        if !lbs_trivial
            betastar = copy(beta)
            beta_out, repar_pb = ChainRulesCore.rrule(reparameterization, betastar, lbs, parameter_order)
            beta .= beta_out
        end

        # γ-only quadratic terms
        mul!(buf_γ, XX_γγ_sum, gamma)
        val_γ = dot(gamma, buf_γ)
        @. ∂gamma = -yZX_γ_sum - XZy_γ_sum + 2*buf_γ
        val = -dot(yZX_γ_sum, gamma) - dot(gamma, XZy_γ_sum) + val_γ
        fill!(∂beta, 0)

        # Per-product terms
        for i in 1:J
            g   = gfp[i]
            nb  = nbeta_per_g[g]
            β_i = @view beta[starts[g]:ends[g]]
            ∂β_g = @view ∂beta[starts[g]:ends[g]]
            ββ_v = @view buf_ββ[1:nb]
            βγ_v = @view buf_βγ[1:nb]

            mul!(ββ_v, XX_ββ[i], β_i)
            mul!(βγ_v, XX_βγ[i], gamma)

            @. ∂β_g += -yZX_β[i] - XZy_β[i] + 2*ββ_v + 2*βγ_v
            val    += -dot(yZX_β[i], β_i) - dot(β_i, XZy_β[i]) +
                       dot(β_i, ββ_v) + 2*dot(β_i, βγ_v)

            mul!(buf_γ, XX_βγ[i]', β_i)
            @. ∂gamma += 2*buf_γ
        end

        # Chain rule through reparameterization then NCP
        if lbs_trivial
            @. grad[1:nbeta]    = -z_β - 0.5*∂beta*sqrt_vbeta
        else
            _, ∂betastar, _, _ = repar_pb(∂beta)
            @. grad[1:nbeta]    = -z_β - 0.5*∂betastar*sqrt_vbeta
        end
        @. grad[nbeta+1:end] = -z_γ - 0.5*∂gamma*sqrt_vgamma

        return -0.5*(dot(z_β, z_β) + dot(z_γ, z_γ)) - 0.5*val
    end

    # --- Leapfrog HMC with optional dual-averaging adaptation ---
    rng      = isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed)
    n_store  = n_samples + burn_in
    samples  = Matrix{Float64}(undef, n_store, n)
    z        = isnothing(z_init) ? zeros(n) : float(vec(z_init))[1:n]
    z_prop   = similar(z);  grad = zeros(n);  grad_prop = zeros(n);  p = similar(z)

    logp      = logpost_grad!(grad, z)
    n_accept  = 0
    ε         = float(step_size)
    L         = n_leapfrog
    τ_target  = ε * L           # fixed trajectory length; L adapts to preserve this

    # Nesterov dual-averaging state (Hoffman & Gelman 2014, Algorithm 5)
    μ       = log(10 * ε)      # log of 10× initial step (target for averaging)
    H̄       = 0.0              # running mean of (target_accept - α)
    log_ε̄   = log(ε)           # ergodic mean of log ε (used after adaptation)
    t₀      = 10;  γ_da = 0.05;  κ = 0.75

    iter = verbose ? ProgressBar(1:n_store) : 1:n_store
    for s in iter
        randn!(rng, p)
        H_old = -logp + 0.5*dot(p, p)
        z_prop .= z;  grad_prop .= grad
        logp_prop = logp

        # Leapfrog
        p .+= 0.5 .* ε .* grad_prop
        for l in 1:L
            z_prop    .+= ε .* p
            logp_prop  = logpost_grad!(grad_prop, z_prop)
            p .+= (l < L ? ε : 0.5*ε) .* grad_prop
        end

        H_new = -logp_prop + 0.5*dot(p, p)
        # Treat NaN/Inf proposals as hard rejections so α=0 drives ε down
        α = (isfinite(H_new) && isfinite(logp_prop)) ? min(1.0, exp(H_old - H_new)) : 0.0
        if log(rand(rng)) < log(α + eps())
            z .= z_prop;  grad .= grad_prop
            logp = logp_prop;  n_accept += 1
        end
        samples[s, :] .= z

        # Dual-averaging step-size adaptation during warm-up
        if n_adapt > 0 && s <= n_adapt
            m     = float(s)
            H̄     = (1 - 1/(m + t₀)) * H̄ + (1/(m + t₀)) * (target_accept - α)
            log_ε = μ - (sqrt(m) / γ_da) * H̄
            log_ε̄ = m^(-κ) * log_ε + (1 - m^(-κ)) * log_ε̄
            ε     = clamp(exp(log_ε), 1e-6, 10.0)
            adapt_L && (L = clamp(round(Int, τ_target / ε), 1, 200))
        elseif n_adapt > 0 && s == n_adapt + 1
            ε = clamp(exp(log_ε̄), 1e-6, 10.0)
            adapt_L && (L = clamp(round(Int, τ_target / ε), 1, 200))
        end

        if verbose
            phase = (n_adapt > 0 && s <= n_adapt) ? "adapt" : "sample"
            set_description(iter, @sprintf("HMC [%s] ε=%.3g L=%d acc=%.0f%%",
                phase, ε, L, n_accept/s*100))
        end
    end

    verbose && @info "analytical_hmc acceptance rate: $(round(n_accept/n_store*100, digits=1))%"

    param_names = vcat(
        [Symbol("z_beta[$i]")  for i in 1:nbeta],
        [Symbol("z_gamma[$i]") for i in 1:ngamma])
    return MCMCChains.Chains(
        reshape(samples, n_store, n, 1),
        param_names)
end

function smc_logtarget_z(z, problem::NPDemand.NPDProblem, nbetas, lbs, parameter_order,
        beta_mean, sqrt_vbeta, gamma_mean, sqrt_vgamma, gmm_loglike,
        penalty::Real, smoothness::Real, sieve_type, prices, shares)
    nbeta = length(beta_mean)
    z_beta = @view z[1:nbeta]
    z_gamma = @view z[(nbeta+1):length(z)]
    betastar = beta_mean .+ sqrt_vbeta .* z_beta
    gamma = gamma_mean .+ sqrt_vgamma .* z_gamma
    beta = sieve_type == "bernstein" ? reparameterization(betastar, lbs, parameter_order) : betastar
    sieve = map_to_sieve(beta, gamma, problem.exchange, nbetas, problem; sieve_type = sieve_type)
    return gmm_loglike(beta, gamma) - 0.5 * dot(z, z) +
           logpenalty_smc(smooth_constraint_distances(sieve, problem, prices, shares; smoothness = smoothness), penalty)
end

function smc_hmc_rejuvenate!(z, logtarget_grad!, rng, n_steps::Int, step_size::Real, n_leapfrog::Int)
    grad = similar(z)
    grad_prop = similar(z)
    z_prop = similar(z)
    momentum = similar(z)
    logp = logtarget_grad!(grad, z)
    isfinite(logp) || return logp, 0

    n_accept = 0
    for _ in 1:n_steps
        randn!(rng, momentum)
        h_old = -logp + 0.5 * dot(momentum, momentum)
        z_prop .= z
        grad_prop .= grad
        logp_prop = logp

        momentum .+= 0.5 .* step_size .* grad_prop
        valid = true
        for leapfrog_step in 1:n_leapfrog
            z_prop .+= step_size .* momentum
            logp_prop = logtarget_grad!(grad_prop, z_prop)
            if !isfinite(logp_prop)
                valid = false
                break
            end
            momentum .+= (leapfrog_step < n_leapfrog ? step_size : 0.5 * step_size) .* grad_prop
        end

        if valid
            h_new = -logp_prop + 0.5 * dot(momentum, momentum)
            log_accept = h_old - h_new
            if isfinite(log_accept) && log(rand(rng)) < log_accept
                z .= z_prop
                grad .= grad_prop
                logp = logp_prop
                n_accept += 1
            end
        end
    end
    return logp, n_accept
end

function smc_z_to_state!(theta_row, sieve_row, z, problem::NPDemand.NPDProblem, nbetas,
        lbs, parameter_order, beta_mean, sqrt_vbeta, gamma_mean, sqrt_vgamma, sieve_type)
    nbeta = length(beta_mean)
    @views begin
        theta_row[1:nbeta] .= beta_mean .+ sqrt_vbeta .* z[1:nbeta]
        theta_row[(nbeta+1):length(theta_row)] .= gamma_mean .+ sqrt_vgamma .* z[(nbeta+1):length(z)]
        beta = sieve_type == "bernstein" ? reparameterization(theta_row[1:nbeta], lbs, parameter_order) : theta_row[1:nbeta]
        sieve_row .= vec(map_to_sieve(beta, theta_row[(nbeta+1):length(theta_row)], problem.exchange, nbetas, problem; sieve_type = sieve_type))
    end
    return nothing
end

function smc_gradient_workspace(problem::NPDemand.NPDProblem, nbeta::Int, ngamma::Int, nsieve::Int)
    J = length(problem.Xvec)
    return (;
        betastar = zeros(nbeta),
        gamma = zeros(ngamma),
        grad_beta = zeros(nbeta),
        grad_gamma = zeros(ngamma),
        sieve = zeros(nsieve),
        grad_sieve = zeros(nsieve),
        inverse_derivative = zeros(J, J),
        jacobian_inverse = zeros(J, J),
        grad_jacobian = zeros(J, J),
        grad_inverse = zeros(J, J),
        grad_A = zeros(J, J),
        tmp_A = zeros(J, J))
end

function gmm_loglike_grad!(grad_beta, grad_gamma, beta, gamma,
        yZX_β, XZy_β, XX_ββ, XX_βγ,
        yZX_γ_sum, XZy_γ_sum, XX_γγ_sum,
        starts_params, ends_params, group_for_product, J;
        workspace = nothing)

    value, pullback = ChainRulesCore.rrule(gmm_fast_v2, beta, gamma,
        yZX_β, XZy_β, XX_ββ, XX_βγ,
        yZX_γ_sum, XZy_γ_sum, XX_γγ_sum,
        starts_params, ends_params, group_for_product, J)
    partials = pullback(-0.5)
    grad_beta .= partials[2]
    grad_gamma .= partials[3]
    return -0.5 * value
end

function add_sieve_gradient_to_beta!(grad_beta, grad_sieve, problem::NPDemand.NPDProblem, nbetas, group_for_product)
    starts_sieve = [1; cumsum(size.(problem.Xvec, 2))[1:end-1] .+ 1]
    ends_sieve = cumsum(size.(problem.Xvec, 2))
    starts_params = [1; cumsum(nbetas)[1:end-1] .+ 1]
    ends_params = cumsum(nbetas)

    for product in 1:length(problem.Xvec)
        group = group_for_product[product]
        @views grad_beta[starts_params[group]:ends_params[group]] .+=
            grad_sieve[starts_sieve[product]:ends_sieve[product]]
    end
    return grad_beta
end

function smc_logtarget_grad!(grad_z, z, problem::NPDemand.NPDProblem, nbetas, lbs, parameter_order,
        beta_mean, sqrt_vbeta, gamma_mean, sqrt_vgamma,
        yZX_β, XZy_β, XX_ββ, XX_βγ,
        yZX_γ_sum, XZy_γ_sum, XX_γγ_sum,
        starts_params, ends_params, group_for_product, J,
        penalty::Real, smoothness::Real, sieve_type, prices, shares,
        workspace = nothing)

    nbeta = length(beta_mean)
    z_beta = @view z[1:nbeta]
    z_gamma = @view z[(nbeta+1):length(z)]
    if workspace === nothing
        betastar = beta_mean .+ sqrt_vbeta .* z_beta
        gamma = gamma_mean .+ sqrt_vgamma .* z_gamma
    else
        betastar = workspace.betastar
        gamma = workspace.gamma
        @. betastar = beta_mean + sqrt_vbeta * z_beta
        @. gamma = gamma_mean + sqrt_vgamma * z_gamma
    end

    beta = betastar
    reparameterization_pullback = nothing
    if sieve_type == "bernstein"
        beta, reparameterization_pullback = ChainRulesCore.rrule(reparameterization, betastar, lbs, parameter_order)
    end

    grad_beta = workspace === nothing ? zeros(eltype(z), length(beta)) : workspace.grad_beta
    grad_gamma = workspace === nothing ? zeros(eltype(z), length(gamma)) : workspace.grad_gamma
    value = gmm_loglike_grad!(grad_beta, grad_gamma, beta, gamma,
        yZX_β, XZy_β, XX_ββ, XX_βγ,
        yZX_γ_sum, XZy_γ_sum, XX_γγ_sum,
        starts_params, ends_params, group_for_product, J;
        workspace = workspace)

    if workspace === nothing
        sieve = vec(map_to_sieve(beta, gamma, problem.exchange, nbetas, problem; sieve_type = sieve_type))
        grad_sieve = zeros(eltype(z), length(sieve))
    else
        sieve = workspace.sieve
        sieve .= vec(map_to_sieve(beta, gamma, problem.exchange, nbetas, problem; sieve_type = sieve_type))
        grad_sieve = workspace.grad_sieve
    end
    logpenalty = smooth_logpenalty_grad!(grad_sieve, sieve, problem, prices, shares, penalty;
        smoothness = smoothness, workspace = workspace)
    if !isfinite(logpenalty)
        fill!(grad_z, 0)
        return -Inf
    end
    value += logpenalty - 0.5 * dot(z, z)

    add_sieve_gradient_to_beta!(grad_beta, grad_sieve, problem, nbetas, group_for_product)
    grad_betastar = grad_beta
    if reparameterization_pullback !== nothing
        _, grad_betastar, _, _ = reparameterization_pullback(grad_beta)
    end

    @views begin
        grad_z[1:nbeta] .= -z_beta .+ sqrt_vbeta .* grad_betastar
        grad_z[(nbeta+1):length(z)] .= -z_gamma .+ sqrt_vgamma .* grad_gamma
    end
    return value
end

function smc(problem::NPDemand.NPDProblem;
    grid_points::Int    = 50, 
    max_penalty::Real   = 5, 
    ess_threshold::Real = 100, 
    step_size::Real     = 0.1, 
    skip::Int           = 5,
    burn_in::Int        = 5000, 
    mh_steps            = 10,
    smc_method          = :grid, 
    seed                = 4132, 
    max_iter            = 1000, 
    adaptive_tolerance  = false, 
    max_violations      = 0.01,
    modulo_num          = 1, 
    smc_kernel          = :mh,
    penalty_type        = smc_kernel == :hmc ? :smooth : :count,
    hmc_step_size::Real = step_size,
    hmc_leapfrog_steps::Int = 5,
    smoothness::Real    = 20.0,
    approximation_details::Dict{Symbol, Any} = Dict()
    )

    # Define inputs to quasi-bayes sampling 
    prior           = problem.sampling_details.prior;
    lbs             = prior["lbs"]
    parameter_order = prior["parameter_order"]
    nbetas          = prior["nbetas"]
    gamma_length    = size(problem.Bvec[1],2);

    # 1. Run MCMC on unconstrained model
    start_row       = burn_in+1;
    skiplen         = skip;

    particles       = problem.chain;
    nbetas          = NPDemand.get_nbetas(problem);
    nbeta           = length(lbs) == 0 ? sum(nbetas) : length(lbs);
    
    _betabar        = prior["betabar"]
    _gammabar       = prior["gammabar"]
    _vbeta          = prior["vbeta"]
    _vgamma         = prior["vgamma"]
    z_betadraws     = hcat([particles["z_beta[$i]"]  for i in 1:sum(nbetas)]...)[start_row:end,:]
    z_gammadraws    = hcat([particles["z_gamma[$i]"] for i in 1:gamma_length-1]...)[start_row:end,:]
    betastardraws   = _betabar' .+ sqrt.(_vbeta') .* z_betadraws
    gammadraws      = _gammabar' .+ sqrt(_vgamma) .* z_gammadraws
    betadraws       = NPDemand.reparameterization_draws(betastardraws, lbs, parameter_order)

    # thin the markov chain
    L               = size(betastardraws,1);
    skip_inds       = 1:skiplen:L
    betadraws       = betadraws[skip_inds,:];
    gammadraws      = gammadraws[skip_inds,:];
    betastardraws   = betastardraws[skip_inds,:];
    nparticles      = size(betastardraws,1);

    # Format parameter vec so that gmm can use it
    thetas          = [betastardraws gammadraws]
    # betas           = NPDemand.reparameterization_draws(
    #     thetas[:,1:nbeta], 
    #     lbs, 
    #     parameter_order
    #     )
    st            = approximation_details[:sieve_type]
    thetas_sieve  = vcat([map_to_sieve(
                            betadraws[i,:], 
                            gammadraws[i,:],
                            problem.exchange, 
                            nbetas, 
                            problem;
                            sieve_type=st)
                         for i in 1:nparticles]...)
    matrix_storage_dict = gmm_fast_blocks(problem, nbetas)
    nproducts = length(problem.Avec)
    yZX_β, XZy_β = matrix_storage_dict["yZX_β"], matrix_storage_dict["XZy_β"]
    XX_ββ, XX_βγ = matrix_storage_dict["XX_ββ"], matrix_storage_dict["XX_βγ"]
    yZX_γ_sum, XZy_γ_sum = matrix_storage_dict["yZX_γ_sum"], matrix_storage_dict["XZy_γ_sum"]
    XX_γγ_sum = matrix_storage_dict["XX_γγ_sum"]
    starts_params = matrix_storage_dict["starts_params_v2"]
    ends_params = matrix_storage_dict["ends_params_v2"]
    group_for_product = matrix_storage_dict["group_for_product"]
    gmm_loglike(beta, gamma) = -0.5 * gmm_fast_v2(beta, gamma, yZX_β, XZy_β, XX_ββ, XX_βγ,
        yZX_γ_sum, XZy_γ_sum, XX_γγ_sum, starts_params, ends_params, group_for_product, nproducts)
    smc_prices = Matrix(problem.data[!,r"prices"])
    smc_shares = Matrix(problem.data[!,r"shares"])
    
    # 2. Set initial weights
    smc_weights     = fill(1.0 / nparticles, nparticles)
    penalty         = range(0, max_penalty, length = grid_points);

    # violation_dict = report_constraint_violations(problem, 
    #         params = mean(thetas_sieve, StatsBase.weights(smc_weights), dims = 1))
    violation_dict_array = [
        report_constraint_violations_inner(problem, 
            params = thetas_sieve[i,:], verbose = false) for i in axes(thetas_sieve,1)
    ];
    violation_dict = Dict{Symbol, Float64}()
    for k in keys(violation_dict_array[1])
        push!(violation_dict, k => mean([violation_dict_array[i][k] for i in axes(thetas_sieve,1)]));
    end
    t = 1;

    viol_store = [];
    ess_store  = [];
    penalty_vec = [];
    
    Random.seed!(seed)
    prev_penalty = 0.01;
    new_penalty  = 1e-6;
    x_pen = range(1e-3, Float64.(max_penalty), length = Int(grid_points)); 

    if smc_method == :linear_grid
        penalty_list = x_pen;
    elseif smc_method == :logit_grid
        penalty_list = maximum(x_pen) ./ (1 .+ exp.(-2 .*(x_pen .- median(x_pen))));
    elseif smc_method == :geometric_grid
        penalty_list = geometric_grid(1e-3, Float64.(max_penalty), Int(grid_points));
    end

    t = 1;
    beta_μ, gamma_μ = _betabar, _gammabar
    beta_inv_var, beta_logdet = inv.(_vbeta), sum(log.(_vbeta))
    gamma_inv_var, gamma_logdet = fill(inv(_vgamma), gamma_length-1), (gamma_length-1) * log(_vgamma)
    sqrt_vbeta, sqrt_vgamma = sqrt.(_vbeta), sqrt(_vgamma)
    n_kernel_steps = Int(mh_steps)
    gradient_workspaces = smc_kernel == :hmc ?
        [smc_gradient_workspace(problem, nbeta, length(gamma_μ), size(thetas_sieve, 2)) for _ in 1:Threads.maxthreadid()] :
        nothing
    failure_count = 0;
    while (violation_dict[:any] > max_violations) & (prev_penalty < max_penalty) & (t < max_iter)
        t = t+1
        print("\n Iteration "*string(t-1)*"...\r")
        penalty_distances = smc_penalty_distances(thetas_sieve, problem;
            multithread = true, penalty_type = penalty_type, smoothness = smoothness)
        current_logprior = zeros(eltype(thetas_sieve), nparticles)

        if (smc_method == :adaptive) & (mod(t,modulo_num) == 0)
            # Solve for next step penalty
            print("\n Optimizing penalty... \r")
            # try 
                _unused, current_logprior = get_importance_weights(thetas_sieve, smc_weights, Float64(prev_penalty), Float64(prev_penalty), problem,
                    penalty_distances = penalty_distances, penalty_type = penalty_type, smoothness = smoothness);
                if adaptive_tolerance 
                    new_penalty = find_zero(x -> f_ess(x, thetas_sieve, smc_weights, prev_penalty, problem, ess_threshold,
                        logprior_t_minus_1 = current_logprior, penalty_distances = penalty_distances,
                        penalty_type = penalty_type, smoothness = smoothness),
                        (prev_penalty, max_penalty), Bisection(); xatol = get_tolerance(prev_penalty))
                else
                    custom_ub = max(prev_penalty * 10.0, prev_penalty + 0.01);
                    xatol = 0.0002;
                    try 
                        new_penalty = find_zero(x -> f_ess(x, thetas_sieve, smc_weights, prev_penalty, problem, ess_threshold,
                            logprior_t_minus_1 = current_logprior, penalty_distances = penalty_distances,
                            penalty_type = penalty_type, smoothness = smoothness),
                            (prev_penalty, min(custom_ub, max_penalty)), Bisection(); xatol = xatol, verbose = false)
                        new_penalty = new_penalty - xatol;
                    catch 
                        new_penalty = min(custom_ub, max_penalty);
                    end
                    if abs(prev_penalty - new_penalty) > 2 * xatol # adjustment to make sure tolerance isn't an issue
                        new_penalty = new_penalty - xatol;
                    end
                end
            # catch
                # error("Solving for penalty failed. Try reducing `ess_threshold` or increasing the number of particles (by reducing `burn_in` or `skip`)")
            # end
        elseif (smc_method == :adaptive) # every other iteration, use the previous penalty and re-sample again
            new_penalty = prev_penalty;
        else
            new_penalty = penalty_list[t]
        end
        
        # Calculate normalized importance weights
        log_smc_weights, _logprior_t = get_importance_weights(thetas_sieve, smc_weights, prev_penalty, new_penalty, problem,
            logprior_t_minus_1 = current_logprior, penalty_distances = penalty_distances,
            penalty_type = penalty_type, smoothness = smoothness)

        smc_weights .= exp.(log_smc_weights .- maximum(log_smc_weights))
        smc_weights .= smc_weights ./ sum(smc_weights)
        ess = 1 / sum(smc_weights.^2)
        push!(ess_store, ess)

        # Always resample under adaptive approach
        indices = 1:nparticles;
        try 
            indices = wsample(1:nparticles, smc_weights, nparticles)
        catch
            failure_count +=1;
            if failure_count == 1
                @warn "Resampling failed. Running new MH steps without resampling..."
                indices = 1:nparticles;
            else
                @warn "Resampling failed twice. Exiting..."
                continue
            end
        end

        thetas = thetas[indices,:]
        thetas_sieve = thetas_sieve[indices,:]
        smc_weights .= 1.0 / nparticles

        # 3.4 Perturb particles via MH step(s)
        n_accept = zeros(nparticles);
        proposal_distribution = nothing;
        if smc_kernel == :mh
            try
                Sigma = cov(thetas); # covariance matrix parameters, used to improve proposals
                Sigma = Sigma .* 2.38^2 ./ size(Sigma,1); # scale covariance matrix
                proposal_distribution = MvNormal(zeros(length(thetas[1,:])), step_size .* Sigma);
            catch
                @warn "Covariance matrix is not positive definite. Using identity matrix instead. If using a non-adaptive grid, you may wish to increase `grid_points`."
                Sigma = Diagonal(ones(size(cov(thetas),1)))
                proposal_distribution = MvNormal(zeros(length(thetas[1,:])), step_size .* Sigma);
            end
        end
        
        # One RNG per thread — created before the parallel region so each thread
        # gets an independent, reproducible stream. mh_iter is mixed into the seed
        # so successive MH steps within a particle draw different proposals.
        thread_rngs = [MersenneTwister(seed + tid) for tid in 1:Threads.maxthreadid()]
        reparameterization_bufs = [zeros(eltype(thetas_sieve), sum(nbetas)) for _ in 1:Threads.maxthreadid()]

        prog = Threads.Atomic{Int}(0)

        # Async monitor: reads the atomic counter and overwrites the progress line.
        monitor = @async begin
            while (n_done = prog[]) < nparticles
                print(@sprintf("\r  MH steps: %d/%d (%.0f%%)", n_done, nparticles, 100.0*n_done/nparticles))
                sleep(0.05)
            end
            println(@sprintf("\r  MH steps: %d/%d (100%%)  ", nparticles, nparticles))
        end

        Threads.@threads for i in axes(thetas,1)
            tid  = Threads.threadid()
            rng  = thread_rngs[tid]
            reparameterization_storage = reparameterization_bufs[tid]
            if smc_kernel == :hmc
                gradient_workspace = gradient_workspaces[tid]
                Random.seed!(rng, seed + i)
                z = Vector{Float64}(undef, size(thetas, 2))
                @views begin
                    z[1:nbeta] .= (thetas[i,1:nbeta] .- beta_μ) ./ sqrt_vbeta
                    z[(nbeta+1):length(z)] .= (thetas[i,(nbeta+1):size(thetas,2)] .- gamma_μ) ./ sqrt_vgamma
                end
                logtarget_grad! = (grad_current, z_current) -> smc_logtarget_grad!(grad_current, z_current,
                    problem, nbetas, lbs, parameter_order,
                    beta_μ, sqrt_vbeta, gamma_μ, sqrt_vgamma,
                    yZX_β, XZy_β, XX_ββ, XX_βγ,
                    yZX_γ_sum, XZy_γ_sum, XX_γγ_sum,
                    starts_params, ends_params, group_for_product, nproducts,
                    new_penalty, smoothness, st, smc_prices, smc_shares,
                    gradient_workspace)
                _logp, accepted = smc_hmc_rejuvenate!(z, logtarget_grad!, rng, n_kernel_steps, hmc_step_size, hmc_leapfrog_steps)
                smc_z_to_state!(@view(thetas[i,:]), @view(thetas_sieve[i,:]), z, problem, nbetas,
                    lbs, parameter_order, beta_μ, sqrt_vbeta, gamma_μ, sqrt_vgamma, st)
                n_accept[i] = accepted
            else
                betai_old = NPDemand.reparameterization(thetas[i,1:nbeta], lbs, parameter_order, buffer_beta = reparameterization_storage)
                gamma_old = @view thetas[i,(nbeta+1):size(thetas,2)]
                logprior_old = logprior_smc(thetas[i,1:nbeta], gamma_old, beta_μ, beta_inv_var, beta_logdet, gamma_μ, gamma_inv_var, gamma_logdet) +
                               logpenalty_smc(thetas_sieve[i,:], new_penalty, problem;
                                   penalty_type = penalty_type, smoothness = smoothness,
                                   prices = smc_prices, shares = smc_shares)
                loglike_old = gmm_loglike(betai_old, gamma_old)

                for mh_iter in 1:n_kernel_steps
                    # Re-seed per (particle, step) so each draw is independent
                    Random.seed!(rng, seed + i + mh_iter * nparticles)

                    # Propose new values + reparameterize + map to sieve
                    thetai_new       = thetas[i,:] + rand(rng, proposal_distribution)
                    betai_new        = NPDemand.reparameterization(thetai_new[1:nbeta], lbs, parameter_order, buffer_beta = reparameterization_storage)
                    gamma_new        = @view thetai_new[(nbeta+1):length(thetai_new)]
                    thetai_sieve_new = NPDemand.map_to_sieve(betai_new, gamma_new, problem.exchange, nbetas, problem, sieve_type=st)

                    # Evaluate prior
                    logprior_new     = logprior_smc(thetai_new[1:nbeta], gamma_new, beta_μ, beta_inv_var, beta_logdet, gamma_μ, gamma_inv_var, gamma_logdet) +
                                       logpenalty_smc(thetai_sieve_new, new_penalty, problem;
                                           penalty_type = penalty_type, smoothness = smoothness,
                                           prices = smc_prices, shares = smc_shares)

                    # Evaluate likelihood
                    loglike_new = gmm_loglike(betai_new, gamma_new)

                    # Calculate MH acceptance ratio
                    logratio = loglike_new + logprior_new - loglike_old - logprior_old
                    if log(rand(rng)) < logratio
                        thetas[i,:]         = thetai_new
                        thetas_sieve[i,:]   = thetai_sieve_new
                        logprior_old        = logprior_new
                        loglike_old         = loglike_new
                        n_accept[i]        += 1
                    end
                    GC.safepoint()
                end
            end
            Threads.atomic_add!(prog, 1)
        end
        wait(monitor)
        n_accept = n_accept ./ n_kernel_steps
        accept_rate = round(mean(n_accept), digits = 2);
        # println("\n Average MH acceptance rate: $(accept_rate)")

        # Check constraints 
        violation_dict_array = [report_constraint_violations_inner(problem, params = thetas_sieve[i,:], verbose = false) for i in axes(thetas_sieve,1)];
        violation_dict = Dict{Symbol, Float64}()
        for k in keys(violation_dict_array[1])
            push!(violation_dict, k => 
                round(
                    mean([violation_dict_array[i][k] for i in axes(thetas_sieve,1)]),
                    digits = 3)
            );
        end
        push!(viol_store, violation_dict[:any])

        # Store and update penalty
        push!(penalty_vec, new_penalty)
        prev_penalty = new_penalty;

        println("|--------------------------------|--------|")
        println("| Iteration results              |        |")
        println("|--------------------------------|--------|")
        println(@sprintf("| %-30s | %.4f   |", "Current Penalty", new_penalty))
        if !(isnan(ess) | isinf(ess))
            println(@sprintf("| %-30s | %.2f |", "ESS", Int(floor(ess))))
        else
            println(@sprintf("| %-30s | %.2f |", "ESS", NaN))
        end
        println(@sprintf("| %-30s | %.2f   |", "Average MH Acceptance rate", accept_rate))
        println("| Violations                     |        |")
        for (key, value) in violation_dict
            println(@sprintf("| %-30s | %.3f  |", key, value))
        end
    end

    return (; thetas, smc_weights, violations = viol_store, ess = ess_store, penalties = penalty_vec);
end

function logpdf_mvn(mu::Vector{T}, Sigma::Matrix{T}, theta::Vector{T}) where T<:Real
    n = length(mu)

    # Ensure that Sigma is positive definite
    L = cholesky(Sigma).L
    diff = theta - mu
    quadratic_form = sum((L \ diff) .^ 2)

    logdetSigma = 2.0 * sum(log.(diag(L))) # log determinant of Sigma
    logpdf = -0.5 * (n * log(2 * π) + logdetSigma + quadratic_form)

    return logpdf
end

function logpdf_mvn(mu::Vector, chol::Cholesky, logdet_sigma::Real, theta::AbstractVector)
    diff = theta .- mu
    quadratic_form = sum((chol.L \ diff) .^ 2)
    return -0.5 * (length(mu) * log(2π) + logdet_sigma + quadratic_form)
end

function logpdf_diag_mvn(mu::AbstractVector, inv_var::AbstractVector, logdet_sigma::Real, theta::AbstractVector)
    q = zero(promote_type(eltype(theta), eltype(inv_var)))
    @inbounds for i in eachindex(mu, inv_var, theta)
        d = theta[i] - mu[i]
        q += d * d * inv_var[i]
    end
    return -0.5 * (length(mu) * log(2π) + logdet_sigma + q)
end

function particle_logpenalty(thetas_sieve, penalty_distances, i, penalty, problem;
    penalty_type = :count,
    smoothness::Real = 20.0)
    if penalty_distances === nothing
        return logpenalty_smc(thetas_sieve[i,:], penalty, problem;
            penalty_type = penalty_type, smoothness = smoothness)
    end
    return logpenalty_smc(view(penalty_distances, i, :), penalty)
end

# function f_ess(p::T, thetas_sieve::Matrix{T}, smc_weights::Vector{T}, 
#     prev_penalty::Real, problem::NPDemand.NPDProblem, 
#     ess_threshold::Real;
#     logprior_t_minus_1 = zeros(T, size(thetas_sieve,1))) where T<:Real

#     if sum(smc_weights)==0 
#         ess = 0
#     else 
#         wts, ~ = get_importance_weights(thetas_sieve, smc_weights, prev_penalty, p[1], problem, logprior_t_minus_1 = logprior_t_minus_1)
#         if sum(wts) == 0
#             ess = Inf; # 0
#         else 
#             ess = (1 / sum(wts.^2))
#         end
#     end
#     return ess - ess_threshold
# end

function f_ess(p::T, thetas_sieve::Matrix{T}, smc_weights::Vector{T}, 
    prev_penalty::Real, problem::NPDemand.NPDProblem, 
    ess_threshold::Real;
    logprior_t_minus_1 = zeros(T, size(thetas_sieve,1)),
    penalty_distances = nothing,
    penalty_type = :count,
    smoothness::Real = 20.0) where T

    logwts, _unused = get_importance_weights(thetas_sieve, smc_weights, prev_penalty, p[1],
            problem, 
            logprior_t_minus_1 = logprior_t_minus_1, 
            multithread = true,
            penalty_distances = penalty_distances,
            penalty_type = penalty_type,
            smoothness = smoothness)

    num = 2*maximum(logwts) + 2*log(sum(exp.(logwts .- maximum(logwts))))
    denom = maximum(2*logwts) + log(sum(exp.(2*logwts .- maximum(2*logwts))))
    log_ess = num - denom
    exp_log_ess = exp(log_ess) 
    if isfinite(exp_log_ess)
        return exp_log_ess - ess_threshold
    else
        return -ess_threshold;
    end
end

function get_importance_weights(thetas_sieve::Matrix{T}, smc_weights::Vector{T},
    penalty_prev::Real, penalty_new::Real, problem::NPDemand.NPDProblem;
    new_log_weights     = similar(smc_weights),
    logprior_t          = zeros(T, size(thetas_sieve,1)),
    logprior_t_minus_1  = zeros(T, size(thetas_sieve,1)),
    multithread         = false,
    penalty_distances   = nothing,
    penalty_type        = :count,
    smoothness::Real    = 20.0) where T<:Real

    has_old_logprior = any(!iszero, logprior_t_minus_1)
    if multithread == false 
        for i in axes(thetas_sieve,1)
            logprior_t[i]             = particle_logpenalty(thetas_sieve, penalty_distances, i, penalty_new, problem;
                penalty_type = penalty_type, smoothness = smoothness)
            logprior_old              = has_old_logprior ? logprior_t_minus_1[i] : particle_logpenalty(thetas_sieve, penalty_distances, i, penalty_prev, problem;
                penalty_type = penalty_type, smoothness = smoothness)
            log_prior_ratio           = logprior_t[i] - logprior_old
            new_log_weights[i]            = log(smc_weights[i]) + log_prior_ratio;
        end
    else
        Threads.@threads for i in axes(thetas_sieve,1)
            logprior_t[i]             = particle_logpenalty(thetas_sieve, penalty_distances, i, penalty_new, problem;
                penalty_type = penalty_type, smoothness = smoothness)
            logprior_old              = has_old_logprior ? logprior_t_minus_1[i] : particle_logpenalty(thetas_sieve, penalty_distances, i, penalty_prev, problem;
                penalty_type = penalty_type, smoothness = smoothness)
            log_prior_ratio           = logprior_t[i] - logprior_old
            new_log_weights[i]            = log(smc_weights[i]) + log_prior_ratio;
        end
    end
    # println("weights before normalizing")
    # println(mean(new_weights))
    # if sum(new_weights) > 0 
    #     new_weights = new_weights ./ sum(new_weights);
    # end
    # println("weights after normalizing")
    # println(mean(new_weights))

    return new_log_weights, logprior_t
end

function geometric_grid(A::Float64, B::Float64, n::Int)
    r = (B / A)^(1 / (n - 1))  # Common ratio for geometric progression
    grid = [A * r^(i - 1) for i in 1:n]
    return grid
end

function make_prior_dists(prior, gamma_length)
    betabar     = prior["betabar"]
    gammabar    = prior["gammabar"]
    vbeta       = prior["vbeta"]
    vgamma      = prior["vgamma"]    
    ngamma      = gamma_length-1;
    
    beta_dist   = MvNormal(betabar, Diagonal(vbeta))
    gamma_dist  = MvNormal(gammabar, Diagonal(fill(vgamma, ngamma)))

    return beta_dist, gamma_dist
end

# function logprior_smc(particle_betastar::AbstractArray{T}, particle_gamma::AbstractArray{T}, beta_dist, gamma_dist) where T
function logprior_smc(particle_betastar, particle_gamma, beta_μ, beta_Σ, gamma_μ, gamma_Σ)
    out_beta    = logpdf_mvn(beta_μ, beta_Σ, particle_betastar)
    out_gamma   = logpdf_mvn(gamma_μ, gamma_Σ, particle_gamma)
    return out_beta + out_gamma
end

function logprior_smc(particle_betastar, particle_gamma,
                      beta_μ, chol_beta::Cholesky, logdet_beta::Real,
                      gamma_μ, chol_gamma::Cholesky, logdet_gamma::Real)
    out_beta  = logpdf_mvn(beta_μ, chol_beta,  logdet_beta,  particle_betastar)
    out_gamma = logpdf_mvn(gamma_μ, chol_gamma, logdet_gamma, particle_gamma)
    return out_beta + out_gamma
end

function logprior_smc(particle_betastar, particle_gamma,
                      beta_μ, beta_inv_var::AbstractVector, beta_logdet::Real,
                      gamma_μ, gamma_inv_var::AbstractVector, gamma_logdet::Real)
    return logpdf_diag_mvn(beta_μ, beta_inv_var, beta_logdet, particle_betastar) +
           logpdf_diag_mvn(gamma_μ, gamma_inv_var, gamma_logdet, particle_gamma)
end

function smooth_positive(x::Real, smoothness::Real)
    sx = smoothness * x
    if sx > 0
        return x + log1p(exp(-sx)) / smoothness
    else
        return log1p(exp(sx)) / smoothness
    end
end

smooth_abs(x::Real, eps::Real) = sqrt(x * x + eps * eps)

function smooth_positive_derivative(x::Real, smoothness::Real)
    sx = smoothness * x
    if sx >= 0
        return inv(1 + exp(-sx))
    else
        exp_sx = exp(sx)
        return exp_sx / (1 + exp_sx)
    end
end

smooth_abs_derivative(x::Real, eps::Real) = x / smooth_abs(x, eps)

function smooth_constraint_distance_grad!(grad_jacobian, jacobian::AbstractMatrix,
        constraints, exchange, smoothness::Real, abs_eps::Real)
    fill!(grad_jacobian, 0)
    distance = zero(eltype(jacobian))
    denom = length(constraints)
    denom == 0 && return distance
    J = size(jacobian, 1)

    if :monotone in constraints
        scale = inv(denom * J)
        @inbounds for j in 1:J
            x = jacobian[j,j]
            distance += scale * smooth_positive(x, smoothness)
            grad_jacobian[j,j] += scale * smooth_positive_derivative(x, smoothness)
        end
    end

    if :all_substitutes in constraints
        distance += smooth_signed_offdiag_grad!(grad_jacobian, jacobian, smoothness, -1.0, denom)
    end
    if :all_complements in constraints
        distance += smooth_signed_offdiag_grad!(grad_jacobian, jacobian, smoothness, 1.0, denom)
    end
    if :subs_in_group in constraints
        distance += smooth_signed_group_grad!(grad_jacobian, jacobian, exchange, smoothness, -1.0, denom)
    end
    if :complements_in_group in constraints
        distance += smooth_signed_group_grad!(grad_jacobian, jacobian, exchange, smoothness, 1.0, denom)
    end
    if :subs_across_group in constraints
        distance += smooth_signed_across_group_grad!(grad_jacobian, jacobian, exchange, smoothness, -1.0, denom)
    end
    if :complements_across_group in constraints
        distance += smooth_signed_across_group_grad!(grad_jacobian, jacobian, exchange, smoothness, 1.0, denom)
    end

    if :diagonal_dominance_all in constraints
        scale = inv(denom * J)
        @inbounds for col in 1:J
            offdiag_sum = zero(eltype(jacobian))
            for row in 1:J
                row == col || (offdiag_sum += jacobian[row,col])
            end
            offdiag_abs = smooth_abs(offdiag_sum, abs_eps)
            diag_abs = smooth_abs(jacobian[col,col], abs_eps)
            margin = offdiag_abs - diag_abs
            slope = scale * smooth_positive_derivative(margin, smoothness)
            distance += scale * smooth_positive(margin, smoothness)

            offdiag_derivative = smooth_abs_derivative(offdiag_sum, abs_eps)
            for row in 1:J
                row == col || (grad_jacobian[row,col] += slope * offdiag_derivative)
            end
            grad_jacobian[col,col] -= slope * smooth_abs_derivative(jacobian[col,col], abs_eps)
        end
    end
    return distance
end

function smooth_signed_offdiag_grad!(grad_jacobian, jacobian, smoothness::Real, sign::Real, denom::Int)
    J = size(jacobian, 1)
    n = J * (J - 1)
    n == 0 && return zero(eltype(jacobian))
    scale = inv(denom * n)
    distance = zero(eltype(jacobian))
    @inbounds for col in 1:J, row in 1:J
        row == col && continue
        x = sign * jacobian[row,col]
        distance += scale * smooth_positive(x, smoothness)
        grad_jacobian[row,col] += scale * sign * smooth_positive_derivative(x, smoothness)
    end
    return distance
end

function smooth_signed_group_grad!(grad_jacobian, jacobian, exchange, smoothness::Real, sign::Real, denom::Int)
    n = sum(length(group) * (length(group) - 1) for group in exchange)
    n == 0 && return zero(eltype(jacobian))
    scale = inv(denom * n)
    distance = zero(eltype(jacobian))
    @inbounds for group in exchange, col in group, row in group
        row == col && continue
        x = sign * jacobian[row,col]
        distance += scale * smooth_positive(x, smoothness)
        grad_jacobian[row,col] += scale * sign * smooth_positive_derivative(x, smoothness)
    end
    return distance
end

function smooth_signed_across_group_grad!(grad_jacobian, jacobian, exchange, smoothness::Real, sign::Real, denom::Int)
    if length(exchange) != 2
        error("Cannot use `across_group` constraints with only one exchangeable group")
    end
    J = maximum(union(exchange[1], exchange[2]))
    n = sum(length(group) * (J - length(group)) for group in exchange)
    n == 0 && return zero(eltype(jacobian))
    scale = inv(denom * n)
    distance = zero(eltype(jacobian))
    @inbounds for group in exchange, row in group, col in 1:J
        col in group && continue
        x = sign * jacobian[row,col]
        distance += scale * smooth_positive(x, smoothness)
        grad_jacobian[row,col] += scale * sign * smooth_positive_derivative(x, smoothness)
    end
    return distance
end

function smooth_diag_distance(jacobian::AbstractMatrix, smoothness::Real)
    total = zero(eltype(jacobian))
    J = size(jacobian, 1)
    @inbounds for j in 1:J
        total += smooth_positive(jacobian[j,j], smoothness)
    end
    return total / J
end

function smooth_offdiag_distance(jacobian::AbstractMatrix, smoothness::Real, sign::Real)
    total = zero(eltype(jacobian))
    J = size(jacobian, 1)
    n = 0
    @inbounds for col in 1:J, row in 1:J
        row == col && continue
        total += smooth_positive(sign * jacobian[row,col], smoothness)
        n += 1
    end
    return n == 0 ? total : total / n
end

function smooth_group_distance(jacobian::AbstractMatrix, exchange, smoothness::Real, sign::Real)
    total = zero(eltype(jacobian))
    n = 0
    @inbounds for group in exchange, col in group, row in group
        row == col && continue
        total += smooth_positive(sign * jacobian[row,col], smoothness)
        n += 1
    end
    return n == 0 ? total : total / n
end

function smooth_across_group_distance(jacobian::AbstractMatrix, exchange, smoothness::Real, sign::Real)
    if length(exchange) != 2
        error("Cannot use `across_group` constraints with only one exchangeable group")
    end
    total = zero(eltype(jacobian))
    J = maximum(union(exchange[1], exchange[2]))
    n = 0
    @inbounds for group in exchange, row in group, col in 1:J
        col in group && continue
        total += smooth_positive(sign * jacobian[row,col], smoothness)
        n += 1
    end
    return n == 0 ? total : total / n
end

function smooth_diagonal_dominance_distance(jacobian::AbstractMatrix, smoothness::Real, abs_eps::Real)
    total = zero(eltype(jacobian))
    J = size(jacobian, 1)
    @inbounds for col in 1:J
        offdiag_sum = zero(eltype(jacobian))
        for row in 1:J
            row == col || (offdiag_sum += jacobian[row,col])
        end
        margin = smooth_abs(offdiag_sum, abs_eps) - smooth_abs(jacobian[col,col], abs_eps)
        total += smooth_positive(margin, smoothness)
    end
    return total / J
end

function smooth_constraint_distances(particle_sieve::AbstractArray{T},
        problem::NPDemand.NPDProblem,
        prices::AbstractMatrix,
        shares::AbstractMatrix;
        smoothness::Real = 20.0,
        abs_eps::Real = 1e-8) where T
    constraints = problem.constraints
    distances = Vector{promote_type(T, Float64)}(undef, size(problem.data, 1))
    if isempty(constraints) || constraints == [:exchangeability]
        fill!(distances, zero(eltype(distances)))
        return distances
    end

    jacobians = elast_mat_zygote(particle_sieve, problem, problem.tempmats;
        at = prices,
        s = shares)
    denom = length(constraints)

    @inbounds for i in eachindex(jacobians)
        jacobian = jacobians[i]
        distance = zero(eltype(distances))
        if :monotone in constraints
            distance += smooth_diag_distance(jacobian, smoothness)
        end
        if :all_substitutes in constraints
            distance += smooth_offdiag_distance(jacobian, smoothness, -1.0)
        end
        if :diagonal_dominance_all in constraints
            distance += smooth_diagonal_dominance_distance(jacobian, smoothness, abs_eps)
        end
        if :subs_in_group in constraints
            distance += smooth_group_distance(jacobian, problem.exchange, smoothness, -1.0)
        end
        if :subs_across_group in constraints
            distance += smooth_across_group_distance(jacobian, problem.exchange, smoothness, -1.0)
        end
        if :all_complements in constraints
            distance += smooth_offdiag_distance(jacobian, smoothness, 1.0)
        end
        if :complements_in_group in constraints
            distance += smooth_group_distance(jacobian, problem.exchange, smoothness, 1.0)
        end
        if :complements_across_group in constraints
            distance += smooth_across_group_distance(jacobian, problem.exchange, smoothness, 1.0)
        end
        distances[i] = distance / denom
    end
    return distances
end

function smooth_constraint_distances(particle_sieve::AbstractArray{T},
        problem::NPDemand.NPDProblem;
        smoothness::Real = 20.0,
        abs_eps::Real = 1e-8) where T
    return smooth_constraint_distances(particle_sieve, problem,
        Matrix(problem.data[!,r"prices"]),
        Matrix(problem.data[!,r"shares"]);
        smoothness = smoothness,
        abs_eps = abs_eps)
end

function approx_cdf_normal01(x::Real)
    # Constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p  = 0.3275911
    
    sign = x < 0 ? -1 : 1
    abs_x = abs(x) / sqrt(2.0)
    
    # Approximation of the error function using a series expansion
    t = 1.0 / (1.0 + p * abs_x)
    y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    erf_approx = 1.0 - y * exp(-abs_x^2)
    
    return 0.5 * (1.0 + sign * erf_approx)
end

function approx_logcdf_normal01_grad(x::Real)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p  = 0.3275911

    sign = x < 0 ? -1 : 1
    abs_x = abs(x) / sqrt(2.0)
    t = 1.0 / (1.0 + p * abs_x)
    y = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    exp_term = exp(-abs_x^2)
    erf_approx = 1.0 - y * exp_term
    cdf = 0.5 * (1.0 + sign * erf_approx)
    if cdf <= 0 || !isfinite(cdf)
        return -Inf, zero(x)
    end

    dy_dt = a1 + 2 * a2 * t + 3 * a3 * t^2 + 4 * a4 * t^3 + 5 * a5 * t^4
    dy_dabs = -p * t^2 * dy_dt
    dq_dabs = exp_term * (dy_dabs - 2 * abs_x * y)
    dcdf_dx = -0.5 * dq_dabs / sqrt(2.0)
    return log(cdf), dcdf_dx / cdf
end

function get_tolerance(p::Float64)
    mintol = 1e-4
    if p==0
        out = mintol
    else
        ndecimals = floor(log10(p))
        out = max(mintol, floor(10^ndecimals, digits=Int(abs(ndecimals))))
    end
    return out
end

function smc_penalty_distances(thetas_sieve::Matrix, problem::NPDemand.NPDProblem;
    multithread = false,
    penalty_type = :count,
    smoothness::Real = 20.0)
    out = Matrix{Float64}(undef, size(thetas_sieve, 1), size(problem.data, 1))
    prices = penalty_type == :smooth ? Matrix(problem.data[!,r"prices"]) : nothing
    shares = penalty_type == :smooth ? Matrix(problem.data[!,r"shares"]) : nothing
    loop = axes(thetas_sieve, 1)
    if multithread
        Threads.@threads for i in loop
            out[i,:] .= penalty_type == :smooth ?
                smooth_constraint_distances(thetas_sieve[i,:], problem, prices, shares; smoothness = smoothness) :
                report_constraint_violations_inner(problem, params = thetas_sieve[i,:], verbose = false, output = "frac")
        end
    else
        for i in loop
            out[i,:] .= penalty_type == :smooth ?
                smooth_constraint_distances(thetas_sieve[i,:], problem, prices, shares; smoothness = smoothness) :
                report_constraint_violations_inner(problem, params = thetas_sieve[i,:], verbose = false, output = "frac")
        end
    end
    return out
end

function logpenalty_smc(distance::AbstractVector, penalty::Real)
    return sum(log.(approx_cdf_normal01.(-2 .* penalty .* distance)))
end

function smooth_logpenalty_grad!(grad_sieve, particle_sieve::AbstractArray,
        problem::NPDemand.NPDProblem, prices::AbstractMatrix, shares::AbstractMatrix,
        penalty::Real; smoothness::Real = 20.0, abs_eps::Real = 1e-8,
        workspace = nothing)
    fill!(grad_sieve, 0)
    constraints = problem.constraints
    if isempty(constraints) || constraints == [:exchangeability]
        return size(problem.data, 1) * log(approx_cdf_normal01(0.0))
    end

    J = length(problem.Xvec)
    indexes = [0; cumsum(size.(problem.Xvec, 2))]
    if workspace === nothing
        jacobian_inverse = Matrix{eltype(particle_sieve)}(undef, J, J)
        inverse_derivative = similar(jacobian_inverse)
        grad_jacobian = similar(jacobian_inverse)
        grad_inverse = similar(jacobian_inverse)
        grad_A = similar(jacobian_inverse)
        tmp_A = similar(jacobian_inverse)
    else
        jacobian_inverse = workspace.jacobian_inverse
        inverse_derivative = workspace.inverse_derivative
        grad_jacobian = workspace.grad_jacobian
        grad_inverse = workspace.grad_inverse
        grad_A = workspace.grad_A
        tmp_A = workspace.tmp_A
    end
    logpenalty = zero(eltype(particle_sieve))

    for market in axes(prices, 1)
        @inbounds for row in 1:J, col in 1:J
            theta_block = @view particle_sieve[(indexes[row]+1):indexes[row+1]]
            temp_row = @view problem.tempmats[row,col][market,:]
            inverse_derivative[row,col] = dot(temp_row, theta_block)
        end

        try
            jacobian_inverse .= -inv(inverse_derivative)
        catch
            fill!(grad_sieve, 0)
            return -Inf
        end

        distance = smooth_constraint_distance_grad!(grad_jacobian, jacobian_inverse,
            constraints, problem.exchange, smoothness, abs_eps)
        logcdf, dlogcdf_dx = approx_logcdf_normal01_grad(-2 * penalty * distance)
        if !isfinite(logcdf)
            fill!(grad_sieve, 0)
            return -Inf
        end
        logpenalty += logcdf

        grad_inverse .= (-2 * penalty * dlogcdf_dx) .* grad_jacobian
        mul!(tmp_A, transpose(jacobian_inverse), grad_inverse)
        mul!(grad_A, tmp_A, transpose(jacobian_inverse))

        @inbounds for row in 1:J, col in 1:J
            theta_grad = @view grad_sieve[(indexes[row]+1):indexes[row+1]]
            temp_row = @view problem.tempmats[row,col][market,:]
            theta_grad .+= grad_A[row,col] .* temp_row
        end
    end
    return logpenalty
end

function logpenalty_smc(particle_sieve::Array{T}, penalty::Real, problem::NPDemand.NPDProblem;
    penalty_type = :count,
    smoothness::Real = 20.0,
    prices = nothing,
    shares = nothing) where T
    distance = penalty_type == :smooth ?
        (prices === nothing || shares === nothing ?
            smooth_constraint_distances(particle_sieve, problem; smoothness = smoothness) :
            smooth_constraint_distances(particle_sieve, problem, prices, shares; smoothness = smoothness)) :
        report_constraint_violations_inner(problem, params = particle_sieve, verbose = false, output = "frac")
    return logpenalty_smc(distance, penalty)
end

function loglikelihood(problem::NPDemand.NPDProblem, particle_betastar::Vector{T}, particle_gamma::Vector{T}, nbetas) where T
    nbetas          = NPDemand.get_nbetas(problem)
    lbs             = NPDemand.get_lower_bounds(problem)
    parameter_order = NPDemand.get_parameter_order(lbs)
    particle_beta   = NPDemand.reparameterization_draws(reshape(particle_betastar,1,sum(nbetas)), lbs, parameter_order);

    x = NPDemand.map_to_sieve(particle_beta, particle_gamma, problem.exchange, nbetas, problem);
   
    return -0.5 * gmm(x, problem, problem.weight_matrices)
end


function fe_posteriors(problem; FE::Union{Array, String} = [])
    if FE==[] 
        error("Please provide name of FE (the `FE` keyword argument)")
    end
    if (problem.chain ==[])
        error("No Markov chain found in the problem")
    end

    coef_names = [problem.fe_param_mapping[i].name for i in 1:length(problem.fe_param_mapping)]
    coefs_for_this_fe = findall(coef_names .== FE)
    all_gammas = problem.results.filtered_chain[:,end-size(problem.Bvec[1],2)+1:end];
    num_index_vars = length(problem.index_vars);
    
    df_fe = DataFrame()
    for (_, i) in enumerate(coefs_for_this_fe)
        val = problem.fe_param_mapping[i].value
        column_name = "Value$val"
        if i ==1 
            df_fe[!, column_name] = all_gammas[:, num_index_vars + i]
        else 
            df_fe[!, column_name] = all_gammas[:, num_index_vars + i]
        end
    end
    return df_fe
end
