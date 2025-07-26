using NPDemand 
using DataFrames
using Plots

# Define the approximation details for Bernstein and non-tensor polynomial sieves
approximation_details_bernstein = Dict(
    :order => 2, 
    :max_interaction => 2, 
    :sieve_type => "bernstein", # "bernstein" or "polynomial"
    :tensor => true # NOTE: tensor overrides max_interaction
)

approximation_details_nontensor = Dict(
    :order => 2, 
    :max_interaction => 100, 
    :sieve_type => "polynomial", # "bernstein" or "polynomial"
    :tensor => false # NOTE: tensor overrides max_interaction
)


# Count the number of coefficients for different numbers of products (J)
sieve_coef_counts = [
    NPDemand.count_params(
        n_products = J,
        exchange = [collect(1:J)],
        approximation_details = approximation_details_bernstein
    ) for J in 2:13
] |> DataFrame

sieve_coef_counts_notensor = [
    NPDemand.count_params(
        n_products = J,
        exchange = [collect(1:J)],
        approximation_details = approximation_details_nontensor
    ) for J in 2:13
] |> DataFrame

# Plot the number of coefficients for the Bernstein sieve
plot(
    Int.(sieve_coef_counts.n_products), 
    sieve_coef_counts.unique_params, 
    xlabel = "Number of Products (J)", 
    ylabel = "Number of Coefficients", 
    label = "Unique Coefficients", 
    yscale = :log10, 
    xticks = 2:13  
)
plot!(
    sieve_coef_counts.n_products, 
    sieve_coef_counts.total_params, 
    label = "Total Coefficients"
)

# Plot the number of coefficients for the non-tensor polynomial sieve
# ------------------------------------
# Q: Why is the graph below flat? 
# A: Because the polynomial approximation is much simpler and the exchangeability 
# is very strong. The terms we need are: 
# (constant), own_share, own_share^2, own_share*cross_share, cross_share_a * cross_share_b, cross_share, and cross_share^2

plot(
    Int.(sieve_coef_counts_notensor.n_products), 
    sieve_coef_counts_notensor.total_params, 
    xlabel = "Number of Products (J)", 
    ylabel = "Number of Coefficients", 
    label = "Unique Coefficients (no tensor)", 
    xticks = 2:13  # Force ticks at every integer from 2 to 13
)