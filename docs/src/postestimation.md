# Post-Estimation 
## Estimated Price Elasticities 
The main output that can be calculated from the results of `estimate!` is the full set of price elasticities of demand. Calculating all price elasticities requires 
a single call to the `price_elasticities!` function. After this call, you can access the full set of elasticities via the field `npd_problem.all_elasticities`, which is an Array of matrices. 
Each element of the array is the matrix of price elasticities for a market, where the $(i,j)$ element of each matrix denotes the elasticity of demand for product $i$ with respect to the price of product $j$.
Alternatively, you can extract the own-price elasticities for each product using the `own_elasticities` function or summarize elasticities across markets using the `summarize_elasticities` function. Some example usage: 
```julia    
price_elasticities!(npd_problem); # Calculate all elasticities 

own = own_elasticities(npd_problem) # Extract own-price elasticities 

# Summaries of elasticities
summarize_elasticities(npd_problem, "median")
summarize_elasticities(npd_problem, "mean")
summarize_elasticities(npd_problem, "quantile"; q = 0.3) # 30th percentile
```

## Estimated Demand Functions
One of the benefits of nonparametric demand estimation is that it can allow us to study the shape of the demand function and compare it to the demand functions implied by alternative models. For this purpose, or for evaluating the impact of counterfactual prices, it may be useful to estimate the demand function for a given set of prices of one good while keeping prices of other goods fixed, or at a vector of alternative prices. 

This can be done using the `compute_demand_function!` function, which takes in a `NPDProblem` and a `DataFrame` which contains prices and market shares. The function then modifies the provided DataFrame, filling in the columns `shares0`, `shares1`, ..., `sharesJ` with the estimated shares for each market. This is a sightly more complicated task than in many structural models, because the parameters we are estimating govern the inverse demand system rather than the demand system itself. In order to calculate the (non-inverted) demand function, we have to perform a nonlinear search over market shares to match the user-provided prices and other index covariates. 

!!! warning

    `compute_demand_function!` can currently only calculate counterfacual market shares assuming that all demand shocks ($\xi$) are set to zero.  

Here is an example of how to use this function to compute the demand function, varying the price of product 1 while keeping prices of other goods fixed. 

```julia
# Create a new DataFrame which can be filled in with estimated demand functions
alt_price_df = DataFrame(); # Initialize dataframe 
alt_price_df.prices1 = 1.1 .* ones(10); alt_price_df.prices2 .= 1.1; alt_price_df.prices3 .= 1.1;
alt_price_df.prices0 .= collect(range(0.7,1.3, length=10));
for j = 0:3
    alt_price_df[!,"shares$j"] .=0; # shares set to zero to initialize the fields-- compute_demand_function will ignore and replace these values
    alt_price_df[!,"x$j"] .= 0.5; 
end

compute_demand_function!(npd_problem, alt_price_df; max_iter = 1000, show_trace = false);
```
As mentioned in the note above, if you run this code you will notice that the output above generates a warning that the function "Assumes residual demand shifters set to zero." This refers to the values of the residuals $\xi$. Other alternative values for $\xi$ will likely be implemented in future versions of the package. After running `compute_demand_function!`, we can then plot the estimated demand functions (stored in `shares0`...`shares3`). Here is some example code which plots all counterfactual shares: 

```julia, label = "testing"
plot(p, alt_price_df.prices0, alt_price_df.shares0, 
    color = :red, 
    lw = 1.5, label = "Est. Share 1", 
    xlabel = "Price 1", 
    ylabel = "Market Share");
plot!(alt_price_df.prices0, alt_price_df.shares1, color = :grey, label = "Est. Share 2");
plot!(alt_price_df.prices0, alt_price_df.shares2, color = :grey, label = "Est. Share 3");
plot!(alt_price_df.prices0, alt_price_df.shares3, color = :grey, label = "Est. Share 4");
```