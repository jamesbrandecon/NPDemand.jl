# Basic Usage  
Having described the some of the estimation details, we now describe how to use the package.

### Defining Problems
The first step to estimating demand is to define the problem. The `define_problem` function takes the following arguments:
- `df`: a DataFrame containing the data to be used in estimation.
- `exchange`: a list of lists of integers, where each list of integers corresponds to a group of products that are exchangeable.
- `index_vars`: a list of strings, where each string corresponds to a column in `df` that contains the index variables for the demand system. 
- `constraints`: a list of symbols, where each symbol corresponds to a constraint to be imposed on the demand system.
- `bO`: an integer indicating the order of Bernstein polynomials to be used in the demand system. The default value is 2, and larger values will result in (significantly) more parameters.
- `FE`: a list of strings, where each string corresponds to a column in `df` that contains fixed effects to be included in the demand system.


#### Constraints 
So far, we have implemented the following constraints: 
- `:exchangeability`: Products given by `exchange` are exchangeable, meaning that product identities do not affect demand. Product fixed effects can stil be estimated
- `:monotone`: All demand functions are monotonic in their own index 
- `:diagonal_dominance_all`: All demand functions are diagonally dominant
- `:diagonal_dominance_group`: All demand functions are diagonally dominant within exchangeable groups
- `:all_substitutes`: All inverse demand functions are increasing in all indexes, which is a necessary condition for all products to be substitutes
- `:subs_in_group`: Cross-price elasticities of demand are positive for every pair of goods within an exchangeable group. No restrictions on the Jacobian are made across groups

All of these constraints are linear except for the last. 

#### Example 
Here is an example of a problem definition using many of the options described above: 
```julia
bO = 2; 
exchange = [[1 2], [3 4]] # the first and second products are exchangeable, as are the third and fourth
index_vars = ["prices", "x"]

constraints = [:exchangeability, :diagonal_dominance_all, :monotone]; 
npd_problem = define_problem(df;  
                            exchange = exchange,
                            index_vars = index_vars, 
                            constraints = constraints,
                            bO = bO,
                            FE = []); 
```
After running this command, `npd_problem` is now of type `NPDemand.NPDProblem`. The problem contains many components that would be difficult for users to read and understand, so for ease of use we include a `show` method for `NPDemand.NPDProblem` which prints out the core components of the problem definition. 

```julia
@show npd_problem;
```

### One-line estimation: estimate! 
Estimation of a defined problem can be done via a call to the `NPDemand.estimate!` function: 
```julia 
estimate!(npd_problem)
``` 

### Price elasticities
Similarly, we can then calculate all price elasticities between products in all markets via

```julia
price_elasticities!(problem)
```

We can also quickly summarize the median of each element of the elasticity matrix: 
```julia
summarize_elasticities(problem, "matrix", "quantile", q = 0.5)
```