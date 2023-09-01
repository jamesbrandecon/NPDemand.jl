# Usage  
Having described the some of the estimation details, we now describe how to use the package.

### Defining Problems
The first step to estimating demand is to define the problem. The `define_problem` function takes the following arguments:
- `df`: a DataFrame containing the data to be used in estimation.
- `exchange`: a list of lists of integers, where each list of integers corresponds to a group of products that are exchangeable.
- `index_vars`: a list of strings, where each string corresponds to a column in `df` that contains the index variables for the demand system. 
- `constraints`: a list of symbols, where each symbol corresponds to a constraint to be imposed on the demand system.
- `bO`: an integer indicating the order of Bernstein polynomials to be used in the demand system. The default value is 2, and larger values will result in (significantly) more parameters.
- `FE`: a list of strings, where each string corresponds to a column in `df` that contains fixed effects to be included in the demand system.
- `constraint_tol`: the tolerance for the constraint satisfaction problem.
- `obj_xtol`: the tolerance for the unconstrained optimization problem.
- `obj_ftol`: the tolerance for the unconstrained optimization problem.


#### Constraints 
So far, we have implemented the following constraints: 
- `:exchangeability`: Products given by `exchange` are exchangeable, meaning that product identities do not affect demand. Product fixed effects can stil be estimated
- `:monotone`: All demand functions are monotonic in their own index 
- `:diagonal\_dominance\_all`: All demand functions are diagonally dominant
- `:diagonal\_dominance\_group`: All demand functions are diagonally dominant within exchangeable groups
- `:all_substitutes`: All inverse demand functions are increasing in all indexes, which is a necessary condition for all products to be substitutes
- `:subs\_in\_group`: Cross-price elasticities of demand are positive for every pair of goods within an exchangeable group. No restrictions on the Jacobian are made across groups

All of these constraints are linear except for the last. 

#### Example 
Here is an example of a problem definition using many of the options described above: 
```julia
bO = 2; 
exchange = [[1 2], [3 4]]
index_vars = ["prices", "x"]
constraint_tol = 1e-10;
obj_xtol = 1e-5;
obj_ftol = 1e-5;

constraints = [:exchangeability, :diagonal_dominance_all, :monotone, :subs_in_group]; 
npd_problem = define_problem(df;  
                            exchange = exchange,
                            index_vars = index_vars, 
                            constraints = constraints,
                            bO = bO,
                            FE = [],
                            constraint_tol = constraint_tol,
                            obj_xtol = obj_xtol, 
                            grid_size = 2,
                            obj_ftol = obj_ftol); 
```
After running this command, `npd_problem` is now of type `NPDemand.NPDProblem`. The problem contains many components that would be difficult for users to read and understand, 
so for ease of use we have included a `show` method for `NPDemand.NPDProblem` which prints out the core components of the problem definition. 

```julia
@show npd_problem;
```

### One-line estimation: estimate! 
Estimation of a defined problem can be done via a call to the `NPDemand.estimate!` function, as shown here: 
```julia 
estimate!(npd_problem, max_outer_iterations = 10, show_trace = false, verbose = true)
``` 
The key options and inputs to this function are as follows: 
    - `npd_problem`: the problem to be estimated, of type `NPDemand.NPDProblem`.
    - `max_outer_iterations`: the maximum number of iterations to be used in the outer loop (the loop over which the penalty which enforces constraints is increased) of the estimation procedure. 
    - `max_inner_iterations`: the maximum number of iterations to be used in the inner optimizer. This should be set to a large number. 
    - `verbose`: a boolean indicating whether to print the steps of estimation, including any pre-processing and the number of iterations of the outer loop. The default value is true.
    - `show_trace`: a boolean indicating whether to print the trace of the inner loop of the estimation procedure. The default value is true.
