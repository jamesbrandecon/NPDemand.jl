# NPDemand.jl

```@meta
CurrentModule = NPDemand
```

Documentation for NPDemand.jl, a package for nonparametric demand estimation. This version of the package will be introduced formally and used by (in-progress) Brand and Smith (2024).  

## Package Framework 
This package is meant to make the estimation of price elasticities as easy, and as close to `reg y x, r`, as possible. There are three necessary steps: 

1. **Define** a problem (an `NPDProblem`): at this stage, you specify your data, your model, and any econometric choices that are required up front. 

2. **Estimate** the problem: in the simplest case, this is a single call to the `estimate!` function. Otherwise, all that needs to be specified are details about how long to let the estimation run and other similar controls.

3. **Process** results: the problem has now stored the estimated parameters internally, and we've tried to provide functions to calculate the main objects of interest (price elasticities and demand functions) that do not require the user to know anything about what is happening under the hood. 

This is similar to, and inspired by, the structure of the (much more extensive) [PyBLP package](https://pyblp.readthedocs.io/en/stable/) in Python. Ideally, this structure makes the code modular enough that our package could be appended (either by the user or by us) with additional functions for more complicated processing and other results. 


## Documentation Contents 
```@contents
```

