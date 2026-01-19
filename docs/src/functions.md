# Function documentation
## Problem construction and manipulation
```@docs
define_problem
```

```@docs
update_constraints!
```

```@docs
NPDemand.list_constraints
```

## Estimation
```@docs
estimate!
```

```@docs
smc!
```

## Post-estimation tools
```@docs
price_elasticities!
```

```@docs
elasticity_quantiles
```

```@docs
report_constraint_violations
```

```@docs
compute_demand_function!
```

```@docs
summarize_elasticities
```

## Back-end functions
```@docs
NPDemand.NPD_parameters
```

```@docs
simulate_logit
```

```@docs
toDataFrame
```

```@docs
bern
```

```@docs
dbern
```

```@docs
NPDemand.poly
```

```@docs
NPDemand.dpoly
```

```@docs
NPDemand.PolyRecipe
```

```@docs
NPDemand.build_poly_recipe
```

```@docs
NPDemand.poly_features
```

```@docs
NPDemand.poly_features_derivative
```