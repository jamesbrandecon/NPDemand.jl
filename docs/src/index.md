# Documentation for NPDemand.jl
This is the documentation for the Julia Package `NPDemand.jl`, which is designed to make flexible demand estimation quick and easy for economists to implement.   

### Data Structure
In order to use any of the commands in this package, the data must be in the form of a DataFrame with the following column names

- shares: market shares for each product
- prices: prices for each product
- demand\_instruments: instruments for market shares. These can include observed exogenous characteristics or excluded cost shifters which shift market shares through price.  

Each row should represent a distinct market. Columns should be numbered starting from zero, with one column for each product, e.g. `shares0`, `shares1`, etc.

### Functions
#### Main User Functions
```@docs
inverse_demand()
```

```@docs
price_elasticity()
```

```@docs
hierNet()
```

```@docs
hierNet_boot()
```


#### Helper Functions
```@docs
bern(t,order)
```

```@docs
dbern(t,order)
```

```@docs
b(t,n,k)
```

```@docs
db(t,n,k)
```
