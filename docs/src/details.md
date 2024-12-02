# Implementation Details

This page explains both the underlying economic model that our package assumes and some of the details concerning our approach, which implements a nonparametric estimator as in [Compiani, 2022](https://drive.google.com/file/d/1GTDJ7W9Fu0mugQsm14125jYJkPbTvGFh/view?pli=1) with a more general suite of (linear and nonlinear) constraints and implementation/estimation options. 

## The Nonparametric Demand Estimation Problem 
Our package estimates a demand function for $J$ goods, allowing flexible substitution patterns including some forms of complementarities. Letting $s$ denote market shares, the assumed demand function takes the following form: 

```math
    s_{jt} = \sigma_j(\delta_{jt})
```
where $j,t$ index products and markets respectively. Note that $\sigma$ is indexed by $j$, allowing the demand functions of different goods to take different forms. The arguments of $\sigma$ are $\delta$, which we assume takes the following form: 

```math
    \delta_{jt} = \beta x_{jt} + \xi_{jt}
```
where $x_{jt}$ denote observable product characteristics including price, and $\xi_{jt}$ denotes unobservable demand shifters at the product-market level. Direct estimation of $\sigma$ would be complicated here due to the fact that $\xi$ enters $\sigma$ nonlinearly. Instead, under conditions stated in [Compiani, 2022](https://drive.google.com/file/d/1GTDJ7W9Fu0mugQsm14125jYJkPbTvGFh/view?pli=1), we can **invert** $\sigma$ and estiamte the resulting inverse. The inverse demand equation is then: 

```math
    x_{jt} = \sigma^{-1}_{j}(s_{jt}) - \xi_{jt}
```

The **inverse** demand functions can then be estimated via Nonparametric Instrumental Variables (NPIV). We discuss the exact implementation below. 

## Bernstein Polynomials 
One key decision in estimating the functions $\sigma^{-1}_j$ is how to approximate them. In this package we approximate each function using [Bernstein polynomials](https://en.wikipedia.org/wiki/Bernstein_polynomial). A Bernstein polynomial of order $n$ takes the form of a sum of basis functions:  

```math
B_n(x) = \sum_{k=1}^n \theta_k b_{k,n}(x)
```
where the basis functions are defined as:

```math
b_{k,n}(x) =  {n \choose k} x^k (1-x)^{n-k}
```

Although one could make other reasonable choices for approximating $\sigma^{-1}_j$, Bernstein polynomials were chosen because they an approximation for which it is uniquely easy to impose constraints, which is a central focus of our package. 

Whenever you run `define_problem`, we generate two Bernstein polynomials. The first is in market shares, and serves as the design matrix for the NPIV regression. The second is in the vector of instruments. 

## GMM Estimation
The estimation problem takes the following form: 

```math
    \min_{\theta} \sum_{j=1}^J \Big [\sum_{t=1}^T \tilde \xi_{jt} \Big ]'(A_j'A_j)^- \Big [\sum_{t=1}^T \tilde \xi_{jt} \Big ]
```
where $\tilde \xi_{jt}$ are the values of $\xi_{jt}$ implies by the current estimate of the demand system. The matrix $A_j$ is the aforementioned Bernstein polynomial in demand instruments for product $j$.

When we are solving this linear problem using JuMP.jl, we solve it as written. When we are using more general approaches (Optim.jl or Turing.jl), we found this problem to be difficult to efficiently solve as written. Instead, during the construction of the problem, we multiply out the matrices above, allowing us to store and manipulate much smaller matrices for evaluating our objective functions. In particular, for appropriately defined values of $y,X$ and $Z$, we can rewrite the objective function as 

```math 
    \min_\theta (y - X\theta)' Z (y - X \theta)
```

For complicated problems, $\theta$ can include hundreds of parameters, meaning that $Z$ and $X$ may have hundreds of columns (and, in the case of $Z$, rows). Manipulating 
these large matrices directly resulted in an estimation process that was quite slow even with analytic gradients. However, note that if we multiply out the product above, we get 
```math 
    y'Zy + y'Z X\theta - \theta'X'Zy + \theta'X'X \theta 
```
Note, then, that the first term is constant with respect to $\theta$, and the rest of the terms are relatively small matrices which can be pre-computed prior to estimation. 
In practice, to handle normalizations and to treat the components $\theta$ that correspond to the index $\delta$ differently from those that correspond to the inverse demand functions themselves, 
we use a slightly different formulation, but the impact is the same. By pre-computing these matrices and storing them during problem construction, we find that estimation is dramatically faster. 
The only cost we pay is that this process introduces a small amount of (floating-point) errors in our objective function construction, but we find that these are negligible even in simulated data with unreasonably high signal to noise ratios (i.e., small optimized objective values).  

## Fixed Effects
Fixed effects are estimated as parameters, not absorbed from the data. So, be careful including fixed effects with too many values, as this may both slow down the optimization and require more memory.

To include fixed effects (categorical variables), use the option `FE` to provide a vector of strings, where each element of the vector is a name of a column in the provided data `df`. Note however that at present, variables that are included as fixed effect must be constant across products within a market. The only exception to this rule is `"product"` which is a keyword which will produce product fixed-effects. There need not be a column named `product` in the data, and in fact the code will ignore it if it's there. 

!!! warning 
    Given how commmon fixed effect regression packages have become, users may expect the package to smartly identify and drop singletons and perform colinearity checks. This it not yet implemented-- at present, the only thing we do beyond including all fixed effects as dummy variables is to drop one level per dimension of fixed effects. 
