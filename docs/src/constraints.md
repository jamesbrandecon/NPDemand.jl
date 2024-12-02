## Constraints
Below is a list of constraints which have been implemented in NPDemand so far. We impose these constraints on our nonparametric demand functions in two ways: one which is linear in parameters, and another which requires quasi-Bayes sampling over our objective function. We discuss both approaches below. 

| Constraint    | Symbol | Linear constraint |
| -------- | ------- | ------- |
| Negative own-price elasticities | :monotone | Yes |
| Diagonal Dominance within exchangeable groups  | :diagonal\_dominance\_in\_group    | Yes |
| Diagonal Dominance among all products | :diagonal\_dominance\_all     | Yes |
| All products substitutes    | :all\_substitutes    | Yes |
| Products in the same group are substitutes | :subs\_in\_group | No |
| Products in different groups are substitutes | :subs\_across\_group | No |
| Products in the same group are substitutes | :complements\_in\_group | No |
| Products in different groups are substitutes | :complements\_across\_group | No |

### Linear constraints
Our first constraint implementation is a linear constraints as developed by Compiani (2022). Each of these takes advantage of the fact that coefficients of Bernstein polynomials approximate values of the target function (the inverse demand function) on a known grid. As a result, we can easily impose some constraints, like monotonicity on the inverse demand function, by imposing linear relationships between sieve parameters. Constraints which fall into this category are denoted above in the **Linear** column
 
It is important to note that these linear constraints on our sieve parameters are, in general, **necessary** for the desired economic constraints to be satisfied by the resulting demand function, but not sufficient. However, they are much easier to impose than the more complex constraints we discuss below. In practice, we find that these linear constraints are often sufficient to provide reasonable orders of magnitude and relationships between products at the median, but insufficient for other moments of the elasticity distribution to be well-behaved. For more information, see discussions in Compiani (2022) and Brand and Smith (2024).

 ### Quasi-bayes constraints 

 #### Dogmatic Priors
The first way we allow users to impose constraints using our quasi-Bayes approach is via a penalty term (the `penalty` argument in `estimate!`). The default value of this penalty is 0, but when the user increases this value, we evaluate a modified objective function. Rather than sampling according to the quasi-likelihood, we penalize that likelihood by the provided penalty:

```julia
Turing.@addlogprob! (-0.5 * objective(...) - penalty)
```

We find that this approach has an imperfect success rate. If multiple Markov chains are constructed for the same problem via this method, some may find the constrained space quickly and remain there, and others will never find the space even after many tens of thousands of samples. Still, we include this option because of its ease of use and its speed relative to our final approach below. 

 #### Sequentially Constrained Monte Carlo (SMC)
 Our final option for imposing constraints is implemented in the `smc!` function. This function implements "Sequentially Constrained Monte Carlo," an iterative method which slowly reshapes an unconstrained Markov chain into a potentially complex constrained space. The approach is intutively simple. We specify an increasing sequence of penalty terms $\lambda_i$, which (after smoothing) define a sequence of posteriors $\pi_i$ from which we can sample (with likelihood $l_n$ and unconstrained posterior $\bar\pi$): 

$$\pi_0(\theta^*|\text{data})\propto e^{l_n(g^{-1}(\theta^*))} \bar{\pi}(\theta^*)$$
$$\pi_1(\theta^*|\text{data})\propto e^{l_n(g^{-1}(\theta^*))} \bar{\pi}(\theta^*) \lVert\theta\rVert_\mathcal{C}^{\lambda_1}$$
$$\pi_2(\theta^*|\text{data})\propto e^{l_n(g^{-1}(\theta^*))} \bar{\pi}(\theta^*) \lVert\theta\rVert_\mathcal{C}^{\lambda_2}$$ 
$$ \vdots $$ 
$$\pi_M(\theta^*|\text{data})\propto e^{l_n(g^{-1}(\theta^*))} \bar{\pi}(\theta^*) \lVert\theta\rVert_\mathcal{C}^{\lambda_M}$$

We smooth the $\lambda$ penalty terms via the following, where $\Phi$ is the standard normal CDF: 
$$||\theta||_\mathcal{C}^\lambda = \prod_{t=1}^T \Phi(-\lambda c_t(\theta))$$

We define this sequence adaptively by default, though we offer users other options in the package. At each step in the sequence, we take a relatively small number of Metropolis-Hastings steps for each element of our chain, sampling from the posterior corresponding to the next element $\lambda_{i+1}$. As we demonstrate in Brand and Smith (2024), appropriately defining this sequence allows us to enforce the desired constraints globally on our data. The corresponding cost is that this method is by far the most time consuming to run, often taking many minutes or even hours, depending on the problem's complexity. 