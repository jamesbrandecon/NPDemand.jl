# Quasi-Bayes: Priors and Sampling

## Quasi-Bayes at a high level
Let $l_n(\theta) denote a scaled version of our GMM objective function. Additionally let the transformation $e^{l_n(\theta)}$ denote the quasi-likelihood function and $\pi(\theta)$ denote a prior density. Then the quasi-posterior takes the form:
$$
\pi(\theta|\text{data}) = \frac{e^{\ell_n(\theta)} \pi(\theta)}{\int e^{\ell_n(\theta)} \pi(\theta) d\theta } 
$$
and is a valid density over $\theta$. \cite{Chernozhukov.2003} 

## Priors 
For problems without constraints and with only exchangeability constraints, we specify a simple and dispersed prior:
$$ \theta \sim N(\vec 0, I * v), $$ 
where $I$ is the identity matrix and $v$ is a large prespecified constant. Whenever we impose constraints linearly, we modify our prior to encode the constraint in the prior directly. For example, if our constraints imply that $\theta_1$ (the first element of our parameter vector) must be larger than $\theta_2$, then we introduce an auxiliary parameter $\theta^*_2$ and specify priors on $(\theta_1, \theta_2)$ as

$$ \theta_1, \theta^*_2 \sim N(0,v)$$
$$\theta_2 = \theta_1 + exp(\theta^*_2) $$


## Sampling -- maybe delete
When users call `estimate!` with `quasibayes = true`, we provide flexible tools 

- Turing 

!!!note
