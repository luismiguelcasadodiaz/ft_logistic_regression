# Multivariate logistic regression
The difference between the linear regression and the logistic regression is that the logistic regression transforms the linear regression's lineal combinations with a sigmoid function
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

to get a probability.

$$h_\theta(x) = \sigma(\boldsymbol{x}\boldsymbol\theta)$$
$$h_\theta(x) = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n \qquad (1)$$

Such change, transforms the linear regression's cost function 

```math
J(\boldsymbol{\theta})
=
\frac{1}{2m} \quad (\boldsymbol{X} \boldsymbol{\theta} - \boldsymbol{y})^{T}(\boldsymbol{X} \boldsymbol{\theta} - \boldsymbol{y})
```
into

```math
J(\boldsymbol{\theta})
=
- \frac{1}{m} \quad \sum_{i=1}^{m}[\hat{y}^{(i)} \log(h_\theta(x^{(i)})) + (1 - \hat{y}^{(i)}) \log(1 - h_\theta(x^{(i)}))]
```
This function is known as `Log-Loss` or Binary Cross-entropy. It comes from a fundamental statistical principle: Maximum likelihood Estimation or MLE.

In contrast with the linear regression, where errors follow a normal distribution, in the logistic regression the output is binary $y \in {0, 1}$ which corresponds with a Bernouilli distribution.

### 1.-Model of the probability of a sample.

Let's define the ouput of the hypotesys $$h_{\boldsymbol{\theta}}(x) = \sigma(\boldsymbol{x}\boldsymbol\theta)$$ as the conditional probability of the label be 1.

```math
\begin{aligned}
P( y = 1 \mid \mathbf{x}; \boldsymbol{\theta}) &= h_{\boldsymbol{\theta}}(x) \\
P( y = 0 \mid \mathbf{x}; \boldsymbol{\theta}) &= 1 -h_{\boldsymbol{\theta}}(x) \\
\end{aligned}
```

### 2.-Join two equations into one.
In order to work analytically either with 1 or with 0, the combination results into
$$P( y \mid \mathbf{x}; \boldsymbol{\theta}) = [ \quad h_{\boldsymbol{\theta}}(x) \quad]^{y} * [ \quad 1 -h_{\boldsymbol{\theta}}(x) \quad]^{1 - y}, \quad y \in \{0, 1 \}$$

in such a way that when `y = 1` :$$[ \quad h_{\boldsymbol{\theta}}(x) \quad]^{1} * [ \quad 1 -h_{\boldsymbol{\theta}}(x) \quad]^{0} = h_{\boldsymbol{\theta}}(x)$$
or when `y = 0`: $$[ \quad h_{\boldsymbol{\theta}}(x) \quad]^{0} * [ \quad 1 -h_{\boldsymbol{\theta}}(x) \quad]^{1} = 1 - h_{\boldsymbol{\theta}}(x)$$

it is a mathematical convenience.

### 3.- Likelihood.
Assuming that the m dataset's samples are independent and identically distributed (i.i.d.), the joint probability of observing all the actual labels `y` given the inputs `x` is the product of their individual probabilities.

```math
\begin{aligned}
L(\boldsymbol{\theta})
&= P(\mathbf{y} \mid \mathbf{X};\boldsymbol{\theta}) \\[1ex]
&= P(y^{(1)}, y^{(2)}, \dots, y^{(m)} \mid \mathbf{X}; \boldsymbol{\theta]})\\[1ex]
&= \prod_{i=1}^{m} P(y^{(i)} \mid \mathbf{x}^{(i)};\boldsymbol{\theta})\\[1ex]
&= \prod_{i=1}^{m} \left( h_{\boldsymbol{\theta}}(\mathbf{x^{(i)}}) \right)^{y^{(i)}} * \left( 1 -h_{\boldsymbol{\theta}}(\mathbf{x^{(i)}}) \right)^{1 - y}\\[1ex]

\end{aligned}
```
The statistical goal is to find the parameters $\boldsymbol{\theta}$ that maximice this likelihood $L(\boldsymbol{\theta})$.


## Prediction