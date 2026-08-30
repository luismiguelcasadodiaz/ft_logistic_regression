# Multivariate linear regression
## Prediction
The equation to predict the value (a real number) of the target variable for one element from the dataset, one student, is:

$$h_\theta(x) = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n \qquad (1)$$

where $\hat{y} = h_\theta(x)$ is the target feature or prediction. Such prediction is a real number in linear regression. Each coefficient $\theta_j$ measures how much the prediction changes when feature $x_j$ increases by one unit, holding the other features constant. Training the model means finding the values of $\theta_0, \theta_1, \dots, \theta_n$ that make these predictions as close as possible to the real target values across the whole dataset.

We can consider the values of the features as a vector, as well as the coefficients' values:

```math
\boldsymbol{\theta} 
= 
\begin{bmatrix} 
    \theta_0 \\ 
    \theta_1 \\ 
    \theta_2 \\ 
    \vdots  \\ 
    \theta_n 
\end{bmatrix} 
\in \mathbb{R}^{n+1} , \qquad 
\boldsymbol{x} 
= 
\begin{bmatrix} 
    x_0 \\
    x_1 \\ 
    x_2 \\ 
    \vdots  
    \\ x_n 
\end{bmatrix}
\in \mathbb{R}^{n+1}
```

and transform (1) into a vector multiplication. Writing the equation as a dot product lets it handle any number of features with a single expression, and it maps directly onto how NumPy computes it efficiently as a matrix operation instead of a Python loop over each feature.

$$h_\theta(x) = \sum_{j=0}^{n}\theta_j x_j$$

```math
\boldsymbol{\theta} \cdot \boldsymbol{x}
=
\boldsymbol{\theta}^{T} \boldsymbol{x}
=
\begin{bmatrix} 
    \theta_0 & \theta_1  & \dots & \theta_n 
\end{bmatrix}
\begin{bmatrix} 
    x_0 \\ 
    x_1  \\ 
    \vdots \\ 
    x_n 
\end{bmatrix} = \sum_{j=0}^{n}\theta_j x_j
```


Always $x_0 = 1$, so the product $x_0\theta_0$ returns $\theta_0$. This is a common trick for folding the intercept $\theta_0$ into the same dot product as the rest of the coefficients: by prepending a constant feature $x_0 = 1$ to every example, $\theta_0$ behaves exactly like any other coefficient, and no separate additive term is needed outside the sum.

When the dataset's size is $m$, we can write the vector holding $m$ predictions as the product of the feature matrix and the coefficients vector.
$\boldsymbol{X}$ is called the design matrix: each row is one example (one student) and each column is one feature, with the first column fixed at 1 for the bias term. Multiplying $\boldsymbol{X}$ by $\boldsymbol{\theta}$ computes all $m$ predictions in one matrix operation instead of looping row by row.
```math
\boldsymbol{\hat{y}} = \boldsymbol{X} \boldsymbol{\theta}, \qquad 
\begin{bmatrix} 
    \hat{y}^{(1)} \\ 
    \hat{y}^{(2)} \\ 
    \vdots \\ 
    \hat{y}^{(m)} 
\end{bmatrix} 
= 
\begin{bmatrix}
    1       & x_1^{(1)} & x_2^{(1)} & \dots     & x_n^{(1)} \\ 
    1       & x_1^{(2)} & x_2^{(2)} & \dots     & x_n^{(2)} \\ 
    \vdots  & \vdots    & \vdots    &\ddots\    & \vdots \\ 
    1       & x_1^{(m)} & x_2^{(m)} & \dots     & x_n^{(m)}
\end{bmatrix}
\begin{bmatrix} 
    \theta_0 \\ 
    \theta_1 \\ 
    \vdots \\ 
    \theta_n 
\end{bmatrix}
```

where

$$\boldsymbol{\hat{y}} \in \mathbb{R}^{m}, \qquad \boldsymbol{\theta} \in \mathbb{R}^{(n + 1)}, \qquad \boldsymbol{X} \in \mathbb{R}^{m \times (n+1)}$$

## Cost function
To know whether a given set of coefficients $\boldsymbol{\theta}$ produces good predictions, I need one number that summarizes how wrong the model is across the whole dataset — that's what the cost function $J(\boldsymbol{\theta})$ measures: the average squared distance between each prediction $\hat{y}^{(i)}$ and the real value $y^{(i)}$. Squaring keeps the error positive (so errors in opposite directions don't cancel out) and penalizes large errors more than small ones. Dividing by $2m$ averages over the dataset size $m$; the extra factor of 2 doesn't change which $\boldsymbol{\theta}$ minimizes $J$, but it will cancel out neatly once I differentiate.

```math
J(\boldsymbol{\theta})
= 
\frac{1}{2m} \sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2
=
\frac{1}{2m} \sum_{i=1}^{m} \left(h_\theta\left(\boldsymbol{x}^{(i)}\right) - y^{(i)}\right)^2
=
\frac{1}{2m} \sum_{i=1}^{m} \left(\boldsymbol{\theta}^{T}\boldsymbol{x}^{(i)} - y^{(i)}\right)^2
```
The addition of the squares of the components of a vector equals the scalar product of such vector with itself. Instead of computing each error $e^{(i)} = \hat{y}^{(i)} - y^{(i)}$ one at a time, I can stack them into a single error vector $\boldsymbol{e}$. That lets the whole cost function collapse into one compact expression, $\boldsymbol{e}^T\boldsymbol{e}$, instead of an explicit sum over $m$ terms.

```math
\boldsymbol{e}
=
\begin{bmatrix} 
    e^{(1)} \\ 
    e^{(2)} \\ 
    \vdots \\ 
    e^{(m)} 
\end{bmatrix}
=
\begin{bmatrix} 
    \hat{y}^{(1)} - {y}^{(1)} \\ 
    \hat{y}^{(2)} - {y}^{(2)} \\ 
    \vdots \\ 
    \hat{y}^{(m)} -{y}^{(m)} 
\end{bmatrix}
=
\begin{bmatrix} 
    h_\theta\left(\boldsymbol{x}^{(1)}\right) - {y}^{(1)} \\ 
    h_\theta\left(\boldsymbol{x}^{(2)}\right) - {y}^{(2)} \\ 
    \vdots \\ 
    h_\theta\left(\boldsymbol{x}^{(m)}\right) - {y}^{(m)} 
\end{bmatrix}
=
\begin{bmatrix}
    \boldsymbol{\theta}^{T}\boldsymbol{x}^{(1)} - {y}^{(1)} \\ 
    \boldsymbol{\theta}^{T}\boldsymbol{x}^{(2)} - {y}^{(2)} \\ 
    \vdots \\ 
    \boldsymbol{\theta}^{T}\boldsymbol{x}^{(m)} - {y}^{(m)} 
\end{bmatrix}
\implies
\boldsymbol{e} = \boldsymbol{X} \boldsymbol{\theta} - \boldsymbol{y}
```
 
```math
J(\boldsymbol{\theta})
=
\frac{1}{2m} \quad (\boldsymbol{X} \boldsymbol{\theta} - \boldsymbol{y})^{T}(\boldsymbol{X} \boldsymbol{\theta} - \boldsymbol{y})
= 
\frac{1}{2m} \quad \boldsymbol{e}^{T}\boldsymbol{e}
=
\frac{1}{2m} \quad (e^{(1)}e^{(1)} + e^{(2)}e^{(2)} + \dots + e^{(m)}e^{(m)})
=
\frac{1}{2m} \quad \sum_{i=1}^{m} \left(e^{(i)}\right)^2
=
\frac{1}{2m} \sum_{i=1}^{m} \left(\boldsymbol{\theta}^{T}\boldsymbol{x}^{(i)} - y^{(i)}\right)^2
```
## Partial derivative of cost function

To find the coefficients that minimize the cost function, I need to know how $J(\boldsymbol{\theta})$ changes as each individual coefficient $\theta_j$ changes — that's exactly what $\frac{\partial}{\partial \theta_j}J(\boldsymbol{\theta})$ gives me. The steps below expand $J(\boldsymbol{\theta})$ back into its sum-of-squares form and apply the chain rule to reach a closed expression for this derivative.

```math
\frac{\partial}{\partial \theta_j}J(\boldsymbol{\theta})
=
\frac{1}{m} \sum_{i=1}^{m} \left(h_\theta\left(\boldsymbol{x}^{(i)}\right) - y^{(i)}\right)x_j^{(i)}
```
Let's see the maths below it.
```math
J(\boldsymbol{\theta})
=
\frac{1}{2m} \sum_{i=1}^{m} \left(h_\theta\left(\boldsymbol{x}^{(i)}\right) - y^{(i)}\right)^2
=
\frac{1}{2m} \sum_{i=1}^{m} \left(\theta_0 x_0^{(i)} + \theta_1 x_1^{(i)} + \theta_2 x_2^{(i)} + \dots + \theta_n x_n^{(i)}  - y^{(i)}\right)^2
```
### The derivative of a sum is the sum of derivatives
This uses the sum rule: since $J(\boldsymbol{\theta})$ is a sum over $m$ terms, each term can be differentiated independently and the results added, instead of differentiating the whole sum at once.
```math
\frac{\partial}{\partial \theta_j}J(\boldsymbol{\theta})
=
\frac{1}{2m} \sum_{i=1}^{m} \frac{\partial}{\partial \theta_j}\left[\theta_0 x_0^{(i)} + \theta_1 x_1^{(i)} + \theta_2 x_2^{(i)} + \dots + \theta_n x_n^{(i)}  - y^{(i)}\right]^2
```

### The derivative of the power of a function
Applying the chain rule with $u$ equal to the expression inside the parentheses brings the exponent down as a factor of 2 — which is exactly what will cancel the $\frac{1}{2m}$ from the cost function in the next step.
Where $$f(u)= u^{2}$$ its derivative $$\frac{\partial}{\partial u} \left(u^{2}\right) = 2u\frac{\partial}{\partial u}$$ so:

```math
\frac{\partial}{\partial \theta_j}J(\boldsymbol{\theta})
=
\frac{1}{2m} \sum_{i=1}^{m} 2 \left(\theta_0 x_0^{(i)} + \theta_1 x_1^{(i)} + \theta_2 x_2^{(i)} + \dots + \theta_n x_n^{(i)}  - y^{(i)}\right)\frac{\partial}{\partial \Theta_j}\left(\theta_0 x_0^{(i)} + \theta_1 x_1^{(i)} + \theta_2 x_2^{(i)} + \dots + \theta_n x_n^{(i)}  - y^{(i)}\right)
```
### One more time: The derivative of a sum is the sum of derivatives
$$\frac{\partial}{\partial \theta_j}\left(\theta_0 x_0^{(i)} + \theta_1 x_1^{(i)} + \theta_2 x_2^{(i)} + \dots + \theta_n x_n^{(i)}  - y^{(i)}\right)=x_j^{(i)}$$

```math
\frac{\partial}{\partial \theta_j}J(\boldsymbol{\theta})
=
\frac{1}{m} \sum_{i=1}^{m}  \left(\theta_0 x_0^{(i)} + \theta_1 x_1^{(i)} + \theta_2 x_2^{(i)} + \dots + \theta_n x_n^{(i)}  - y^{(i)}\right)x_j^{(i)}
=
\frac{1}{m} \sum_{i=1}^{m}  e^{(i)}x_j^{(i)}
```
We can affirm that the partial derivative is the scalar product of errors vector with one columns of features'value matrix. In other words, the partial derivative with respect to $\theta_j$ is the dot product between the error vector and the column of $\boldsymbol{X}$ holding feature $x_j$ across all examples — it captures how strongly the errors correlate with that particular feature.

## The gradient
The gradient is the vector that gathers partial derivatives. Rather than computing one partial derivative at a time, I can stack all $n+1$ of them into a single vector: the gradient $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$. This vector points in the direction where the cost function increases fastest; gradient descent uses it to update $\boldsymbol{\theta}$ in the opposite direction, step by step, until the cost is minimized.

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})\in \mathbb{R}^{(n+1) \times 1}$$

```math
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})
=
\begin{bmatrix}
    \frac{\partial}{\partial \theta_0} \\
    \frac{\partial}{\partial \theta_1} \\
    \vdots \\
    \frac{\partial}{\partial \theta_n} \\
\end{bmatrix}
=
\frac{1}{m}
\begin{bmatrix}
    \sum_{i=1}^{m}  e^{(i)}x_0^{(i)} \\
    \sum_{i=1}^{m}  e^{(i)}x_1^{(i)} \\
    \vdots \\
    \sum_{i=1}^{m}  e^{(i)}x_n^{(i)} \\
\end{bmatrix}
```
I repeat here some elements i need in a matrix multiplication
```math
\boldsymbol{X}
= 
\begin{bmatrix}
    1       & x_1^{(1)} & x_2^{(1)} & \dots     & x_n^{(1)} \\ 
    1       & x_1^{(2)} & x_2^{(2)} & \dots     & x_n^{(2)} \\ 
    \vdots  & \vdots    & \vdots    &\ddots\    & \vdots \\ 
    1       & x_1^{(m)} & x_2^{(m)} & \dots     & x_n^{(m)}
\end{bmatrix}
\in \mathbb{R}^{m \times (n + 1)}
, \qquad
\boldsymbol{X}^{T}
=
\begin{bmatrix}
    1           & 1         & 1         & \dots     & 1 \\
    x_1^{(1)}   & x_1^{(2)} & x_1^{(3)} & \dots     & x_1^{(m)} \\
    x_2^{(1)}   & x_2^{(2)} & x_2^{(3)} & \dots     & x_2^{(m)} \\
    \dots       & \dots     & \dots     & \ddots    & \dots \\
    x_n^{(1)}   & x_n^{(2)} & x_n^{(3)} & \dots     & x_n^{(m)} \\        
\end{bmatrix}
\in \mathbb{R}^{(n + 1) \times m}
, \qquad
\boldsymbol{e}
=
\begin{bmatrix} 
    e^{(1)} \\ 
    e^{(2)} \\ 
    \vdots \\ 
    e^{(m)} 
\end{bmatrix}
\in \mathbb{R}^{m \times 1}
```
To prove that the Gradient $$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$$ result from matrix multiplication
```math
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})
=
\frac{1}{m} \boldsymbol{X}^{T}\boldsymbol{e}
=
\frac{1}{m}
\begin{bmatrix}
    1           & 1         & 1         & \dots     & 1 \\
    x_1^{(1)}   & x_1^{(2)} & x_1^{(3)} & \dots     & x_1^{(m)} \\
    x_2^{(1)}   & x_2^{(2)} & x_2^{(3)} & \dots     & x_2^{(m)} \\
    \dots       & \dots     & \dots     & \ddots    & \dots \\
    x_n^{(1)}   & x_n^{(2)} & x_n^{(3)} & \dots     & x_n^{(m)} \\        
\end{bmatrix}
\begin{bmatrix} 
    e^{(1)} \\ 
    e^{(2)} \\ 
    \vdots \\ 
    e^{(m)} 
\end{bmatrix}
=
\frac{1}{m}
\begin{bmatrix}
    (1       \times e^{(1)}) & +(1         \times e^{(2)})    & \dots     & + (1         \times e^{(m)}) \\
    (x_1^{(1)} \times e^{(1)}) & +(x_1^{(2)} \times e^{(2)})    & \dots     & + (x_1^{(m)} \times e^{(m)}) \\
    \dots               & \dots                     & \ddots    & + \dots \\
    (x_n^{(1)} \times e^{(1)}) & +(x_n^{(2)} \times e^{(2)})    & \dots     & + (x_n^{(m)} \times e^{(m)})   
\end{bmatrix}
=
\frac{1}{m}
\begin{bmatrix}
    \sum_{i=1}^{m}  e^{(i)}x_0^{(i)} \\
    \sum_{i=1}^{m}  e^{(i)}x_1^{(i)} \\
    \vdots \\
    \sum_{i=1}^{m}  e^{(i)}x_n^{(i)} \\
\end{bmatrix}
```
With a unique matrix operation we calculate simultaneously the adjust for all parameters.

After the replacement of $$\boldsymbol{e} = \boldsymbol{X} \boldsymbol{\theta} - \boldsymbol{y}$$ the gradient becomes $$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})=\frac{1}{m}\boldsymbol{X}^{T}\left(\boldsymbol{X} \boldsymbol{\theta} - \boldsymbol{y} \right)$$

This is the practical payoff of the vectorized form: one matrix multiplication computes the gradient for every coefficient at once, without an explicit loop over $n$ or $m$ — simpler to implement and much faster than a naive Python loop.

## Gradient descent
Gradient descent is the algorithm that actually uses this gradient to find the coefficients $\boldsymbol{\theta}$ that minimize the cost function. Since $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ points in the direction where $J$ increases fastest, moving $\boldsymbol{\theta}$ a small step in the opposite direction is guaranteed to decrease the cost, at least locally. Starting from some initial guess (often all zeros), the algorithm repeatedly applies the update rule $\boldsymbol{\theta_{k + 1}} := \boldsymbol{\theta_{k}} - \alpha \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$, where $\alpha$ is the learning rate: a small positive number that controls how large each step is. Too small an $\alpha$ makes convergence painfully slow; too large an $\alpha$ can overshoot the minimum and cause the cost to oscillate or even diverge. Because $J(\boldsymbol{\theta})$ for linear regression is a convex bowl-shaped function, this process is guaranteed to converge toward the single global minimum — the point where the gradient is zero and no further step improves the predictions — as long as $\alpha$ is chosen well and the algorithm is run for enough iterations. Since this derivation uses the whole dataset $\boldsymbol{X}$ in every update, each iteration here is also a full pass over the data — what's commonly called an epoch.

$$\boldsymbol{\theta_{k+1}} := \boldsymbol{\theta_{k}} - \alpha \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) :=\boldsymbol{\theta_{k}} - \alpha \cdot \frac{1}{m}\boldsymbol{X}^{T}(\boldsymbol{X}\boldsymbol{\theta} - \boldsymbol{y})$$

### Stoppping Criteria
#### Change threshold