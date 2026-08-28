# ft_logistic_regression
Data Science × Logistic Regression - "Harry Potter and the Data Scientist"

## Data Analysis

I figured out the data structure supporting my describe() output as a Pandas DataFrame. So I created a Zero-filled DataFrame whose index labels had the name of the calculated statistic.

```python
df = pd.read_csv(dataset_path)
desc_index = ['count', 'unique', 'top', 'freq', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
desc_columns = df.columns.tolist()
desc= pd.DataFrame(np.zeros((len(desc_index), len(desc_columns))), index=desc_index, columns=desc_columns)
# Initialize an empty DataFrame for descriptive stats
```

### top
I started this calculation with a discrepancy from Pandas' describe() output. My first approach returned `Allan` as the top/most frequent value with two occurrences. This contradicted Pandas' official describe() output, which returned `Nathanael`.

Pandas' describe() tie-breaks with first-seen order. This is the same behaviour as the `Counter` class from the `collections` library, which I used.
```python
counter = Counter(data)
most_common = counter.most_common(1)
return most_common[0][0] if most_common else None
```
My mistake was that I had sorted the feature's values for the numeric ones. I was also using the sorted values for the categorical features. It was incorrect.

At this point, I realised that using the Counter class might be considered cheating, so I refactored the code.

```python
stat={}
for e in data:
    stat[e] = stat.get(e, 0) + 1
stat_sorted = sorted(stat.items(), key=lambda item: item[1],reverse=True)
unique_count = len(stat_sorted)
top_value = stat_sorted[0][0]
frequency = stat_sorted[0][1]
return (unique_count, top_value, frequency)
```


### Standard deviation
Heads up here. I got a difference of 0.144608 from the std calculated by the original pandas DataFrame describe() method.

By default, pandas' `.std()` (and `.describe()`, which calls it internally) computes the sample standard deviation, dividing by `N-1` instead of `N`.

#### 1.-Mean
$$\bar{x}= \frac{1}{n} \sum_{i=1}^{n} x_i$$
```python
mean = sum(data) / n
```

#### 2.-Deviations from the mean
$$d_i = x_i - \bar{x}, \quad i = 1, \quad \dots \quad, n$$


```python
v_minus_mean = [x - mean for x in data]
```

#### 3.-Squared deviations
$$d_i^2 = (x_i - \bar{x})^2$$
```python
squared_minus_mean = [x * x for x in v_minus_mean]
```

#### 4.- Sample standard deviation (n - 1) vs Population standard deviation (n)



##### 4a.- Sample standard deviation (n - 1)
I got 115.614301, the right value.

$$s^2 = \frac{1}{n - 1} \sum_{i=1}^{n}(x_i - \bar{x})^2$$
```python
ft_mean(squared_minus_mean, n - 1)
```

##### 4b.- Population standard deviation (n)
I got 115.469693, a small error.

$$s^2 = \frac{1}{n} \sum_{i=1}^{n}(x_i - \bar{x})^2$$
```python
ft_mean(squared_minus_mean, n)
```

# Multivariate linear regression
## Prediction
The equation to predict the value (a real number) of the target variable for one element from the dataset, one student, is:

$$h_\theta(x) = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n \qquad (1)$$

where $\hat{y} = h_\theta(x)$ the target feature or prediction. Such prediction is a real number in linear regression.

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

and transform (1) into a vector multiplication.

$$h_\theta(x) = \sum_{j=0}^{n}\theta_j x_j$$

```math\boldsymbol{\theta} \cdot \boldsymbol{x} = \boldsymbol{\theta}^{T} \boldsymbol{x} = \begin{bmatrix} \theta_0 & \theta_1  & \dots & \theta_n \end{bmatrix} \begin{bmatrix} x_0 \\ x_1  \\ \vdots \\ x_n \end{bmatrix} = \sum_{j=0}^{n}\theta_j x_j```


Always $x_0 = 1$, so the product $x_0\theta_0$ returns $\theta_0$.

When the dataset's size is $m$, we can write the vector holding $m$ predictions as the product of the feature matrix and the coefficients vector.
$$\hat{y} = \boldsymbol{X} \boldsymbol{\theta}, \qquad 
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
    \theta_m 
\end{bmatrix}$$

where
$$\hat{y}, \boldsymbol{\theta} \in \mathbb(R)^{m}, \qquad \boldsymbol{X} \in \mathbb{R}^{m \times (n+1)}$$
## Cost function
The cost function calculates the error between the predicted value and the real one.
```math
J(\boldsymbol{\theta})
= 
\frac{1}{2m} \sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2
=
\frac{1}{2m} \sum_{i=1}^{m} \left(h_\theta{(x)^{(i)}} - y^{(i)}\right)^2
=
\frac{1}{2m} \sum_{i=1}^{m} \left(\boldsymbol{\theta}^{T}x^{(i)} - y^{(i)}\right)^2
```
The addition of the squares of the components of a vector equals the scalar product of such vector with itself.

$$
\boldsymbol{e}
=
\begin{bmatrix} 
    e^{(1)} \\ 
    e^{(2)} \\ 
    \vdots \\ 
    e^{(m)} 
\end{bmatrix} 

