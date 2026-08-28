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

