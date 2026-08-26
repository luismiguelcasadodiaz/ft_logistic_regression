"""Descriptive statistics module.

Provides functions for computing common statistical measures — mean,
median, quartiles, variance, standard deviation, and arbitrary
percentiles — on numeric datasets. Includes a dispatcher function
that accepts data as positional arguments and selects operations
via keyword arguments.
"""
from typing import Any
import numpy as np


def ft_percentile(percentile: float, data: Any, n: int) -> float:
    """Compute a percentile value using linear interpolation.

    If the computed index falls exactly on a data point, that value is
    returned. Otherwise, linearly interpolates between the two nearest
    data points.

    Args:
        percentile: A float between 0 and 1 representing the desired
            percentile (e.g., 0.25 for the 25th percentile).
        data: A sorted sequence of numeric values.
        n: The number of elements in data.

    Returns:
        The interpolated value at the given percentile.
    """
    index = percentile * (n - 1)
    if index % 1 == 0:
        return float(data[int(index)])
    else:
        idx = int(index)
        a = data[idx]
        b = data[idx + 1]
        diff = b - a
        return a + diff * (index % 1)


def ft_quartile(data: Any, n: int) -> list:
    """Compute the first and third quartiles (Q1 and Q3).

    Args:
        data: A sorted sequence of numeric values.
        n: The number of elements in data.

    Returns:
        A list of two floats: [Q1, Q3].
    """
    return [ft_percentile(0.25, data, n), ft_percentile(0.75, data, n)]


def ft_25_percentile(data: Any, n: int) -> float:
    """Compute the 25th percentile (first quartile).

    Args:
        data: A sorted sequence of numeric values.
        n: The number of elements in data.

    Returns:
        The 25th percentile as a float.
    """
    return ft_percentile(0.25, data, n)


def ft_50_percentile(data: Any, n: int) -> float:
    """Compute the 50th percentile (median).

    Args:
        data: A sorted sequence of numeric values.
        n: The number of elements in data.

    Returns:
        The 50th percentile as a float.
    """
    return ft_percentile(0.50, data, n)


def ft_75_percentile(data: Any, n: int) -> float:
    """Compute the 75th percentile (third quartile).

    Args:
        data: A sorted sequence of numeric values.
        n: The number of elements in data.

    Returns:
        The 75th percentile as a float.
    """
    return ft_percentile(0.75, data, n)


def ft_min(data: Any, n: int) -> float:
    """Compute the minimum value.

    Args:
        data: A sorted sequence of numeric values.
        n: The number of elements in data.

    Returns:
        The minimum value as a float.
    """
    partial_min = float('inf')
    for num in data:
        if num < partial_min:
            partial_min = num
    return partial_min


def ft_max(data: Any, n: int) -> float:
    """Compute the maximum value.

    Args:
        data: A sorted sequence of numeric values.
        n: The number of elements in data.

    Returns:
        The maximum value as a float.
    """
    partial_max = float('-inf')
    for num in data:
        if num > partial_max:
            partial_max = num
    return partial_max


def ft_count(data: Any, n: int) -> float:
    """Compute the count of elements in the dataset.

    Args:
        data: A sequence of numeric values.
        n: The number of elements in data.

    Returns:
        The arithmetic mean as a float.
    """
    return sum(data) / n


def ft_mean(data: Any, n: int) -> float:
    """Compute the arithmetic mean.

    Args:
        data: A sequence of numeric values.
        n: The number of elements in data.

    Returns:
        The arithmetic mean as a float.
    """
    return sum(data) / n


def ft_median(data: Any, n: int) -> float:
    """Compute the median of a sorted dataset.

    For odd-length data, returns the middle element. For even-length
    data, returns the average of the two middle elements.

    Args:
        data: A sorted sequence of numeric values.
        n: The number of elements in data.

    Returns:
        The median value as a float.
    """
    if n % 2 == 1:
        return data[n // 2]
    else:
        return (data[(n // 2) - 1] + data[n // 2]) / 2


def ft_variance(data: Any, n: int) -> float:
    """Compute the population variance.

    Calculates the average of the squared deviations from the mean.

    Args:
        data: A sequence of numeric values.
        n: The number of elements in data.

    Returns:
        The population variance as a float.
    """
    mean = ft_mean(data, n)
    v_minus_mean = [x - mean for x in data]
    squared_minus_mean = [x * x for x in v_minus_mean]
    return (ft_mean(squared_minus_mean, n - 1))


def ft_std(data: Any, n: int) -> float:
    """Compute the population standard deviation.

    Returns the square root of the population variance.

    Args:
        data: A sequence of numeric values.
        n: The number of elements in data.

    Returns:
        The population standard deviation as a float.
    """
    return ft_variance(data, n) ** (1/2)


def ft_unique(data: Any, n: int) -> float:
    """Compute the number of unique values in the dataset.

    Args:
        data: A sequence of numeric values.
        n: The number of elements in data.

    Returns:
        The count of unique values as a float.
    """
    return ft_category_statistics(data, n)[0]


def ft_top(data: Any, n: int) -> float:
    """Compute the most frequent value (mode) in the dataset.

    Args:
        data: A sequence of numeric values.
        n: The number of elements in data.

    Returns:
        The most frequent value as a float. If there are multiple modes,
        returns one of them arbitrarily.
    """
    return ft_category_statistics(data, n)[1]


def ft_freq(data: Any, n: int) -> float:
    """Compute the frequency of the most frequent value (mode) in the dataset.

    Args:
        data: A sequence of numeric values.
        n: The number of elements in data.

    Returns:
        The frequency of the most frequent value as a float. If there are
        multiple modes, returns the frequency of one of them arbitrarily.
    """
    return ft_category_statistics(data, n)[2]


def ft_category_statistics(data: Any, n: int) -> tuple:
    """Compute the number of unique values, the most frequent value (mode),
    and the frequency of the most frequent value in a categorical dataset.

    Args:
        data: A sequence of categorical values.
        n: The number of elements in data.
    Returns:
        A tuple containing:
            - The count of unique values (int)
            - The most frequent value (mode) (any type)
            - The frequency of the most frequent value (int)
    """
    d = {}
    for e in data:
        d[e] = d.get(e, 0) + 1
    stat_sorted = sorted(d.items(), key=lambda item: item[1], reverse=True)
    unique_count = len(stat_sorted)
    top_value = stat_sorted[0][0]
    frequency = stat_sorted[0][1]
    return (unique_count, top_value, frequency)


def ft_statistics(*args: Any, n: int, **kwargs: Any) -> np.float64:
    """Compute and print selected statistics on a numeric dataset.

    Accepts numeric values as positional arguments and operation names
    as keyword argument values. Sorts the data, then dispatches each
    requested operation via an internal lookup table.

    Valid operation names: "mean", "median", "quartile", "std", "var".

    Args:
        *args: Numeric values (int or float) forming the dataset.
        n: The number of elements in the dataset.
        **kwargs: Keyword arguments whose values specify which
            statistics to compute. The keys are ignored; only the
            values are used for dispatch.

    Raises:
        AssertionError: If any positional argument is not an int
            or float.

    Note:
        Prints "ERROR" and returns early if no data is provided,
        or prints "ERROR" for any unrecognized operation name.
    """
    """assert all(isinstance(valor, int) or
               isinstance(valor, float) for valor in args), \
        "Not all values are numbers"
    """
    dispatcher = {"mean": ft_mean,
                  "median": ft_median,
                  "quartile": ft_quartile,
                  "std": ft_std,
                  "var": ft_variance,
                  "count": ft_count,
                  'min': ft_min,
                  '25%': ft_25_percentile,
                  '50%': ft_50_percentile,
                  '75%': ft_75_percentile,
                  'max': ft_max,
                  'unique': ft_unique,
                  'top': ft_top,
                  'freq': ft_freq,
                  'category': ft_category_statistics}
    for k, v in kwargs.items():
        if v in dispatcher:
            return dispatcher[v](args[0], n)
        else:
            return np.nan


def main():
    """Run example statistics computations to demonstrate ft_statistics."""
    ft_statistics(1, 42, 360, 11, 64,
                  toto="mean", tutu="median", tata="quartile")
    print("-----")
    ft_statistics(5, 75, 450, 18, 597, 27474, 48575,
                  hello="std", world="var")
    print("-----")
    ft_statistics(5, 75, 450, 18, 597, 27474, 48575,
                  ejfhhe="heheh", ejdjdejn="kdekem")
    print("-----")
    ft_statistics(toto="mean", tutu="median", tata="quartile")


if __name__ == "__main__":
    main()
