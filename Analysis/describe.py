import sys
import os
import pandas as pd
import numpy as np
from statistics import ft_statistics, ft_category_statistics


def path_test(path: str) -> str:
    """Validate that a given path points to a readable CSV file.

    Resolves the provided path to an absolute path and runs a series of
    checks to ensure the file exists, is a regular file (not a directory),
    is readable by the current user, and has a .csv extension.

    Args:
        path: A relative or absolute filesystem path to validate.

    Returns:
        The resolved absolute path to the validated CSV file.

    Raises:
        AssertionError: If the path does not exist, is not a regular file,
            is not readable, or does not have a .csv extension.
    """
    abspath = os.path.abspath(path)
    assert os.path.exists(abspath), f"Wrong Path {path}"
    assert os.path.isfile(abspath), f"{path} is not a file"
    assert os.access(abspath, os.R_OK), f"User can not read permit on {path}"
    _, ext = os.path.splitext(abspath)
    assert ext.lower() in (".csv",), \
        f"Expected a CSV file, got '{ext[1:]}'"
    return abspath


def fmt(x):
    """Format a single value for display in the descriptive statistics table.

    Formats floats to six decimal places, renders integers as plain
    strings, and shows missing values as "Nan". Any other type falls
    back to str(). Intended for use as a per-column formatter passed to
    DataFrame.to_string().

    Args:
        x: The value to format. May be a float, an integer, a missing
            value (NaN), or any other type.

    Returns:
        The string representation of x, formatted according to its type.
    """
    if pd.api.types.is_float(x):
        if pd.isna(x):
            return "Nan"
        return f"{x: .6f}"
    if pd.api.types.is_integer(x):
        if pd.isna(x):
            return "Nan"
        return f"{x}"
    if pd.isna(x):
        return "Nan"
    return str(x)


def ft_describe_numeric(feature: str, data: pd.Series, desc: pd.DataFrame):
    """Compute descriptive statistics for a numeric feature.

    This function calculates various descriptive statistics for a given
    numeric feature in a pandas Series, including count, mean, standard
    deviation, minimum, maximum, and percentiles (25%, 50%, 75%).

    Args:
        feature: The name of the feature (column) being analyzed.
        data: A pandas Series containing the numeric data for the feature.
        desc: A pandas DataFrame to store the computed statistics.

    Returns:
        None. The function updates the 'desc' DataFrame in place with
        the computed statistics for the specified feature.
    """
    values = data.dropna().values  # Drop NaN values for accurate statistics
    n = len(values)
    desc.at['count', feature] = float(n)  # Count of non-NaN values
    if n == 0:
        for stat in desc.index[1:]:
            desc.at[stat, feature] = np.nan
        return  # Skip empty columns

    sort_values = sorted(values)
    for stat in desc.index[1:]:
        if stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            desc.at[stat, feature] = ft_statistics(sort_values, n=n, stat=stat)
        else:
            # For 'unique', 'top', 'freq', set as NaN
            desc.at[stat, feature] = np.nan


def ft_describe_categorical(feature: str, data: pd.Series, desc: pd.DataFrame):
    """Compute descriptive statistics for a categorical feature.

    This function calculates various descriptive statistics for a given
    categorical feature in a pandas Series, including count, unique values,
    top value, and frequency of the top value.

    Args:
        feature: The name of the feature (column) being analyzed.
        data: A pandas Series containing the categorical data for the feature.
        desc: A pandas DataFrame to store the computed statistics.

    Returns:
        None. The function updates the 'desc' DataFrame in place with
        the computed statistics for the specified feature.
    """
    values = data.dropna().values
    n = len(values)
    desc.at['count', feature] = n  # Count of non-NaN values
    if n == 0:
        for stat in desc.index[1:]:
            desc.at[stat, feature] = np.nan
        return  # Skip empty columns
    unique_count, top_value, frequency = ft_category_statistics(values, n=n)
    desc.at['unique', feature] = unique_count
    desc.at['top', feature] = top_value
    desc.at['freq', feature] = frequency
    # For 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
    # set as NaN or handle separately if needed

    for stat in desc.index[1:]:
        if stat not in ['unique', 'top', 'freq']:
            desc.at[stat, feature] = np.nan


def ft_describe_other(feature: str, data: pd.Series, desc: pd.DataFrame):
    """Compute descriptive statistics for a feature of other types.

    This function calculates various descriptive statistics for a given
    feature in a pandas Series that is neither numeric nor categorical.
    It computes the count of non-NaN values and sets other statistics to NaN.

    Args:
        feature: The name of the feature (column) being analyzed.
        data: A pandas Series containing the data for the feature.
        desc: A pandas DataFrame to store the computed statistics.

    Returns:
        None. The function updates the 'desc' DataFrame in place with
        the computed statistics for the specified feature.
    """
    values = data.dropna().values
    n = len(values)
    desc.at['count', feature] = n  # Count of non-NaN values
    if n == 0:
        for stat in desc.index[1:]:
            desc.at[stat, feature] = np.nan
        return  # Skip empty columns

    # Set all other stats to NaN for unsupported types
    for stat in desc.index[1:]:
        desc.at[stat, feature] = np.nan


def ft_describe(dataset_path: str) -> pd.DataFrame:
    """Perform descriptive analysis on a CSV dataset.

    This function reads a CSV file into a pandas DataFrame and computes
    descriptive statistics for each column, including count, unique, top, freq,
    mean, std, min, 25%, 50%, 75%, and max values.

    Args:
        dataset_path: The path to the CSV dataset file.

    Returns:
        A pandas DataFrame containing the dataset's descriptive statistics.
    """
    df = pd.read_csv(dataset_path)
    desc_index = ['count', 'unique', 'top', 'freq',
                  'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    desc_columns = df.columns.tolist()
    # Initialize an empty DataFrame for descriptive stats

    desc = pd.DataFrame(np.zeros((len(desc_index), len(desc_columns))),
                        index=desc_index, columns=desc_columns,
                        dtype=object)
    for feature in desc_columns:
        if pd.api.types.is_numeric_dtype(df[feature].dtype):
            ft_describe_numeric(feature, df[feature], desc)
        elif pd.api.types.is_string_dtype(df[feature].dtype):
            ft_describe_categorical(feature, df[feature], desc)
        else:
            ft_describe_other(feature, df[feature], desc)
    return desc


def main(path: str):
    print(f"Descriptive Analysis of {path}")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)   # don't wrap based on terminal width
    pd.set_option('display.max_rows', None)
    desc = ft_describe(path)
    print(desc.to_string(formatters={col: fmt for col in desc.columns}))
    file = os.path.splitext(os.path.basename(path))[0] + "_describe.txt"
    with open(file, "w") as f:
        f.write(desc.to_string())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset_file>")
        sys.exit(1)

    try:
        main(path_test(sys.argv[1]))
    except AssertionError as e:
        print(f"Error: {e}")
        sys.exit(1)
