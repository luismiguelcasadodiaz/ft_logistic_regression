import sys
import os
import pandas as pd
import numpy as np
from statistics import ft_statistics


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
    assert ext.lower() in (".csv"), \
        f"Expected a CSV file, got '{ext[1:]}'"
    return abspath


def ft_describe(dataset_path: str) -> pd.DataFrame:
    """Perform descriptive analysis on a CSV dataset.

    This function reads a CSV file into a pandas DataFrame and computes
    descriptive statistics for each column, including count, mean, std,
    min, 25%, 50%, 75%, and max values.

    Args:
        dataset_path: The path to the CSV dataset file.

    Returns:
        A pandas DataFrame containing the descriptive statistics of the dataset.
    """
    df = pd.read_csv(dataset_path)
    desc_index = ['count', 'unique', 'top', 'freq', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    desc_columns = df.columns.tolist()
    desc= pd.DataFrame(np.zeros((len(desc_index), len(desc_columns))), \
                       index=desc_index, columns=desc_columns, \
                        dtype=object)  # Initialize an empty DataFrame for descriptive stats

    for feature in desc_columns:
  
        data = df[feature].dropna().values  # Drop NaN values for accurate statistics
        sorted_values = sorted(data) # Sort the data for percentile calculations
        n = len(sorted_values)
        if n == 0:
            continue  # Skip empty columns
        desc.at[desc_index[0], feature] = n  # Count of non-NaN values
        print(f"Column: {feature}, Type: {df[feature].dtype}")
        for stat in desc_index[1:]:
            if df[feature].dtype in [np.float64, np.float32, np.int64, np.int32] \
                  and stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                desc.at[stat, feature] = ft_statistics(sorted_values, n=n, stat=stat)  # Use custom statistics function for each stat
            elif df[feature].dtype == "str" \
                  and stat in ['unique', 'top', 'freq']:
                desc.at[stat, feature] = ft_statistics(sorted_values, n=n, stat=stat)  # Use custom statistics function for each stat
            else:
                desc.at[stat, feature] = np.nan  # For 'unique', 'top', 'freq', set as NaN or handle separately if needed
    return desc

def main(dataset_path: str):
    print(f"Descriptive Analysis of {dataset_path}")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)       # don't wrap based on terminal width
    pd.set_option('display.max_rows', None)
    desc = ft_describe(dataset_path)
    print(desc)
    output_file = os.path.splitext(os.path.basename(dataset_path))[0] + "_describe.txt"
    with open(output_file, "w") as f:
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
