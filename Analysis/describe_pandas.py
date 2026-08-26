import sys
import os
import pandas as pd


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


def main(path: str):
    print(f"Descriptive Analysis of {path}")
    df = pd.read_csv(path)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)   # don't wrap based on terminal width
    pd.set_option('display.max_rows', None)
    desc = df.describe(include='all')
    print(desc)
    file = os.path.splitext(os.path.basename(path))[0] + "_describe_pandas.txt"
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
