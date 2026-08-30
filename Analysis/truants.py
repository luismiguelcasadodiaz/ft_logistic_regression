import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from aux_funcs.file_func import path_test  # noqa: E402


def ft_truants(df: pd.DataFrame) -> pd.DataFrame:
    """Report missing values ("truants") per course and per student.

    For each feature column, finds the students (by "Index") missing a
    value and prints, per course, how many are missing and the resulting
    non-null count. It then inverts that mapping to count, per student,
    how many courses they're missing a value in, groups students by that
    count, and prints how many students fall into each group along with
    the dataset size that would remain after dropping every row with at
    least one NaN. Finally, prints the name of the student with the most
    missing values.

    Args:
        df: The dataset as a pandas DataFrame. Must contain an "Index"
            column identifying each row/student, plus the feature columns
            to check for missing values.

    Returns:
        None. All results are printed directly to stdout.
    """
    margin = 26
    features = df.columns.tolist()
    # detect absentees by feature
    absentees = {}
    for feat in features[1:]:
        nan_index = df[df[feat].isna()]["Index"].values.tolist()
        absentees[feat] = (len(nan_index), nan_index)
    print("=" * margin + " Absentees by course " + "=" * margin)
    for feature, students in absentees.items():
        print(f"{feature:32} had {students[0]:2} absentees, ", end="")
        print(f"describe counts {df.shape[0] - students[0]:4}.")
    students_ns = {}
    for k, v in absentees.items():
        for absentee in v[1]:
            students_ns[absentee] = students_ns.get(absentee, 0) + 1
    print("=" * margin + " Different absentees " + "=" * margin)
    print(f"{len(students_ns)} students has at least one no show")
    truant = {}
    for k, v in students_ns.items():
        # truant[v] = truant.get(v, []).append(k)
        truant.setdefault(v, []).append(k)
    for k, v in truant.items():
        print(f"{len(v):3} students did not show to {k} features")
    print("After droping all NaNs the remainig dataset will have ", end="")
    print(f"{df.shape[0] - len(students_ns)} rows")
    print("=" * margin + " The ghost  students " + "=" * margin)
    print("The winner in the truant ranking is")
    print(df[df['Index'] == list(truant.keys())[-1]][['First Name',
                                                     'Last Name']])


def main(dataset_path):
    """Run the missing-value ("truant") analysis for a CSV dataset.

    Reads the dataset into a DataFrame and prints a per-course and
    per-student report of missing values via ft_truants(), then reports
    the shape of the dataset after dropping every row that contains at
    least one NaN value.

    Args:
        dataset_path: The path to the CSV dataset file to analyze.

    Returns:
        None. Prints the truancy report and the post-dropna dataset shape
        to stdout.
    """
    print("NaN analysys")
    df = pd.read_csv(dataset_path)
    data = ft_truants(df)
    data = df.dropna()
    print(f"Data shape after dropping NaN values: {data.shape}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset_file>")
        sys.exit(1)

    try:
        main(path_test(sys.argv[1]))
    except AssertionError as e:
        print(f"Error: {e}")
        sys.exit(1)
