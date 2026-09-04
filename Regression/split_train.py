import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from aux_funcs.file_func import path_test  # noqa: E402


def main(dataset_path: str, percentage: float):
    """Split a dataset into stratified train/test CSVs by Hogwarts House.

    Reads the dataset and, for each house, shuffles its rows (fixed
    random_state=42 for reproducibility) and splits them into a training
    and a testing subset according to percentage, so that each
    house's original class balance is preserved in both subsets
    (stratified split). The per-house train and test subsets are then
    concatenated and reshuffled into two combined DataFrames. For each
    house, prints the row count and percentage of the total in the
    original dataset, the training set, and the test set. Finally, saves
    the training and test DataFrames as "<dataset_basename>_to_train.csv"
    and "<dataset_basename>_to_test.csv" in the same directory as the
    input file.

    Args:
        dataset_path: The path to the CSV dataset file to split.
        percentage: The percentage (0-100) of each house's rows to
            allocate to the training set; the remainder goes to the test
            set.

    Returns:
        None. Prints a per-house split summary to stdout and writes the
        train and test subsets to two CSV files.
    """
    print(f"Logistic Regression Training with {percentage}% for training")
    df = pd.read_csv(dataset_path)
    metrics = {}
    train_subsets = {}
    test_subsets = {}
    for house, group in df.groupby('Hogwarts House'):
        # metrics[house] accumulates [count, pct] triples in this order:
        # [0] = original dataset, [1] = training set, [2] = test set
        metrics[house] = [[len(group), len(group)/len(df)*100]]
        # Shuffle the group
        group = group.sample(frac=1, random_state=42).reset_index(drop=True)
        train_size = int(len(group) * (percentage / 100))
        train_subsets[house] = group.iloc[:train_size]
        test_subsets[house] = group.iloc[train_size:]

    # Re-shuffle after concatenating so rows aren't grouped by house
    train_df = pd.concat(
        train_subsets.values()).sample(
            frac=1,
            random_state=42).reset_index(drop=True)

    for house, group in train_df.groupby('Hogwarts House'):
        metrics[house].append([len(group), len(group)/len(train_df)*100])

    # Re-shuffle after concatenating so rows aren't grouped by house
    test_df = pd.concat(
        test_subsets.values()).sample(
            frac=1,
            random_state=42).reset_index(drop=True)
    for house, group in test_df.groupby('Hogwarts House'):
        metrics[house].append([len(group), len(group)/len(test_df)*100])
    for k, v in metrics.items():
        print(f"{k:<15}: ",
              f"Original: {v[0][0]:>5} ({v[0][1]:.2f}%),"
              f"Train: {v[1][0]:>5} ({v[1][1]:.2f}%),"
              f"Test: {v[2][0]:>5} ({v[2][1]:.2f}%)")
    new_path = os.path.join(
        os.path.dirname(dataset_path),
        os.path.splitext(os.path.basename(dataset_path))[0] + "_to_train.csv")
    train_df.to_csv(new_path, index=False)
    print(f"Training set saved to: {new_path}")
    new_path = os.path.join(
        os.path.dirname(dataset_path),
        os.path.splitext(os.path.basename(dataset_path))[0] + "_to_test.csv")
    test_df.to_csv(new_path, index=False)
    print(f"Testing  set saved to: {new_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_train.py <dataset_file> <percentage>")
        sys.exit(1)

    try:
        main(path_test(sys.argv[1]), float(sys.argv[2]))
    except AssertionError as e:
        print(f"Error: {e}")
        sys.exit(1)
