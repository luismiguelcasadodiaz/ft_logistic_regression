import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from aux_funcs.file_func import path_test  # noqa: E402
from aux_funcs.logrec_func import load_parameters  # noqa: E402
from aux_funcs.logrec_func import predict  # noqa: E402
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../Analysis")))
from Analysis.describe import ft_normalize_data  # noqa: E402


def main(dataset_path: str, weight_file_path: str, describe_path: str):
    print("Logistic Regression Prediction")
    desc = pd.read_csv(describe_path, index_col=0)
    print(desc.columns)
    df_norm = ft_normalize_data(dataset_path, desc)
    df_norm_num = df_norm.select_dtypes(include=['float64', 'int64'])
    df_norm_num["Index"] = 1  # bias term
    df_norm_num.rename(columns={"Index": "Bias"}, inplace=True)
    # Drop columns in test dataset that do not exist in trained model
    # Remember that records with NaN were substituted by average, and a missing
    # bit added in a column.
    drop = ["Hogwarts House", "Hogwarts House_missing",
            "Charms_missing", "Flying_missing"]
    df_norm_num = df_norm_num.drop(columns=drop)
    m, n = df_norm_num.shape
    X = df_norm_num.to_numpy()
    parametres = load_parameters(weight_file_path)
    predictions = predict(X, parametres)
    print(predictions)
    for i, prediction in enumerate(predictions):
        print(f"{i:3}.-{prediction}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python test.py <dataset_file> <weight_file>"
              "<describe info>")
        sys.exit(1)

    try:
        main(path_test(sys.argv[1]),  # dataset path
             path_test(sys.argv[2]),  # weight file path
             path_test(sys.argv[3]))  # descibe info path
    except AssertionError as e:
        print(f"Error: {e}")
        sys.exit(1)
