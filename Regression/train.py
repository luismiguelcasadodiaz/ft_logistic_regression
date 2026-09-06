import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from aux_funcs.file_func import path_test  # noqa: E402
from aux_funcs.logrec_func import train_one_vs_all  # noqa: E402
from aux_funcs.logrec_func import save_parameters  # noqa: E402
from aux_funcs.logrec_func import join_saved_parameters_to_json  # noqa: E402


def main(path: str, epochs: int, learning_rate: float, tolerance: float):
    print("Logistic Regression Training")
    df = pd.read_csv(path)
    df_Y = df['Hogwarts House']
    df_num = df.select_dtypes(include=['float64', 'int64'])
    df_num["Index"] = 1  # bias term
    df_num.rename(columns={"Index": "Bias"}, inplace=True)
    m, n = df_num.shape
    print(f"Rows in the dataset(m): {m} features(n): {n - 1}   ")
    print(df_num.columns)
    X = df_num.to_numpy()
    X_transpose = X.T
    for class_label in df_Y.unique():
        class_parameters = train_one_vs_all(X, X_transpose, df_Y, class_label,
                                            epochs, learning_rate, tolerance)
        save_parameters(class_label, class_parameters)
    join_saved_parameters_to_json(df_Y.unique())


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python train.py <dataset_file> <epochs> <learning_rate>"
              "<tolerance>")
        sys.exit(1)

    try:
        main(path_test(sys.argv[1]),  # dataset path
             int(sys.argv[2]),        # epoch
             float(sys.argv[3]),      # learnining_rate
             float(sys.argv[4]))      # tolerance
    except AssertionError as e:
        print(f"Error: {e}")
        sys.exit(1)
