import sys
import os
import pandas as pd
from sklearn.metrics import accuracy_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from aux_funcs.file_func import path_test  # noqa: E402
from aux_funcs.logrec_func import load_parameters  # noqa: E402
from aux_funcs.logrec_func import predict  # noqa: E402


def main(dataset_path: str, weight_file_path: str):
    print("Logistic Regression test")
    df = pd.read_csv(dataset_path)
    Y = df['Hogwarts House']
    df_num = df.select_dtypes(include=['float64', 'int64'])
    df_num["Index"] = 1  # bias term
    df_num.rename(columns={"Index": "Bias"}, inplace=True)

    m, n = df_num.shape
    print(f"Rows in the dataset(m): {m} features(n): {n - 1}   ")
    X = df_num.to_numpy()
    parametres = load_parameters(weight_file_path)
    predictions = predict(X, parametres)
    print(predictions)
    for i, prediction in enumerate(predictions):
        print(f"{i:3}.-{prediction}")
    print(f"Accuracy Score {accuracy_score(Y, predictions):.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test.py <dataset_file> <weight_file>"
              "<tolerance>")
        sys.exit(1)

    try:
        main(path_test(sys.argv[1]),  # dataset path
             path_test(sys.argv[2]))        # weight file path
    except AssertionError as e:
        print(f"Error: {e}")
        sys.exit(1)
