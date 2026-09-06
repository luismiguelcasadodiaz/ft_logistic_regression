import numpy as np
from numpy.typing import NDArray
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from aux_funcs.Stopping_criteria import Magnitude_criteria  # noqa: E402


def sigmoid(z):
    """Compute the sigmoid of z."""
    return 1 / (1 + np.exp(-z))


def save_parameters(class_label, parameters: NDArray[np.float64]):
    """
    Save the learned parameters for a specific class label to a JSON file.

    Args:
        class_label: The class label for which the parameters are saved.
        parameters: The learned parameters (weights) for the class label.

    Returns:
        None. Saves the parameters to a JSON file named
        "weights_<class_label>.json".
    """
    import json
    filename = f"weights_{class_label}.json"
    with open(filename, 'w') as f:
        json.dump(parameters.tolist(), f)


def join_saved_parameters_to_json(class_labels):
    """
    Join the saved parameters for all class labels into a single JSON file.

    Args:
        class_labels: List of class labels for which the parameters are saved.

    Returns:
        None. Saves the joined parameters to a JSON file named "weights.json".
        deletes the individual weight files after joining.
    """

    all_parameters = {}
    for class_label in class_labels:
        filename = f"weights_{class_label}.json"
        with open(filename, 'r') as f:
            parameters = json.load(f)
            all_parameters[class_label] = parameters
        os.remove(filename)  # Delete the individual weight file after joining

    with open("datasets/weights.json", 'w') as f:
        json.dump(all_parameters, f)


def load_parameters(weight_file_path: str):
    """
    Load the learned parameters from a JSON file.

    Args:
        weight_file_path: The path to the JSON file containing the learned
        parameters.

    Returns:
        A dictionary where keys are class labels and values are the learned
        parameters (weights) for each class label.
    """
    with open(weight_file_path, 'r') as f:
        all_parameters = json.load(f)
    return all_parameters


def train_one_vs_all(X, X_transpose, df_Y, class_label,
                     epochs, learning_rate, tolerance) -> np.ndarray:
    """
    Train a logistic regression model for a specific class label using the
    one-vs-all approach.

    Args:
        X: The feature matrix (np.array) of shape (m, n).
        X_transpose: The transposed feature matrix (np.array) of shape (n, m).
        df_Y: The target labels (pandas Series) of shape (m,).
        class_label: The class label for which the model is trained.
        epochs: The number of training epochs.
        learning_rate: The learning rate for gradient descent.
        tolerance: The tolerance for convergence (stopping criteria).

    Returns:
        The learned parameters (weights) for the specific class label as a
        numpy array of shape (n,).
    """
    m, n = X.shape
    print(f"training {m} x {n}")
    # Initialize parameters (weights) to zeros
    parameters: NDArray = np.zeros(n)

    # Create binary labels for the current class label
    y_binary = (df_Y == class_label).astype(int).to_numpy()
    required_epochs = epochs
    for epoch in range(epochs):
        # Compute the linear combination of inputs and weights
        z = np.dot(X, parameters)
        # Apply the sigmoid function to get predictions
        predictions = sigmoid(z)

        # Compute the gradient
        gradient = np.dot(X_transpose, (predictions - y_binary)) / m

        # Update parameters using gradient descent
        parameters -= learning_rate * gradient

        # Check for convergence using the threshold criteria
        if Magnitude_criteria(gradient, tolerance):
            print(f"Converged at {epoch + 1} epoch for class '{class_label}'.")
            required_epochs = epoch + 1
            break
    print(f"'{class_label}' class trained in {required_epochs} epochs.")
    return parameters


def predict(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """
    Predict the class labels for the given feature matrix using the learned
    parameters.

    Args:
        X: The feature matrix (np.array) of shape (m, n).
        parameters: The learned parameters (weights) for each class label as a
        dictionary where keys are class labels and values are numpy arrays of
        shape (n,).

    Returns:
        A numpy array of predicted class labels of shape (m,).
    """
    m, n = X.shape
    print(f"predicting {m} x {n}")
    predictions = np.zeros(m, dtype=object)

    # Compute the probability for each class label
    probabilities = {}
    for class_label, param in parameters.items():
        z = np.dot(X, param)  # (m, n) dot (n,) -> (m,)
        probabilities[class_label] = sigmoid(z)

    # Assign the class label with the highest probability to each sample
    for i in range(m):
        max_prob = -1
        best_class = None
        for class_label, prob in probabilities.items():
            if prob[i] > max_prob:
                max_prob = prob[i]
                best_class = class_label
        predictions[i] = best_class

    return predictions
