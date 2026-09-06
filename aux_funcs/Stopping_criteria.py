import numpy as np


def threshold_criteria(cost: float, p_cost: float, threshold: float) -> bool:
    """
    Determines whether the stopping criteria for an optimization algorithm have
    been met.

    Parameters:
    - cost: The current cost value.
    - prev_cost: The previous cost value.
    - threshold: The threshold for determining convergence.

    Returns:
    - True if the stopping criteria are met (i.e., the change in cost is less
    than the threshold), False otherwise.
    """
    cost_diff = abs(cost - p_cost)
    print(f"Cost difference: {cost_diff}, Threshold: {threshold:.6f}")
    return cost_diff < threshold


def Magnitude_criteria(gradient: np.ndarray, threshold: float) -> bool:
    """
    Determines whether the stopping criteria for an optimization algorithm have
    been met based on the magnitude of the gradient.

    Parameters:
    - gradient: The current gradient vector.
    - threshold: The threshold for determining convergence.

    Returns:
    - True if the stopping criteria are met (i.e., gradiente magnitude
      is less than the threshold), False otherwise.
    """
    # normal = np.linalg.norm(gradient)
    # print(f"Gradient magnitude: {normal}, Threshold: {threshold:.6f}")
    return np.linalg.norm(gradient) < threshold


def parameter_change_criteria(parameters: np.ndarray,
                              prev_parameters: np.ndarray,
                              threshold: float) -> bool:
    """
    Determines whether the stopping criteria for an optimization algorithm have
    been met based on the change in parameters.

    Parameters:
    - parameters: The current parameter vector.
    - prev_parameters: The previous parameter vector.
    - threshold: The threshold for determining convergence.

    Returns:
    - True if the stopping criteria are met (i.e., the change in parameters is
    less than the threshold), False otherwise.
    """
    return np.linalg.norm(parameters - prev_parameters) < threshold


def early_stopping_criteria(loss: float, prev_loss: float,
                            pat_counter: int, pat_limit: int) -> bool:
    """
    Determines whether the stopping criteria for an optimization algorithm have
    been met based on early stopping.

    Parameters:
    - loss: The current validation loss value.
    - prev_loss: The previous validation loss value.
    - pat_counter: The current count of epochs without improvement.
    - pat_limit: The maximum number of epochs to wait for improvement
      before stopping.

    Returns:
    - True if the stopping criteria are met (i.e., the validation loss has not
      improved for a number of epochs equal to the patience limit),
      False otherwise.
    """
    if loss < prev_loss:
        return False  # Reset patience counter if there's an improvement
    else:
        pat_counter += 1
        return pat_counter >= pat_limit
