import sys
import numpy as np

from util import dataprocessing as dp
from util import perceptron_util
import os

# lines 10, 11 imported because they're used in original code
import math
import matplotlib.pyplot as plt


class state_class:
    def __init__(self, beta, x):
        self.beta = beta
        self.x = x

def initial_state():
    """Return the initial state for the perceptron.

    This function computes and then returns the initial state of the perceptron.
    Feel free to use any data type (dicts, lists, tuples, or custom classes) to
    contain the state of the perceptron.

    """
    # *** START CODE HERE ***
    return []
    # *** END CODE HERE ***


def predict(state, kernel, x_i):
    """Peform a prediction on a given instance x_i given the current state
    and the kernel.

    Args:
        state: The state returned from initial_state()
        kernel: A binary function that takes two vectors as input and returns
            the result of a kernel
        x_i: A vector containing the features for a single instance

    Returns:
        Returns the prediction (i.e 0 or 1)
    """
    # *** START CODE HERE ***
    sum = 0
    for s in state:
        sum += (s.beta * (kernel(s.x, x_i)))
    return sign(sum)
    # *** END CODE HERE ***


def update_state(state, kernel, learning_rate, x_i, y_i):
    """Updates the state of the perceptron.

    Args:
        state: The state returned from initial_state()
        kernel: A binary function that takes two vectors as input and returns the result of a kernel
        learning_rate: The learning rate for the update
        x_i: A vector containing the features for a single instance
        y_i: A 0 or 1 indicating the label for a single instance
    """
    # *** START CODE HERE ***
    sum = 0
    for i in range(len(state)):
        sum += (state[i].beta * kernel(state[i].x, x_i))
    beta = learning_rate * (y_i - sign(sum))
    new_state = state_class(beta, x_i)
    state.append(new_state)
    return state
    # *** END CODE HERE ***


def sign(a):
    """Gets the sign of a scalar input."""
    if a >= 0:
        return 1
    else:
        return 0

def rbf_kernel(a, b, sigma=1):
    """An implementation of the radial basis function kernel.

    Args:
        a: A vector
        b: A vector
        sigma: The radius of the kernel
    """
    distance = (a - b).dot(a - b)
    scaled_distance = -distance / (2 * (sigma) ** 2)
    return math.exp(scaled_distance)

def train_perceptron(kernel_name, kernel, learning_rate, train_path, save_path, pos):
    """Train a perceptron with the given kernel.

    This function trains a perceptron with a given kernel and then
    uses that perceptron to make predictions.
    The output predictions are saved to src/perceptron/perceptron_{kernel_name}_predictions.txt.
    The output plots are saved to src/perceptron/perceptron_{kernel_name}_output.pdf.

    Args:
        kernel_name: The name of the kernel.
        kernel: The kernel function.
        learning_rate: The learning rate for training.
    """
    x_train, y_train, x_valid, y_valid, x_test, y_test = dp.load_dataset(train_path, pos, add_intercept=False)
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    # x_train[:, 0] *= 0.5
    # x_train[:, 1] *= 10
    # x_test[:, 0] *= 0.5
    # x_test[:, 1] *= 10

    state = initial_state()

    for x_i, y_i in zip(x_train, y_train):
        update_state(state, kernel, learning_rate, x_i, y_i)

    #predict_y = [predict(state, kernel, x_test[i, :]) for i in range(y_test.shape[0])]

    # plot function using sam's util
    #plots.plot(x_test, y_test, predict_y, 'PERCEPTRON.png')
    #plots.plot_with_pca(x_test, y_test, predict_y, 'PERCEPTRON.png')
    #plots.plot_all_feature_pairs(x_test, y_test, predict_y, 'PERCEPTRON.png')
    
    # plot functions using class util
    plt.figure(figsize=(12, 8))
    perceptron_util.plot_contour(lambda a: predict(state, kernel, a))
    perceptron_util.plot_points(x_test, y_test)
    os.makedirs(os.path.dirname(f"{save_path}"), exist_ok=True)
    plt.savefig("perceptron.png", format='png')

    predict_y = [predict(state, kernel, x_test[i, :]) for i in range(y_test.shape[0])]

    np.savetxt(save_path, predict_y)


def main(save_path, train_path, pos):
    """Problem: Kernelizing the Perceptron using rbf as kernel function.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    train_perceptron('rbf', rbf_kernel, 0.5, train_path, save_path, pos)



if __name__ == "__main__":
    main()