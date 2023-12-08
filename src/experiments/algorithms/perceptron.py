import math

import matplotlib.pyplot as plt
import numpy as np

import util
import numpy as np
import util
from sklearn.naive_bayes import GaussianNB

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from util import dataprocessing as dp
from sklearn import metrics

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


def predict(state, kernel, feature1, feature2, x_i):
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
        a = np.array([s.x[feature1], s.x[feature2]])
        sum += (s.beta * (kernel(a, x_i)))
    return sign(sum)
    # *** END CODE HERE ***


def predict_all(state, kernel, x_i):
    sum = 0
    for s in state:
        sum += (s.beta * kernel(s.x, x_i))
    return sign(sum)

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


def train_perceptron(kernel_name, kernel, learning_rate, train_path, pos):
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
    x_train, y_train, x_valid, y_valid, x_test, y_test = dp.load_dataset(train_path, pos, add_intercept=True)
    y_train = np.squeeze(y_train)
    state = initial_state()

    index = 0
    for x_i, y_i in zip(x_train, y_train):
        print(index)
        index += 1
        update_state(state, kernel, learning_rate, x_i, y_i)

    return state


def plot_contour(predict_fn):
    """Plot a contour given the provided prediction function"""
    x, y = np.meshgrid(np.linspace(-10, 10, num=20), np.linspace(-10, 10, num=20))
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            print((i, j))
            z[i, j] = predict_fn([x[i, j], y[i, j]])

    plt.contourf(x, y, z, levels=[-float('inf'), 0, float('inf')], colors=['orange', 'cyan'])


def plot_points(x, y, feature1, feature2):
    """Plot some points where x are the coordinates and y is the label"""
    x_one = x[y == 0, :]
    x_two = x[y == 1, :]

    plt.scatter(x_one[:,feature1], x_one[:,feature2], marker='x', color='red')
    plt.scatter(x_two[:,feature1], x_two[:,feature2], marker='o', color='blue')


def plot_pairwise_relationships(state, kernel, test_x, test_y, feature1, feature2, title):
    plot_contour(lambda a: predict(state, kernel, feature1, feature2, a))
    plot_points(test_x, test_y, feature1, feature2)


def main(save_path, train_path, pos):
    """
    Validation Accuracy:                    Training Accuracy:
        WR: 0.520618556701031                   WR: 0.5618115055079559
        TE: 0.5081521739130435                  TE: 0.5515405527865881
        RB: 0.4972714870395634                  RB: 0.6726931604980385
    """
    state = train_perceptron('rbf', rbf_kernel, 0.5, train_path, pos)
    
    x_train, y_train, x_valid, y_valid, x_test, y_test = dp.load_dataset(train_path, pos, add_intercept=True)
    y_train = np.squeeze(y_train)
    y_valid = np.squeeze(y_valid)

    y_predict = [predict_all(state, rbf_kernel, x_valid[i, :]) for i in range(y_valid.shape[0])]
    print("Validation Accuracy:", metrics.accuracy_score(y_valid, y_predict))

    y_predict_train = [predict_all(state, rbf_kernel, x_train[i, :]) for i in range(y_train.shape[0])]
    print("Training Accuracy:", metrics.accuracy_score(y_train, y_predict_train))

    # Visualize pairwise relationships
    '''for feature1 in range(x_valid.shape[1]):
        for feature2 in range(feature1 + 1, x_valid.shape[1]):
            title = f'Pairwise Relationship: X{feature1} vs X{feature2}'
            plot_pairwise_relationships(state, rbf_kernel, x_valid, y_valid, feature1, feature2, title)
            plt.show()'''



if __name__ == "__main__":
    main()