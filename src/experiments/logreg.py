import sys
import numpy as np

sys.path.append('/Users/Sam/Desktop/FFML')

from src.util import dataprocessing as dp
from src.util import plots

rb_train_path = "src/input_data/RBs/all_rb_stats.csv"
te_train_path = "src/input_data/tight_ends/all_te_stats.csv"
wr_train_path = "src/input_data/wrs/all_wr_stats.csv"


def main(dataset_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train, x_valid, y_valid, x_test, y_test = dp.load_dataset(te_train_path, 'te')

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_train)
    # plots.plot(x_test, y_test, clf.theta, 'GDA.png')
    # plots.plot_with_pca(x_test, y_test, clf.theta, 'GDA.png')
    plots.plot_all_feature_pairs(x_test, y_test, clf.theta, 'GDA.png')
    # *** END CODE HERE ***

class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n_examples, dim = x.shape
        y = y.reshape(-1)
        self.theta = np.zeros(dim) # init theta to zeros (dim,)

        for i in range(self.max_iter):
            h_theta = 1 / (1 + np.exp(-np.dot(x, self.theta)))
            H = np.matmul(x.T * (h_theta * (1 - h_theta)), x) / n_examples
            l = np.matmul(x.T, h_theta - y) / n_examples

            prevTheta = self.theta.copy()
            loss = np.matmul(np.linalg.inv(H), l)
            self.theta -= loss

            normDiff = np.linalg.norm(self.theta - prevTheta, 1)
            if self.verbose:
                print(f"Loss vector for iteration {i}: {loss}")
                print(f"1-Norm difference for iteration {i}: {normDiff}")

            # check each iteration starting with right after first
            if normDiff < self.eps:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-np.dot(x, self.theta)))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(te_train_path)
