import sys
import numpy as np

sys.path.append('/Users/Sam/Desktop/FFML')

from src.util import dataprocessing as dp
from src.util import plots


rb_train_path = "src/input_data/RBs/all_rb_stats.csv"
te_train_path = "src/input_data/tight_ends/all_te_stats.csv"
wr_train_path = "src/input_data/wrs/all_wr_stats.csv"

def main(save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train, x_valid, y_valid, x_test, y_test = dp.load_dataset(rb_train_path, 'rb')
    clf = GDA()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_train)
    plots.plot(x_test, y_test, clf.theta, 'GDA.png')
    plots.plot_with_pca(x_test, y_test, clf.theta, 'GDA.png')
    plots.plot_all_feature_pairs(x_test, y_test, clf.theta, 'GDA.png')
    np.savetxt(save_path, predictions)


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        x_1 = x[y.flatten() == 1]
        x_0 = x[y.flatten() == 0]
        phi_1, phi_0 = x_1.shape[0] / x.shape[0], x_0.shape[0] / x.shape[0]
        mu_1, mu_0 = np.mean(x_1, axis=0), np.mean(x_0, axis=0)
        Sigma = np.zeros((x.shape[1], x.shape[1]))

        for i in range(x.shape[0]):
            mu = mu_1 if y[i] == 1 else mu_0
            Sigma += (np.outer((x[i] - mu), (x[i] - mu).T) / x.shape[0])

        

        theta_0 = 0.5 * (mu_0.T @ np.linalg.inv(Sigma) @ mu_0 - mu_1.T @ np.linalg.inv(Sigma) @ mu_1) - np.log(phi_0 / phi_1)
        theta = -np.linalg.inv(Sigma) @ (mu_0 - mu_1)
        self.theta = np.append(theta_0, theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-((x.dot(self.theta[1:].T))+ self.theta[0])))
        # *** END CODE HERE

if __name__ == '__main__':
    main('experiments/test')