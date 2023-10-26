import numpy as np
import matplotlib as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def plot(X: np.ndarray,
         thetas: np.ndarray, 
         save_path: str,
         correction=1.0):
    """
    Args:
        x: input examples, (n, d)
        y: labels, (n,)
        thetas: weights, (,d)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    y = sigmoid(X @ thetas) > 0.5
    # Color based on the fourth feature
    colors = X[:, 3]

    # Scatter plot
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, cmap=plt.hot())
    plt.colorbar(sc)

    # Creating a meshgrid for the decision boundary visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))

    # Calculating corresponding z values based on theta and the sigmoid function
    zz = -(thetas[0] * xx + thetas[1] * yy + thetas[3] * np.mean(colors)) / thetas[2]

    # Decision boundary plot
    ax.plot_surface(xx, yy, zz, color='c', alpha=0.2)

    plt.show()