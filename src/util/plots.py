import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import itertools

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def plot(X: np.ndarray,
         Y: np.ndarray,
         thetas: np.ndarray, 
         save_path: str,
         correction=1.0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Color based on the fourth feature
    # Use the fourth dimension (colors) as the colormap
    colors = np.array(X[:, 3])
    
    # Normalize colors to map to the entire color spectrum
    norm = plt.Normalize(colors.min(), colors.max())
    cmap = plt.cm.viridis  # You can choose a different colormap

    sc = ax.scatter(X[Y.ravel() == 0, 0], X[Y.ravel() == 0, 1], X[Y.ravel() == 0, 2], c=colors[Y.ravel() == 0], cmap=cmap, marker='o', norm=norm, label='Y=0 (Circles)')
    sc = ax.scatter(X[Y.ravel() == 1, 0], X[Y.ravel() == 1, 1], X[Y.ravel() == 1, 2], c=colors[Y.ravel() == 1], cmap=cmap, marker='s', norm=norm, label='Y=1 (Squares)')

    plt.xlabel('X1')
    plt.ylabel('X2')
    ax.set_zlabel('X3')

    # Creating a meshgrid for the decision boundary visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))

    # Calculating corresponding z values based on theta and the sigmoid function
    zz = -(thetas[0] * xx + thetas[1] * yy + thetas[3] * np.mean(colors)) / thetas[2]

    # Decision boundary plot
    ax.plot_surface(xx, yy, zz, color='c', alpha=0.2, label='Decision Boundary')
    cbar = plt.colorbar(sc)
    cbar.set_label('X4')

    plt.show()

def plot_with_pca(X: np.ndarray,
                  Y: np.ndarray,
                  thetas: np.ndarray, 
                  save_path: str,
                  correction=1.0):
    fig, ax = plt.subplots()

    # Apply PCA to reduce dimensionality to 2 dimensions
    pca = PCA(n_components=2)
    reduced_X = pca.fit_transform(X)

    # Use the reduced data for plotting
    reduced_colors = reduced_X[:, 1]  # You can choose any dimension as the color
    
    # Scatter plot with color mapping
    sc = ax.scatter(reduced_X[Y.ravel() == 0, 0], reduced_X[Y.ravel() == 0, 1], c=reduced_colors[Y.ravel() == 0], cmap=plt.cm.viridis, marker='o', label='Y=0 (Circles)')
    sc = ax.scatter(reduced_X[Y.ravel() == 1, 0], reduced_X[Y.ravel() == 1, 1], c=reduced_colors[Y.ravel() == 1], cmap=plt.cm.viridis, marker='s', label='Y=1 (Squares)')

    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')

    # To plot the decision boundary in 2D
    x_min, x_max = reduced_X[:, 0].min() - 1, reduced_X[:, 0].max() + 1
    y_min, y_max = reduced_X[:, 1].min() - 1, reduced_X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = thetas[0] + thetas[1] * xx + thetas[2] * yy
    ax.contour(xx, yy, Z, [0], colors='c', linewidths=0.5)

    # Create colorbar for the reduced color dimension
    cbar = plt.colorbar(sc)
    cbar.set_label('PCA Color')

    ax.legend()
    plt.show()

def plot_all_feature_pairs(x, y, theta, save_path, correction=1.0):
    """Plot dataset and fitted logistic regression parameters.

    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply, if any.
    """
    # Plot dataset
    features = [i for i in range(x.shape[1])]
    all_feature_pairs = list(itertools.combinations(features, 2))
    for (feat_1, feat_2) in all_feature_pairs:
        x_copy = x[:, [feat_1, feat_2]]
        print(y.shape)
        theta_copy = [theta[feat_1], theta[feat_2]]
        plt.figure()
        plt.plot(x_copy[y == 1, -2], x_copy[y == 1, -1], 'bx', linewidth=2)
        plt.plot(x_copy[y == 0, -2], x_copy[y == 0, -1], 'go', linewidth=2)

        # Plot decision boundary (found by solving for theta^T x = 0)
        x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
        x2 = -(theta[0] / theta_copy[2] + theta_copy[1] / theta_copy[2] * x1
            + np.log((2 - correction) / correction) / theta_copy[2])
        plt.plot(x1, x2, c='red', linewidth=2)
        plt.xlim(x[:, -2].min()-.1, x[:, -2].max()+.1)
        plt.ylim(x[:, -1].min()-.1, x[:, -1].max()+.1)

        # Add labels and save to disk
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()