import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import itertools
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
import dataprocessing as dp

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

    plt.savefig(f"{save_path}.png", format='png')

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
    os.makedirs(os.path.dirname(f"{save_path}.png"), exist_ok=True)
    plt.savefig("pca.png", format='png')

def plot_all_feature_pairs(x, y, theta, save_path, log_reg, correction=1.0):
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
    y = y.flatten()
    all_feature_pairs = list(itertools.combinations(features, 2))
    for (feat_1, feat_2) in all_feature_pairs:
        x_copy = x[:, [feat_1, feat_2]]
        # Create a new figure
        theta_copy = [theta[0], theta[feat_1 + 1], theta[feat_2 + 1]]
        plt.figure(figsize=(8, 6))
        # Plot the points for y = 1 (blue crosses) and y = 0 (green circles)
        plt.plot(x_copy[y == 1, 0], x_copy[y == 1, 1], 'bo', label='Y=1', markersize=4)
        plt.plot(x_copy[y == 0, 0], x_copy[y == 0, 1], 'gx', label='Y=0', markersize=8)

        # Plot the decision boundary
        x1 = np.arange(x_copy[:, 0].min() - 0.1, x_copy[:, 0].max() + 0.1, 0.01)
        x2 = x2 = -(theta_copy[0] / theta_copy[2] + theta_copy[1] / theta_copy[2] * x1
            + np.log((2 - correction) / correction) / theta_copy[2])
        plt.plot(x1, x2, c='red', label='Decision Boundary', linewidth=2)
        # Set axis limits and labels
        for axis in ['top','bottom','left','right']:
            plt.gca().spines[axis].set_linewidth(2)
        plt.xlim(x_copy[:, 0].min() - 0.1, x_copy[:, 0].max() + 0.1)
        plt.ylim(x_copy[:, 1].min() - 0.1, x_copy[:, 1].max() + 0.1)
        plt.xlabel('x{}'.format(feat_1 + 1), fontsize=16)
        plt.ylabel('x{}'.format(feat_2 + 1), fontsize=16)
        plt.title("Prediction Boundary")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # Add a legend
        plt.legend(loc='upper right')

        # Show the plot
        os.makedirs(os.path.dirname(f"{save_path}_{feat_1}_{feat_2}.png"), exist_ok=True)
        plt.savefig(f"pairs_{feat_1}_{feat_2}.png", format="png")

def plot_with_PCA(pos="rb"):
    if pos == "rb":
        d_path = "../input_data/RBs/rb_13-22_final.csv"
    elif pos == "te":
        d_path = "../input_data/TEs/te_13-22_final.csv"
    else:
        d_path = "../input_data/WRs/wr_13-22_final.csv"

    palette = sns.color_palette("mako", 2)
    x_train, y_train, _, _, _, _= dp.load_dataset(d_path, "", True)
    df = pd.DataFrame(x_train)
    labels = pd.DataFrame(y_train)
    features = df.columns
    X = df.loc[:, features].values
    X = StandardScaler().fit_transform(X)

    # Run PCA to reduce to 2 components for plotting
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)

    # Create a DataFrame with the principal components and the labels
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    principalDf['label'] = labels.values

    # Plot the two principal components by the class label
    fig, ax = plt.subplots()

    # Scatter plot for class label 0
    ax.scatter(principalDf.loc[principalDf['label'] == 0, 'principal component 1'],
            principalDf.loc[principalDf['label'] == 0, 'principal component 2'],
            c=palette[0], s=50, label='Class 0', alpha=0.1)

    # Scatter plot for class label 1
    ax.scatter(principalDf.loc[principalDf['label'] == 1, 'principal component 1'],
            principalDf.loc[principalDf['label'] == 1, 'principal component 2'],
            c=palette[1], s=50, label='Class 1', alpha=0.1)

    # Labels, title and legend
    ax.set_xlabel('Principal Component 1', font="Avenir", size=16)
    ax.set_ylabel('Principal Component 2', font="Avenir", size=16)
    ax.set_title('Two component PCA', font="Avenir", size=24)
    ax.legend(
        fontsize=14,
    )

    plt.savefig("outputs/PCA.pdf", format="pdf")
    plt.clf()

def compare_test_and_train(pos="rb"):
    pass

def plot_bar_comparison():
    algorithms = ['GDA', 'LogReg', 'Rand Forests', 'NN', 'SVC']
    train_accuracies = [0.571, 0.563, 0.599, 0.598, 0.615]
    test_accuracies = [0.587, 0.588, 0.589, 0.583, 0.597]

    # Define the x locations for the groups
    ind = np.arange(len(algorithms))

    # The width of the bars
    width = 0.35       
    palette = sns.color_palette("mako", 8)
    # Plotting
    fig, ax = plt.subplots()

    # Plot the train accuracies
    train_bars = ax.bar(ind - width/2, train_accuracies, width, label='Train', color=palette[4])

    # Plot the test accuracies
    test_bars = ax.bar(ind + width/2, test_accuracies, width, label='Test', color=palette[7])

    # Adding labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Train versus Test Accuracy by Algorithm')
    ax.set_xticks(ind)
    ax.set_xticklabels(algorithms, fontsize = 10)
    ax.legend()

    plt.savefig("outputs/comparison.pdf", format="pdf")
    plt.clf()
if __name__ == "__main__":
    plot_with_PCA()
