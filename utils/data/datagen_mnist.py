import os
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Function to downsample MNIST dataset
def downsample_mnist_with_pca(save_train_path, save_test_path, n_components=20):
    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)

    # Split into training (50%) and test (50%) sets
    np.random.seed(42)
    train_size = int(len(X) * 0.5)
    indices = np.random.permutation(len(X))
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Downsample to 100 instances per digit for training
    train_downsampled = []
    for digit in range(10):
        digit_indices = np.where(y_train == digit)[0]
        selected_indices = np.random.choice(digit_indices, size=100, replace=False)
        train_downsampled.append((X_train[selected_indices], y_train[selected_indices]))
    X_train_downsampled = np.vstack([x[0] for x in train_downsampled])
    y_train_downsampled = np.hstack([x[1] for x in train_downsampled])

    # Downsample to 100 instances per digit for testing
    test_downsampled = []
    for digit in range(10):
        digit_indices = np.where(y_test == digit)[0]
        selected_indices = np.random.choice(digit_indices, size=100, replace=False)
        test_downsampled.append((X_test[selected_indices], y_test[selected_indices]))
    X_test_downsampled = np.vstack([x[0] for x in test_downsampled])
    y_test_downsampled = np.hstack([x[1] for x in test_downsampled])

    # Apply PCA to reduce features to 20 dimensions
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_downsampled)
    X_test_pca = pca.transform(X_test_downsampled)

    # Save downsampled datasets to CSV
    train_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    train_df['label'] = y_train_downsampled
    train_df.to_csv(save_train_path, index=False)

    test_df = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    test_df['label'] = y_test_downsampled
    test_df.to_csv(save_test_path, index=False)

    print(f"Saved downsampled MNIST training set with PCA to {save_train_path}")
    print(f"Saved downsampled MNIST test set with PCA to {save_test_path}")

# Call the function
downsample_mnist_with_pca(os.path.join("data", "mnist_train_downsampled_pca.csv"), os.path.join("data", "mnist_test_downsampled_pca.csv"))

