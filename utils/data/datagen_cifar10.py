import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import cifar10

# Function to downsample CIFAR-10 dataset with PCA
def downsample_cifar10_with_pca(save_train_path, save_test_path, n_components=20):
    # Load CIFAR-10 dataset
    (X_train_full, y_train_full), (X_test_full, y_test_full) = cifar10.load_data()
    y_train_full, y_test_full = y_train_full.flatten(), y_test_full.flatten()

    # Create a medium class imbalance for training (e.g., 125-175 per class)
    np.random.seed(42)
    class_distribution = [150, 175, 150, 150, 125, 125, 175, 150, 150, 150]  # Medium imbalance
    train_downsampled = []
    for label, n_samples in enumerate(class_distribution):
        class_indices = np.where(y_train_full == label)[0]
        selected_indices = np.random.choice(class_indices, size=n_samples, replace=False)
        train_downsampled.append((X_train_full[selected_indices], y_train_full[selected_indices]))
    X_train_downsampled = np.vstack([x[0] for x in train_downsampled])
    y_train_downsampled = np.hstack([x[1] for x in train_downsampled])

    # Downsample test set to ~150 samples per class
    test_downsampled = []
    for label in range(10):
        class_indices = np.where(y_test_full == label)[0]
        selected_indices = np.random.choice(class_indices, size=150, replace=False)
        test_downsampled.append((X_test_full[selected_indices], y_test_full[selected_indices]))
    X_test_downsampled = np.vstack([x[0] for x in test_downsampled])
    y_test_downsampled = np.hstack([x[1] for x in test_downsampled])

    # Flatten images for PCA
    X_train_flat = X_train_downsampled.reshape(X_train_downsampled.shape[0], -1)
    X_test_flat = X_test_downsampled.reshape(X_test_downsampled.shape[0], -1)

    # Apply PCA to reduce features to 20 dimensions
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)

    # Save downsampled datasets to CSV
    train_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    train_df['label'] = y_train_downsampled
    train_df.to_csv(save_train_path, index=False)

    test_df = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    test_df['label'] = y_test_downsampled
    test_df.to_csv(save_test_path, index=False)

    print(f"Saved downsampled CIFAR-10 training set with PCA to {save_train_path}")
    print(f"Saved downsampled CIFAR-10 test set with PCA to {save_test_path}")

# Call the function
downsample_cifar10_with_pca(os.path.join("data", "cifar10_train_downsampled_pca.csv"), os.path.join("data", "cifar10_test_downsampled_pca.csv"))

