import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
import numpy as np

class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
    def fit(self, X):
        n_samples = X.shape[0]
        self.labels = -np.ones(n_samples, dtype=int)  # Initialize labels to -1 (noise)
        cluster_id = 0

        for i in range(n_samples):
            if self.labels[i] != -1:  # Skip if already labeled
                continue

            # Get neighbors for the current point
            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                self.labels[i] = -1  # Mark as noise
            else:
                # Expand the cluster starting from this core point
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1
    def _get_neighbors(self, X, point_idx):
        # Calculate distances between point_idx and all other points
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        # Return indices of neighbors within eps distance
        return np.where(distances < self.eps)[0]
    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        self.labels[point_idx] = cluster_id
        queue = list(neighbors)

        while queue:
            current_point = queue.pop(0)

            # Only process if the point is not already part of a cluster
            if self.labels[current_point] == -1:
                self.labels[current_point] = cluster_id  # Assign to the current cluster

                current_neighbors = self._get_neighbors(X, current_point)
                if len(current_neighbors) >= self.min_samples:
                    # Add only new neighbors that haven't been assigned to any cluster
                    queue.extend([n for n in current_neighbors if self.labels[n] == -1])

            # Visualize the clustering process (slowmo)
            #self.visualize_clusters(X, cluster_id)
    def visualize_clusters(self, X, current_cluster):
        plt.clf()  # Clear the current figure
        unique_labels = set(self.labels)
        colors = plt.cm.get_cmap('tab20', len(unique_labels))

        for label in unique_labels:
            if label == -1:
                color = 'black'  # Black for noise
                label_name = 'Noise'
            else:
                color = colors(label)
                label_name = f'Cluster {label}'

            plt.scatter(X[self.labels == label][:, 0], X[self.labels == label][:, 1],
                        color=color, label=label_name, alpha=0.8, edgecolor='k', s=50)

        plt.title(f"Custom DBSCAN Clustering - Current Cluster: {current_cluster}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid()
        plt.pause(0.1)  # Pause for visualization
    def fit_predict(self, X):
        self.fit(X)
        return self.labels
def find_best_dbscan_params(X_test, y_test, eps_start, eps_end, eps_step, min_samples_range, use_custom=False):

    best_eps = 0
    best_min_samples = 0
    best_score = 0

    DBSCAN_Class = CustomDBSCAN if use_custom else DBSCAN

    for eps in np.arange(eps_start, eps_end, eps_step):
        for min_samples in min_samples_range:
            dbscan = DBSCAN_Class(eps=eps, min_samples=min_samples)
            predicted = dbscan.fit_predict(X_test)

            labels = np.zeros_like(predicted)
            for cluster_id in range(len(np.unique(predicted))):
                mask = (predicted == cluster_id)
                if np.sum(mask) > 0:
                    labels[mask] = np.argmax(np.bincount(y_test[mask]))
            score = metrics.adjusted_rand_score(y_test, labels)
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples

    return {
        "best_eps": best_eps,
        "best_min_samples": best_min_samples,
        "best_score": best_score,
    }


# Load the digits dataset
digits = datasets.load_digits()

# Flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

# Optimize DBSCAN parameters
result = find_best_dbscan_params(X_test, y_test, 20.9, 21, 0.01, range(2,5), use_custom=True)
best_eps = result["best_eps"]
best_min_samples = result["best_min_samples"]

# Create a DBSCAN clustering model with optimized parameters
dbscan = CustomDBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan_inbuild= DBSCAN(eps=best_eps, min_samples=best_min_samples)
# Fit the model and predict clusters on the test subset
predicted = dbscan.fit_predict(X_test)
predicted_inbuild = dbscan_inbuild.fit_predict(X_test)


# Map the cluster labels to the true labels for CustomDBSCAN
labels_custom = np.zeros_like(predicted)
for cluster_id in range(len(np.unique(predicted))):
    mask = (predicted == cluster_id)
    if np.sum(mask) > 0:
        labels_custom[mask] = np.argmax(np.bincount(y_test[mask]))

# Map the cluster labels to the true labels for DBSCAN
labels_inbuild = np.zeros_like(predicted_inbuild)
for cluster_id in range(len(np.unique(predicted_inbuild))):
    mask = (predicted_inbuild == cluster_id)
    if np.sum(mask) > 0:
        labels_inbuild[mask] = np.argmax(np.bincount(y_test[mask]))

# Plot real labels
fig_real, axes_real = plt.subplots(nrows=5, ncols=10, figsize=(12, 6))  # Adjusted figure and subplot size
fig_real.suptitle("Real Labels", fontsize=16)  # Title for the figure
axes_real = axes_real.ravel()
for ax, image, label in zip(axes_real, digits.images[n_samples // 2:], y_test):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Label: {label}", fontsize=8)  # Smaller font for titles
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to leave space for suptitle

# Plot predicted labels for CustomDBSCAN
fig_custom, axes_custom = plt.subplots(nrows=5, ncols=10, figsize=(12, 6))  # Adjusted figure and subplot size
fig_custom.suptitle("Predicted Labels - CustomDBSCAN", fontsize=16)  # Title for the figure
axes_custom = axes_custom.ravel()
for ax, image, label in zip(axes_custom, digits.images[n_samples // 2:], labels_custom):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Pred: {label}", fontsize=8)  # Smaller font for titles
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Plot predicted labels for DBSCAN
fig_inbuild, axes_inbuild = plt.subplots(nrows=5, ncols=10, figsize=(12, 6))  # Adjusted figure and subplot size
fig_inbuild.suptitle("Predicted Labels - DBSCAN", fontsize=16)  # Title for the figure
axes_inbuild = axes_inbuild.ravel()
for ax, image, label in zip(axes_inbuild, digits.images[n_samples // 2:], labels_inbuild):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Pred: {label}", fontsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Plot confusion matrices for CustomDBSCAN
fig_cm_custom, ax_cm_custom = plt.subplots(figsize=(6, 6))  # Adjusted figure size
fig_cm_custom.suptitle("Confusion Matrix - CustomDBSCAN", fontsize=14)  # Title for the figure
metrics.ConfusionMatrixDisplay.from_predictions(y_test, labels_custom, ax=ax_cm_custom)  # Use custom axis
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to leave space for suptitle

# Plot confusion matrices for DBSCAN
fig_cm_inbuild, ax_cm_inbuild = plt.subplots(figsize=(6, 6))  # Adjusted figure size
fig_cm_inbuild.suptitle("Confusion Matrix - DBSCAN", fontsize=14)  # Title for the figure
metrics.ConfusionMatrixDisplay.from_predictions(y_test, labels_inbuild, ax=ax_cm_inbuild)  # Use custom axis
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to leave space for suptitle

# Show all figures
plt.show()

