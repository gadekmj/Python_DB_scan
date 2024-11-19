import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN as SklearnDBSCAN


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

# Generate data points for a smiling face
def create_smiling_face_data():
    np.random.seed(42)  # For reproducibility
    # Create eyes
    left_eye = np.random.normal(loc=[2, 7], scale=0.2, size=(200, 2))
    right_eye = np.random.normal(loc=[5, 7], scale=0.2, size=(200, 2))

    # Create smile curve (semi-circle)
    theta = np.linspace(0, np.pi, 200)
    smile_x = 3.5 + 2.5 * np.cos(theta)
    smile_y = 4 + 2.5 * np.sin(theta)
    smile = np.column_stack((smile_x, smile_y))

    # Combine all parts
    data = np.vstack([left_eye, right_eye, smile])
    return data


# Plotting the generated data
data = create_smiling_face_data()
plt.scatter(data[:, 0], data[:, 1], s=10, color='blue')
plt.title("Generated Smiling Face Data")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Apply Sklearn's DBSCAN to the data
sklearn_dbscan = SklearnDBSCAN(eps=0.2, min_samples=6, algorithm = 'kd_tree' )
sklearn_labels = sklearn_dbscan.fit_predict(data)

# Apply Custom DBSCAN
custom_dbscan = CustomDBSCAN(eps=0.2, min_samples=7)
custom_dbscan.fit(data)



# Plot results of  DBSCAN
plt.figure(figsize=(14, 7))
# Sklearn DBSCAN results
plt.subplot(1, 2, 1)
unique_labels_sklearn = set(sklearn_labels)
colors_sklearn = plt.cm.get_cmap('tab10', len(unique_labels_sklearn))

for label in unique_labels_sklearn:
    if label == -1:
        color = 'black'  # Black for noise
        label_name = 'Noise'
    else:
        color = colors_sklearn(label)
        label_name = f'Cluster {label}'

    plt.scatter(data[sklearn_labels == label][:, 0], data[sklearn_labels == label][:, 1],
                color=color, label=label_name, alpha=0.8, s=20)

plt.title("Sklearn DBSCAN Clustering Results")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
# Custom DBSCAN results
plt.subplot(1, 2, 2)
unique_labels_custom = set(custom_dbscan.labels)
colors_custom = plt.cm.get_cmap('tab10', len(unique_labels_custom))

for label in unique_labels_custom:
    if label == -1:
        color = 'black'  # Black for noise
        label_name = 'Noise'
    else:
        color = colors_custom(label)
        label_name = f'Cluster {label}'

    plt.scatter(data[custom_dbscan.labels == label][:, 0], data[custom_dbscan.labels == label][:, 1],
                color=color, label=label_name, alpha=0.8, s=20)

plt.title("Custom DBSCAN Clustering Results")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()

plt.tight_layout()
plt.show()
