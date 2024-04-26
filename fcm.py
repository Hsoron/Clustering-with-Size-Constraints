import sys
import time

from scipy.spatial.distance import cdist
import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt

class FCM:
    def __init__(self, n_clusters, max_iters=1000, m=4, epsilon=1e-5, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.m = m
        self.epsilon = epsilon
        self.random_state = random_state
        np.random.seed(random_state)
        self.distance_func = cdist
        self.u = None
        self.cluster_centers_ = None
        self.beta = None
        self.labels_ = None

    def kmeans_plusplus_init(self, X):
        n_samples, _ = X.shape
        centers = np.empty((self.n_clusters, X.shape[1]))

        # Randomly select the first cluster center
        center_idx = np.random.choice(n_samples)
        centers[0] = X[center_idx]

        # For the remaining cluster centers
        for i in range(1, self.n_clusters):
            # Calculate the shortest distance from all points to the current cluster center
            distances = cdist(X, centers[:i], metric='euclidean').min(axis=1)
            # Calculate the square of the distance as a weight
            weights = distances ** 2
            # Randomly select the next cluster center based on weights
            center_idx = np.random.choice(n_samples, p=weights / np.sum(weights))
            centers[i] = X[center_idx]

        return centers

    def calculate_alpha(self, X):
        epsilon = 1e-10
        distances = self.distance_func(X, self.cluster_centers_)
        # Add a small value to avoid dividing by 0
        distances_with_epsilon = distances + epsilon
        denominator = np.sum(1 / distances_with_epsilon, axis=1)
        numerator = np.sum(self.beta / distances_with_epsilon, axis=1)
        alpha = (2 - numerator) / denominator
        return alpha

    def update_membership(self, X, alpha):
        epsilon = 1e-10
        distances = self.distance_func(X, self.cluster_centers_) + epsilon  # Add a small value to avoid dividing by 0
        new_u = (alpha[:, np.newaxis] + self.beta) / (2 * distances)
        new_u /= np.sum(new_u, axis=1, keepdims=True)  # Normalize membership values
        return new_u

    def solve_for_beta(self, X):
        assert self.cluster_centers_ is not None, "Cluster centers have not been initialized!"

        # Calculate the distance between data points and clustering centers
        distances = self.distance_func(X, self.cluster_centers_) + 1e-10  # Add a small value to avoid dividing by 0

        A = np.zeros((self.n_clusters, self.n_clusters))
        b = np.zeros(self.n_clusters)

        # Compute each element of the coefficient matrix A and the constant vector b
        for k in range(self.n_clusters):
            for m in range(self.n_clusters):
                if k == m:
                    A[k, m] = np.sum(1 / (2 * distances[:, k])) - np.sum(
                        (1 / distances[:, m]) / (2 * distances[:, k] * np.sum(1 / distances, axis=1)))
                else:
                    A[k, m] = -np.sum(1 / (2 * distances[:, m]) / (distances[:, k] * np.sum(1 / distances, axis=1)))
            b[k] = distances.shape[0] / self.n_clusters - np.sum(
                1 / (distances[:, k] * np.sum(1 / distances, axis=1)))

        # solve the linear equation system
        # self.beta = np.linalg.solve(A, b)
        if np.linalg.cond(A) < 1 / sys.float_info.epsilon:
            # A is not a singular matrix, solve directly
            self.beta = np.linalg.solve(A, b)
        else:
            # A is a singular matrix, solve using the pseudoinverse
            pinv = np.linalg.pinv(A)
            self.beta = np.dot(pinv, b)
        return self.beta

    def update_centers(self, X):
        um = np.power(self.u, self.m)
        centers = np.dot(um.T, X) / np.sum(um, axis=0)[:, None]
        return centers

    def fit(self, X, name):
        start_time = time.time()
        # Random initialization
        # self.u = np.random.random(size=(X.shape[0], self.n_clusters))
        # self.u /= np.sum(self.u, axis=1)[:, None]
        # self.cluster_centers_ = np.random.random(size=(self.n_clusters, X.shape[1]))

        # Initialize the cluster centers using K-Means++
        self.cluster_centers_ = self.kmeans_plusplus_init(X)


        # Initialize membership matrix with small random values
        self.u = np.random.rand(X.shape[0], self.n_clusters)
        self.u /= np.sum(self.u, axis=1)[:, None]

        for itr in range(self.max_iters):
            last_u = self.u.copy()
            self.beta = self.solve_for_beta(X)
            alpha = self.calculate_alpha(X)
            self.u = self.update_membership(X, alpha)

            # If the change in membership is less than the convergence threshold, we're done
            if norm(self.u - last_u) < self.epsilon:
                break

            # Update cluster centers based on the new memberships
            self.cluster_centers_ = self.update_centers(X)

        end_time = time.time()

        # Calculate and print runtime
        print(f"Algorithm running time in {name}: {end_time - start_time:.2f} seconds")
        self.labels_ = np.argmax(self.u, axis=1)

def plot_clusters_3D(X, labels, centers, name):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(centers)):
        mask = labels == i
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], s=30, label=f'Cluster {i + 1}')

    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=100, color='black', marker='X', label='Centers')
    max_range = np.array(
        [X[:, 0].max() - X[:, 0].min(), X[:, 1].max() - X[:, 1].min(), X[:, 2].max() - X[:, 2].min()]).max() / 2.0
    mid_x = (X[:, 0].max() + X[:, 0].min()) * 0.5
    mid_y = (X[:, 1].max() + X[:, 1].min()) * 0.5
    mid_z = (X[:, 2].max() + X[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_title(f'3D Visualization of Clusters of {name} by FCM')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.legend()
    # plt.savefig(f'3D Visualization of Clusters of {name} by FCM.png', dpi=600)
    plt.show()


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import seaborn as sns

    n_clusters = 10
    # X, _ = make_blobs(n_samples=100, centers=1, random_state=42)
    pointClouds = ['bigbutterfly', 'butterfly', 'cat', 'chess', 'dragon', 'hat', 'racecar', 'skateboard', 'teapot']
    for pointCloud_name in pointClouds:
        X = np.loadtxt(f'./data/{pointCloud_name}.txt', delimiter=',')

        fcm = FCM(n_clusters=n_clusters)
        fcm.fit(X, pointCloud_name)

        fcm_centers = fcm.cluster_centers_
        fcm_labels = fcm.labels_

        cluster_sizes = np.bincount(fcm.labels_, minlength=fcm.n_clusters)
        group_labels = [f'Cluster {i + 1}' for i in range(fcm.n_clusters)]

        plt.figure(figsize=(10, 6))
        plt.bar(group_labels, cluster_sizes, color='blue')
        plt.title(f'Count of Items in Each Cluster of {pointCloud_name} by FCM')
        plt.xlabel('Clusters')
        plt.ylabel('Counts')
        # plt.savefig(f'Count of Items in Each Cluster of {pointCloud_name} by FCM.png', dpi=600)
        plt.show()
