import numpy as np
from tqdm import tqdm

class KMeans_Custom:
    def __init__(self, n_clusters=5, max_iter=100, tol=1e-4, random_state=42):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = 0

    def fit(self, X):
        """
        Compute k-means clustering.
        X: array-like of shape (n_samples, n_features)
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]
        
        for i in tqdm(range(self.max_iter)):
            distances = self._compute_distances(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros((self.k, n_features))
            for cluster_idx in range(self.k):
                cluster_points = X[self.labels == cluster_idx]
                if len(cluster_points) > 0:
                    new_centroids[cluster_idx] = cluster_points.mean(axis=0)
                else:
                    new_centroids[cluster_idx] = X[np.random.choice(n_samples)]
            
          
            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            
            if shift < self.tol:
                print(f"Converged at iteration {i}")
                break
                
        final_distances = self._compute_distances(X, self.centroids)
        min_distances = np.min(final_distances, axis=1)
        self.inertia_ = np.sum(min_distances ** 2)
        
        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        """
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

    def _compute_distances(self, X, centroids):
        """
        Compute Euclidean distance between each point in X and each centroid.
        Returns matrix of shape (n_samples, n_clusters)
        """

        n_samples = X.shape[0]
        n_clusters = centroids.shape[0]
        distances = np.zeros((n_samples, n_clusters))
        
        for k in range(n_clusters):
            centroid = centroids[k]
            dist = np.linalg.norm(X - centroid, axis=1)
            distances[:, k] = dist
            
        return distances