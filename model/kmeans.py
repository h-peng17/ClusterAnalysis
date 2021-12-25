import numpy as np
from copy import deepcopy
import logging

class KMeans():
    def __init__(self, n_clusters: int=10, init: str="k-means++", max_iter: int=300, tol: float=1e-4) -> None:
        assert init in ['k-means++', 'random'], "init for KMeans should be \"k-means++\" or \"random\""
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.labels = None
        self.centroids = None

    def fit(self, x):
        """
            x: [data_num, dim]
                all data
        """
        self.centroids = self.init_centroid(x)       # [n_clusters, dim]
        for iter in range(self.max_iter):
            distance = np.linalg.norm(x[:, None, :] - self.centroids[None, :, :], axis=-1)
            self.labels = np.argmin(distance, axis=-1)
            new_centroids = deepcopy(self.centroids)
            for i in range(self.n_clusters):
                index = self.labels == i
                if np.sum(index) > 0:
                    new_centroids[i] = np.mean(x[index], axis=0)
            diff = np.linalg.norm(self.centroids - new_centroids)
            if diff < self.tol:
                break
            self.centroids = new_centroids
        if iter == self.max_iter:
            logging.warning("Reach the iteration limitation. May not converged.")

    def predict(self, x):
        distance = np.linalg.norm(x[:, None, :] - self.centroids[None, :, :], axis=-1)
        return np.argmin(distance, axis=-1)

    def init_centroid(self, x):
        if self.init == 'k-means++':
            index = np.random.randint(0, x.shape[0], 1)
            centroids = x[index]
            for i in range(self.n_clusters - 1):
                distance = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
                min_distance = distance.min(axis=-1)
                index = np.random.choice(x.shape[0], 1, p=min_distance / min_distance.sum())
                centroids = np.concatenate([centroids, x[index]])
        else:
            choice = np.random.choice(np.arange(x.shape[0]), self.n_clusters)
            centroids = x[choice]
        return centroids
