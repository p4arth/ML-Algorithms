import numpy as np
from scipy.spatial.distance import cdist

class KMeans():
    def __init__(self, 
                 n_clusters = 3,
                 max_iter = 300,
                 random_state = None):
        np.random.seed(random_state)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    
    def assign_clusters(self, X, centroids):
        # for every point calculate the distance from each centroid
        distance_matrix = cdist(X, centroids, metric='euclidean')
        # assign the point the cluster they are most close to
        labels = np.argmin(distance_matrix, axis = 1)
        return labels
    
    def recompute_centroids(self, X, labels):
        # take the mean of all the points belonging to a certain class
        new_centroids = []
        X = np.column_stack((X, labels))
        for i in range(self.n_clusters):
            new_centroids.append(np.mean(X[X[:, -1] == i][:,:-1], axis = 0))
        return np.array(new_centroids)
    
    def fit(self, X):
        centroids = np.random.normal(0, 1, size=(self.n_clusters, self.n_clusters))
        labels = []
        for i in range(self.max_iter):
            if i == 0:
                labels = self.assign_clusters(X, centroids)
            else:
                centroids = self.recompute_centroids(X, labels)
                labels = self.assign_clusters(X, centroids)
        self.centroids = centroids
        self.labels = labels