import random
import numpy as np
import pandas as pd


class KMeans:
    def __init__(self, n_clusters=4, max_iter=2000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fitPredict(self, X):

        random.seed(42)
        rand_idx = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X.iloc[rand_idx, :].values

        for i in range(self.max_iter):    

            # assign clusters
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids

            # move centroids
            self.move_centroids = self.move_centroids(X, cluster_group)

            # check and finish
            if(old_centroids == self.centroids).all():
                break

        print(cluster_group)
        return cluster_group


    def assign_clusters(self, X):
        cluster_group = []
        distances = []

        for row in X.values:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row - centroid, row - centroid)))
            # distances array will contain 4 elements only
            min_distance = min(distances)
            idx_pos = distances.index(min_distance) # It will give the answer in terms of 0, 1, 2, 3, which is the cluster number
            cluster_group.append(idx_pos)
            distances.clear()

        return np.array(cluster_group)
    
    def move_centroids(self, X, cluster_group):
        new_centroids = []

        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            new_centroids.append(np.array(X)[cluster_group == type].mean(axis=0))

        return np.array(new_centroids)