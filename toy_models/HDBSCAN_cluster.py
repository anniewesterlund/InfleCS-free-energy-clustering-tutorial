import numpy as np
import hdbscan
from sklearn.neighbors import KNeighborsClassifier

class HDBSCANCluster():

    def __init__(self, min_samples=None, min_cluster_size=5):
        self.min_samples_ = min_samples
        self.min_cluster_size_ = min_cluster_size
        self.labels_ = None
        self.classifier = KNeighborsClassifier(n_neighbors=3)
        self.name='HDBSCAN'
        return

    def cluster(self, x):
        print('Cluster data with HDBSCAN')

        cl = hdbscan.HDBSCAN(min_samples=self.min_samples_,min_cluster_size=self.min_cluster_size_)
        cl.fit(x)
        self.labels_ = cl.labels_+1

        # Train kNN classifier
        self.classifier.fit(x, self.labels_)
        print('Cluster labels: '+str(np.unique(self.labels_)))
        return self.labels_

    def assign_cluster_labels(self, x):
        return self.classifier.predict(x)
