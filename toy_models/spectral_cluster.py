import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier

class SpectralCluster():

    def __init__(self,n_min_clusters,n_max_clusters):
        self.n_min_clusters_ = n_min_clusters
        self.n_max_clusters_ = n_max_clusters
        self.labels_ = None
        self.classifier = KNeighborsClassifier(n_neighbors=3)
        self.name = 'spectral'
        return

    def transition_matrix(self, A):
        for i in range(A.shape[0]):
            A[i,i] = 0
        D = np.sum(A, axis=1)
        D_inv = np.diag(1 / np.sqrt(D))
        T = np.dot(D_inv, np.dot(A, D_inv))
        return T

    def get_n_clusters(self, transition_mat):
        print('Spectral embedding')
        eigenvalues, eigenvectors = np.linalg.eig(transition_mat)#eigsh(transition_mat, k=(self.n_max_clusters_ + 1));

        # Sort in descending order
        ind_sort = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[ind_sort]

        # Get largest eigengap
        eigengaps = -np.diff(eigenvalues)
        ind = np.argmax(eigengaps[self.n_min_clusters_:self.n_max_clusters_+1])
        n_clusters = ind+self.n_min_clusters_
        embedding = eigenvectors[:, ind_sort[0:n_clusters+1]]
        for i in range(embedding.shape[0]):
            embedding[i] /= np.linalg.norm(embedding[i])

        return n_clusters, embedding

    def cluster(self, x):

        # Set affinity matrix
        distances = cdist(x,x)
        distSort = np.sort(distances, axis=1)
        gamma = np.max(distSort[:,1])**2

        dist_squared = np.multiply(distances, distances)

        A = np.exp(-dist_squared/(2*gamma))

        print('Cluster data with spectral clustering')
        # Get transition matrix, select number of dimensions/clusters and project data
        transition_mat = self.transition_matrix(A)
        n_clusters, embedding = self.get_n_clusters(transition_mat)

        print('Cluster data with '+str(n_clusters)+' clusters.')
        km = KMeans(n_clusters=n_clusters).fit(embedding)
        self.labels_ = km.labels_+1

        # Train kNN classifier
        self.classifier.fit(x, self.labels_)
        print('Cluster labels: '+str(np.unique(self.labels_)))
        return self.labels_

    def assign_cluster_labels(self, x):
        return self.classifier.predict(x)