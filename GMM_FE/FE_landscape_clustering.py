import numpy

class landscape_clustering():

    def __init__(self):
        self.cluster_centers = None
        return

    def get_cluster_representative(self, x, labels, free_energies):
        """
        Get one point in each cluster that has minimum FE in that cluster
        """
        n_components = np.max(labels) + 1
        n_points = x.shape[0]

        min_FE_inds = np.zeros(n_components)
        all_inds = np.arange(n_points)
        for i_cluster in range(n_components):
            cluster_inds = all_inds[labels == i_cluster]

            min_FE_inds[i_cluster] = cluster_inds[np.argmin(free_energies)]

        self.cluster_centers = min_FE_inds.astype(int)
        return self.cluster_centers