import numpy as np
from GMM_FE.GMM import GaussianMixture

class toy_model_2D_GMM(GaussianMixture):

    def __init__(self):
        GaussianMixture.__init__(self, n_components=9)

        self.n_dims_ = 2
        self._set_parameters()
        return

    def sample(self, n_points):
        """
        Sample points from the density model.
        :param n_points:
        :return:
        """
        sampled_points = np.zeros((n_points, self.n_dims_))
        prob_component = np.cumsum(self.weights_)
        r = np.random.uniform(size=n_points)

        is_point_sampled = np.zeros((n_points),dtype=int)

        for i_point in range(n_points):
            for i_component in range(self.n_components_):
                if r[i_point] <= prob_component[i_component]:
                    sampled_points[i_point,:] = np.random.multivariate_normal(self.means_[i_component], self.covariances_[i_component],1)
                    is_point_sampled[i_point] = 1
                    break
        return sampled_points

    def _set_cov(self, x11,x12,x22):
        tmp_cov = np.zeros((self.n_dims_, self.n_dims_))

        tmp_cov[0, 0] = x11
        tmp_cov[0, 1] = x12
        tmp_cov[1, 0] = x12
        tmp_cov[1, 1] = x22
        return tmp_cov

    def _set_parameters(self):

        self.means_ = np.asarray([ np.asarray([0.8,0.35]), np.asarray([0.45,0.52]), np.asarray([0.2,0.6]),
                                   np.asarray([0.05,0.8]), np.asarray([0.52,0.27]), np.asarray([0.48,0.27]),
                                   np.asarray([0.49, 0.24]), np.asarray([0.4, 0.34]), np.asarray([0.8,0.5])])

        covs = [np.zeros((2,2))]*self.n_components_

        covs[0] = self._set_cov(0.0021, 0.0005, 0.002)
        covs[1] = self._set_cov(0.001, 0.0009, 0.001)
        covs[2] = self._set_cov(0.002, 0.001, 0.002)
        covs[3] = self._set_cov(0.003, 0.0008, 0.003)
        covs[4] = self._set_cov(0.0012, 0.0005, 0.0012)
        covs[5] = self._set_cov(0.005, 0.0, 0.0015)
        covs[6] = self._set_cov(0.002, 0.0, 0.002)
        covs[7] = self._set_cov(0.0012, 0.00, 0.002)
        covs[8] = self._set_cov(0.001, 0.0009, 0.001)

        self.covariances_ = covs

        self.weights_ = np.asarray([0.15,0.1,0.3,0.25,0.1,0.05,0.1,0.05,0.4])
        self.weights_ /= np.sum(self.weights_)

        return