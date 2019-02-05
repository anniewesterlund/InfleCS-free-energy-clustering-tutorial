import numpy as np
from GMM_FE.GMM import GaussianMixture
from GMM_FE.stack_landscapes import LandscapeStacker

class MultipleGMMs(LandscapeStacker):
    def __init__(self):
        data = np.zeros((3,2))
        LandscapeStacker.__init__(self, data, [], [])
		
        self.n_dims_ = 2
        self._set_parameters()
        return

    def sample(self, n_points):
        """
		Sample from stacked GMMs.
		"""
        sampled_points = np.zeros((n_points, self.n_dims_))
        prob_model = np.cumsum(self.model_weights_)

        r = np.random.uniform(size=n_points)
        is_point_sampled = np.zeros((n_points), dtype=int)

        for i_point in range(n_points):
            for i_model in range(self.n_models_):
                if r[i_point] <= prob_model[i_model]:
                    is_point_sampled[i_model] += 1
                    sampled_points[i_point,:] = self.GMM_list_[i_model].sample(1)
                    break
        print('Sampled: '+str(is_point_sampled.sum())+'/'+str(n_points))
        return sampled_points

    def _set_cov(self, x11,x12,x22):
        tmp_cov = np.zeros((self.n_dims_, self.n_dims_))

        tmp_cov[0, 0] = x11
        tmp_cov[0, 1] = x12
        tmp_cov[1, 0] = x12
        tmp_cov[1, 1] = x22
        return tmp_cov

    def _set_GMM1(self):
        n_components = 9
        means = np.asarray([ np.asarray([0.8,0.35]), np.asarray([0.45,0.52]), np.asarray([0.2,0.6]),
                                   np.asarray([0.05,0.8]), np.asarray([0.52,0.27]), np.asarray([0.48,0.27]),
                                   np.asarray([0.49, 0.24]), np.asarray([0.4, 0.34]), np.asarray([0.8,0.5])])

        covs = [np.zeros((2,2))]*n_components

        covs[0] = self._set_cov(0.0021, 0.0005, 0.002)
        covs[1] = self._set_cov(0.001, 0.0009, 0.001)
        covs[2] = self._set_cov(0.002, 0.001, 0.002)
        covs[3] = self._set_cov(0.003, 0.0008, 0.003)
        covs[4] = self._set_cov(0.0012, 0.0005, 0.0012)
        covs[5] = self._set_cov(0.005, 0.0, 0.0015)
        covs[6] = self._set_cov(0.002, 0.0, 0.002)
        covs[7] = self._set_cov(0.0012, 0.00, 0.002)
        covs[8] = self._set_cov(0.001, 0.0009, 0.001)
  		
        weights = np.asarray([0.15,0.1,0.5,0.25,0.1,0.05,0.1,0.05,0.4])
        weights /= weights.sum()
        return means, covs, weights

    def _set_GMM2(self):

        n_components = 3
        means = np.asarray([ np.asarray([0.6,0.35]), np.asarray([0.45,0.52]), np.asarray([0.19,0.62])])
		
        covs = [np.zeros((2,2))]*n_components

        covs[0] = self._set_cov(0.003, 0.0008, 0.003)
        covs[1] = self._set_cov(0.005, 0.0, 0.0015)
        covs[2] = self._set_cov(0.0012, 0.00, 0.002)
		
        weights = np.asarray([0.5,0.2,0.3])
        weights /= weights.sum()
        return means, covs, weights


    def _set_GMM3(self):
        n_components = 5
        means = np.asarray([ np.asarray([0.08,0.78]),np.asarray([0.05,0.8]), np.asarray([0.52,0.25]), np.asarray([0.52,0.27]), np.asarray([0.49, 0.24])])
		
        covs = [np.zeros((2,2))]*n_components
		
        covs[0] = self._set_cov(0.0021, 0.0005, 0.002)
        covs[1] = self._set_cov(0.001, 0.0009, 0.001)
        covs[2] = self._set_cov(0.002, 0.001, 0.002)
        covs[3] = self._set_cov(0.003, 0.0008, 0.003)
        covs[4] = self._set_cov(0.0012, 0.0005, 0.0012)
  		
        weights = np.asarray([0.15,0.25,0.10,0.3,0.2])
        weights /= weights.sum()
        return means, covs, weights
		
    def _set_parameters(self):
		
        self.GMM_list_ = []
		
        means, covs, weights = self._set_GMM1()
        GMM1 = GaussianMixture(means.shape[0])
        GMM1.means_ = means
        GMM1.covariances_ = covs
        GMM1.weights_ = weights
        self.GMM_list_.append(GMM1)

        means, covs, weights = self._set_GMM2()
        GMM2 = GaussianMixture(means.shape[0])
        GMM2.means_ = means
        GMM2.covariances_ = covs
        GMM2.weights_ = weights
        self.GMM_list_.append(GMM2)

        means, covs, weights = self._set_GMM3()
        GMM3 = GaussianMixture(means.shape[0])
        GMM3.means_ = means
        GMM3.covariances_ = covs
        GMM3.weights_ = weights
        self.GMM_list_.append(GMM3)

        self.n_models_ = len(self.GMM_list_)
        self.model_weights_ = np.asarray([0.25,0.5,0.25])
        self.n_components_list_ = np.asarray([9,3,5])
        return
