import numpy as np
import GMM_FE.GMM as gmm

class ensemble_of_GMMs(object):
    def __init__(self, data, n_min_components, n_max_components, convergence_tol=1e-4):
        """
        Class for weighting density estimators with EM, based on how well they describe the data.
        :param data: [n_samples x n_dimensions]
        :param list_of_estimators:
        :param convergence_tol:
        """
        self.GMM_list_ = []
        for n_components in range(n_min_components,n_max_components+1):
            self.GMM_list_.append(gmm.GaussianMixture(n_components, convergence_tol))
            self.GMM_list_[-1]._initialize_parameters(data)
        
        self.data_ = data
        self.convergence_tol_ = convergence_tol
        self.n_models_ = len(self.GMM_list_)

        # Initlialize weights over GMMs
        self.model_weights_ = 1.0 / self.n_models_ * np.ones(self.n_models_)
        return

    def fit(self):
        loglikelihood = -np.inf
        prev_loglikelihood = 0
        while (np.abs(prev_loglikelihood - loglikelihood) > self.convergence_tol_):
            beta, gamma = self._expectation()
            self._maximization(beta, gamma)
            prev_loglikelihood = loglikelihood
            loglikelihood = self.loglikelihood(self.data_)
        return

    def _expectation(self):
        n_points = self.data_.shape[0]

        beta = np.zeros((self.n_models_, n_points))
        gamma = [0]*self.n_models_
        for i_model in range(self.n_models_):
            beta[i_model, :] = self.model_weights_[i_model]*self.GMM_list_[i_model].density(self.data_)
            gamma[i_model] = self.GMM_list_[i_model]._expectation(self.data_)

        beta /= np.sum(beta, axis=0)
        return beta, gamma

    def _maximization(self, beta, gamma):
        """
        Update model weights and GMM parameters
        """
        self._update_model_weights(beta)
        self._update_GMM_parameters(beta, gamma)
        return

    def _update_model_weights(self,beta):
        """
        Update density model weights.
        """
        self.model_weights_ = np.sum(beta, axis=1)

        # Normalize Cat-distibution
        self.model_weights_ /= np.sum(self.model_weights_)
        return

    def _update_GMM_parameters(self, beta, gamma):
        """
        Update the parameters of the i_model:th GMM using the weights from both density model and Gaussians
        within that model.
        :param i_model:
        :param beta:
        :param gamma:
        :return:
        """
        for i_model in range(self.n_models_):
            update_gamma = np.multiply(beta[i_model,:],gamma[i_model])
            self.GMM_list_[i_model]._maximization(self.data_, update_gamma)
        return

    def density(self, x):
        """
        Compute mixture of landscape density at the given points, x.
        x is either a numpy-array of size [n_samples x n_dims] or a list of
        validation datasets with length [self.n_models_].
        """
        density = np.zeros(x.shape[0])
        for i_model in range(self.n_models_):
            density += self.model_weights_[i_model]*self.GMM_list_[i_model].density(x)
        return density

    def loglikelihood(self, x):
        """
        Compute log-likelihood.
        """
        density = self.density(x)
        return np.mean(density)