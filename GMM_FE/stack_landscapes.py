import numpy as np

class LandscapeStacker(object):
    def __init__(self, list_of_validation_data, list_of_estimators, convergence_tol=5e-3):
        """
        Class for weighting density estimators with EM, based on how well they describe the validation dataset.
        :param data: [n_samples x n_dimensions]
        :param list_of_estimators:
        :param convergence_tol:
        """
        self.GMM_list_ = list_of_estimators
        self.val_data_list_ = list_of_validation_data
        self.convergence_tol_ = convergence_tol
        self.n_models_ = len(list_of_estimators)

        # Initlialize weights
        self.model_weights_ = 1.0 / self.n_models_ * np.ones(self.n_models_)
        return

    def fit(self):
        loglikelihood = -np.inf
        prev_loglikelihood = 0
        while (np.abs(prev_loglikelihood - loglikelihood) > self.convergence_tol_):
            gamma = self._expectation()
            self._maximization(gamma)
            prev_loglikelihood = loglikelihood
            loglikelihood = self.loglikelihood(self.val_data_list_, list_of_validation_data=True)

        # Keep only models with non-zero weight
        self._sparisify_model()
        return

    def _expectation(self):
        n_points = self.val_data_list_[0].shape[0]

        gamma = np.zeros((self.n_models_, n_points))

        for i_estimator in range(self.n_models_):
            gamma[i_estimator, :] = self.model_weights_[i_estimator]*self.GMM_list_[i_estimator].density(self.val_data_list_[i_estimator])

        gamma /= np.sum(gamma, axis=0)
        return gamma

    def _maximization(self, gamma):
        """
        Update density estimator weights.
        """
        self.model_weights_ = np.sum(gamma, axis=1)

        # Normalize Cat-distibution
        self.model_weights_ /= np.sum(self.model_weights_)

        # Remove models with too low weight
        self.model_weights_[self.model_weights_< 1e-4] = 0

        # Normalize Cat-distibution again
        self.model_weights_ /= np.sum(self.model_weights_)

        return

    def _sparisify_model(self):
        """
        Remove all models with zero-weights (done after converged optimization).
        :return:
        """
        print('Removing zero-weighted models.')
        n_models = np.sum(self.model_weights_>0)
        new_weights = []
        new_models = []

        for i_model in range(self.n_models_):
            if self.model_weights_[i_model] > 0:
                new_weights.append(self.model_weights_[i_model])
                new_models.append(self.GMM_list_[i_model])

        self.n_models_ = n_models
        self.GMM_list_ = new_models
        self.model_weights_ = np.asarray(new_weights)
        return

    def density(self, x, list_of_validation_data=False):
        """
        Compute mixture of landscape density at the given points, x.
        x is either a numpy-array of size [n_samples x n_dims] or a list of
        validation datasets with length [self.n_models_].
        """

        if list_of_validation_data:
            density = np.zeros(x[0].shape[0])
            for i_estimator in range(self.n_models_):
                density += self.model_weights_[i_estimator]*self.GMM_list_[i_estimator].density(x[i_estimator])
        else:
            density = np.zeros(x.shape[0])
            for i_estimator in range(self.n_models_):
                if self.model_weights_[i_estimator] > 0:
                    density += self.model_weights_[i_estimator]*self.GMM_list_[i_estimator].density(x)
        return density

    def loglikelihood(self, x, list_of_validation_data=False):
        """
        Compute log-likelihood.
        """
        density = self.density(x, list_of_validation_data=list_of_validation_data)
        return np.sum(density)
