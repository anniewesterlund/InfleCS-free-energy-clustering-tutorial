import numpy as np
import toy_models as tm
import GMM_FE
import matplotlib.pyplot as plt

class MethodEvaluator(object):

    def __init__(self, toy_model='GMM_2D', x_lims=None, n_grids=30):

        if toy_model == 'GMM_2D':
            self.toy_model_ = tm.toy_model_2D_GMM()

        self.cluster_scores_GMM_CV_ = None
        self.density_errors_GMM_CV_ = None
        self.FE_errors_GMM_CV_ = None
        self.loglikelihoods_GMM_CV_ = None

        self.cluster_scores_mix_models_ = None
        self.density_errors_mix_models_ = None
        self.FE_errors_mix_models_ = None
        self.loglikelihoods_mix_models_ = None

        self.x_lims_ = x_lims
        self.n_grids_ = n_grids

        self.true_FE_ = None
        self.true_density_ = None
        self.test_set_ = None
        self.min_FE_ = None
        self.set_true_free_energy()

        return

    def set_true_free_energy(self):
        """
        Create a free energy object that contains the true free energy and density on the given grid.
        :return:
        """
        # Create grid and evaluate density on it
        self.test_set_ = self.toy_model_.sample(1000)
        self.true_FE_ = GMM_FE.free_energy(self.test_test_, x_lims=self.x_lims_, n_grids=self.n_grids_)
        self.true_FE_.density_est_ = self.toy_model_

        coords, self.true_density_ = self.true_FE_.density_landscape()

        # Compute true free energy
        FE_landscape = self.true_FE_._free_energy(self.true_density_)
        self.min_FE_= np.min(FE_landscape)
        FE_landscape = FE_landscape - self.min_FE_

        # Set true free energy
        self.true_FE_.coords_ = coords
        self.true_FE_.FE_landscape_ = FE_landscape
        return

    def run_evaluation(self, n_runs, n_points, n_iterations=1, min_n_components=2, max_n_components=15, n_splits=3):
        """
        Run multiple free energy estimations and evaluate performance.
        :param n_runs:
        :return:
        """
        self.cluster_scores_GMM_CV_ = np.zeros(n_runs)
        self.density_errors_GMM_CV_ = np.zeros(n_runs)
        self.FE_errors_GMM_CV_ = np.zeros(n_runs)
        self.loglikelihoods_GMM_CV_ = np.zeros(n_runs)

        self.cluster_scores_mix_models_ = np.zeros(n_runs)
        self.density_errors_mix_models_ = np.zeros(n_runs)
        self.FE_errors_mix_models_ = np.zeros(n_runs)
        self.loglikelihoods_mix_models_ = np.zeros(n_runs)

        data = self.toy_model_.sample(3)

        # Create free energy estimators
        gmm_FE_CV = GMM_FE.free_energy(data, min_n_components=min_n_components, max_n_components=max_n_components,
                                     x_lims=self.x_lims_, n_grids=self.n_grids_, stack_landscapes=False,
                                     n_splits=n_splits, n_iterations=n_iterations)

        gmm_FE_mix_models = GMM_FE.free_energy(data, min_n_components=min_n_components, max_n_components=max_n_components,
                                     x_lims=self.x_lims_, n_grids=self.n_grids_, stack_landscapes=True,
                                     n_splits=n_splits, n_iterations=n_iterations)

        all_data = []
        for i_run in range(n_runs):
            # Sample data
            data = self.toy_model_.sample(n_points)
            all_data.append(data)

            # Set data in model
            gmm_FE_CV.data_ = data
            gmm_FE_mix_models.data_ = data

            # Estimate free energy of sampled data
            coords, est_FE_landsc_CV, est_FE_points_CV = gmm_FE_CV.landscape()
            _, est_FE_landsc_mix_models, est_FE_points_mix_models = gmm_FE_mix_models.landscape()

            # Compute free energy errors
            self.FE_errors_GMM_CV_[i_run] = self._FE_error(est_FE_landsc_CV)
            self.FE_errors_mix_models_[i_run] = self._FE_error(est_FE_landsc_mix_models)

            # Compute density errors

            # Compute loglikelihood of test set

        np.save('sampled_data_'+file_label+'.npy',all_data)
        np.save('free_energy_errors_CV_'+file_label+'.npy',self.FE_errors_GMM_CV_)
        np.save('free_energy_errors_mix_models_' + file_label + '.npy', self.FE_errors_GMM_mix_models_)
        return

    def _score_clustering(self):
        # Cluster data using the estimated density models, and the original density model
        est_labels_CV, est_cl_centers_CV = gmm_FE_CV.cluster(coords, est_FE_points_CV, data)
        est_labels_mix_models, est_cl_centers_mix_models = gmm_FE_mix_models.cluster(coords, est_FE_points_mix_models, data)

        # Add clustering of points in original density model

        return

    def _FE_error(self, estimated_FE_landscape):
        error = np.mean(np.abs(estimated_FE_landscape-self.true_FE_.FE_landscape_))
        return error

    def _density_error(self, estimated_density):
        error = np.mean(np.abs(estimated_density - self.true_density_))
        return error

    def visualize(self):
        """
        Visualizing the quantities from estimations.
        :return:
        """
        plt.figure(1)
        ax1 = plt.add_suplot(1,2,1)
        # Plot free energy error
        ax1.plot(self.FE_errors_GMM_CV_, linewidth=4, label='GMM with cross-validation')
        ax1.plot(self.FE_errors_GMM_mix_models_, linewidth=4, label='GMM with mixture of models')
        plt.legend()

        # Plot density error

        # Plot log-likelihood of test set

        # Plot clustering score

        # (Plot kinetics in case of HMM)
        plt.show()

        return
