import sys
import numpy as np
import toy_models as tm
import GMM_FE
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import fowlkes_mallows_score

class MethodEvaluator(object):

    def __init__(self, toy_model='GMM_2D', x_lims=None, n_grids=30, convergence_tol=1e-4,verbose=False):

        if toy_model == 'GMM_2D':
            self.toy_model_ = tm.GMM2D()
        elif toy_model == 'mGMMs':
            self.toy_model_ = tm.MultipleGMMs()
        elif toy_model == 'moons':
            self.toy_model_ = tm.Moons()
        else:
            print('Toy model: '+str(toy_model)+' does not exist')
            sys.exit(0)
		
        self.cluster_scores_GMM_CV_ = None
        self.density_errors_GMM_CV_ = None
        self.FE_errors_GMM_CV_ = None
        self.loglikelihoods_GMM_CV_ = None
        self.convergence_tol_ = convergence_tol

        self.cluster_scores_mix_models_ = None
        self.density_errors_mix_models_ = None
        self.FE_errors_mix_models_ = None
        self.loglikelihoods_mix_models_ = None

        self.x_lims_ = x_lims
        self.n_grids_ = n_grids

        self.true_FE_ = None
        self.true_density_ = None
        self.true_labels_ = None
        self.test_set_ = None
        self.min_FE_ = None
        self.verbose_ = verbose
        self.set_true_free_energy()

        return

    def set_true_free_energy(self):
        """
        Create a free energy object that contains the true free energy and density on the given grid.
        :return:
        """
        # Create grid and evaluate density on it
        print('Setting true model.')
        self.test_set_ = self.toy_model_.sample(1000)
        self.true_FE_ = GMM_FE.FreeEnergy(self.test_set_, x_lims=self.x_lims_, n_grids=self.n_grids_,verbose=False,
                                          convergence_tol=self.convergence_tol_)
        self.true_FE_.density_est_ = self.toy_model_

        coords, self.true_density_ = self.true_FE_.density_landscape()

        # Compute true free energy
        FE_landscape = self.true_FE_._free_energy(self.true_density_)
        self.min_FE_= np.min(FE_landscape)
        FE_landscape = FE_landscape - self.min_FE_

        # Set true free energy
        self.true_FE_.coords_ = coords
        self.true_FE_.FE_landscape_ = FE_landscape

        if hasattr(self.toy_model_,"assign_cluster_labels"):
            self.true_labels_ = self.toy_model_.assign_cluster_labels(self.test_set_)
        else:
            self.true_labels_, _ = self.true_FE_.cluster(coords, np.zeros(self.test_set_.shape[0]), self.test_set_)
        return

    def run_evaluation(self, n_runs, n_points, n_iterations=1, min_n_components=2, max_n_components=15,
                       n_splits=3, save_data=False, file_label=''):
        """
        Run multiple free energy estimations and evaluate performance.
        :param n_runs:
        :return:
        """

        self.cluster_scores_GMM_clusters_ = np.zeros(n_runs)

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
        gmm_FE_CV = GMM_FE.FreeEnergy(data, min_n_components=min_n_components, max_n_components=max_n_components,
                                     x_lims=self.x_lims_, n_grids=self.n_grids_, stack_landscapes=False,
                                     n_splits=n_splits, n_iterations=n_iterations,convergence_tol=self.convergence_tol_,
                                      verbose=self.verbose_)

        gmm_FE_mix_models = GMM_FE.FreeEnergy(data, min_n_components=min_n_components, max_n_components=max_n_components,
                                     x_lims=self.x_lims_, n_grids=self.n_grids_, stack_landscapes=True,
                                     n_splits=n_splits, n_iterations=n_iterations,convergence_tol=self.convergence_tol_,
                                              verbose=self.verbose_)

        all_data = []
        for i_run in range(n_runs):
            print("Run: "+str(i_run+1)+'/'+str(n_runs))
            # Sample data
            data = self.toy_model_.sample(n_points)
            all_data.append(data)

            # Set data in model
            gmm_FE_CV.data_ = data
            gmm_FE_mix_models.data_ = data

            # Estimate free energy of sampled data
            coords, est_FE_landsc_CV, est_FE_points_CV = gmm_FE_CV.landscape()
            #_, est_FE_landsc_mix_models, est_FE_points_mix_models = gmm_FE_mix_models.landscape()

            # Compute free energy errors
            self.FE_errors_GMM_CV_[i_run] = self._FE_error(est_FE_landsc_CV)
            #self.FE_errors_mix_models_[i_run] = self._FE_error(est_FE_landsc_mix_models)

            # Compute loglikelihood of test set
            self.loglikelihoods_GMM_CV_[i_run] = gmm_FE_CV.density_est_.loglikelihood(self.test_set_)
            #self.loglikelihoods_mix_models_[i_run] = gmm_FE_mix_models.density_est_.loglikelihood(self.test_set_)

            # Score clustering
            est_labels_CV, _ = gmm_FE_CV.cluster(self.test_set_, est_FE_points_CV)
            #est_labels_mix_models, _ = gmm_FE_mix_models.cluster(coords, est_FE_points_mix_models, self.test_set_)

            self.cluster_scores_GMM_CV_[i_run] = self._score_clustering(est_labels_CV)
            self.cluster_scores_GMM_clusters_[i_run] = self._score_clustering(gmm_FE_CV.density_est_.predict(self.test_set_))
            #self.cluster_scores_mix_models_[i_run] = self._score_clustering(gmm_FE_mix_models)

        if save_data:
            np.save('sampled_data_'+file_label+'.npy',all_data)
            np.save('free_energy_errors_CV_'+file_label+'.npy',self.FE_errors_GMM_CV_)
            np.save('free_energy_errors_mix_models_' + file_label + '.npy', self.FE_errors_GMM_mix_models_)
            np.save('loglikelihood_test_set_CV_'+file_label+'.npy',self.loglikelihoods_GMM_CV_)
            np.save('loglikelihood_test_set_mix_models_'+file_label+'.npy',self.loglikelihoods_mix_models_)
            np.save('cluster_correlations_FE_min_'+file_label+'.npy',self.cluster_scores_GMM_CV_)
            np.save('cluster_correlations_GMM_' + file_label + '.npy', self.cluster_scores_GMM_clusters_)
        return

    def _score_clustering(self, labels):
        # Score clustering compared to true model
        score = fowlkes_mallows_score(self.true_labels_, labels)
        return score

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
