import numpy as np
from scipy.spatial.distance import cdist

import GMM_FE.GMM as GMM
import GMM_FE.cross_validation as CV
import GMM_FE.ensemble_of_GMMs as eGMM
import GMM_FE.FE_landscape_clustering as FE_cluster

import matplotlib
import matplotlib.pyplot as plt

class FreeEnergy(object):

	def __init__(self, data, min_n_components=8, max_n_components=None, x_lims=None, temperature=300.0, n_grids=50,
				 n_splits=3, shuffle_data=False, n_iterations=1, covergence_tol=1e-6, ensemble_of_GMMs=False):
		"""
		Class for computing free energy landscape in [kcal/mol] using probabilistic PCA.
		- observed_data has dimensionality [N x d]
		"""
		self.data_ = data
		self.shuffle_data = shuffle_data
		self.n_splits_ = n_splits
		self.n_iterations_ = n_iterations
		self.convergence_tol_ = covergence_tol
		self.ensemble_of_GMMs_ = ensemble_of_GMMs

		self.min_n_components = min_n_components
		self.max_n_components = max_n_components

		self.FE_points_ = None
		self.FE_landscape_ = None
		self.coords_ = None

		self.cl_ = None # Clustering object
		self.labels_ = None
		self.cluster_centers_ = None
		self.pathways_ = None

		if x_lims is not None:
			self.x_lim = x_lims
			self.n_dims = len(self.x_lim)
		else:
			if len(data.shape) > 1:
				self.x_lim = []
				for i in range(data.shape[1]):
					self.x_lim.append([data[:,i].min(),data[:,i].max()])
				self.n_dims = len(self.x_lim)
			else:
				self.x_lim = [[data.min(),data.max()]]
				self.n_dims = 1

		self.temperature_ = temperature # [K]
		self.boltzmann_constant = 0.0019872041 # [kcal/(mol K)]
		self.density_est_ = None
		self.min_FE_ = None
		self.nx_ = n_grids

		self.test_set_loglikelihood = None

		print('*----------------Gaussian mixture model free energy estimator----------------*')
		print('   n_splits = '+str(n_splits))
		print('   shuffle_data = ' + str(shuffle_data))
		print('   n_iterations = ' + str(n_iterations))
		print('   n_grids = ' + str(n_grids))
		print('   covergence_tol = ' + str(covergence_tol))
		print('   ensemble_of_GMMs = ' + str(ensemble_of_GMMs))
		print('   x_lims (axes limits) = ' + str(self.x_lim))
		print('   temperature = ' + str(temperature)+' K')
		print('   min_n_components = ' + str(min_n_components))
		print('   max_n_components = ' + str(max_n_components))
		print('*----------------------------------------------------------------------------*')
		return

	def density_landscape(self):

		x = []
		n_grids = []
		for i_dim in range(self.n_dims):
			n_grids.append(self.nx_)
			x.append(np.linspace(self.x_lim[i_dim][0], self.x_lim[i_dim][1], self.nx_))

		if self.n_dims == 1:
			densities = np.zeros(self.nx_)
			for j_x in range(self.nx_):
				point = np.asarray([x[0][j_x]])
				densities[j_x] = self.density_est_.density(point[np.newaxis,:])
			return x, densities

		coords = np.meshgrid(*x)
		densities = np.zeros(n_grids)
		
		print('Density grid shape: '+str(densities.shape))
		grid_points_flatten = []
		for x in coords:
			grid_points_flatten.append(np.ravel(x))
		points = np.asarray(grid_points_flatten).T
		
		densities = self.density_est_.density(points)
		densities = np.reshape(densities, n_grids)
		
		return coords, densities

	def _free_energy(self,density):
		density[density < 1e-8] = 1e-8
		FE = -self.temperature_ * self.boltzmann_constant * np.log(density)
		return FE

	def estimate_standard_error(self):
		# TODO: implement
		return

	def _fit_FE(self, data):
		"""
		Fit density to data points.
		:param data: [n_samples x n_dims]
		:return: free energy of points
		"""
		best_loglikelihood = -np.inf
		best_n_components = self.min_n_components

		# Extract test set from the dataset
		n_points_test = int(0.05*data.shape[0])
		test_data = data[-n_points_test::,:]
		input_data = np.copy(data)
		data = np.copy(data[0:-n_points_test, :])

		if self.ensemble_of_GMMs_:
			print('Estimating density with ensembles of GMMs.')

			self.density_est_ = eGMM.ensemble_of_GMMs(data, self.min_n_components, self.max_n_components, self.convergence_tol_)
			self.density_est_.fit()
			density = self.density_est_.density(input_data)
		else:
			print('Estimating density with GMM.')

			list_of_GMMs = []
			list_of_validation_data = []

			# Get indices of training and validation datasets
			train_inds, val_inds = CV.split_train_validation(data, self.n_splits_, self.shuffle_data)

			# Determine number of components with k-fold cross-validation,
			# or store all estimated densities and then weight together.
			if self.max_n_components is not None:
				for n_components in range(self.min_n_components,self.max_n_components+1):
					print('# Components = '+str(n_components))
					gmm = GMM.GaussianMixture(n_components=n_components,convergence_tol=self.convergence_tol_)

					loglikelihood = 0
					for i_split in range(self.n_splits_):
						training_data, validation_data = CV.get_train_validation_set(data, train_inds[i_split], val_inds[i_split])

						# Train model on the current training data
						gmm.fit(training_data)
						# Check log-likelihood of validation data
						loglikelihood += gmm.loglikelihood(validation_data)

					if loglikelihood > best_loglikelihood:
						best_n_components = n_components
						best_loglikelihood = loglikelihood

			print('Estimating final density with '+str(best_n_components)+' components.')
			# Estimate FE with best number of components
			best_loglikelihood = -np.inf
			for i_iter in range(self.n_iterations_):
				self.density_est_ = GMM.GaussianMixture(n_components=best_n_components,convergence_tol=self.convergence_tol_)
				self.density_est_.fit(data)
				loglikelihood = self.density_est_.loglikelihood(data)
				if  loglikelihood > best_loglikelihood:
					best_loglikelihood = loglikelihood
					density = self.density_est_.density(input_data)

		# Compute test set loglikelihood on the test set
		self.test_set_loglikelihood = self.density_est_.loglikelihood(test_data)
		return self._free_energy(density)

	def landscape(self):
		"""
		Computing free energy landscape with
		G(x) = -kT*log(p(x|T))
		Returns the X,Y coordinate matrices (meshgrid) and 
		their corresponding free energy.
		"""

		if self.n_dims == 1:
			FE_points = self._fit_FE(self.data_[:,np.newaxis])
		else:
			FE_points = self._fit_FE(self.data_)
		
		print('Evaluating density in landscape')
		coords, density = self.density_landscape()

		FE_landscape = self._free_energy(density)

		# Shift to zero
		self.min_FE_ = np.min(FE_landscape)
		FE_landscape = FE_landscape-self.min_FE_
		FE_points = FE_points-self.min_FE_
		
		self.FE_points_ = FE_points
		self.FE_landscape_ = FE_landscape
		self.coords_ = coords

		return coords, FE_landscape, FE_points

	def evaluate_free_energy(self,data):
		"""
		Evaluate the free energy of given data in the current free energy model.
		"""
		density = self.density_est_.density(data)
		free_energy = self._free_energy(density)
		if self.min_FE_ is not None:		
			free_energy -= self.min_FE_
		return free_energy
	
	def evaluate_clustering(self, points):
		"""
		Assign cluster indices to points based on precomputed density model clustering.
		"""
		if self.cl_ is not None and self.cl_.clusterer_ is not None:
			labels = self.cl_.clusterer_.data_cluster_indices(cdist(points, self.cl_.clusterer_.grid_points_), self.cl_.clusterer_.grid_cluster_inds_)
		return labels

	def cluster(self, points, free_energies, eval_points=None, return_center_coords=False):
		"""
		Cluster points according to estimated density.
		"""
		print('Clustering free energy landscape...')
		self.cl_ = FE_cluster.LandscapeClustering(self.ensemble_of_GMMs_)

		if eval_points is not None:
			if len(points[0].shape)>1:
				tmp_points = []
				for x in points:
					tmp_points.append(np.ravel(x))
				points = np.asarray(tmp_points).T

		self.labels_ = self.cl_.cluster(self.density_est_, points, eval_points=eval_points)

		if eval_points is not None:
			self.cluster_centers_ = self.cl_.get_cluster_representative(eval_points, self.labels_, free_energies)
		else:
			self.cluster_centers_ = self.cl_.get_cluster_representative(points, self.labels_, free_energies)
		print('Done clustering.')
		
		if return_center_coords:
			return self.labels_, eval_points[self.cluster_centers_,:]
		else:
			return self.labels_, self.cluster_centers_

	def visualize(self,title="Free energy landscape", fontsize=22, savefig=True, xlabel='x', ylabel='y', vmax=7.5, n_contour_levels=15, show_data=False):
		# Set custom colormaps
		my_cmap = matplotlib.cm.get_cmap('jet')
		my_cmap.set_over('white')
		my_cmap_cont = matplotlib.colors.ListedColormap(['black'])
		my_cmap_cont.set_over('white')

		plt.rcParams['figure.figsize'] = [7, 6]
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		ax.tick_params(labelsize=fontsize - 2)

		# Plot free energy landscape
		FE_landscape_ = np.copy(self.FE_landscape_)
		FE_landscape_[self.FE_landscape_ > vmax+0.5] = vmax+0.5

		if self.n_dims == 2:
			plt.contourf(self.coords_[0], self.coords_[1], FE_landscape_, n_contour_levels, cmap=my_cmap, vmin=0, vmax=vmax)
			cb=plt.colorbar(label='[kcal/mol]')
			cb.ax.tick_params(labelsize=fontsize-2)
			ax.set_ylim([self.coords_[1].min(), self.coords_[1].max()])
			plt.ylabel(ylabel, fontsize=fontsize - 2)
		elif self.n_dims == 1:
			plt.plot(self.coords_[0], FE_landscape_, linewidth=2,color='k')
			plt.ylabel('Free energy [kcal/mol]',fontsize=fontsize-2)
		else:
			print('Plotting does not support > 2 dimensions')
			return
		ax.set_xlim([self.coords_[0].min(), self.coords_[0].max()])

		# Plot projected data points
		if show_data:
			# Plot projected data points
			if self.labels_ is not None:
				ax.scatter(self.data_[self.labels_==0, 0], self.data_[self.labels_==0, 1], s=10, c=[0.67,0.67,0.65],
						   edgecolor='', label='Transition point', alpha=0.3)
				ax.scatter(self.data_[self.labels_>0, 0], self.data_[self.labels_>0, 1], s=20, c=self.labels_[self.labels_>0],
						   edgecolor='k', cmap=my_cmap, label='Intermediate state')
				plt.legend()
			else:
				ax.scatter(self.data_[:, 0], self.data_[:, 1], s=10, c=[0.67, 0.67, 0.65], edgecolor='', alpha=0.3)

			# Plot minimum pathways between states
			if self.pathways_ is not None:
				for p in self.pathways_:
					ax.plot(p[:, 0], p[:, 1], color='k', linewidth=2, marker='o', label='Pathway')
				plt.legend()

			# Plot cluster centers in landscape
			if self.cluster_centers_ is not None:
				ax.scatter(self.data_[self.cluster_centers_,0], self.data_[self.cluster_centers_,1], marker='s', s=30,
						   linewidth=2, facecolor='',edgecolor='w', label='Cluster centers')
				plt.legend()
		plt.title(title, fontsize=fontsize)
		plt.xlabel(xlabel, fontsize=fontsize - 2)
		plt.rc('xtick', labelsize=fontsize-2)
		plt.rc('ytick', labelsize=fontsize-2)



		if savefig:
			plt.savefig(title + '.eps')
			plt.savefig(title + '.png')
		return
