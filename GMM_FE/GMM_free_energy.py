import numpy as np
from scipy.spatial.distance import cdist

import GMM_FE.GMM as GMM
from sklearn.mixture import GaussianMixture
import GMM_FE.cross_validation as CV
import GMM_FE

import matplotlib
import matplotlib.pyplot as plt

class FreeEnergy(object):

	def __init__(self, data, min_n_components=8, max_n_components=None, n_components_step=1, x_lims=None, temperature=300.0, n_grids=50,
				 n_splits=1, shuffle_data=False, n_iterations=1, convergence_tol=1e-4, stack_landscapes=False, verbose=True, test_set_perc=0.0):
		"""
		Class for computing free energy landscape in [kcal/mol].
		- observed_data has dimensionality [N x d].
		"""
		self.data_ = data
		self.shuffle_data = shuffle_data
		self.n_splits_ = n_splits
		self.n_iterations_ = n_iterations
		self.convergence_tol_ = convergence_tol
		self.stack_landscapes_ = stack_landscapes

		self.min_n_components = min_n_components
		self.max_n_components = max_n_components
		self.n_components_step = n_components_step

		self.FE_points_ = None
		self.FE_landscape_ = None
		self.coords_ = None

		self.cl_ = None # Clustering object
		self.labels_ = None
		self.cluster_centers_ = None
		self.pathways_ = None

		if x_lims is not None:
			self.x_lims_ = x_lims
			self.n_dims_ = len(self.x_lims_)
		else:
			if len(data.shape) > 1:
				self.x_lims_ = []
				for i in range(data.shape[1]):
					self.x_lims_.append([data[:,i].min(),data[:,i].max()])
				self.n_dims_ = len(self.x_lims_)
			else:
				self.x_lims_ = [[data.min(),data.max()]]
				self.n_dims_ = 1

		self.temperature_ = temperature # [K]
		self.boltzmann_constant = 0.0019872041 # [kcal/(mol K)]
		self.density_est_ = None
		self.standard_error_FE_ = None
		self.nx_ = n_grids
		self.n_grids_ = [self.nx_]*self.n_dims_
		self.test_set_perc_ = test_set_perc
		self.verbose_ = verbose

		self.test_set_loglikelihood = None
		if verbose:
			print('*----------------Gaussian mixture model free energy estimator----------------*')
			print('   n_splits = '+str(n_splits))
			print('   shuffle_data = ' + str(shuffle_data))
			print('   n_iterations = ' + str(n_iterations))
			print('   n_grids = ' + str(n_grids))
			print('   covergence_tol = ' + str(convergence_tol))
			print('   stack_landscapes = ' + str(stack_landscapes))
			print('   x_lims (axes limits) = ' + str(self.x_lims_))
			print('   temperature = ' + str(temperature))
			print('   min_n_components = ' + str(min_n_components))
			print('   max_n_components = ' + str(max_n_components))
			print('   n_components_step = ' + str(n_components_step))
			print('*----------------------------------------------------------------------------*')
		return

	def _get_grid_coords(self):
		x = []
		self.n_grids_ = []
		for i_dim in range(self.n_dims_):
			self.n_grids_.append(self.nx_)
			x.append(np.linspace(self.x_lims_[i_dim][0], self.x_lims_[i_dim][1], self.nx_))
		
		if self.n_dims_ == 1:
			return x
		coords = np.meshgrid(*x)
		return coords

	def _density_landscape(self, density_est):
		"""
		Evaluate density model at the grid points.
		"""
		if self.coords_ is None:
			coords = self._get_grid_coords()
		else:
			coords = self.coords_
		
		if self.n_dims_ == 1:
			densities = density_est.density(coords[0])
			return coords, densities
		
		densities = np.zeros(self.n_grids_)
		
		print('Density grid shape: '+str(densities.shape))
		grid_points_flatten = []
		for x in coords:
			grid_points_flatten.append(np.ravel(x))
		points = np.asarray(grid_points_flatten).T
		
		densities = density_est.density(points)
		densities = np.reshape(densities, self.n_grids_)
		
		return coords, densities

	def _free_energy(self,density):
		density[density < 1e-8] = 1e-8
		FE = -self.temperature_ * self.boltzmann_constant * np.log(density)
		return FE

	def standard_error(self, n_data_blocks=3):
		"""
		Estimating standard error.
		"""
		print('Estimating standard error.')
		n_points = self.data_.shape[0]
		n_data_points = int(n_points/n_data_blocks)
		
		free_energies = []
	
		for i in range(n_data_blocks):

			if i != n_data_blocks-1:
				data = np.copy(self.data_[i*n_data_points:(i+1)*n_data_points])
			else:
				data = np.copy(self.data_[i*n_data_points::])

			if self.n_dims_ == 1:
				data = data[:,np.newaxis]
			
			_, density_model = self._fit_FE(data, set_density_model=False)
			_, density = self._density_landscape(density_model)
			free_energies.append(self._free_energy(density))
		
		free_energies = np.asarray(free_energies)
		self.standard_error_FE_ = np.std(free_energies,axis=0)/np.sqrt(n_data_blocks-1)
		print('Standard error estimation done.')
		return self.standard_error_FE_

	def _fit_FE(self, data, set_density_model=True):
		"""
		Fit density to data points.
		:param data: [n_samples x n_dims]
		:return: free energy of points
		"""

		best_n_components = self.min_n_components

		# Extract test set from the dataset
		n_points_test = int(self.test_set_perc_*data.shape[0])
		data_orig = np.copy(data)
		if n_points_test > 0:
			test_data = data[-n_points_test::,:]
			data = np.copy(data[0:-n_points_test, :])
		else:
			test_data = np.zeros((0,self.n_dims_))

		if self.stack_landscapes_:
			print('Estimating density with stacked GMMs.')
		else:
			print('Estimating density with GMM.')

		best_loglikelihood = -np.inf
		list_of_GMMs = []
		list_of_validation_data = []
		ICs = []
		n_points, n_dims = data.shape

		# Get indices of training and validation datasets
		if self.n_splits_ > 1:
			train_inds, val_inds = CV.split_train_validation(data, self.n_splits_, self.shuffle_data)

		# Determine number of components with k-fold cross-validation,
		# or store all estimated densities and then weight together.
		if self.max_n_components is not None:
			for n_components in range(self.min_n_components,self.max_n_components+1,self.n_components_step):
				if self.verbose_:
					print('# Components = '+str(n_components))

				if self.n_splits_ > 1 and not(self.stack_landscapes_):
					loglikelihood = 0
					for i_split in range(self.n_splits_):
						gmm = GaussianMixture(n_components=n_components,tol=self.convergence_tol_,max_iter=1000)

						training_data, validation_data = CV.get_train_validation_set(data, train_inds[i_split], val_inds[i_split])

						# Train model on the current training data
						gmm.fit(training_data)

						# Check log-likelihood of validation data
						loglikelihood += gmm.score(validation_data)

					# Keep best model
					if loglikelihood > best_loglikelihood:
						best_loglikelihood = loglikelihood
						best_n_components = n_components
				else:
					best_loglikelihood = -np.inf
					for i_iter in range(self.n_iterations_):
						gmm = GaussianMixture(n_components=n_components, tol=self.convergence_tol_)
						
						gmm.fit(data)
						loglikelihood = gmm.score(data)
						# Compute average AIC/BIC over iterations
						if i_iter == 0:
							if self.stack_landscapes_:
								ICs.append(gmm.aic(data))
							else:
								ICs.append(gmm.bic(data))

						# Keep best model
						if loglikelihood > best_loglikelihood:
							best_loglikelihood = loglikelihood
							if i_iter == 0:
								list_of_GMMs.append(GMM.GaussianMixture(n_components=n_components))

							if self.stack_landscapes_:
								ICs[-1] = gmm.aic(data)
							else:
								ICs[-1] = gmm.bic(data)

							list_of_GMMs[-1].weights_ = gmm.weights_
							list_of_GMMs[-1].means_ = gmm.means_
							list_of_GMMs[-1].covariances_ = gmm.covariances_

		if self.stack_landscapes_:
			if  self.max_n_components is None:
				gmm = GMM.GaussianMixture(n_components=self.min_n_components,convergence_tol=self.convergence_tol_)
				gmm.fit(data)
				list_of_GMMs.append(gmm)

			ICs = np.asarray(ICs)
			model_weights = np.exp(-0.5 *(ICs-ICs.min()))
			model_weights /= model_weights.sum()
			
			# Fit mixture of density estimators using the validation data
			density_est = GMM_FE.LandscapeStacker(data, list_of_validation_data, list_of_GMMs, n_splits=1,
														convergence_tol=self.convergence_tol_, n_iterations=self.n_iterations_,
														model_weights=model_weights)
			
			density = density_est.density(data_orig)
			if set_density_model:
					self.density_est_ = density_est
		else:
			# Estimate FE with best number of components (deduced from cross-validation)
			if self.n_splits_ > 1:
				print('Training final model with ' + str(best_n_components) + ' components.')
				
				density_est = GMM.GaussianMixture(n_components=best_n_components)
				# Fit multiple times to
				for i_iter in range(self.n_iterations_):
					gmm = GaussianMixture(n_components=best_n_components,tol=self.convergence_tol_)
					gmm.fit(data)
					loglikelihood = gmm.score(data)
					if  loglikelihood > best_loglikelihood:
						best_loglikelihood = loglikelihood
						density_est.weights_ = gmm.weights_
						density_est.means_ = gmm.means_
						density_est.covariances_ = gmm.covariances_
				
				if set_density_model:
					self.density_est_ = density_est
			else:
				ICs = np.asarray(ICs)
				model_ind = ICs.argmin()
				gmm = list_of_GMMs[model_ind]
				best_n_components = gmm.weights_.shape[0]
				density_est = GMM.GaussianMixture(n_components=best_n_components)
	
				print('Identifying final model with ' + str(density_est.n_components_) + ' components.')
				
				density_est.weights_ = gmm.weights_
				density_est.means_ = gmm.means_
				density_est.covariances_ = gmm.covariances_

			density = density_est.density(data_orig)
		
			if set_density_model:
				self.density_est_ = density_est
		
		if set_density_model:
			# Compute test set loglikelihood on the test set if test set exists
			if n_points_test > 0:
				self.test_set_loglikelihood = self.density_est_.loglikelihood(test_data)
			return self._free_energy(density)
		else:
			return self._free_energy(density), density_est

	def landscape(self):
		"""
		Computing free energy landscape with
		G(x) = -kT*log(p(x|T))
		Returns the X,Y coordinate matrices (meshgrid) and 
		their corresponding free energy.
		"""

		if self.n_dims_ == 1:
			FE_points = self._fit_FE(self.data_[:,np.newaxis])
		else:
			FE_points = self._fit_FE(self.data_)
		
		print('Evaluating density in landscape')
		coords, density = self._density_landscape(self.density_est_)

		FE_landscape = self._free_energy(density)

		# Shift to zero
		min_FE = np.min(FE_landscape)
		FE_landscape = FE_landscape-min_FE
		FE_points = FE_points-min_FE

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
	
	def evaluate_clustering(self, points, assign_transition_points=False):
		"""
		Assign cluster indices to points based on precomputed density model clustering.
		"""
		print('Assigning cluster labels based on precomputed density model clustering.')
		if self.cl_ is not None and self.cl_.clusterer_ is not None:
			labels = self.cl_.clusterer_.data_cluster_indices(cdist(points, self.cl_.clusterer_.grid_points_), self.cl_.clusterer_.grid_cluster_inds_)

		if assign_transition_points:
				labels = self.cl_.assign_transition_points(labels, points, self.density_est_)

		return labels

	def cluster(self, points, free_energies, eval_points=None, return_center_coords=False, assign_transition_points=False):
		"""
		Cluster points according to estimated density.
		"""
		print('Clustering free energy landscape...')
		self.cl_ = GMM_FE.LandscapeClustering(self.stack_landscapes_)
		
		if eval_points is not None:
			tmp_points = []
			for x in points:
				tmp_points.append(np.ravel(x))
			points = np.asarray(tmp_points).T

		if len(points.shape) == 1:
			points = points[:,np.newaxis]
		
		if eval_points is not None:
			if len(eval_points.shape) == 1:
				eval_points = eval_points[:,np.newaxis]
		
		self.labels_, self.is_FE_min = self.cl_.cluster(self.density_est_, points, eval_points=eval_points)

		if eval_points is not None:
			self.cluster_centers_ = self.cl_.get_cluster_representative(eval_points, self.labels_, free_energies)
		else:
			self.cluster_centers_ = self.cl_.get_cluster_representative(points, self.labels_, free_energies)

		if assign_transition_points:
			if eval_points is not None:
				self.labels_ = self.cl_.assign_transition_points(self.labels_, eval_points, self.density_est_)
			else:
				self.labels_ = self.cl_.assign_transition_points(self.labels_, points, self.density_est_)

		print('Done clustering.')
		if return_center_coords:
			return self.labels_, eval_points[self.cluster_centers_,:]
		else:
			return self.labels_, self.cluster_centers_


	def visualize(self,title="Free energy landscape", fontsize=30, savefig=True, xlabel='x', ylabel='y', vmax=7.5, n_contour_levels=15, show_data=False):
		# Set custom colormaps
		my_cmap = matplotlib.cm.get_cmap('jet')
		my_cmap.set_over('white')
		my_cmap_cont = matplotlib.colors.ListedColormap(['black'])
		my_cmap_cont.set_over('white')

		plt.rcParams['figure.figsize'] = [13, 10]
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		ax.tick_params(labelsize=fontsize - 2)

		plt.tick_params(axis='both', which='major', labelsize=fontsize-4)

		for tick in ax.get_xticklabels():
			tick.set_fontname("Serif")
			tick.set_fontweight('light')
		
		for tick in ax.get_yticklabels():
			tick.set_fontname("Serif")
			tick.set_fontweight('light')

		# Plot free energy landscape
		FE_landscape = np.copy(self.FE_landscape_)
		FE_landscape[self.FE_landscape_ > vmax+0.5] = vmax+0.5

		if self.n_dims_ == 2:
			plt.contourf(self.coords_[0], self.coords_[1], FE_landscape, n_contour_levels, cmap=my_cmap, vmin=0, vmax=vmax)
			cb=plt.colorbar(label='[kcal/mol]')
			text = cb.ax.yaxis.label
			font = matplotlib.font_manager.FontProperties(size=fontsize-3,family='serif',weight='light')
			text.set_font_properties(font)
			cb.ax.tick_params(labelsize=fontsize-2)

			for tick in cb.ax.get_yticklabels():
				tick.set_fontname("Serif")
				tick.set_fontweight('light')

			ax.set_ylim([self.coords_[1].min(), self.coords_[1].max()])
			plt.ylabel(ylabel, fontsize=fontsize - 2,fontname='serif',fontweight='light')
		elif self.n_dims_ == 1:
			if self.standard_error_FE_ is not None:
				plt.fill_between(self.coords_[0], FE_landscape - self.standard_error_FE_, FE_landscape + self.standard_error_FE_, color='k', alpha=0.2,zorder=2)
			plt.plot(self.coords_[0], FE_landscape, linewidth=3,color='k',zorder=1)
			plt.ylabel('Free energy [kcal/mol]',fontsize=fontsize-2,fontname='serif',fontweight='light')
		else:
			print('Plotting does not support > 2 dimensions')
			return
		ax.set_xlim([self.coords_[0].min(), self.coords_[0].max()])
		
		# Plot projected data points
		if show_data:

			# Plot projected data points
			if self.labels_ is not None:
				if self.n_dims_ > 1:
					ax.scatter(self.data_[self.labels_==0, 0], self.data_[self.labels_==0, 1], s=30, c=[0.67, 0.67, 0.65],alpha=0.6)
					ax.scatter(self.data_[self.labels_>0, 0], self.data_[self.labels_>0, 1], s=50, c=self.labels_[self.labels_>0],
						   edgecolor='k', cmap=my_cmap, label='Intermediate state',alpha=0.8)
				else:
					ax.scatter(self.data_[self.labels_==0], self.FE_points_[self.labels_==0], s=30, c=[0.67, 0.67, 0.65],alpha=0.6,zorder=3)
					ax.scatter(self.data_[self.labels_>0], self.FE_points_[self.labels_>0], s=50, c=self.labels_[self.labels_>0],
						   edgecolor='k', cmap=my_cmap, label='Intermediate state',alpha=0.8,zorder=4)
				if fontsize > 18:
					plt.legend(fontsize=fontsize-10,facecolor=[0.9,0.9,0.92])
				else:
					plt.legend(fontsize=fontsize-4,facecolor=[0.9,0.9,0.92])
			else:
				if self.n_dims_ > 1:
					ax.scatter(self.data_[:, 0], self.data_[:, 1], s=30, c=[0.67, 0.67, 0.65])
				else:
					ax.scatter(self.data_, self.FE_points_[:, 1], s=30, c=[0.67, 0.67, 0.65])
			# Plot minimum pathways between states
			if self.pathways_ is not None and self.dims > 1:
				for p in self.pathways_:
					ax.plot(p[:, 0], p[:, 1], color='k', linewidth=2, marker='o', label='Pathway')
				if fontsize > 18:
					plt.legend(fontsize=fontsize-10,facecolor=[0.9,0.9,0.92])
				else:
					plt.legend(fontsize=fontsize-4,facecolor=[0.9,0.9,0.92])
			
			# Plot cluster centers in landscape
			if self.cluster_centers_ is not None:
				if self.n_dims_ > 1:
					ax.scatter(self.data_[self.cluster_centers_,0], self.data_[self.cluster_centers_,1], marker='s', s=80,
						   linewidth=3, facecolor='',edgecolor='w', label='Cluster centers')
				else:
					ax.scatter(self.data_[self.cluster_centers_], self.FE_points_[self.cluster_centers_], marker='s', s=80,
						   linewidth=3, facecolor='',edgecolor='w', label='Cluster centers',zorder=5)					
				if fontsize > 18:
					plt.legend(fontsize=fontsize-10,facecolor=[0.9,0.9,0.92])
				else:
					plt.legend(fontsize=fontsize-4,facecolor=[0.9,0.9,0.92])
		plt.title(title, fontsize=fontsize,fontname='serif',fontweight='light')
		plt.xlabel(xlabel, fontsize=fontsize - 2,fontname='serif',fontweight='light')
		plt.rc('xtick', labelsize=fontsize-2)
		plt.rc('ytick', labelsize=fontsize-2)
		matplotlib.rc('font',family='Serif')


		if savefig:
			plt.savefig(title + '.eps')
			plt.savefig(title + '.png')
		return
