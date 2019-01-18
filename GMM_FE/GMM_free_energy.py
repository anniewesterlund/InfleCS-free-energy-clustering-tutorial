import numpy as np
import GMM_FE.GMM as GMM
import GMM_FE.cross_validation as CV
import GMM_FE.FE_landscape_clustering as FE_cluster
import matplotlib
import matplotlib.pyplot as plt

class free_energy(object):

	def __init__(self, data, min_n_components=8, max_n_components=None, x_lims=None, temperature=300.0, n_grids=50,
				 n_splits=3, shuffle_data=False, n_iterations=1):
		"""
		Class for computing free energy landscape in [kcal/mol] using probabilistic PCA.
		- observed_data has dimensionality [N x d]
		"""
		self.data_ = data
		self.shuffle_data = shuffle_data
		self.n_splits_ = n_splits
		self.n_iterations_ = n_iterations

		self.min_n_components = min_n_components
		self.max_n_components = max_n_components

		self.FE_points_ = None
		self.FE_landscape_ = None
		self.coords_ = None

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
		self.boltzmann_constant = 0.0019872041; # [kcal/(mol K)]
		self.gmm = None
		self.nx = n_grids
	
	def density_landscape(self):

		x = []
		for i_dim in range(self.n_dims):
			x.append(np.linspace(self.x_lim[i_dim][0], self.x_lim[i_dim][1], self.nx))

		if self.n_dims == 1:
			densities = np.zeros(self.nx)
			for j_x in range(self.nx):
				point = np.asarray([x[0][j_x]])
				densities[j_x] = self.gmm.density(point[np.newaxis,:])
			return x, densities

		X, Y = np.meshgrid(*x)
		densities = np.zeros((self.nx,self.nx))
		
		for i_x in range(self.nx):

			for j_x in range(self.nx):
				point = np.asarray([X[i_x,j_x], Y[i_x, j_x]])
				densities[i_x,j_x] = self.gmm.density(point[np.newaxis,:])
		
		return [X, Y], densities

	def _free_energy(self,density):
		density[density < 1e-8] = 1e-8
		FE = -self.temperature_ * self.boltzmann_constant * np.log(density)
		return FE

	def estimate_standard_error(self):
		# TODO: implement
		return

	def _fit_FE(self, data):

		train_inds, val_inds = CV.split_train_validation(data, self.n_splits_, self.shuffle_data)

		best_loglikelihood = -np.inf
		best_n_components = self.min_n_components
		print('Estimating density with GMM.')
		# Determine number of components with k-fold cross-validation
		if self.max_n_components is not None:
			for n_components in range(self.min_n_components,self.max_n_components+1):
				print('# Components = '+str(n_components))
				self.gmm = GMM.GaussianMixture(n_components=n_components)
				loglikelihood = 0
				for i_split in range(self.n_splits_):
					training_data, validation_data = CV.get_train_validation_set(data, train_inds[i_split], val_inds[i_split])

					self.gmm.fit(training_data)
					loglikelihood += self.gmm.loglikelihood(validation_data)

				if loglikelihood > best_loglikelihood:
					best_n_components = n_components
					best_loglikelihood = loglikelihood

		print('Estimating final density.')
		# Estimate FE with best number of components
		best_loglikelihood = -np.inf
		for i_iter in range(self.n_iterations_):
			self.gmm = GMM.GaussianMixture(n_components=best_n_components)
			self.gmm.fit(data)
			loglikelihood = self.gmm.loglikelihood(data)
			if  loglikelihood > best_loglikelihood:
				best_loglikelihood = loglikelihood
				density = self.gmm.density(data)

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
		FE_landscape = FE_landscape-np.min(FE_landscape)
		FE_points = FE_points-np.min(FE_landscape)

		self.FE_points_ = FE_points
		self.FE_landscape_ = FE_landscape
		self.coords_ = coords

		return coords, FE_landscape, FE_points
	
	def cluster(self, points, free_energies, eval_points=None):
		"""
		Cluster points according to estimated density.
		"""
		print('Clustering free energy landscape...')
		cl = FE_cluster.landscape_clustering()

		if eval_points is not None:
			if len(points[0].shape)>1:
				points = np.asarray([np.ravel(points[0]),np.ravel(points[1])]).T

		self.labels_ = cl.cluster(self.gmm, points, eval_points=eval_points)

		if eval_points is not None:
			self.cluster_centers_ = cl.get_cluster_representative(eval_points, self.labels_, free_energies)
		else:
			self.cluster_centers_ = cl.get_cluster_representative(points, self.labels_, free_energies)
		print('Done clustering.')
		return self.labels_, self.cluster_centers_

	def visualize(self,title="Free energy landscape", fontsize=22, savefig=True, xlabel='x', ylabel='y', vmax=7.5):
		# Set custom colormaps
		my_cmap = matplotlib.cm.get_cmap('jet')
		my_cmap.set_over('white')
		my_cmap_cont = matplotlib.colors.ListedColormap(['black'])
		my_cmap_cont.set_over('white')

		plt.rcParams['figure.figsize'] = [7, 6]
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)

		# Plot free energy landscape
		self.FE_landscape_[self.FE_landscape_ > vmax+0.5] = vmax+0.5

		if self.n_dims == 2:
			plt.contourf(self.coords_[0], self.coords_[1], self.FE_landscape_, 15, cmap=my_cmap, vmin=0, vmax=vmax)
			cb=plt.colorbar(label='[kcal/mol]')
			cb.ax.tick_params(labelsize=fontsize-2)
			ax.set_ylim([self.coords_[1].min(), self.coords_[1].max()])
			plt.ylabel(ylabel, fontsize=fontsize - 2)
		elif self.n_dims == 1:
			plt.plot(self.coords_[0], self.FE_landscape_, linewidth=2,color='k')
			plt.ylabel('Free energy [kcal/mol]',fontsize=fontsize-2)
		else:
			print('Plotting does not support > 2 dimensions')
			return
		ax.set_xlim([self.coords_[0].min(), self.coords_[0].max()])
		
		# Plot projected data points
		if self.labels_ is not None:
			ax.scatter(self.data_[self.labels_==0, 0], self.data_[self.labels_==0, 1], s=10, c=[0.67,0.67,0.65], edgecolor='', label='Transition point')
			ax.scatter(self.data_[self.labels_>0, 0], self.data_[self.labels_>0, 1], s=20, c=self.labels_[self.labels_>0], edgecolor='k', cmap=my_cmap, label='Intermediate state')
			plt.legend()
		# Plot minimum pathways between states
		if self.pathways_ is not None:
			for p in self.pathways_:
				ax.plot(p[:, 0], p[:, 1], color='k', linewidth=2, marker='o', label='Pathway')
			plt.legend()

		# Plot cluster centers in landscape
		if self.cluster_centers_ is not None:
			ax.scatter(self.data_[self.cluster_centers_,0], self.data_[self.cluster_centers_,1], marker='s', s=30, linewidth=2, facecolor='',edgecolor='w', label='Cluster centers')
			plt.legend()
		plt.title(title, fontsize=fontsize)
		plt.xlabel(xlabel, fontsize=fontsize - 2)
		plt.rc('xtick', labelsize=fontsize-2)
		plt.rc('ytick', labelsize=fontsize-2)



		if savefig:
			plt.savefig(title + '.eps')
			plt.savefig(title + '.png')
		return
