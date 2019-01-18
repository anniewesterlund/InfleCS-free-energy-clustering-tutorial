import numpy as np
import GMM_FE.cluster_density as cluster
from scipy.stats import multivariate_normal

class landscape_clustering():

	def __init__(self):
		self.cluster_centers_ = None
		self.labels_ = None
		return
	
	def get_cluster_representative(self, x, labels, free_energies):
		"""
		Get one point in each cluster that has minimum FE in that cluster
		"""
		n_components = int(np.max(labels) + 1)
		n_points = x.shape[0]

		min_FE_inds = np.zeros(n_components)
		all_inds = np.arange(n_points)
		for i_cluster in range(1,n_components):
			cluster_inds = all_inds[labels == i_cluster]
			min_FE_inds[i_cluster] = cluster_inds[np.argmin(free_energies[cluster_inds])]

		self.cluster_centers_ = min_FE_inds.astype(int)
		return self.cluster_centers_
		
	def _Hessian_def(self, GMM_estimator, points):
		"""
		Compute the Hessian in every point to check whether they belong to a 
		free energy minimum or not.
		"""
		n_points = points.shape[0]
		n_dims = points.shape[1]
		n_components = GMM_estimator.n_components_
		
		means = GMM_estimator.means_
		covs = GMM_estimator.covariances_
		weights = GMM_estimator.weights_
		
		is_FE_min = [False]*n_points
		
		gradients = np.zeros((n_points,n_dims))
		
		inv_covs = [np.zeros((n_dims,n_dims))]*n_components
		print('Computing gradients.')
		for i_component in range(n_components):
			inv_covs[i_component] = np.linalg.inv(covs[i_component])
			
			devs = points-means[i_component]
			exp_deriv = -devs.dot(inv_covs[i_component])
			for i_point in range(n_points):
				gradients[i_point,:] += weights[i_component]*exp_deriv[i_point,:]*multivariate_normal.pdf(points[i_point,:], mean=means[i_component], cov=covs[i_component])
		
		# Computing Hessian to determine whether point belongs to FE min or not
		# neg definite => -1, pos definite => 1, other => 0
		print('Computing Hessians.')
		for i_point, x in enumerate(points):
			hessian = np.zeros((n_dims,n_dims))
			for i_component in range(n_components):
				devs = x-means[i_component]
				exp_deriv = -devs.dot(inv_covs[i_component])
				
				# Compute Hessian at current point
				for i_dim in range(n_dims):
					for j_dim in range(n_dims):
						post_weight = weights[i_component]*multivariate_normal.pdf(x, mean=means[i_component],
																	  cov=covs[i_component])
						hessian[i_dim,j_dim] += post_weight*(-inv_covs[i_component][i_dim,j_dim] + exp_deriv[i_dim]*exp_deriv[j_dim])
			
			# Compute Hessian eigenvalues
			eigvals = np.linalg.eigvals(hessian)
						
			# Check if Hessian is negative definite, the point is at a free energy minimum
			if eigvals.max() < 0:
				is_FE_min[i_point] = True
		
		return is_FE_min
	
	def cluster(self, GMM_estimator, points, eval_points=None):
		# Indicate whether points are at free energy minimum or not
		is_FE_min = self._Hessian_def(GMM_estimator, points)
		# Cluster free energy landscape
		cl = cluster.cluster_density(points, eval_points)
		self.labels_ = cl.cluster_data(is_FE_min)
		return self.labels_
		