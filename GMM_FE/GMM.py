import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixture():
	
	def __init__(self,n_components=2, convergence_tol=1e-4):
		self.n_components_ = n_components
		self.weights_ = np.ones(n_components)/float(n_components)
		self.means_ = np.zeros(n_components)
		self.covariances_ = [np.zeros((n_components,n_components))]*n_components
		self.tol_ = convergence_tol
		return
	
	def fit(self,x):
		"""
		Fit GMM to points in x with EM.
		"""
		prev_loglikelihood = np.inf
		loglikelihood = 0
		self._initialize_parameters(x)
		while(np.abs(prev_loglikelihood-loglikelihood) > self.tol_):
			gamma = self._expectation(x)
			self._maximization(x, gamma)
			prev_loglikelihood= loglikelihood
			loglikelihood = self.loglikelihood(x)
		
		return self
	
	def predict(self,x):
		gamma = self._expectation(x)
		labels = np.argmax(gamma,axis=0)
		return labels
	
	def _initialize_parameters(self,x):
		"""
		Initialize component means and covariances
		"""
		n_points = x.shape[0]
		inds = np.random.randint(n_points,size=self.n_components_)
		
		# Initialize means
		self.means_ = x[inds,:]
		
		# Initialize covariances
		tmp_cov = np.cov(x.T)
		for i_component in range(self.n_components_):
			self.covariances_[i_component] = tmp_cov
		return
	
	def _expectation(self,x):
		"""
		Perform expecation step
		"""
		n_points = x.shape[0]
		gamma = np.zeros((self.n_components_,n_points));
				
		for i_component in range(self.n_components_):
			gamma[i_component,:] = self.weights_[i_component]*multivariate_normal.pdf(x, mean=self.means_[i_component],
		                                                              cov=self.covariances_[i_component])
		gamma /= np.sum(gamma,axis=0)
		return gamma
	
	def _maximization(self,x, gamma):
		"""
		Update parameters with maximization step
		"""
		self._update_weights(x, gamma)
		self._update_means(x, gamma)
		self._update_covariances(x, gamma)
		return
	
	def _update_weights(self,x, gamma):
		"""
		Update each component amplitude.
		"""
		n_points = x.shape[0]
		self.weights_ = np.sum(gamma,axis=1)
		
		# Normalize Cat-distibution
		self.weights_ /= np.sum(self.weights_)
		return
	
	def _update_means(self,x, gamma):
		"""
		Update each component mean.
		"""
		Nk = np.sum(gamma,axis=1)
		for i_component in range(self.n_components_):
			self.means_[i_component,:] = np.dot(x.T,gamma[i_component])/Nk[i_component]
		return
	
	def _update_covariances(self,x, gamma):
		"""
		Update each component covariance
		"""
		Nk = np.sum(gamma,axis=1)
		n_dims = x.shape[1]
		
		for i_component in range(self.n_components_):
			y = x - self.means_[i_component]
			y2 = np.multiply(gamma[i_component,:,np.newaxis],y).T
			self.covariances_[i_component] = y2.dot(y)/Nk[i_component] + 1e-7*np.eye(n_dims)
		return
	
	def density(self,x):
		"""
		Compute GMM density at given points, x.
		"""
		density = np.zeros(x.shape[0])
		for i_component in range(self.n_components_):
			density += self.weights_[i_component]*multivariate_normal.pdf(x, mean=self.means_[i_component],
		                                                              cov=self.covariances_[i_component])
		return density
	
	def loglikelihood(self,x):
		"""
		Compute log-likelihood.
		"""
		density = self.density(x)
		return np.sum(density)