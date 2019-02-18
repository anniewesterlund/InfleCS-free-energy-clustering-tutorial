import numpy as np
from sklearn import datasets, manifold
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier

class Digits():
	def __init__(self):
		self.labels_ = None
		self.data_ = None
		self.classifier = KNeighborsClassifier(n_neighbors=3)
		self.name = 'digits'

		self.full_data_, self.full_labels_ = datasets.load_digits(return_X_y=True)
		self.n_full_dataset_ = self.full_data_.shape[0]

		# Reduce dimensionality
		self.full_data_ = PCA(n_components=2).fit_transform(self.full_data_)
		return
	
	def sample(self, n_points):
		"""
		Bootstrap samples from digits.
		"""
		inds = np.random.randint(self.n_full_dataset_,size=(n_points))
		self.data_ = self.full_data_[inds,:]
		self.labels_ = self.full_labels_[inds]+1
		self.classifier.fit(self.data_,self.labels_)
		return self.data_

	def density(self,x):
		min_dist = cdist(x,self.data_).min(axis=1)
		density = np.zeros(x.shape[0])
		density[min_dist < 5e-2] = 0.5
		density /= density.sum()
		return density

	def assign_cluster_labels(self, x):
		return self.classifier.predict(x)

