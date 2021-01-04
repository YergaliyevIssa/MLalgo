import numpy as np
import random


def sign(x):
	if x > 0:
		return 1
	elif x == 0:
		return 0
	return -1


class LinearRegression:
	"""
		This class provides simple implementation of 
		linear regression algorithm. 
	"""
	def __init__(self, n_features = 1):
		self.n_features = n_features + 1
		self.weights = np.random.rand(n_features + 1)
	

	def fit(self, X, y, mode='GD', learning_rate=0.01, reg=None, beta):
		""" """
		if mode == 'GD':
			if reg == 'L1':
				self.GradDescL1(X, y, learning_rate,  beta)
			elif reg == 'L2':
				self.GradDescL2(X, y, learning_rate, beta)
			else:
				self.GradDesc(X, y, learning_rate)
		elif mode == 'SGD':
			if reg == 'L1':
				self.StohasticGradDescL1(X, y, learning_rate,  beta)
			elif reg == 'L2':
				self.StohasticGradDescL2(X, y, learning_rate, beta)
			else:
				self.StohasticGradDesc(X, y, learning_rate)
		else:
			self.GaussMethod(X, y)

	# compute gradient for one object
	def gradient(self, x, y):
		delta_w = np.zeros(self.n_features)
		dot_prod = self.weights[1:] * x + self.weights[0]
		delta_w[0] = (dot_prod - y)
		for k in range(1, self.n_features):
			delta_w[k] = (dot_pr - y) * x[k]
		
		return delta_w

			 

	def GradDesc(self, X, y, alpha):

		while True:
			grad = 0
			for i in range(X.shape[0]):
				grad += self.gradient(X[i], y[i])
			grad /= X.shape[0]
			grad *= 2
			self.weights  -= alpha * grad
	
	def GradDescL1(self, X, y, alpha, beta): 
		
		while True:

			grad = sum([self.gradient(X[i], y[i]) for i in range(X.shape[0])])
			grad /= X.shape[0]
			grad *= 2
			grad += beta * sign(self.weights)
			self.weights -= alpha * grad
		
	
	def GradDescL2(self, X, y, alpha, beta):
		while True:
			grad = sum([self.gradient(X[i], y[i]) for i in range(X.shape[0])])
			grad /= X.shape[0]
			grad *= 2
			grad += 2 * beta * self.weights
			self.weights -= alpha * grad
	

	
	def StohasticGradDesc(self, X, alpha):
		while True:
			i = random.randint(0, X,shape[0] - 1)
			grad = self.gradient(X[i], y[i])
			self.weights -= alpha * 2 * grad

			

	def StohasticGradDescL1(self,  X, y, alpha, beta):
		while True:
			i = random.randint(0, X,shape[0] - 1)
			grad = self.gradient(X[i], y[i])
			grar += beta * sign(self.weights)
			self.weights -= alpha * 2 * grad



	def StohasticGradDescL2(self,  X, y, alpha, beta):
		while True:
			i = random.randint(0, X,shape[0] - 1)
			grad = self.gradient(X[i], y[i])
			grad += 2 * beta * self.weights
			self.weights -= alpha * 2 * grad


	def GaussMethod(self, X, y):



	def predict(self, X):
		return np.array([np.sum(x * self.weights[1:]) + self.weights[0] for x in X])

	


class LogisticRegression:
	"""docstring for LogisticReggression"""
	def __init__(self, n_features):
		self.n_features = n_features + 1
		self.weights = np.random.rand(n_features + 1)
		

	def fit(self, X, y, learning_rate=0.01, reg=None, beta=0.1):
		if mode == 'GD':
			if reg == 'L1':
				self.GradDescL1(X, y, learning_rate,  beta)
			elif reg == 'L2':
				self.GradDescL2(X, y, learning_rate, beta)
			else:
				self.GradDesc(X, y, learning_rate)
		elif mode == 'SGD':
			if reg == 'L1':
				self.StohasticGradDescL1(X, y, learning_rate,  beta)
			elif reg == 'L2':
				self.StohasticGradDescL2(X, y, learning_rate, beta)
			else:
				self.StohasticGradDesc(X, y, learning_rate)
		else:
			self.GaussMethod(X, y)


	# compute gradient for one object
	def gradient(self, x, y):
		delta_w = np.zeros(self.n_features)
		tmp = np.e ** (-y * (self.weights[1:] * x + self.weights[0]))
		tmp2 = tmp + 1
		delta_w[0] = tmp / tmp2 * -y
		for k in range(1, self.n_features):
			delta_w[k] = tmp / tmp2 * -y * x[k]

		return delta_w

	def GradDesc(self, X, y, alpha):
		while True:
			grad = 0
			for i in range(X.shape[0]):
				grad += self.gradient(X[i], y[i])
			grad /= X.shape[0]
			grad *= 2
			self.weights -= alpha * grad

	def GradDescL1(self, X, y, alpha, beta):
		while True:

			grad = sum([self.gradient(X[i], y[i]) for i in range(X.shape[0])])
			grad /= X.shape[0]
			grad *= 2
			grad += beta * sign(self.weights)
			self.weights -= alpha * grad
		

	def GradDescL2(self, X, y, alpha, beta):
		while True:
			grad = sum([self.gradient(X[i], y[i]) for i in range(X.shape[0])])
			grad /= X.shape[0]
			grad *= 2
			grad += 2 * beta * self.weights
			self.weights -= alpha * grad

	def StohasticGradDesc(self, X, y, alpha): 
		while True:
			i = random.randint(0, X,shape[0] - 1)
			grad = self.gradient(X[i], y[i])
			self.weights -= alpha * 2 * grad

	def StohasticGradDescL1(self, X, y, alpha, beta):
		while True:
			i = random.randint(0, X,shape[0] - 1)
			grad = self.gradient(X[i], y[i])
			grar += beta * sign(self.weights)
			self.weights -= alpha * 2 * grad

	def StohasticGradDescL2(self, X, y, alpha, beta):
		while True:
			i = random.randint(0, X,shape[0] - 1)
			grad = self.gradient(X[i], y[i])
			grad += 2 * beta * self.weights
			self.weights -= alpha * 2 * grad



	def predict(self, X):
		return np.array([sign(np.sum(x * self.weights[1:]) + self.weights[0]) for x in X])


	
	def predict_proba(self, X):
		return [1 / (np.e ** -(np.sum(self.weights[1:] * x) + self.weights[0]) + 1) for x in X]	
