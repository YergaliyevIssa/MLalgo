import numpy as np
import random



class LinearRegression:
	""""""
	def __init__(self, n_features = 1):
		self.n_features = n_features + 1
		self.weights = np.random.rand(n_features + 1)
	

	def fit(self, X, y, adjust='GD', learning_rate = 0.01, reg=None, beta):
		""" """
		if adjust == 'GD':
			if reg == 'L1':
				self.GD_L1(X, y, learning_rate,  beta)
			elif reg == 'L2':
				self.GD_L2(X, y, learning_rate, beta)
			else:
				self.GD(X, y, learning_rate)
		elif adjust == 'SGD':
			if reg == 'L1':
				self.SGD_L1(X, y, learning_rate,  beta)
			elif reg == 'L2':
				self.SGD_L2(X, y, learning_rate, beta)
			else:
				self.SGD(X, y, learning_rate)
		else:
			self.


	def gradient(self, obj_id):
		delta_w = np.zeros(self.n_features)
		dot_pr = self.weights[1:] * X[obj_id] + self.weights[0]
		delta_w[0] = (dot_pr - y[obj_id])
		for k in range(1, self.n_features):
			delta_w[i] = (dot_pr - y[obj_id]) * X[obj_id][i]
		
		return delta_w

			 

	def GD(self, X, y, alpha):

		while True:
			grad = 0
			for k in range(X.shape[0]):
				grad += self.gradient(i)
			grad /= X.shape[0]
			grad *= 2
			self.weights  -= alpha * grad
	
	def GD_L1(self, X, y, alpha, beta): 
		while True:
			grad = 0
			for k in range(X.shape[0]):
				grad += self.gradient(i)
			grad /= X.shape[0]
			grad *= 2
			self.weights  -= alpha * grad
		
	
	def GD_L2(self, X, y, alpha, beta):
		while True:
			grad = 0
			for k in range(X.shape[0]):
				grad += self.gradient(i)
			grad /= X.shape[0]
			grad *= 2
			grad += 2 * beta * self.weights
			self.weights -= alpha * grad
	

	
	def SGR(self, X, alpha):
		while True:
			k = random.randint(0, X,shape[0] - 1)
			grad = self.gradient(k)
			self.weights -= alpha * 2 * grad

			

	def SGD_L1(self,  X, y, alpha, beta):

	def SGD_L2(self,  X, y, alpha, beta):
		while True:
			k = random.randint(0, X,shape[0] - 1)
			grad = self.gradient(k)
			grad += 2 * beta * self.weights
			self.weights -= alpha * 2 * grad




	 