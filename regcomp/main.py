import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from .regression_methods import * 


class RegCompare: 

	def __init__(self, X, Y, methods=["OMP", "ElasticNet"], methods_options={"OMP":{}, "ElasticNet":{}}, data_test_frac=0.33, 
		random_state=None, *args, **kwargs): 

		self.methods = methods 
		self.methods_options = methods_options
		self.data_test_frac = data_test_frac

		# Feature Engineering (Future)

		# Define training and testing set of data 
		self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(X, Y, test_size=data_test_frac, random_state=random_state)

		# Fit regression models - create objects for each 
		if "OMP" in self.methods: 

			if "OMP" not in self.methods_options: 
				self.methods_options["OMP"] = {}

			self.OMPModel = omp.OMP(self.Xtrain, self.Ytrain, OMP_options=self.methods_options["OMP"])

		if "ElasticNet" in self.methods: 

			if "ElasticNet" not in self.methods_options: 
				methods_options["ElasticNet"] = {} 

			if random_state != None: 
				methods_options["ElasticNet"]["random_state"] = random_state

			self.ElasticNetModel = elasticnet.ElasticNet(self.Xtrain, self.Ytrain, ElasticNet_options=self.methods_options["ElasticNet"])





	def Comparisons(self, Xtest=np.array([]), Ytest=np.array([]), true_variables=[], true_coefficients=[], which_compare=[], *args, **kwargs): 

		# Error - RMSE 

		# Support Recovery Metrics 

		# Residual Plot 

		pass