import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split


class RegCompare: 

	def __init__(self, X, Y, methods=["OMP", "ElasticNet"], data_test_frac=0.33, random_seed=0, *args, **kwargs): 

		self.methods = methods 
		self.data_test_frac = data_test_frac

		#Define training and testing set of data 
		self.Xtrain, self.Ytrain, self.Xtest, self.Ytest = train_test_split(X, Y, test_size=data_test_frac, random_seed=random_seed)





	def 