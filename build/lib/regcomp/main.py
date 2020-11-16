import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, precision_score, confusion_matrix, recall_score, \
explained_variance_score, mean_squared_error, r2_score
from .regression_methods import * 


class RegCompare: 

	def __init__(self, X, Y, methods=["OMP", "ElasticNet"], methods_options={"OMP":{}, "ElasticNet":{}}, data_test_frac=0.33, 
		random_state=None, *args, **kwargs): 

		self.methods = methods 
		self.methods_options = methods_options
		self.data_test_frac = data_test_frac
		self.random_state = random_state

		# Feature Engineering (Future)

		# Define training and testing set of data 
		self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(X, Y, test_size=data_test_frac, random_state=random_state)

		# Define list of model objects 

		self.models = []

	def fit(self): 
		# Fit regression models - create objects for each 
		if "OMP" in self.methods: 

			if "OMP" not in self.methods_options: 
				self.methods_options["OMP"] = {}

			self.OMPModel = omp.OMP(self.Xtrain, self.Ytrain, OMP_options=self.methods_options["OMP"])
			self.OMPModel.SelectedVars = np.nonzero(self.OMPModel.coef_)[0]
			self.models.append(self.OMPModel)


		if "ElasticNet" in self.methods: 

			if "ElasticNet" not in self.methods_options: 
				self.methods_options["ElasticNet"] = {} 

			if self.random_state != None: 
				self.methods_options["ElasticNet"]["random_state"] = self.random_state

			self.ElasticNetModel = elasticnet.ElasticNet(self.Xtrain, self.Ytrain, ElasticNet_options=self.methods_options["ElasticNet"])
			self.ElasticNetModel.SelectedVars = np.nonzero(self.ElasticNetModel.coef_)[0]
			self.models.append(self.ElasticNetModel)


	def comparisons(self, Xtest=np.array([]), Ytest=np.array([]), true_variables_list=[], true_coefficients=[],
		residual_plot_variables=[], *args, **kwargs): 

		self.SummaryTable = pd.DataFrame(index=self.methods)
		# Sparsity metric 

		for model in self.models: 

			model.sparsity = int(len(model.SelectedVars))

		self.SummaryTable = pd.concat([self.SummaryTable, pd.DataFrame([model.sparsity for model in self.models], index=self.methods, columns=["Sparsity"])], axis=1)

		# Regression metrics - RMSE/Explained Variance/R^2

		if len(Ytest) != 0 and len(Xtest) != 0: 

			self.Xtest = Xtest
			self.Ytest = Ytest 

		for model in self.models: 

			model.Y_predict =  model.predict(self.Xtest)
			model.RMSE = mean_squared_error(self.Ytest, model.Y_predict) ** 0.5 
			model.ExplainedVariance = explained_variance_score(self.Ytest, model.Y_predict) 
			model.Rsquared = r2_score(self.Ytest, model.Y_predict) 

		self.SummaryTable = pd.concat([self.SummaryTable, pd.DataFrame([model.RMSE for model in self.models], index=self.methods, columns=["RMSE"])], axis=1)
		self.SummaryTable = pd.concat([self.SummaryTable, pd.DataFrame([model.ExplainedVariance for model in self.models], index=self.methods, columns=["ExplainedVariance"])], axis=1)
		self.SummaryTable = pd.concat([self.SummaryTable, pd.DataFrame([model.Rsquared for model in self.models], index=self.methods, columns=["Rsquared"])], axis=1)

		# Support Recovery Metrics 

		if len(true_variables_list) != 0 or len(true_coefficients) != 0: 

			# Make array-like to indicate true variables
			if len(true_coefficients) == 0: 
				true_variables = np.zeros(self.Xtest.shape[1])
				true_variables[true_variables_list] = 1 
			else: 
				true_variables = true_coefficients


			# Calculate Support Recovery Metrics 

			for model in self.models: 

				model_variables = np.zeros(self.Xtest.shape[1])
				model_variables[np.nonzero(model.coef_)[0]] = 1 

				model.AUROC =  roc_auc_score(true_variables, model_variables) 
				model.Accuracy = accuracy_score(true_variables, model_variables) 
				model.MatthewsCoef = matthews_corrcoef(true_variables, model_variables) 
				model.F1Score = f1_score(true_variables, model_variables) 
				model.Precision = precision_score(true_variables, model_variables) 
				model.Recall = recall_score(true_variables, model_variables) 

			self.SummaryTable = pd.concat([self.SummaryTable, pd.DataFrame([model.AUROC for model in self.models], index=self.methods, columns=["AUROC"])], axis=1)
			self.SummaryTable = pd.concat([self.SummaryTable, pd.DataFrame([model.Accuracy for model in self.models], index=self.methods, columns=["Accuracy"])], axis=1)
			self.SummaryTable = pd.concat([self.SummaryTable, pd.DataFrame([model.Rsquared for model in self.models], index=self.methods, columns=["Rsquared"])], axis=1)
			self.SummaryTable = pd.concat([self.SummaryTable, pd.DataFrame([model.MatthewsCoef for model in self.models], index=self.methods, columns=["MatthewsCoef"])], axis=1)
			self.SummaryTable = pd.concat([self.SummaryTable, pd.DataFrame([model.Precision for model in self.models], index=self.methods, columns=["Precision"])], axis=1)
			self.SummaryTable = pd.concat([self.SummaryTable, pd.DataFrame([model.Recall for model in self.models], index=self.methods, columns=["Recall"])], axis=1)

		# Residual Plots

		self.residual_plot_variables = residual_plot_variables

		if self.residual_plot_variables == 'all': 
			self.residual_plot_variables = [i for i in range(self.Xtest.shape[1])]

		if len(self.residual_plot_variables) != 0: 

			self.residual_figures = {}
			self.residual_axes = []
			for f in self.residual_plot_variables: 
				fig = plt.figure()
				ax = fig.add_subplot(1,1,1)
				self.residual_figures[f] = fig 
				self.residual_axes.append(ax)


			for idx, method in enumerate(self.methods): 

				model.residuals = self.Ytest - model.Y_predict
				for var, axis in enumerate(self.residual_axes): 
					axis.plot(self.Xtest[:, var], model.residuals, ls="", marker="o", label= self.methods[idx]+" Residuals")


			for var, axis in enumerate(self.residual_axes):
				axis.set_xlabel("Variable " + str(self.residual_plot_variables[var]))
				axis.set_ylabel("Residual")
				axis.set_title("Residual Plot for Variable " + str(self.residual_plot_variables[var]))
				axis.legend()
