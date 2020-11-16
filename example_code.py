from regcomp import * 
from sklearn.datasets import make_regression 

## Generate Data 

X, y = make_regression(n_samples=10, n_features=10)


#Initialize RC object 

# as an example, set optional parameters data_test_frac = 0.5 (50% of data is set to training and 50% testing) and change CV-folds for OMP to 2 and ENet to 3


rc = RegCompare(X, y, data_test_frac=0.5, methods_options={"OMP":{"cv":2}, "ElasticNet":{"cv": 3}})

# view training set and test set if desired 

print("Xtrain" = rc.Xtrain)
print("Ytest" = rc.Ytest)

## Fit Model 

rc.fit()

## View some RC object model attributes 

print("Variables selected by OMP:", rc.OMPModel.SelectedVars)
print("Coefficinets of model selected by ElasticNet:", rc.ElasticNetModel.coef_)

#Perform Comparisons 

# optionally add true variables for support recovery 

true_variables_list = [1, 4]

# optionally plot residuals for variables 2 and 4

residual_plot_variables = [2, 4]

rc.comparisons(true_variables_list=true_variables_list, residual_plot_variables=residual_plot_variables)

## View some Regression and Support Recovery Metrics 

print("RMSE of OMP Model:", rc.OMPModel.RMSE)
print("Recall of ElasticNet Model:", rc.ElasticNetModel.Recall) 

# Or see summary 

print(rc.SummaryTable)

#View residual plots 

# Look at residual plot for variable 4 

rc.residual_figures[4].show()