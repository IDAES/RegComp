# RegressionCompare
*Owais Sarwar, Carnegie Mellon University*
*Nick Sahinidis, Georgia Institue of Technology* 

A straightforward tool for automatic building of and comparison of linear regression models. Models are built using cross-validation and compared 'under one hood.' 

## Overview 

There are numerous methods for building linear regression models, and none is guaranteed to perform well for any given problem. Our goal is to provide an easy framework to build and compare linear regression models built using various methods. For more information on the benefits and drawbacks of major methods, see *A Discussion on Practical Considerations with Sparse Regression Methodologies* by Sarwar, Sauk, and Sahinidis (*Statistical Science*, 2020). 

### Current Methods Supported (November 2020) 
- [Orthogonal Matching Pursuit](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV.html#sklearn.linear_model.OrthogonalMatchingPursuitCV)
- [Elastic-Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)

## Install 
The user will need to install [Python](https://www.python.org/downloads/).

The RegComp package can be installed using [pip](https://pip.pypa.io/en/stable/), running the following code from the command line: 

```bash 
pip install regcomp
``` 

## Useage 
Please see `example_code.py` for an example. It is reproduced below: 
```python 
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

rc.residual_figures[2].show()

```
### Initialize RegCompare() object 

#### Mandatory Inputs to Initialize 

-`X`, `Y`: Design matrix of input data (X) and response variable that you are trying to predict (Y). No need to separate training/test data.

#### Optional Inputs to Initialize 

-`data_test_frac`: Fraction between 0 - 1 of input data that should be reserved for test set (default: `0.33`) 

-`methods`: List of methods to use (default: `["OMP", "ElasticNet"]`)

-`random_state`: Int for reproduceability (default: `None`)

-`methods options`: Dictionary of dictionaries for changing options for regression methods (e.g. `{"ElasticNet":{"cv":6, "max_iter":200}, "OMP":{"intercept":False}}`) for defaults, please see links for each algorithm above


#### Attributes of RegCompare object after Initialization 

-`Xtrain, Xtest, Ytrain, Ytest`: Data split 

### Fit model - determines best model by K-fold cross validaiton 

```python
rc = RegCompare(X, Y)
rc.fit()
```
#### Attributes of RegCompare object after fitting  

-`rc.XModel` e.g. `rc.OMPModel, rc.ElasticNetModel` - Objects for each model, each which have their own default attributes like `rc.OMPModel.coef_` for intercept. (see algorithm links above for details)

-`rc.XModel.SelectedVars`: variables that are selected by the algorithm

### Compare models 

```python 
rc.comparisons()
```
#### Optional inputs to compare

-`Xtest`, `Ytest`: If `data_test_frac = 0` above and want specific test set

-`true_coefficients`: 1D array-like of true coefficients of all variables (length = number columns of X) 

-`true_variables`: 1D array-like of true variables (as indicies of columns of X, with first index = 0)

-`residual_plot_variables`: 1D array-like of variables for which a residual plot should be generated 

#### Attributes of RegCompare object after comparisons are run 

-`rc.XModel.sparsity` (e.g. `rc.OMPModel.sparisty`):  number of variables in model

-`rc.XModel.RMSE`: root mean squared error on test set of data

-`rc.XModel.XSupportRecoveryMetric`: if `true_variables/true_coefficients` provided, where `XSupportRecoveryMetric = AUROC, Accuracy, Precision, Recall, MatthewsCoef, F1score`

-`rc.SummaryTable`: Pandas DataFrame of key comparison stats

-`rc.residual_figures`: If `residual_plot_variables` provided, a dictionary of `matplotlib` `figure` objects with key values of entries in `residual_plot_variables`. To see a plot for a given variable, run `rc.residual_figures[Int for variable in residual_plot_variables].show()` 

### Future Additions 

Currently, the code is in the early stages. In the near future, we hope to add: 

-Automatic feature engineering support 

-Additional comparions 

-Additional regression methods - in the same framework as the current methods (based on the scikit-learn framework). 

Any and all contributions are welcome. 
## Paper 

For a quick overview on the state-of-the art methods for linear regression, please see the following [paper](https://www.e-publications.org/ims/submission/STS/user/submissionFile/46450?confirm=fccc7ad1). 
