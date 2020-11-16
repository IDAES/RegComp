from sklearn.linear_model import OrthogonalMatchingPursuitCV
def OMP(Xtrain, Ytrain, OMP_options={}, *args, **kwargs): 

	OMPModel = OrthogonalMatchingPursuitCV(**OMP_options)
	OMPModel.fit(Xtrain, Ytrain)

	return OMPModel 