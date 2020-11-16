from sklearn.linear_model import ElasticNetCV
def ElasticNet(Xtrain, Ytrain, ElasticNet_options={}, *args, **kwargs): 

	ElasticNetModel = ElasticNetCV(**ElasticNet_options)
	ElasticNetModel.fit(Xtrain, Ytrain)

	return ElasticNetModel