# RegComp
*Owais Sarwar, Carnegie Mellon University" 
*Nick Sahinidis, Georgia Institue of Technology* 

A straightforward tool for building and comparing linear regression models

## Overview 

There are numerous methods for building linear regression models, and none is guaranteed to perform well for any given problem. Our goal is to provide an easy framework to build and compare linear regression models built using various methods. 

### Current Methods Supported (November 2020) 
- [Orthogonal Matching Pursuit](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV.html#sklearn.linear_model.OrthogonalMatchingPursuitCV)
- [Elastic-Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)

## Install 
The user will need to install [Python](https://www.python.org/downloads/).

The RegComp package can be installed using [pip](https://pip.pypa.io/en/stable/), running the following code from the command line: 

```bash 
pip install regcomp
``` 
## Paper 

For a quick overview on the state-of-the art methods for linear regression, please see the following [paper](https://www.e-publications.org/ims/submission/STS/user/submissionFile/46450?confirm=fccc7ad1). 
