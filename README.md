# BayesianLinearRegression

[![Build Status](https://travis-ci.com/cscherrer/BayesianLinearRegression.jl.svg?branch=master)](https://travis-ci.com/cscherrer/BayesianLinearRegression.jl)
[![Codecov](https://codecov.io/gh/cscherrer/BayesianLinearRegression.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/cscherrer/BayesianLinearRegression.jl)
This library implements Bayesian Linear Regression, as described in Chris Bishop's [Pattern Recognition and Machine Learning](http://research.microsoft.com/en-us/um/people/cmbishop/prml/).

The idea is that if the weights are given iid normal prior distributions and the noise of the response vector is known (also iid normal), the posterior can be found easily in closed form. This leaves the question of how these values should be determined, and Bishop's suggestion is to use [marginal likelihood](http://en.wikipedia.org/wiki/Marginal_likelihood). 

The result of fitting such a model is a form that yields a posterior predictive distribution in closed form, and model evidence that can easily be used for model selection.
