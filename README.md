# BayesianLinearRegression

This library implements Bayesian Linear Regression, as described in Chris Bishop's [Pattern Recognition and Machine Learning](http://research.microsoft.com/en-us/um/people/cmbishop/prml/).

The idea is that if the weights are given iid normal prior distributions and the noise of the response vector is known (also iid normal), the posterior can be found easily in closed form. This leaves the question of how these values should be determined, and Bishop's suggestion is to use [marginal likelihood](http://en.wikipedia.org/wiki/Marginal_likelihood). 
