# BayesianLinearRegression

[![Build Status](https://travis-ci.com/cscherrer/BayesianLinearRegression.jl.svg?branch=master)](https://travis-ci.com/cscherrer/BayesianLinearRegression.jl)
[![Codecov](https://codecov.io/gh/cscherrer/BayesianLinearRegression.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/cscherrer/BayesianLinearRegression.jl)
This library implements Bayesian Linear Regression, as described in Chris Bishop's [Pattern Recognition and Machine Learning](http://research.microsoft.com/en-us/um/people/cmbishop/prml/).

The idea is that if the weights are given iid normal prior distributions and the noise of the response vector is known (also iid normal), the posterior can be found easily in closed form. This leaves the question of how these values should be determined, and Bishop's suggestion is to use [marginal likelihood](http://en.wikipedia.org/wiki/Marginal_likelihood). 

The result of fitting such a model is a form that yields a posterior predictive distribution in closed form, and model evidence that can easily be used for model selection.


## Background 

This package fits the linear regression model

![\\begin{aligned}P(\\boldsymbol{w}\|\\alpha) & =\\text{Normal}(\\boldsymbol{w}\|0,\\alpha\^{-1}\\boldsymbol{I})\\\\
P(\\boldsymbol{y}\|\\boldsymbol{w},\\beta) & =\\text{Normal}(\\boldsymbol{y}\|\\boldsymbol{Xw},\\beta\^{-1}\\boldsymbol{I})\\ .
\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7DP%28%5Cboldsymbol%7Bw%7D%7C%5Calpha%29%20%26%20%3D%5Ctext%7BNormal%7D%28%5Cboldsymbol%7Bw%7D%7C0%2C%5Calpha%5E%7B-1%7D%5Cboldsymbol%7BI%7D%29%5C%5C%0AP%28%5Cboldsymbol%7By%7D%7C%5Cboldsymbol%7Bw%7D%2C%5Cbeta%29%20%26%20%3D%5Ctext%7BNormal%7D%28%5Cboldsymbol%7By%7D%7C%5Cboldsymbol%7BXw%7D%2C%5Cbeta%5E%7B-1%7D%5Cboldsymbol%7BI%7D%29%5C%20.%0A%5Cend%7Baligned%7D "\begin{aligned}P(\boldsymbol{w}|\alpha) & =\text{Normal}(\boldsymbol{w}|0,\alpha^{-1}\boldsymbol{I})\\
P(\boldsymbol{y}|\boldsymbol{w},\beta) & =\text{Normal}(\boldsymbol{y}|\boldsymbol{Xw},\beta^{-1}\boldsymbol{I})\ .
\end{aligned}")

Rather than finding "the" value for
![\\boldsymbol{\\hat{w}}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Chat%7Bw%7D%7D "\boldsymbol{\hat{w}}"),
we take a Bayesian approach and find the posterior distribution, given
by

![P(\\boldsymbol{w}\|\\boldsymbol{y})=\\text{Normal}(\\boldsymbol{w}\|\\boldsymbol{\\mu},\\boldsymbol{H})\\ ,](https://latex.codecogs.com/png.latex?P%28%5Cboldsymbol%7Bw%7D%7C%5Cboldsymbol%7By%7D%29%3D%5Ctext%7BNormal%7D%28%5Cboldsymbol%7Bw%7D%7C%5Cboldsymbol%7B%5Cmu%7D%2C%5Cboldsymbol%7BH%7D%29%5C%20%2C "P(\boldsymbol{w}|\boldsymbol{y})=\text{Normal}(\boldsymbol{w}|\boldsymbol{\mu},\boldsymbol{H})\ ,")

where

![\\begin{aligned}\\boldsymbol{\\mu} & =\\beta\\boldsymbol{H}\^{-1}\\boldsymbol{X}\^{T}\\boldsymbol{y}\\\\
\\boldsymbol{H} & =\\alpha\\boldsymbol{I}+\\beta\\boldsymbol{X}\^{T}\\boldsymbol{X}\\ .
\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%5Cboldsymbol%7B%5Cmu%7D%20%26%20%3D%5Cbeta%5Cboldsymbol%7BH%7D%5E%7B-1%7D%5Cboldsymbol%7BX%7D%5E%7BT%7D%5Cboldsymbol%7By%7D%5C%5C%0A%5Cboldsymbol%7BH%7D%20%26%20%3D%5Calpha%5Cboldsymbol%7BI%7D%2B%5Cbeta%5Cboldsymbol%7BX%7D%5E%7BT%7D%5Cboldsymbol%7BX%7D%5C%20.%0A%5Cend%7Baligned%7D "\begin{aligned}\boldsymbol{\mu} & =\beta\boldsymbol{H}^{-1}\boldsymbol{X}^{T}\boldsymbol{y}\\
\boldsymbol{H} & =\alpha\boldsymbol{I}+\beta\boldsymbol{X}^{T}\boldsymbol{X}\ .
\end{aligned}")

All of the above depends on fixed values for
![\\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\alpha") and
**![\\beta](https://latex.codecogs.com/png.latex?%5Cbeta "\beta")**
being specified in adviance. Alternatively, a "full Bayesian" approach
would specify prior distributions over these, and work in terms of their
posterior distribution for the final result.

*Marginal likelihood* finds a middle ground between these two
approaches, and determines values for the
![\\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\alpha") and
![\\beta](https://latex.codecogs.com/png.latex?%5Cbeta "\beta")
hyperparameters by maximizing

![P(\\boldsymbol{y}\|\\alpha,\\beta)=\\int P(\\boldsymbol{y}\|\\boldsymbol{w},\\alpha,\\beta)\\,d\\boldsymbol{w}\\ .](https://latex.codecogs.com/png.latex?P%28%5Cboldsymbol%7By%7D%7C%5Calpha%2C%5Cbeta%29%3D%5Cint%20P%28%5Cboldsymbol%7By%7D%7C%5Cboldsymbol%7Bw%7D%2C%5Calpha%2C%5Cbeta%29%5C%2Cd%5Cboldsymbol%7Bw%7D%5C%20. "P(\boldsymbol{y}|\alpha,\beta)=\int P(\boldsymbol{y}|\boldsymbol{w},\alpha,\beta)\,d\boldsymbol{w}\ .")

This reduces to

![P(\\boldsymbol{y}\|\\alpha,\\beta)=\\frac{1}{2}\\left\[n\\log\\alpha+p\\log\\beta-\\beta\\left\\Vert \\boldsymbol{y}-\\boldsymbol{X\\mu}\\right\\Vert \^{2}-\\alpha\\left\\Vert \\boldsymbol{\\mu}\\right\\Vert \^{2}-\\log\\left\|\\boldsymbol{H}\\right\|-n\\log2\\pi\\right\]\\ .](https://latex.codecogs.com/png.latex?P%28%5Cboldsymbol%7By%7D%7C%5Calpha%2C%5Cbeta%29%3D%5Cfrac%7B1%7D%7B2%7D%5Cleft%5Bn%5Clog%5Calpha%2Bp%5Clog%5Cbeta-%5Cbeta%5Cleft%5CVert%20%5Cboldsymbol%7By%7D-%5Cboldsymbol%7BX%5Cmu%7D%5Cright%5CVert%20%5E%7B2%7D-%5Calpha%5Cleft%5CVert%20%5Cboldsymbol%7B%5Cmu%7D%5Cright%5CVert%20%5E%7B2%7D-%5Clog%5Cleft%7C%5Cboldsymbol%7BH%7D%5Cright%7C-n%5Clog2%5Cpi%5Cright%5D%5C%20. "P(\boldsymbol{y}|\alpha,\beta)=\frac{1}{2}\left[n\log\alpha+p\log\beta-\beta\left\Vert \boldsymbol{y}-\boldsymbol{X\mu}\right\Vert ^{2}-\alpha\left\Vert \boldsymbol{\mu}\right\Vert ^{2}-\log\left|\boldsymbol{H}\right|-n\log2\pi\right]\ .")

This package maximizes the marginal likelihood using the approach
described in Bishop (2006), which alternates between

1.  Update
    ![\\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\alpha")
    and ![\\beta](https://latex.codecogs.com/png.latex?%5Cbeta "\beta")

2.  Update
    ![\\boldsymbol{\\mu}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cmu%7D "\boldsymbol{\mu}")
    and
    ![\\boldsymbol{H}](https://latex.codecogs.com/png.latex?%5Cboldsymbol%7BH%7D "\boldsymbol{H}").

For details, see

Bishop, C. M. (2006). Pattern Recognition and Machine Learning (M.
Jordan, J. Kleinberg, & B. Scholkopf (eds.); Vol. 53, Issue 9).
Springer. https://doi.org/10.1117/1.2819119

## Example 

First let's generate some fake data 

```julia
using BayesianLinearRegression
n = 20
p = 7;
X = randn(n, p);
β = randn(p);
y = X * β + randn(n);
```

Next we instantiate and fit the model. `fit!` takes a `callback` keyword, which can be used to control stopping criteria, output some function of the model, or even change the model at each iteration.

```julia
julia> m = fit!(BayesianLinReg(X,y);
           callback = 
               # fixedEvidence()
               # stopAfter(2)
               stopAtIteration(50)
       );
```

Once the model is fit, we can ask for the log-evidence (same as the marginal likelihood):

```julia
julia> logEvidence(m)
-37.98301162730431
```

This can be used as a criterion for model selection; adding some junk features tends to reduce the log-evidence:

```julia
julia> logEvidence(fit!(BayesianLinReg([X randn(n,3)],y)))
-41.961261348171064
```

We can also query the values determined for the prior and the noise scales:

```julia
julia> priorScale(m)
1.2784615163925537

julia> noiseScale(m)
0.8520928089955583
```

The `posteriorWeights` include uncertainty, thanks to Measurements.jl:

```julia
julia> posteriorWeights(m)
7-element Array{Measurements.Measurement{Float64},1}:
 -0.35 ± 0.19
  0.07 ± 0.21
 -2.21 ± 0.29
  0.57 ± 0.22
  1.08 ± 0.18
 -2.14 ± 0.24
 -0.15 ± 0.25
```

This uncertainty is propagated for prediciton:

```julia
julia> predict(m,X)
20-element Array{Measurements.Measurement{Float64},1}:
   5.6 ± 1.0
  3.06 ± 0.95
 -3.14 ± 0.98
   3.3 ± 1.0
   4.6 ± 1.1
  -0.2 ± 1.0
 -0.91 ± 0.96
 -3.71 ± 0.99
  -4.0 ± 1.0
 -0.86 ± 0.95
 -5.69 ± 0.95
  4.32 ± 0.94
 -3.49 ± 0.94
  -0.7 ± 1.0
   0.5 ± 1.1
 -0.49 ± 0.92
  0.67 ± 0.91
  0.39 ± 0.95
  -7.3 ± 1.1
  0.11 ± 0.98
```

Finally, we can output the effective number of parameters, which is useful for some computations:

```julia
julia> effectiveNumParameters(m)
6.776655463465779
```
