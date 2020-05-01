using BayesianLinearRegression
using Test

@testset "BayesianLinearRegression.jl" begin
    (n,p) = (10,5)
    X = randn(n,p) 
    β = randn(p)
    ε = randn(n)
    y = X*β + ε
    m = BayesianLinReg(X,y)
    fit!(m)
    
    f(α,β,w) = BayesianLinearRegression._logEv(n, p, α, β, m.X, m.y, m.hessian, w)
    f(α,β,m.weights)
end
