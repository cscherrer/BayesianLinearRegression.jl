module BayesianLinearRegression

using Statistics
using LinearAlgebra
using ApproxFun
using Measurements

include("callbacks.jl")
include("chebyshev.jl")

mutable struct BayesianLinearRegression{T}
    X :: Matrix{T}
    y :: Vector{T}
    XtX :: Matrix{T}
    Xty :: Vector{T}
    XtXeigs :: Vector{T}

    priorPrecision :: T
    updatePrior :: Bool
    
    noisePrecision :: T
    updateNoise :: Bool
    
    weights :: Vector{T}
    hessian :: Matrix{T}

    iterationCount :: Int
    done :: Bool
end



function BayesianLinearRegression(
          X::Matrix{T}
        , y::Vector{T}
        ; updatePrior=true
        , updateNoise=true
        ) where {T}

    (n, p) = size(X)
    
    Xty = X' * y
    XtX = X' * X

    XtXeigs = eigvals(XtX)

    α = 1.0
    β = 1.0

    hessian = α * I + β .* XtX
    weights = β .* (hessian \ Xty)

    BayesianLinearRegression{T}(
          X 
        , y 
        , XtX 
        , Xty 
        , XtXeigs 
        , α 
        , updatePrior
        , β
        , updateNoise
        , weights 
        , hessian 
        , 0
        , false 
    )
end

const normSquared = LinearAlgebra.norm_sqr

function Base.iterate(m::BayesianLinearRegression{T}, iteration=1) where{T}
    m.done && return nothing

    (n, p) = size(m.X)
    α = m.priorPrecision
    β = m.noisePrecision

    α_over_β = α / β
    gamma = sum((λ / (α_over_β + λ) for λ in m.XtXeigs))

    if m.updatePrior
        m.priorPrecision = gamma / dot(m.weights, m.weights)
    end

    if m.updateNoise
        m.noisePrecision = (n - gamma) / normSquared(m.y - m.X * m.weights)
    end

    m.hessian .= α * I + β .* m.XtX;
    m.weights .= β .* (m.hessian \ m.Xty);        
    
    m.iterationCount += 1
    return (m, iteration + 1)
end


function fit!(m::BayesianLinearRegression; kwargs...)
    m.done = false
    callback = get(kwargs, :callback, stopAfter(2))

    try
        for iter in m
            callback(iter)
        end
    catch e
        if e isa InterruptException
            @warn "Computation interrupted"
            return m
        else
            rethrow()
        end    
    end
    return m        
end


function logEvidence(m::BayesianLinearRegression{T}) where {T}
    (n,p) = size(m.X)
    α = m.priorPrecision
    β = m.noisePrecision
    logEv = 0.5 * 
        ( p * log(α) 
        + n * log(β)
        - (β * normSquared(m.y - m.X * m.weights) + α * normSquared(m.weights))
        - logdet(m.hessian)
        - n * log(2π)
        )
    return logEv
end

function effectiveNumParameters(m::BayesianLinearRegression)
    α = m.priorPrecision 
    β = m.noisePrecision
    α_over_β = α/β
    return sum((λ / (α_over_β + λ) for λ in m.XtXeigs))
end


function posteriorPrecision(m::BayesianLinearRegression)
    α = m.priorPrecision
    β = m.noisePrecision
    return α*I + β .* m.XtX
end

posteriorVariance(m::BayesianLinearRegression) = inv(cholesky(posteriorPrecision(m)))

# function posteriorWeights(m)
#     p = size(m.X,2)
#     postVar = posteriorVariance(m)
#     L = cholesky(postVar).L
#     return m.weights + L * (zeros(p) .± 1)
# end

function posteriorWeights(m)
    p = size(m.X,2)
    ϕ = posteriorPrecision(m)
    U = cholesky!(ϕ).U
    return m.weights + inv(U) * (zeros(p) .± 1)
end

function predict(m::BayesianLinearRegression, X)
    w = posteriorWeights(m)
    σ = 0 ± sqrt(1/m.noisePrecision)
    return X * w .+ σ
end


priorPrecision(m::BayesianLinearRegression) = m.priorPrecision
priorVariance(m::BayesianLinearRegression) = 1/m.priorPrecision
priorScale(m::BayesianLinearRegression) = sqrt(priorVariance(m))


noisePrecision(m::BayesianLinearRegression) = m.noisePrecision
noiseVariance(m::BayesianLinearRegression) = 1/m.noisePrecision
noiseScale(m::BayesianLinearRegression) = sqrt(noiseVariance(m))


end # module