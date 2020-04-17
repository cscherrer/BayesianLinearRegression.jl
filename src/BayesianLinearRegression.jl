module BayesianLinearRegression

using Statistics
using LinearAlgebra
using ApproxFun
using Measurements

include("callbacks.jl")
include("chebyshev.jl")

export BayesianLinReg

mutable struct BayesianLinReg{T}
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

    uncertaintyBasis :: Vector{Measurement{Float64}}
end



function BayesianLinReg(
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

    BayesianLinReg{T}(
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
        , zeros(p) .± 1.0
    )
end

const normSquared = LinearAlgebra.norm_sqr

function Base.iterate(m::BayesianLinReg{T}, iteration=1) where{T}
    m.done && return nothing

    (n, p) = size(m.X)
    α = m.priorPrecision
    β = m.noisePrecision

    

    gamma() = let
        α_over_β = m.priorPrecision / m.noisePrecision
        sum((λ / (α_over_β + λ) for λ in m.XtXeigs))
    end

    if m.updatePrior
        m.priorPrecision = gamma() / dot(m.weights, m.weights)
    end

    if m.updateNoise
        m.noisePrecision = (n - gamma()) / normSquared(m.y - m.X * m.weights)
    end

    m.hessian .= α * I + β .* m.XtX;
    m.weights .= β .* (m.hessian \ m.Xty);        
    
    m.iterationCount += 1
    return (m, iteration + 1)
end

export fit!

function fit!(m::BayesianLinReg; kwargs...)
    m.done = false
    callback = get(kwargs, :callback, fixedEvidence())

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

export logEvidence

function logEvidence(m::BayesianLinReg{T}) where {T}
    (n,p) = size(m.X)
    α = m.priorPrecision
    β = m.noisePrecision
    return _logEv(n, p, α, β, m.X, m.y, m.hessian, m.weights) 
end

function _logEv(n, p, α, β, X, y, H, w) 
    logEv = 0.5 * 
        ( p * log(α) 
        + n * log(β)
        - (β * normSquared(y - X * w) + α * normSquared(w))
        - logdet(H)
        - n * log(2π)
        )
    return logEv
end

export effectiveNumParameters

function effectiveNumParameters(m::BayesianLinReg)
    α = m.priorPrecision 
    β = m.noisePrecision
    α_over_β = α/β
    return sum((λ / (α_over_β + λ) for λ in m.XtXeigs))
end

export posteriorPrecision

function posteriorPrecision(m::BayesianLinReg)
    α = m.priorPrecision
    β = m.noisePrecision
    return α*I + β .* m.XtX
end

export posteriorVariance

posteriorVariance(m::BayesianLinReg) = inv(cholesky(posteriorPrecision(m)))

export posteriorWeights

function posteriorWeights(m)
    p = size(m.X,2)
    ϕ = posteriorPrecision(m)
    U = cholesky!(ϕ).U
    return m.weights + inv(U) * m.uncertaintyBasis
end

export predict

function predict(m::BayesianLinReg, X; uncertainty=true, noise=true)
    # dispatch to avoid type instability
    return _predict(m, X, Val(uncertainty), Val(noise))
end

function _predict(m, X, ::Val{true}, ::Val{true})
    yhat = _predict(m, X, Val(true), Val(false))
    n = length(yhat)
    noise = zeros(n) .± noiseScale(m)
    yhat .+= noise
    return yhat
end

_predict(m, X, ::Val{true}, ::Val{false}) = X * posteriorWeights(m)

_predict(m, X, ::Val{false}, ::Val{true}) = @error "Noise requires uncertainty"

_predict(m, X, ::Val{false}, ::Val{false}) = X * m.weights

export priorPrecision
priorPrecision(m::BayesianLinReg) = m.priorPrecision

export priorVariance
priorVariance(m::BayesianLinReg) = 1/m.priorPrecision

export priorScale
priorScale(m::BayesianLinReg) = sqrt(priorVariance(m))



export noisePrecision
noisePrecision(m::BayesianLinReg) = m.noisePrecision

export noiseVariance
noiseVariance(m::BayesianLinReg) = 1/m.noisePrecision

export noiseScale
noiseScale(m::BayesianLinReg) = sqrt(noiseVariance(m))


end # module
