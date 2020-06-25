module BayesianLinearRegression

# using Statistics
using LinearAlgebra
using Measurements
using Printf

include("callbacks.jl")

export BayesianLinReg

mutable struct BayesianLinReg{T}
    X :: Matrix{T}
    y :: Vector{T}

    Xty :: Vector{T}
    
    Λ :: Vector{T}
    Vt :: Matrix{T}

    priorPrecision :: T
    updatePrior :: Bool
    
    noisePrecision :: T
    updateNoise :: Bool
    
    weights :: Vector{T}

    iterationCount :: Int
    done :: Bool

    uncertaintyBasis :: Vector{Measurement{Float64}}

    active :: Vector{Int}
end

function symmetric!(S)
    S .+= S'
    S ./= 2
end

export hessian

function hessian(Λ, Vt, α, β)
    V = Vt'
    D = Diagonal(α .+ β .* Λ)

    H = V * D * Vt
    symmetric!(H)

    return H
end

function hessian(m::BayesianLinReg)
    α = m.priorPrecision
    β = m.noisePrecision
    
    return hessian(m.Λ, m.Vt, α, β)
end

function logdetH(m::BayesianLinReg)
    Λ = m.Λ
    α = m.priorPrecision
    β = m.noisePrecision
    return logdetH(α, β, Λ)
end

function logdetH(α, β, Λ)
    sum(log, α .+ β .* Λ)
end

export hessianinv

function hessianinv(m::BayesianLinReg)
    Λ = m.Λ
    Vt = m.Vt
    V = Vt'

    α = m.priorPrecision
    β = m.noisePrecision

    D = Diagonal(inv.(α .+ β .* Λ))
    Hinv = V * D * Vt
    
    symmetric!(Hinv)
    return Hinv
end

function updateWeights!(m::BayesianLinReg)
    β = m.noisePrecision
    Hinv = hessianinv(m)

    m.weights[m.active] .= β .* (Hinv * m.Xty)
    return view(m.weights, m.active)
end
``
function BayesianLinReg(
          X::Matrix{T}
        , y::Vector{T}
        ; updatePrior=true
        , updateNoise=true
        ) where {T}

    (n, p) = size(X)
    
    Xty = X' * y

    F = svd(X' * X)
    Λ = F.S
    Vt = F.Vt
    V = Vt'

    α = 1.0
    β = 1.0

    D = Diagonal(inv.(α .+ β .* Λ))
    Hinv = V * D * Vt

    weights = β .* (Hinv * Xty)

    BayesianLinReg{T}(
          X 
        , y 
        , Xty 
        , Λ
        , Vt
        , α 
        , updatePrior
        , β
        , updateNoise
        , weights 
        , 0
        , false 
        , zeros(p) .± 1.0
        , [1:p;]
    )
end

function normSquared(v) 
    s = 0.0 
    @inbounds @simd for i ∈ eachindex(v) 
        s += v[i]*v[i]
    end 
    return s 
end
    

function Base.iterate(m::BayesianLinReg{T}, iteration=1) where {T}
    m.done && return nothing

    n = size(m.X,1)
    ps = m.active
    α = m.priorPrecision
    β = m.noisePrecision

    gamma(m) = effectiveNumParameters(m)

    X = view(m.X, :, ps)
    w = view(m.weights, ps)

    if m.updatePrior
        m.priorPrecision = gamma(m) / dot(w,w)
    end

    if m.updateNoise
        m.noisePrecision = (n - gamma(m)) / normSquared(m.y - X * w)
    end

    updateWeights!(m)     
    
    m.iterationCount += 1
    return (m, iteration + 1)
end

export fit!

function fit!(m::BayesianLinReg; kwargs...)
    m.done = false
    callback = get(kwargs, :callback, stopAtIteration(10))

    if m.updatePrior
        m.priorPrecision = 1.0
    end

    if m.updateNoise
        m.noisePrecision = 1.0
    end

    X = view(m.X,:,m.active)
    m.Xty = X' * m.y

    F = svd(X' * X)
    m.Λ = F.S
    m.Vt = F.Vt

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
    n = size(m.X, 1)
    α = m.priorPrecision
    β = m.noisePrecision
    return _logEv(n, m.active, α, β, m.X, m.y, m.Λ, m.Vt, m.weights) 
end

const log2π = log(2π)

function _logEv(n, active, α, β, X, y, Λ, Vt, w) 
    p = length(active)
    X = view(X, :, active)

    H = hessian(Λ, Vt, α, β)
    w = view(w, active)

    logEv = 0.5 * 
        ( p * log(α) 
        + n * log(β)
        - (β * normSquared(y - X * w) + α * normSquared(w))
        - logdetH(α, β, Λ)
        - n * log2π
        )
    return logEv
end

export effectiveNumParameters

function effectiveNumParameters(m::BayesianLinReg)
    α_over_β = m.priorPrecision / m.noisePrecision
    
    Λ = m.Λ

    return sum(λ -> λ / (α_over_β + λ), Λ)
end

export posteriorPrecision

function posteriorPrecision(m::BayesianLinReg)
    return hessian(m)
end

export posteriorVariance

posteriorVariance(m::BayesianLinReg) = hessianinv(m)

export posteriorWeights

function posteriorWeights(m)
    p = length(m.active)
    ϕ = posteriorPrecision(m)
    U = cholesky!(ϕ).U

    w = view(m.weights, m.active)
    ε = inv(U) * view(m.uncertaintyBasis, m.active)
    return w + ε
end

export postWeights

function postWeights(m)
    α = m.priorPrecision
    β = m.noisePrecision
    Λ = m.Λ
    Vt = m.Vt
    V = Vt'

    S = V * diagm(sqrt.(inv.(α .+ β .* Λ))) * Vt
    
    # w = view(m.weights, m.active)
    # ε = S * view(m.uncertaintyBasis, m.active)

    # return w + ε
    return S
end

export predict

function predict(m::BayesianLinReg, X; uncertainty=true, noise=true)
    noise &= uncertainty
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

function Base.show(io::IO, m::BayesianLinReg{T}) where {T}
    @printf io "BayesianLinReg model\n"
    @printf io "\n"
    @printf io "Log evidence: %3.2f\n" logEvidence(m)
    @printf io "Prior scale: %5.2f\n" priorScale(m)
    @printf io "Noise scale: %5.2f\n" noiseScale(m)
    @printf io "\n"
    @printf io "Coefficients:\n"
    weights = posteriorWeights(m)
    for (j,w) in zip(m.active, weights)
        @printf io "%3d: %5.2f ± %4.2f\n" j w.val w.err
    end
end

end # module
