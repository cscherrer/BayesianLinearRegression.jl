module BayesianLinearRegression

# using Statistics
using LinearAlgebra
using Measurements
using Printf
using LazyArrays

include("callbacks.jl")

export BayesianLinReg

mutable struct BayesianLinReg{T}
    XtX :: SubArray{Float64,2,Array{Float64,2},Tuple{Array{Int64,1},Array{Int64,1}},false}
    Xty :: SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false}
    yty :: Float64
    N   :: Int

    Hinv:: SubArray{Float64,2,Array{Float64,2},Tuple{Array{Int64,1},Array{Int64,1}},false}

    eig :: LinearAlgebra.Eigen{Float64,Float64,Array{Float64,2},Array{Float64,1}}

    priorPrecision :: T
    updatePrior :: Bool
    
    noisePrecision :: T
    updateNoise :: Bool
    
    weights

    iterationCount :: Int
    done :: Bool

    uncertaintyBasis :: Vector{Measurement{Float64}}

    active :: Vector{Int}
end

function symmetric!(S)
    S .+= S'
    S ./= 2
end

###########################################################################################
# Hessian

export hessian

function hessian(eig, α, β)
    Q = eig.vectors
    D = Diagonal(α .+ β .* eig.values)

    H = Q * D * Q'
    symmetric!(H)

    return H
end

function hessian(m::BayesianLinReg)
    α = m.priorPrecision
    β = m.noisePrecision
    
    return hessian(m.eig, α, β)
end


###########################################################################################
# Inverse Hessian

function hessianinv!(m::BayesianLinReg)
    Λ = m.eig.values
    Q = m.eig.vectors

    α = m.priorPrecision
    β = m.noisePrecision

    D = Diagonal(inv.(α .+ β .* Λ))

    Hinv = m.Hinv
    Hinv .= @~ Q * D * Q'
    symmetric!(Hinv)

    return Hinv
end

###########################################################################################
# Log-determinant of Hessian

function logdetH(m::BayesianLinReg)
    Λ = m.eig.values
    α = m.priorPrecision
    β = m.noisePrecision
    return logdetH(α, β, Λ)
end

function logdetH(α, β, Λ)
    sum(log, α .+ β .* Λ)
end



function updateWeights!(m::BayesianLinReg)
    β = m.noisePrecision
    Hinv = hessianinv!(m)

    m.weights .= @~ β .* (Hinv * m.Xty)
    return m.weights
end

function BayesianLinReg(
          X::Matrix{T}
        , y::Vector{T}
        ; updatePrior=true
        , updateNoise=true
        ) where {T}

    (N, p) = size(X)
    
    XtX = X' * X
    Xty = X' * y
    yty = normSquared(y)

    symmetric!(XtX)
    
    return BayesianLinReg(XtX, Xty, yty, N)
end

function BayesianLinReg(
      XtX
    , Xty
    , yty
    , N
    ; updatePrior=true
    , updateNoise=true)
    eig = eigen(XtX)

    p = length(Xty)
    ps = [1:p;]

    XtX = view(XtX, ps, ps)
    Xty = view(Xty, ps)

    Λ  = eig.values
    Λ .= max.(Λ, 0.0)
    Q = eig.vectors

    α = 1.0
    β = 1.0

    ps = [1:p;]
    D = Diagonal(inv.(α .+ β .* Λ))
    
    Hinv = Q * D * Q'
    symmetric!(Hinv)
    Hinv = view(Hinv, ps, ps)

    weights = view(β .* (Hinv * Xty), ps)

    BayesianLinReg(
        XtX
      , Xty
      , yty
      , N 
      , Hinv 
      , eig
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

"""
Sum of squared residuals
"""
function ssr(m)
    XtX = m.XtX
    Xty = m.Xty
    yty = m.yty
    w = m.weights
    return yty - 2 * mydot(w, Xty) + w' * XtX * w
end    

function Base.iterate(m::BayesianLinReg{T}, iteration=1) where {T}
    m.done && return nothing

    N = m.N
    ps = m.active
    α = m.priorPrecision
    β = m.noisePrecision

    gamma(m) = effectiveNumParameters(m)

    XtX = m.XtX
    Xty = m.Xty
    yty = m.yty

    w = m.weights

    wᵀw = normSquared(w)
    rᵀr = ssr(m)
    if m.updatePrior
        m.priorPrecision = gamma(m) / wᵀw
    end

    if m.updateNoise
        m.noisePrecision = (N - gamma(m)) / rᵀr
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
    N = m.N
    α = m.priorPrecision
    β = m.noisePrecision
    rtr = ssr(m)
    return _logEv(N, rtr, m.active, α, β, m.eig.values, m.weights) 
end

const log2π = log(2π)

function _logEv(N, rtr, active, α, β, Λ, w) 
    p = length(active)

    logEv = 0.5 * 
        ( p * log(α) 
        + N * log(β)
        - (β * rtr + α * normSquared(w))
        - logdetH(α, β, Λ)
        - N * log2π
        )
    return logEv
end

export effectiveNumParameters

function effectiveNumParameters(m::BayesianLinReg)
    α_over_β = m.priorPrecision / m.noisePrecision
    
    Λ = m.eig.values

    return sum(λ -> λ / (α_over_β + λ), Λ)
end

export posteriorPrecision

function posteriorPrecision(m::BayesianLinReg)
    return hessian(m)
end

export posteriorVariance

posteriorVariance(m::BayesianLinReg) = hessianinv!(m)

export posteriorWeights

function posteriorWeights(m)
    p = length(m.active)
    ϕ = posteriorPrecision(m)
    U = cholesky!(ϕ).U

    w = m.weights
    ε = inv(U) * view(m.uncertaintyBasis, m.active)
    return w + ε
end

# TODO: Why is this slower than `posteriorWeights`?
function postWeights(m)
    α = m.priorPrecision
    β = m.noisePrecision
    Λ = m.Λ
    Vt = m.Vt
    V = Vt'
    
    w = view(m.weights, m.active)

    # S = V * diagm(sqrt.(inv.(α .+ β .* Λ))) * Vt
    # ε = S * view(m.uncertaintyBasis, m.active)

    ε = view(m.uncertaintyBasis, m.active)
    ε .= @~ Vt * ε
    ε ./= sqrt.(α .+ β .* Λ)
    ε .= @~ V * ε

    return w + ε
end

export predict

function predict(m::BayesianLinReg, X; uncertainty=true, noise=false)
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

###########################################################################################
# Helpers

function normSquared(v::AbstractVector{T}) where {T}
    s = zero(T)
    @inbounds @simd for i ∈ eachindex(v) 
        s += v[i]*v[i]
    end 
    return s 
end

mydot(x,y) = mydot(promote(x,y...)...)

function mydot(x::AbstractVector{T},y::AbstractVector{T}) where {T}
    s = zero(T)
    @inbounds @simd for i ∈ eachindex(x) 
        s += x[i]*y[i]
    end 
    return s 
end

export update!
function update!(m)
    ps = m.active
    XtX = m.XtX = view(m.XtX.parent, ps, ps)
    Xty = m.Xty = view(m.Xty.parent, ps)
    weights = m.weights = view(m.weights.parent, ps)
    Hinv = m.Hinv = view(m.Hinv.parent, ps, ps)

    m.eig = eigen(collect(m.XtX))
    m.eig.values .= max.(m.eig.values, 0.0)

    α = m.priorPrecision = 1.0
    β = m.noisePrecision = 1.0

    Q = m.eig.vectors
    Λ = m.eig.values
    D = Diagonal(inv.(α .+ β .* Λ))
    Hinv .= @~ Q * D * Q'

    m.weights .= @~ β .* (Hinv * Xty)
    return m
end

end # module
