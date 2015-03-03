module BayesianLinearRegression

export Fit, fit0, fit1, Prediction, predict

make_counter = function(maxiter)
  count = 0
  function(a...)
    iters = count
    count += 1
    iters > maxiter
  end
end

enoughIters = make_counter(1000)

type Fit
  design                 :: Matrix{Float64}
  standardize
  response               :: Vector{Float64}
  priorPrecision         :: Float64
  noisePrecision         :: Float64
  effectiveNumParameters :: Float64
  logEvidence            :: Float64
  mapWeights             :: Vector{Float64}
  bias                   :: Float64
  hessian                :: Matrix{Float64}
end

function fit0(x, y, studentize=true, enoughIters=enoughIters)
  (n,p) = size(x)
  design = x

  if studentize
    μ = mapslices(mean, x, 1)
    σ = 0μ
    for j in 1:length(σ)
      σ[j] = stdm(x[:,j], μ[j])
    end
    x = (x .- μ) ./ σ
  else
    μ = zeros(p)
    σ = ones(p)
  end

  xtx = x' * x
  xty = x' * y
  
  α0 = 1.0
  β0 = 1.0

  xtxEigs = map(s -> s*s, svdvals(x))

  getHessian(a,b) = a * eye(p) + b * xtx

  normSquared(x) = x ⋅ x

  function go(a0, b0) 
    h0 = getHessian(a0, b0)
    m0 = b0 * (h0 \ xty)
    c = xtxEigs |> (x ->  x ./ (a0 .+ x)) |> sum
    a = c / (m0 ⋅ m0)
    b = (n - c) / normSquared(y - x * m0)
    if enoughIters(a,b)
      (a,b)
    else
      go(a, b)
    end
  end

  (α, β) = go(α0, β0)
  h = getHessian(α, β)
  m = β * (h \ xty)

  γ = xtxEigs |> (x ->  x ./ (α .+ x)) |> sum

  logEv = 0.5 * 
    ( p * log(α) 
    + n * log(β) 
    - (β * normSquared(y - x * m) + α * (m ⋅ m))
    - logdet(h)
    - n * log(2π)
    )

  standardize(x) = (x .- μ) ./ σ

  Fit(
      design # design
    , standardize 
    , y      # response              
    , α      # priorPrecision        
    , β      # noisePrecision        
    , γ      # effectiveNumParameters
    , logEv  # logEvidence           
    , m      # mapWeights            
    , 0      # bias                  
    , h      # hessian               
    )
end

function fit1(x, y, studentize=true, enoughIters=enoughIters)
  ymean = mean(y)
  myFit = fit0(x, y-ymean, studentize, enoughIters)
  myFit.bias = ymean
  myFit
end

type Prediction
  fit            :: Fit
  design         :: Matrix{Float64}
  yhat           :: Vector{Float64}
  parameterVar   :: Vector{Float64}
  noiseVar       :: Float64
end

function predict(fit, x)
  design = x
  (n,p) = size(x)
  x = fit.standardize(x)
  yhat = x * fit.mapWeights + fit.bias

  # σ2 should be the variance of the parameter estimate, and will be computed using the Hessian.
  σ2 = zeros(n)
  for i= 1:n
    σ2[i] = (x[i,:] * (fit.hessian \ x[i,:]'))[1]
  end
  Prediction(fit, design, yhat, σ2, inv(fit.noisePrecision))
end

function sample(pred, x, n=100)

end

end # module
