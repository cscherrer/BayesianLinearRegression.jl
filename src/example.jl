

import Random: seed!
Random.seed!(2)
n = 20;
xx = range(0, 2π, length=n);
y = sin.(xx) .+ 0.1 .* randn(n);

function compareModels(p)
    X = chebyshev(xx,p)
    m = BayesianLinearRegression(X,y)
    fit!(m, callback=stopAtIteration(10))
    return logEvidence(m)
end


for (p,logEv) in enumerate(compareModels.(1:20))
    println("p = ",p,",  evidence = ", exp(logEv))
end

X = chebyshev(xx,6);
@time m = BayesianLinearRegression(X,y) |> fit!;

priorScale(m)

noiseScale(m)

posteriorWeights(m)

effectiveNumParameters(m)

yhat = predict(m,m.X)