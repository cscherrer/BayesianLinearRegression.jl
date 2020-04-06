using BayesianLinearRegression

import Random
import Printf: @printf

Random.seed!(2)
n = 20;
xx = range(0, 2π, length=n);
y = sin.(xx) .+ 0.1 .* randn(n);

function compareModels(p)
    X = chebyshev(xx,p)
    m = BayesianLinReg(X,y)
    fit!(m, callback=stopAtIteration(10))
    return logEvidence(m)
end


for (p,logEv) in enumerate(compareModels.(1:20))
    @printf "p = %2d: evidence = %8.2f\n" p exp(logEv)
end

X = chebyshev(xx,6);
@time m = BayesianLinReg(X,y) |> fit!;

priorScale(m)

noiseScale(m)

posteriorWeights(m)

effectiveNumParameters(m)

yhat = predict(m,m.X)