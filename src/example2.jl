using BayesianLinearRegression

X = randn(100, 80);
β = randn(50);
y = X[:,1:50] * β + randn(100);


@time m = BayesianLinReg(X,y);




@time fit!(m);




logEvidence(m)

effectiveNumParameters(m)

priorScale(m)
noiseScale(m)

posteriorWeights(m)
posteriorVariance(m)

predict(m,X[:,m.active])

function getpath!(m)
    path = Pair{Int, Float64}[]
    while length(m.active) > 1
        bestj = 0
        bestlogEv = -Inf
        activebase = m.active
        for j in activebase
            m.active = setdiff(activebase, j)
            update!(m)
            fit!(m)

            thislogEv = logEvidence(m)
            if thislogEv > bestlogEv
                bestj = j
                bestlogEv = thislogEv
            end
        end

        push!(path, bestj => bestlogEv)
        println("removing ", bestj)
        m.active = setdiff(activebase, bestj)
    end
    return path
end

path = getpath!(m)

logEvs = [getproperty(p,:second) for p in path]

plot(logEvs)

















effectiveNumParameters(m)

priorScale(m)
noiseScale(m)

posteriorWeights(m)
posteriorVariance(m)

predict(m,view(m.X, :, m.active))


function f(n)
    m = BayesianLinReg(X[:,1:n],y)
    fit!(m; callback=stopAtIteration(10))
    return m
end;

ms = f.(1:100);

argmax(logEvidence.(ms))
m = ms[49];


effectiveNumParameters(m)

priorScale(m)
noiseScale(m)

posteriorWeights(m)
posteriorVariance(m)

predict(m,m.X)

using Plots
using Statistics

logevs = logEvidence.(ms)

plot(3:100,logevs[3:100], legend=false)
xlabel!("Number of features used")
ylabel!("Log Evidence")

m = BayesianLinReg(X[:,1:50],y)
fit!(m; callback=stopAtIteration(10))
