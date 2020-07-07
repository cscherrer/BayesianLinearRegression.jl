using BayesianLinearRegression
using Test
using LinearAlgebra

const BLR = BayesianLinearRegression

function runtests()
    X = randn(100,30)
    β = randn(30)
    y = X * β .+ randn(100)

    m = BayesianLinReg(X,y)
    fit!(m)

    @testset "log-determinant of Hessian" begin
        @test BLR.logdetH(m) ≈ logdet(hessian(m))
    end

    @testset "Inverse Hessian" begin
        @test m.Hinv * hessian(m) ≈ I
    end

    @testset "Sum of Squared Residuals" begin
        @test BLR.ssr(m) ≈ BLR.normSquared(y - predict(m,X; uncertainty=false))
    end
end

runtests()
