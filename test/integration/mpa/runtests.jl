# MultiPrecisionArrays.jl integration suite (task's <integration_targets>,
# item 3 -- "the package's flagship use case").
# Run standalone: julia --project=test/integration/mpa
#                        test/integration/mpa/runtests.jl

using Test
using Random
using LinearAlgebra
using MultiPrecisionArrays
using LowPrecisionChop

const LF16 = lowfloattype(Float16Format)

@testset "mplu() convenience constructor: documented API gap, not a bug in us" begin
    # MultiPrecisionArrays.MPArray has exactly 3 constructor methods: a
    # fully generic field constructor (AH,AL,residual,sol,onthefly) and two
    # *concrete-type-only* convenience constructors for AbstractMatrix{Float64}
    # and AbstractMatrix{Float32}. `mplu(A; TF, TR, onthefly)` calls the
    # convenience constructor internally, so it MethodErrors on any other
    # element type, including LowFloat -- confirmed directly, not assumed.
    rng = Xoshiro(1)
    n = 10
    A64 = randn(rng, n, n) + n * I
    A = LF16.(A64)
    err = nothing
    try
        mplu(A; TF=LF16, TR=Float64, onthefly=true)
    catch e
        err = e
    end
    @test err isa MethodError
    # kwargs calls report `Core.kwcall` as `err.f`, not `MPArray` directly;
    # check the message instead for the actual failing constructor.
    @test occursin("MPArray", sprint(showerror, err))
end

@testset "Workaround: direct MPArray field constructor + mplu! + \\ works with LowFloat" begin
    # The underlying package IS fully generic -- MPArray's struct and
    # mplu!(::MPArray)/\\(::MPArray,b) all work with any AbstractFloat once
    # you bypass the convenience constructor's type restriction. This is
    # the package's flagship use case (mixed-precision iterative
    # refinement), demonstrated end to end with LowFloat as the
    # low-precision factorization.
    rng = Xoshiro(1)
    n = 20
    A64 = randn(rng, n, n) + n * I  # diagonally dominant, well conditioned
    b64 = A64 * ones(n)

    AH = copy(A64)
    AL = LF16.(A64)
    residual = zeros(n)
    sol = zeros(n)
    MPA = MultiPrecisionArrays.MPArray(AH, AL, residual, sol, true)
    mplu!(MPA)
    mout = \(MPA, b64; reporting=true)

    # "IR converges to Float64 accuracy in the expected small number of
    # iterations for well-conditioned problems" -- task requirement.
    @test length(mout.rhist) <= 15
    @test mout.rhist[end] < 1e-6
    @test norm(mout.sol - ones(n), Inf) < 1e-6
    @test issorted(mout.rhist; rev=true) || mout.rhist[end] < mout.rhist[1] / 100  # residual trends down
end

@testset "IR diverges/stagnates past the known conditioning threshold (documented, not silently passed)" begin
    # "...and diverges as predicted past the known conditioning threshold" --
    # task requirement. cond(A) ~ 3e9 with an fp16 factorization (unit
    # roundoff ~ 2^-10 ~ 1e-3): IR should stagnate well short of Float64
    # accuracy, not converge, and not silently produce a falsely-small
    # residual either.
    rng = Xoshiro(2)
    n = 20
    U = Matrix(qr(randn(rng, n, n)).Q)
    sigma = [10.0^(-i / 2) for i in 0:(n - 1)]
    A64 = U * Diagonal(sigma) * U'
    b64 = A64 * ones(n)
    @test cond(A64) > 1e8  # confirm the problem is actually ill-conditioned as intended

    AH = copy(A64)
    AL = LF16.(A64)
    residual = zeros(n)
    sol = zeros(n)
    MPA = MultiPrecisionArrays.MPArray(AH, AL, residual, sol, true)
    mplu!(MPA)
    mout = \(MPA, b64; reporting=true)

    @test mout.rhist[end] > 1e-3  # does NOT reach Float64-accuracy residual
    @test norm(mout.sol - ones(n), Inf) > 1e-2  # solution is materially wrong
end
