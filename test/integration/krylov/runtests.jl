# Krylov.jl integration suite (task's <integration_targets>, item 2).
# Run standalone: julia --project=test/integration/krylov
#                        test/integration/krylov/runtests.jl

using Test
using Random
using LinearAlgebra
using Krylov
using LowPrecisionChop

const LFBF16 = lowfloattype(BFloat16Format)
const LF16 = lowfloattype(Float16Format)
const GOLDEN_CG_NITER = 5  # recorded from an initial run (bf16, n=12, seed=1, rtol=atol=1e-2)

@testset "Krylov.cg dispatch: LowFloat routes to the generic path, not BLAS" begin
    # Krylov.jl's own FloatOrComplex{T} = Union{T,Complex{T}} where T<:AbstractFloat
    # -- LowFloat<:AbstractFloat matches the *generic* branch of every k*
    # helper (kdot/knorm/kscal!/kaxpy!/...) rather than the BLAS-specialized
    # one. Confirm the type-level fact directly.
    @test !(LFBF16 <: LinearAlgebra.BLAS.BlasFloat)
    @test !applicable(LinearAlgebra.BLAS.dot, ones(LFBF16, 2), ones(LFBF16, 2))
end

@testset "Krylov.cg on a well-conditioned SPD system at bf16" begin
    rng = Xoshiro(1)
    n = 12
    M = randn(rng, n, n)
    A64 = M' * M + n * I
    b64 = randn(rng, n)
    A = LFBF16.(A64)
    b = LFBF16.(b64)
    # Krylov.jl's `atol`/`rtol` keywords are typed strictly to match the
    # element type T exactly (a plain Float64 literal errors with a
    # TypeError) -- must be passed as LFBF16, not converted internally.
    x, stats = cg(A, b; itmax=50, rtol=LFBF16(1e-2), atol=LFBF16(1e-2))
    resid = norm(Float64.(A * x) .- Float64.(b))
    relres = resid / norm(Float64.(b))
    # Golden values recorded here (bf16 is coarse, 8-bit significand):
    # asserts convergence-or-documented-stagnation.
    @test stats.niter <= 50
    @test relres < 0.5  # loose: bf16 cannot achieve tight residuals: 8-bit significand
    @test isfinite(relres)
end

@testset "Krylov.gmres on a nonsymmetric system at fp16" begin
    rng = Xoshiro(2)
    n = 10
    A64 = randn(rng, n, n) + 3n * I  # diagonally dominant, nonsymmetric, well conditioned
    b64 = randn(rng, n)
    A = LF16.(A64)
    b = LF16.(b64)
    x, stats = gmres(A, b; itmax=30, rtol=LF16(1e-2), atol=LF16(1e-2))
    resid = norm(Float64.(A * x) .- Float64.(b))
    relres = resid / norm(Float64.(b))
    @test stats.niter <= 30
    @test relres < 0.2
    @test isfinite(relres)
end

@testset "Golden iteration counts do not regress (fixed seed)" begin
    # Pinned to the specific seeds/problem sizes/tolerances below -- if
    # these numbers change on a rerun, either the kernel changed behavior
    # or Krylov.jl's algorithm did; investigate rather than just bump them.
    # Recorded once via an initial run, then hardcoded as the regression
    # baseline.
    rng = Xoshiro(1)
    n = 12
    M = randn(rng, n, n)
    A64 = M' * M + n * I
    b64 = randn(rng, n)
    A = LFBF16.(A64)
    b = LFBF16.(b64)
    _, stats = cg(A, b; itmax=50, rtol=LFBF16(1e-2), atol=LFBF16(1e-2))
    @test stats.niter == GOLDEN_CG_NITER
end
