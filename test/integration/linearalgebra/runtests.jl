# LinearAlgebra integration suite (task's <integration_targets>, item 1).
# Run standalone: julia --project=test/integration/linearalgebra
#                        test/integration/linearalgebra/runtests.jl

using Test
using Random
using LinearAlgebra
using LowPrecisionChop

const LF16 = lowfloattype(Float16Format)
const LF32 = lowfloattype(Float32Format)
const LFBF16 = lowfloattype(BFloat16Format)

@testset "LowFloat is not a BlasFloat (routes to generic_lufact!/qrfactUnblocked!)" begin
    @test !(LF16 <: LinearAlgebra.BLAS.BlasFloat)
end

@testset "Generic lu! runs on Matrix{LowFloat} and factors correctly" begin
    rng = Xoshiro(1)
    n = 8
    A64 = randn(rng, n, n) + n * I  # diagonally dominant-ish, well conditioned
    A = LF16.(A64)
    F = lu(A)
    # Reconstruct and check componentwise backward error against theoretical
    # O(n*u) bound (Higham & Pranesh eq. 3.5-style bound for computing wholly
    # in low precision -- our element-wise LowFloat arithmetic is exactly
    # this "Simulation 3.1" regime, not the tighter kernel-level bound).
    u = Float64(eps(LF16))
    resid = Float64.(F.P * A) - Float64.(F.L * F.U)
    gamma_n = n * u / (1 - n * u)
    bound = gamma_n * maximum(abs, Float64.(F.L)) * maximum(abs, Float64.(F.U)) * n
    @test maximum(abs, resid) <= 50 * bound + 10 * u  # generous slack: this is an order-of-magnitude check, not a tight one
end

@testset "Generic qr! runs on Matrix{LowFloat} and factors correctly" begin
    rng = Xoshiro(2)
    n = 6
    A64 = randn(rng, n, n)
    A = LF16.(A64)
    F = qr(A)
    resid = Float64.(Matrix(F.Q) * F.R) - Float64.(A)
    u = Float64(eps(LF16))
    @test maximum(abs, resid) <= 200 * n * u * maximum(abs, Float64.(A))
end

@testset "fp16 LowFloat LU vs native Float16 LU: bitwise identical (Higham-Pranesh Table 6.2)" begin
    # The paper reports lutx_chop (chop-wrapped LU) and lutx (native fp16
    # class) are "bitwise identical" for fp16 -- because t=11 safely
    # satisfies the double-rounding condition (2*11+2=24<=53). If our
    # LowFloat LU disagrees with native Float16 LU here, that is evidence of
    # a bug in us, not an expected divergence -- see docs/src/spec.md §8.
    rng = Xoshiro(3)
    nmismatch = 0
    ntested = 0
    for trial in 1:20
        n = 5
        A64 = randn(rng, n, n)
        Ah = Float16.(A64)
        Alow = LF16.(Float64.(Ah))
        @test Float64.(Alow) == Float64.(Ah)  # construction itself must agree

        Fh = lu(Ah; check=false)
        Flow = lu(Alow; check=false)
        issuccess(Fh) || continue
        issuccess(Flow) || continue
        ntested += 1
        matchL = Float64.(Flow.L) == Float64.(Fh.L)
        matchU = Float64.(Flow.U) == Float64.(Fh.U)
        matchp = Flow.p == Fh.p
        (matchL && matchU && matchp) || (nmismatch += 1)
    end
    @test ntested > 0
    @test nmismatch == 0
end

@testset "Hand-written GMRES on Matrix{LowFloat}: converges on a well-conditioned SPD-adjacent system" begin
    # Minimal unrestarted GMRES (Arnoldi + least-squares), no BLAS calls --
    # exercises LowFloat's arithmetic/comparison interface end to end, not
    # Krylov.jl's implementation (that's the separate krylov/ suite).
    function gmres_lowfloat(
        A::AbstractMatrix{T}, b::AbstractVector{T}; maxiter=length(b), rtol=1e-3
    ) where {T}
        n = length(b)
        x0 = zeros(T, n)
        r0 = b - A * x0
        beta = sqrt(Float64(sum(abs2, r0)))
        beta == 0 && return x0
        V = Vector{Vector{T}}(undef, maxiter + 1)
        V[1] = r0 / T(beta)
        H = zeros(Float64, maxiter + 1, maxiter)
        for j in 1:maxiter
            w = A * V[j]
            for i in 1:j
                H[i, j] = Float64(sum(V[i][k] * w[k] for k in 1:n))
                w = w - T(H[i, j]) * V[i]
            end
            H[j + 1, j] = sqrt(Float64(sum(abs2, w)))
            if H[j + 1, j] < 1e-12
                m = j
                @goto solve
            end
            V[j + 1] = w / T(H[j + 1, j])
        end
        m = maxiter
        @label solve
        e1 = zeros(m + 1)
        e1[1] = beta
        y = H[1:(m + 1), 1:m] \ e1
        x = x0
        for i in 1:m
            x = x + T(y[i]) * V[i]
        end
        return x
    end

    rng = Xoshiro(4)
    n = 10
    M = randn(rng, n, n)
    A64 = M' * M + n * I  # SPD, well conditioned
    b64 = randn(rng, n)
    A = LFBF16.(A64)
    b = LFBF16.(b64)
    x = gmres_lowfloat(A, b; maxiter=n)
    resid64 = Float64.(A * x) .- Float64.(b)
    relres = norm(resid64) / norm(Float64.(b))
    @test relres < 0.1  # bf16 is coarse; loose but non-trivial convergence bound
end
