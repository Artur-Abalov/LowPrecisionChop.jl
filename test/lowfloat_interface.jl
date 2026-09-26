using Test
using Random
using LinearAlgebra
using LowPrecisionChop

const LF16 = lowfloattype(Float16Format)
const LF8 = lowfloattype(Format(3, 3))  # tiny format for exhaustive traversal

@testset "LowFloat: zero/one/promotion" begin
    @test zero(LF16) == LF16(0.0)
    @test one(LF16) == LF16(1.0)
    @test promote_type(LF16, Float64) == LF16
    @test promote_type(LF16, Int) == LF16
    @test (LF16(1.0) + 2) isa LF16
    @test Float64(LF16(1.0) + 2) == 3.0
end

@testset "LowFloat: eps/floatmin/floatmax/precision match format formulas" begin
    @test Float64(eps(LF16)) == 2.0^(-10)
    @test Float64(floatmin(LF16)) == 2.0^(-14)
    @test Float64(floatmax(LF16)) == (2 - 2.0^(-10)) * 2.0^15
    @test precision(LF16) == 11
    @test precision(LF16(1.0)) == 11
end

@testset "LowFloat: typemin/typemax" begin
    @test Float64(typemin(LF16)) == -Inf
    @test Float64(typemax(LF16)) == Inf
end

@testset "LowFloat: nextfloat/prevfloat exhaustive traversal (t=3, emax=3)" begin
    xmax = Float64(floatmax(LF8))
    x = LF8(-2 * xmax)
    vals = Float64[]
    while true
        push!(vals, Float64(x))
        Float64(x) == Inf && break
        x = nextfloat(x)
        length(vals) > 100 && break  # safety valve
    end
    @test issorted(vals)
    @test allunique(filter(isfinite, vals))
    @test vals[end] == Inf
    # Round-trip: prevfloat(nextfloat(x)) == x for every finite value seen.
    for v in vals
        isfinite(v) || continue
        lv = LF8(v)
        @test prevfloat(nextfloat(lv)) == lv
    end
end

@testset "LowFloat: hash/decompose consistent with Float64 value, usable as Dict key" begin
    a = LF16(1.5)
    b = LF16(1.5)
    d = Dict{LF16,Int}()
    d[a] = 1
    @test d[b] == 1
    @test hash(a) == hash(b)
    @test Base.decompose(a) == Base.decompose(1.5)
end

@testset "LowFloat: bitstring matches native Float16 exactly" begin
    for bits in rand(Xoshiro(1), UInt16, 5000)
        h = reinterpret(Float16, bits)
        isnan(h) && continue  # NaN payload encoding is not standardized; skip
        lf = LF16(Float64(h))
        @test bitstring(lf) == bitstring(h)
    end
end

@testset "LowFloat: rand is reproducible from a seed and stays on-grid" begin
    a = [rand(Xoshiro(42), LF16) for _ in 1:100]
    b = [rand(Xoshiro(42), LF16) for _ in 1:100]
    @test a == b
    @test all(x -> LF16(Float64(x)) == x, a)  # idempotent under re-rounding
end

@testset "LowFloat: never promotes to Float64 in mixed reductions (guardrail)" begin
    # Regression guard for the spec.md §9 correctness trap: promote_type
    # must resolve to LowFloat, not Float64, so `sum`/`norm`-style reductions
    # over a LowFloat vector stay in low precision.
    v = LF16.([1.0, 2.0, 3.0])
    s = sum(v)
    @test s isa LF16
end

@testset "LowFloat: not a BlasFloat / FloatOrComplex bypass risk" begin
    @test !(LF16 <: LinearAlgebra.BLAS.BlasFloat)
end
