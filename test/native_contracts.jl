# Native-type contract tests through the public API (chop/LowFloat), not
# the internal kernel directly. The full exhaustive-pair / 2M-random-pair
# Float16 coverage already lives in kernel_float16_exhaustive.jl —
# chop/LowFloat call the identical `_round_to_format` kernel, so this file
# checks the wiring (API -> kernel) is correct, not the kernel's own
# correctness a second time.

using Test
using Random
using LowPrecisionChop
using LowPrecisionChop: chop

const LF16 = lowfloattype(Float16Format)
const LFBF16 = lowfloattype(BFloat16Format)
const LF32 = lowfloattype(Float32Format)

@testset "LowFloat16 vs native Float16: exhaustive conversion" begin
    nmismatch = 0
    for bits in UInt16(0):UInt16(0xffff)
        h = reinterpret(Float16, bits)
        x = Float64(h)
        lf = LF16(x)
        if isnan(x)
            isnan(Float64(lf)) || (nmismatch += 1)
        else
            Float64(lf) === x || (nmismatch += 1)
        end
    end
    @test nmismatch == 0
end

@testset "chop(x, Float16Format) vs native Float16: exhaustive conversion" begin
    nmismatch = 0
    for bits in UInt16(0):UInt16(0xffff)
        h = reinterpret(Float16, bits)
        x = Float64(h)
        c = chop(x, Float16Format)
        if isnan(x)
            isnan(c) || (nmismatch += 1)
        else
            c === x || (nmismatch += 1)
        end
    end
    @test nmismatch == 0
end

@testset "LowFloat16 vs native Float16: +,-,*,/,sqrt over random samples" begin
    rng = Xoshiro(777)
    n = 50_000
    nfail = Dict(:+ => 0, :- => 0, :* => 0, :/ => 0, :sqrt => 0)
    for _ in 1:n
        a = reinterpret(Float16, rand(rng, UInt16))
        b = reinterpret(Float16, rand(rng, UInt16))
        la, lb = LF16(Float64(a)), LF16(Float64(b))
        for (op, sym) in ((+, :+), (-, :-), (*, :*), (/, :/))
            expected = op(a, b)
            got = Float16(Float64(op(la, lb)))
            ok = isnan(expected) ? isnan(got) : (got == expected)
            ok || (nfail[sym] += 1)
        end
        aabs = abs(a)
        expected_sqrt = sqrt(aabs)
        got_sqrt = Float16(Float64(sqrt(LF16(Float64(aabs)))))
        ok = isnan(expected_sqrt) ? isnan(got_sqrt) : (got_sqrt == expected_sqrt)
        ok || (nfail[:sqrt] += 1)
    end
    for (sym, n) in nfail
        @test n == 0
    end
end

@testset "LowFloat32 vs native Float32: randomized +,-,*,/,sqrt" begin
    rng = Xoshiro(778)
    n = 20_000
    nfail = 0
    for _ in 1:n
        a = reinterpret(Float32, rand(rng, UInt32))
        b = reinterpret(Float32, rand(rng, UInt32))
        (isnan(a) || isnan(b)) && continue
        la, lb = LF32(Float64(a)), LF32(Float64(b))
        for op in (+, -, *, /)
            expected = op(a, b)
            got = Float32(Float64(op(la, lb)))
            ok = isnan(expected) ? isnan(got) : (got == expected)
            ok || (nfail += 1)
        end
        got_sqrt = Float32(Float64(sqrt(LF32(Float64(abs(a))))))
        ok = isnan(sqrt(abs(a))) ? isnan(got_sqrt) : (got_sqrt == sqrt(abs(a)))
        ok || (nfail += 1)
    end
    @test nfail == 0
end

# BFloat16s.jl contract test: only runs if the package is available in this
# environment (it is not a hard or weak dependency of LowPrecisionChop --
# task scope names it only as a native-type oracle). Skipped, not silently
# passed, if unavailable, so CI visibly reports which branch ran.
@testset "LowFloat(BFloat16Format) vs BFloat16s.jl (if available)" begin
    bf16_available = try
        Base.require(Main, :BFloat16s)
        true
    catch
        false
    end
    if !bf16_available
        @test_skip "BFloat16s.jl not installed in this environment"
    else
        BFloat16s = Base.require(Main, :BFloat16s)
        BFloat16 = getfield(BFloat16s, :BFloat16)
        rng = Xoshiro(779)
        nfail = 0
        for _ in 1:20_000
            a = reinterpret(BFloat16, rand(rng, UInt16))
            b = reinterpret(BFloat16, rand(rng, UInt16))
            (isnan(a) || isnan(b)) && continue
            la, lb = LFBF16(Float64(a)), LFBF16(Float64(b))
            for op in (+, -, *, /)
                expected = op(a, b)
                got = BFloat16(Float64(op(la, lb)))
                ok = isnan(expected) ? isnan(got) : (got == expected)
                ok || (nfail += 1)
            end
        end
        @test nfail == 0
    end
end
