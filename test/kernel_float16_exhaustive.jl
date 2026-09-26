# Kernel must agree with native Float16 bit-for-bit.
# fp16 format is (t=11, emax=15); round-to-nearest, subnormals on, explim on
# is the closest match to native Float16 semantics (IEEE binary16).

using Test
using Random
using .LowPrecisionChop: _round_to_format

const FP16_T = 11
const FP16_EMAX = 15

_chop16(x::Float64) = _round_to_format(x, FP16_T, FP16_EMAX, RoundNearest, true, true)

@testset "Float16 exhaustive: conversion, all 65536 bit patterns" begin
    nmismatch = 0
    for bits in UInt16(0):UInt16(0xffff)
        h = reinterpret(Float16, bits)
        x = Float64(h)
        c = _chop16(x)
        if isnan(x)
            @test isnan(c)
        else
            if c !== x  # exact bit-for-bit including signed zero
                nmismatch += 1
            end
        end
    end
    @test nmismatch == 0
end

@testset "Float16 exhaustive: +,-,*,/ over representative + random pairs" begin
    reps = Float16[
        Float16(0.0),
        Float16(-0.0),
        Float16(1.0),
        Float16(-1.0),
        Float16(Inf),
        Float16(-Inf),
        Float16(NaN),
        nextfloat(Float16(0.0)),
        prevfloat(Float16(0.0)),  # smallest subnormals
        floatmin(Float16),
        -floatmin(Float16),
        floatmax(Float16),
        -floatmax(Float16),
        Float16(0.5),
        Float16(1.5),
        Float16(2.5),
        Float16(3.14159),
    ]
    all_bits = UInt16(0):UInt16(0xffff)
    rng = Xoshiro(20260901)
    randbits = rand(rng, all_bits, 2_000_000)

    function check_op(op, a::Float16, b::Float16)
        xa, xb = Float64(a), Float64(b)
        expected = op(a, b)             # native Float16 arithmetic
        got64 = _chop16(op(xa, xb))     # Float64 op, then round to fp16 grid
        got = Float16(got64)            # lossless: got64 already lies on the fp16 grid
        if isnan(expected)
            return isnan(got64)
        else
            return got == expected
        end
    end

    nfail_plus = nfail_minus = nfail_times = nfail_div = 0
    # Exhaustive: every one of the 65536 values against every representative.
    for bits in all_bits
        a = reinterpret(Float16, bits)
        for b in reps
            check_op(+, a, b) || (nfail_plus += 1)
            check_op(-, a, b) || (nfail_minus += 1)
            check_op(*, a, b) || (nfail_times += 1)
            check_op(/, a, b) || (nfail_div += 1)
        end
    end
    # Broad random sampling across the full pair space (65536^2 is too large
    # to brute force; this is an honest "large sample", not exhaustive pairs).
    for i in 1:length(randbits)
        a = reinterpret(Float16, randbits[i])
        b = reinterpret(Float16, randbits[mod1(i + 7, length(randbits))])
        check_op(+, a, b) || (nfail_plus += 1)
        check_op(-, a, b) || (nfail_minus += 1)
        check_op(*, a, b) || (nfail_times += 1)
        check_op(/, a, b) || (nfail_div += 1)
    end

    @test nfail_plus == 0
    @test nfail_minus == 0
    @test nfail_times == 0
    @test nfail_div == 0
end
