# Test vectors transcribed from higham/chop's own test suite
# (test_chop.m, test_roundit.m — BSD 2-clause, Higham & Pranesh 2019;
# see test/data/README.md for provenance). Expected values are computed
# from the closed-form descriptions in those files (not copy-pasted MATLAB
# output), so this is a symbolic port, not a numeric transcription.

using Test
using Random
using .LowPrecisionChop: _round_to_format, _round_int

@testset "roundit.m transcribed: round-to-integer per mode" begin
    rng = Random.default_rng()
    # test_roundit.m lines 13-23 (round=1, ties to even)
    A = [0 1.1 1.5; 1.9 2.4 0.5]
    E = [0 1 2; 2 2 0]
    @test [_round_int(Float64(a), RoundNearest, rng) for a in A] == Float64.(E)
    A = [0 -1.1 -1.5; -1.9 -2.4 -0.5]
    E = [0 -1 -2; -2 -2 0]
    @test [_round_int(Float64(a), RoundNearest, rng) for a in A] == Float64.(E)

    # lines 25-33 (round=2, toward +inf)
    A = [0 1.1 1.5; 1.9 2.4 0.5]
    E = [0 2 2; 2 3 1]
    @test [_round_int(Float64(a), RoundUp, rng) for a in A] == Float64.(E)
    A = [0 -1.1 -1.5; -1.9 -2.4 -0.5]
    E = [0 -1 -1; -1 -2 0]
    @test [_round_int(Float64(a), RoundUp, rng) for a in A] == Float64.(E)

    # lines 35-43 (round=3, toward -inf)
    A = [0 1.1 1.5; 1.9 2.4 0.5]
    E = [0 1 1; 1 2 0]
    @test [_round_int(Float64(a), RoundDown, rng) for a in A] == Float64.(E)
    A = [0 -1.1 -1.5; -1.9 -2.4 -0.5]
    E = [0 -2 -2; -2 -3 -1]
    @test [_round_int(Float64(a), RoundDown, rng) for a in A] == Float64.(E)

    # lines 45-48 (round=4, toward zero)
    A = [0 -1.1 -1.5; -2.9 -2 -0.5; 0.5 1.5 3]
    E = [0 -1 -1; -2 -2 0; 0 1 3]
    @test [_round_int(Float64(a), RoundToZero, rng) for a in A] == Float64.(E)
end

@testset "test_chop.m transcribed: pi in fp16 (lines 10-11, 195-199)" begin
    uh = 2.0^(-11)
    pi_h = 6432 * uh
    @test _round_to_format(Float64(pi), 11, 15, RoundNearest, true, true) == pi_h
    @test _round_to_format(-Float64(pi), 11, 15, RoundNearest, true, true) == -pi_h
end

@testset "test_chop.m transcribed: IEEE 754-2019 p.27 overflow rule (lines 273-308)" begin
    for (t, emax) in [(24, 127), (11, 15), (4, 7), (3, 15)]  # single, half, fp8-e4m3, fp8-e5m2
        p = t
        xmax = 2.0^emax * (2 - 2.0^(1 - p))

        # round=1 (nearest): x at the RN-to-infinity boundary rounds to inf,
        # just below (3/4 point) rounds to xmax.
        x_to_inf = 2.0^emax * (2 - 0.5 * 2.0^(1 - p))
        @test _round_to_format(x_to_inf, t, emax, RoundNearest, true, true) == Inf
        @test _round_to_format(-x_to_inf, t, emax, RoundNearest, true, true) == -Inf
        x_to_max = 2.0^emax * (2 - 0.75 * 2.0^(1 - p))
        @test _round_to_format(x_to_max, t, emax, RoundNearest, true, true) == xmax
        @test _round_to_format(-x_to_max, t, emax, RoundNearest, true, true) == -xmax

        # round=2 (+inf): same x_to_inf point overflows to +inf on the
        # positive side but only saturates (not -inf) on the negative side.
        @test _round_to_format(x_to_inf, t, emax, RoundUp, true, true) == Inf
        @test _round_to_format(-x_to_inf, t, emax, RoundUp, true, true) == -xmax

        # round=3 (-inf): mirror of round=2.
        @test _round_to_format(x_to_inf, t, emax, RoundDown, true, true) == xmax
        @test _round_to_format(-x_to_inf, t, emax, RoundDown, true, true) == -Inf

        # round=4 (toward zero): saturates both sides, never inf.
        @test _round_to_format(x_to_inf, t, emax, RoundToZero, true, true) == xmax
        @test _round_to_format(-x_to_inf, t, emax, RoundToZero, true, true) == -xmax
    end
end

@testset "test_chop.m transcribed: underflow/subnormal boundary (lines 318-472)" begin
    for (t, emax) in [(24, 127), (11, 15)]
        p = t
        emin = 1 - emax
        xmin = 2.0^emin
        xmax = 2.0^emax * (2 - 2.0^(1 - p))
        xmins = xmin * 2.0^(1 - p)

        # subnormal=1: xmin itself, and the boundary array from line 327-330.
        # `delta` = eps(1) at t bits (test_chop.m: `double(eps(single(1)))`
        # for single, `2*uh` for fp16 — both equal 2^(1-t)).
        delta1 = 2.0^(1 - t)
        @test _round_to_format(xmin, t, emax, RoundNearest, true, true) == xmin
        x = [xmins, xmin / 2, xmin, 0.0, xmax, 2xmax, 1 - (delta1 / 5), 1 + (delta1 / 4)]
        expected = [x[1], x[2], x[3], x[4], x[5], Inf, 1.0, 1.0]
        got = [_round_to_format(v, t, emax, RoundNearest, true, true) for v in x]
        @test got == expected

        # subnormal=0: line 332-337.
        expected0 = [0.0, xmin, x[3], x[4], x[5], Inf, 1.0, 1.0]
        got0 = [_round_to_format(v, t, emax, RoundNearest, false, true) for v in x]
        @test got0 == expected0

        # Largest subnormal number: xmin - delta (line 339-348).
        delta = xmin * 2.0^(1 - p)
        largest_subnormal = xmin - delta
        @test _round_to_format(largest_subnormal, t, emax, RoundNearest, true, true) ==
            largest_subnormal
        @test _round_to_format(largest_subnormal, t, emax, RoundNearest, false, true) ==
            xmin

        # Flush smallest subnormal to zero when unsupported (line 350-352).
        @test _round_to_format(xmins, t, emax, RoundNearest, false, true) == 0.0

        # A genuine subnormal number is preserved exactly (line 354-357).
        x8 = xmins * 8
        @test _round_to_format(x8, t, emax, RoundNearest, true, true) == x8

        # Numbers smaller than the smallest representable flush correctly
        # (lines 359-386): xmin/2 -> xmin (subnormal off, nearest, exact
        # tie rounds up per >=); xmin/4 -> 0; xmins/2, xmins/4 -> 0 (subnormal on).
        @test _round_to_format(xmin / 2, t, emax, RoundNearest, false, true) == xmin
        @test _round_to_format(-xmin / 2, t, emax, RoundNearest, false, true) == -xmin
        @test _round_to_format(xmin / 4, t, emax, RoundNearest, false, true) == 0.0
        @test _round_to_format(-xmin / 4, t, emax, RoundNearest, false, true) == 0.0
        @test _round_to_format(xmins / 2, t, emax, RoundNearest, true, true) == 0.0
        @test _round_to_format(-xmins / 2, t, emax, RoundNearest, true, true) == 0.0
        @test _round_to_format(xmins / 4, t, emax, RoundNearest, true, true) == 0.0
        @test _round_to_format(-xmins / 4, t, emax, RoundNearest, true, true) == 0.0

        # explim = 0: no range limiting at all (lines 388-399).
        for v in (xmin / 2, -xmin / 2, xmax * 2, -xmax * 2, xmins / 2, -xmins / 2)
            @test _round_to_format(v, t, emax, RoundNearest, true, false) == v
        end

        # round=2 (+inf) underflow boundary (lines 401-423).
        @test _round_to_format(xmin / 2, t, emax, RoundUp, false, true) == xmin
        @test _round_to_format(-xmin / 2, t, emax, RoundUp, false, true) == 0.0
        @test _round_to_format(xmins / 2, t, emax, RoundUp, true, true) == xmins
        @test _round_to_format(-xmins / 2, t, emax, RoundUp, true, true) == 0.0
        @test _round_to_format(xmins / 4, t, emax, RoundUp, true, true) == xmins
        @test _round_to_format(-xmins / 4, t, emax, RoundUp, true, true) == 0.0

        # round=3 (-inf) underflow boundary (lines 425-447).
        @test _round_to_format(xmin / 2, t, emax, RoundDown, false, true) == 0.0
        @test _round_to_format(-xmin / 2, t, emax, RoundDown, false, true) == -xmin
        @test _round_to_format(xmins / 2, t, emax, RoundDown, true, true) == 0.0
        @test _round_to_format(-xmins / 2, t, emax, RoundDown, true, true) == -xmins
        @test _round_to_format(xmins / 4, t, emax, RoundDown, true, true) == 0.0
        @test _round_to_format(-xmins / 4, t, emax, RoundDown, true, true) == -xmins

        # round=4 (toward zero): always flush (lines 449-471).
        for v in (xmin / 2, -xmin / 2)
            @test _round_to_format(v, t, emax, RoundToZero, false, true) == 0.0
        end
        for v in (xmins / 2, -xmins / 2, xmins / 4, -xmins / 4)
            @test _round_to_format(v, t, emax, RoundToZero, true, true) == 0.0
        end
    end
end
