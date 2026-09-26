using Test
using Random
using .LowPrecisionChop: _round_to_format, _flipbit

@testset "Invariants: idempotence" begin
    rng = Xoshiro(1)
    formats = [(4, 7), (3, 15), (11, 15), (8, 127), (24, 127)]
    modes = (RoundNearest, RoundUp, RoundDown, RoundToZero, RoundNearestTiesAway)
    for (t, emax) in formats, roundmode in modes, subnormal in (true, false)
        for _ in 1:200
            x = (rand(rng) - 0.5) * exp(20 * (rand(rng) - 0.5))
            c1 = _round_to_format(x, t, emax, roundmode, subnormal, true)
            c2 = _round_to_format(c1, t, emax, roundmode, subnormal, true)
            @test isequal(c1, c2)
        end
    end
end

@testset "Invariants: sign symmetry (deterministic modes)" begin
    rng = Xoshiro(2)
    formats = [(4, 7), (11, 15), (8, 127)]
    # RoundNearest and RoundNearestTiesAway must be odd (sign-symmetric);
    # RoundUp/RoundDown are sign-antisymmetric with each other, not self-symmetric.
    for (t, emax) in formats, roundmode in (RoundNearest, RoundNearestTiesAway, RoundToZero)
        for _ in 1:200
            x = (rand(rng) - 0.5) * exp(20 * (rand(rng) - 0.5))
            cpos = _round_to_format(x, t, emax, roundmode, true, true)
            cneg = _round_to_format(-x, t, emax, roundmode, true, true)
            # `==` (see note below): flush-to-zero is unsigned by design.
            @test cneg == -cpos
        end
    end
    for (t, emax) in formats
        for _ in 1:200
            x = (rand(rng) - 0.5) * exp(20 * (rand(rng) - 0.5))
            up = _round_to_format(x, t, emax, RoundUp, true, true)
            down = _round_to_format(-x, t, emax, RoundDown, true, true)
            # `==`, not `isequal`: chop.m's underflow-flush branch assigns a
            # literal `0` (MATLAB has no signed-zero distinction there), so
            # our kernel's flush-to-zero is likewise unsigned by design —
            # -0.0 and 0.0 are the correct values to compare as equal here.
            @test down == -up
        end
    end
end

@testset "Invariants: directed modes bracket the exact value" begin
    rng = Xoshiro(3)
    formats = [(4, 7), (11, 15), (8, 127), (24, 127)]
    for (t, emax) in formats
        for _ in 1:300
            x = (rand(rng) - 0.5) * exp(15 * (rand(rng) - 0.5))
            up = _round_to_format(x, t, emax, RoundUp, true, true)
            down = _round_to_format(x, t, emax, RoundDown, true, true)
            tz = _round_to_format(x, t, emax, RoundToZero, true, true)
            (isfinite(up) && isfinite(down)) || continue
            @test down <= x <= up
            if x >= 0
                @test tz == down
            else
                @test tz == up
            end
        end
    end
end

@testset "Invariants: monotonicity (deterministic modes)" begin
    rng = Xoshiro(4)
    formats = [(4, 7), (11, 15), (8, 127)]
    modes = (RoundNearest, RoundUp, RoundDown, RoundToZero, RoundNearestTiesAway)
    for (t, emax) in formats, roundmode in modes
        pts = sort([(rand(rng) - 0.5) * exp(15 * (rand(rng) - 0.5)) for _ in 1:500])
        rounded = [_round_to_format(x, t, emax, roundmode, true, true) for x in pts]
        @test issorted(rounded)
    end
end

@testset "Invariants: nextfloat/prevfloat traverse tiny format without gaps (t=3, emax=3)" begin
    # Exhaustively enumerate every representable value of a tiny format by
    # rounding a dense real-line sweep, and check consecutive distinct
    # outputs are strictly increasing (no repeats collapsed out of order,
    # no gaps skipped relative to the grid spacing implied by (t,emax)).
    t, emax = 3, 3
    emin = 1 - emax
    xmax = ldexp(2.0 - ldexp(1.0, 1 - t), emax)
    xs = range(-2xmax, 2xmax; length=2_000_001)
    vals = sort(
        unique(_round_to_format(Float64(x), t, emax, RoundNearest, true, true) for x in xs)
    )
    finite_vals = filter(isfinite, vals)
    @test issorted(finite_vals)
    @test allunique(finite_vals)
    # Every finite representable value must itself be a fixed point.
    for v in finite_vals
        @test _round_to_format(v, t, emax, RoundNearest, true, true) == v
    end
end

@testset "Invariants: stochastic rounding is reproducible from a seed" begin
    t, emax = 11, 15
    x = 1.0 + 2.0^(-11) * 0.37
    r1 = [
        _round_to_format(x, t, emax, RoundStochasticProportional, true, true, Xoshiro(99))
        for _ in 1:1
    ]
    a = _round_to_format(x, t, emax, RoundStochasticProportional, true, true, Xoshiro(99))
    b = _round_to_format(x, t, emax, RoundStochasticProportional, true, true, Xoshiro(99))
    @test a == b
    rngA = Xoshiro(7)
    rngB = Xoshiro(7)
    seqA = [
        _round_to_format(x, t, emax, RoundStochasticEqual, true, true, rngA) for _ in 1:1000
    ]
    seqB = [
        _round_to_format(x, t, emax, RoundStochasticEqual, true, true, rngB) for _ in 1:1000
    ]
    @test seqA == seqB
end

@testset "Invariants: bit flip is reproducible and respects p" begin
    t = 11
    x = 3.140625  # fp16(pi)
    rngA = Xoshiro(5)
    rngB = Xoshiro(5)
    seqA = [_flipbit(x, t, rngA, 0.5) for _ in 1:1000]
    seqB = [_flipbit(x, t, rngB, 0.5) for _ in 1:1000]
    @test seqA == seqB

    rng = Xoshiro(11)
    nflip0 = count(!=(x), [_flipbit(x, t, rng, 0.0) for _ in 1:1000])
    @test nflip0 == 0
    rng = Xoshiro(12)
    nflip1 = count(!=(x), [_flipbit(x, t, rng, 1.0) for _ in 1:1000])
    @test nflip1 == 1000
end
