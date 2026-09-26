# MATLAB-parity golden test against test/data/chop_golden.csv (generated
# from pychop's FaultChop, higham/chop-compatible port -- see
# test/data/README.md for full provenance and regeneration instructions).
#
# Covers modes 1-4 (deterministic, exact-value comparison against the CSV).
# Mode 7 (RoundNearestTiesAway) has no external oracle -- verified here
# against our own BigFloat oracle instead (same standard as
# kernel_bigfloat_oracle.jl, restricted to this mode). Modes 5-6
# (stochastic) are not represented in the CSV at all -- no cross-language
# RNG equivalence is possible between pychop's NumPy RNG and Julia's -- and
# are instead covered by the property-based tests already in
# kernel_invariants.jl (reproducibility from a seed) and
# kernel_matlab_transcribed.jl (proportion checks transcribed from
# test_roundit.m). This file's own testset names make clear which of the
# "seven rounding modes" each part actually covers.

using Test
using Random
using LowPrecisionChop
using LowPrecisionChop: chop

const GOLDEN_CSV = joinpath(@__DIR__, "data", "chop_golden.csv")
const ROUND_BY_MODE = Dict(
    1 => RoundNearest, 2 => RoundUp, 3 => RoundDown, 4 => RoundToZero
)

function _parse_golden_row(line::AbstractString)
    fields = split(line, ',')
    label = fields[1]
    t = parse(Int, fields[2])
    emax = parse(Int, fields[3])
    round_mode = parse(Int, fields[4])
    subnormal = parse(Int, fields[5]) == 1
    input = parse(Float64, fields[6])
    expected = parse(Float64, fields[7])
    return (; label, t, emax, round_mode, subnormal, input, expected)
end

@testset "chop_golden.csv parity: modes 1-4 (deterministic, exact match against pychop)" begin
    @test isfile(GOLDEN_CSV)
    lines = readlines(GOLDEN_CSV)
    @test length(lines) > 2000  # header + >=2000 data rows
    nchecked = 0
    modes_seen = Set{Int}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        row = _parse_golden_row(line)
        push!(modes_seen, row.round_mode)
        got = chop(
            row.input,
            Format(row.t, row.emax);
            round=ROUND_BY_MODE[row.round_mode],
            subnormal=row.subnormal,
        )
        nchecked += 1
        if isnan(row.expected)
            @test isnan(got)
        else
            @test isequal(got, row.expected)
        end
    end
    @test nchecked > 2000
    @test modes_seen == Set([1, 2, 3, 4])
end

@testset "Mode 7 (RoundNearestTiesAway): no external oracle exists, verified against BigFloat instead" begin
    # Same standard as kernel_bigfloat_oracle.jl -- independent
    # high-precision re-implementation, not a second copy of our own
    # kernel logic under a different name.
    function bigfloat_nearest_away(x::Float64, t::Integer, emax::Integer, subnormal::Bool)
        (isnan(x) || isinf(x) || iszero(x)) && return x
        setprecision(BigFloat, 256) do
            bx = BigFloat(x)
            emin = 1 - emax
            e = Int(floor(log2(abs(bx))))
            while abs(bx) < BigFloat(2)^e
                e -= 1
            end
            while abs(bx) >= BigFloat(2)^(e + 1)
                e += 1
            end
            scale = (e < emin) ? -(emin + 1 - t) : (t - 1 - e)
            c = ldexp(round(ldexp(bx, scale), RoundNearestTiesAway), -scale)
            xmax = ldexp(BigFloat(2) - ldexp(BigFloat(1), 1 - t), emax)
            xboundary = ldexp(BigFloat(2) - ldexp(BigFloat(1), -t), emax)
            if c >= xboundary
                c = BigFloat(Inf)
            elseif c <= -xboundary
                c = BigFloat(-Inf)
            elseif isfinite(c)
                min_rep =
                    subnormal ? ldexp(BigFloat(1), emin + 1 - t) : ldexp(BigFloat(1), emin)
                if abs(c) < min_rep
                    thresh = min_rep / 2
                    ok = subnormal ? abs(c) > thresh : abs(c) >= thresh
                    c = ok ? copysign(min_rep, c) : zero(c)
                end
            end
            return isfinite(c) ? Float64(c) : (c > 0 ? Inf : -Inf)
        end
    end

    rng = Xoshiro(20260901)
    formats = [(4, 7), (3, 15), (11, 15), (8, 127), (24, 127)]
    n = 0
    for (t, emax) in formats, subnormal in (true, false)
        emin = 1 - emax
        for _ in 1:100
            x =
                (1.0 + rand(rng)) *
                2.0^rand(rng, (emin - 8):(emax + 3)) *
                (rand(rng) < 0.5 ? 1 : -1)
            got = chop(x, Format(t, emax); round=RoundNearestTiesAway, subnormal=subnormal)
            want = bigfloat_nearest_away(x, t, emax, subnormal)
            n += 1
            @test isequal(got, want)
        end
    end
    @test n == 5 * 2 * 100
end

@testset "Modes 5-6 (stochastic): no golden CSV possible, see kernel_invariants.jl / kernel_matlab_transcribed.jl" begin
    # Documented pointer, not a duplicate test: reproducibility-from-seed is
    # asserted in kernel_invariants.jl ("stochastic rounding is
    # reproducible from a seed"); the proportion/distance properties
    # transcribed from test_roundit.m are in kernel_matlab_transcribed.jl.
    # This testset exists only so `grep`ing this file for all seven modes
    # finds an explicit, honest answer for 5 and 6, rather than silence.
    @test true
end
