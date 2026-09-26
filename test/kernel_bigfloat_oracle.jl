# BigFloat oracle: an independent re-implementation of the same rounding
# algorithm at very high precision (256 bits), used to check that the
# Float64 kernel never loses precision relative to "effectively exact" real
# arithmetic. Deterministic modes only (stochastic modes have no single
# "correct" answer to check against).

using Test
using Random
using .LowPrecisionChop: _round_to_format

function _oracle_round_to_format(
    x::Float64,
    t::Integer,
    emax::Integer,
    roundmode::RoundingMode,
    subnormal::Bool,
    explim::Bool,
)
    (isnan(x) || isinf(x) || iszero(x)) && return x
    setprecision(BigFloat, 256) do
        bx = BigFloat(x)
        emin = 1 - emax
        e = Int(floor(log2(abs(bx))))
        # exponent() edge case: log2 can be off by one at exact powers of two
        # due to floating point; correct by bisection against 2^e.
        while abs(bx) < BigFloat(2)^e
            e -= 1
        end
        while abs(bx) >= BigFloat(2)^(e + 1)
            e += 1
        end
        scale = (explim && e < emin) ? -(emin + 1 - t) : (t - 1 - e)
        scaled = ldexp(bx, scale)
        rounded = round(scaled, roundmode)
        c = ldexp(rounded, -scale)
        if explim
            xmax = ldexp(BigFloat(2) - ldexp(BigFloat(1), 1 - t), emax)
            xboundary = ldexp(BigFloat(2) - ldexp(BigFloat(1), -t), emax)
            c = _oracle_overflow(c, bx, xmax, xboundary, roundmode)
            if isfinite(c)
                min_rep =
                    subnormal ? ldexp(BigFloat(1), emin + 1 - t) : ldexp(BigFloat(1), emin)
                c = _oracle_underflow(c, min_rep, subnormal, roundmode)
            end
        end
        return isfinite(c) ? Float64(c) : (c > 0 ? Inf : -Inf)
    end
end

function _oracle_overflow(
    c, x, xmax, xboundary, ::Union{RoundingMode{:Nearest},RoundingMode{:NearestTiesAway}}
)
    x >= xboundary && return typeof(c)(Inf)
    x <= -xboundary && return typeof(c)(-Inf)
    return c
end
function _oracle_overflow(c, x, xmax, xboundary, ::RoundingMode{:Up})
    x > xmax && return typeof(c)(Inf)
    (x < -xmax && !isinf(x)) && return -xmax
    return c
end
function _oracle_overflow(c, x, xmax, xboundary, ::RoundingMode{:Down})
    (x > xmax && !isinf(x)) && return xmax
    x < -xmax && return typeof(c)(-Inf)
    return c
end
function _oracle_overflow(c, x, xmax, xboundary, ::RoundingMode{:ToZero})
    (x > xmax && !isinf(x)) && return xmax
    (x < -xmax && !isinf(x)) && return -xmax
    return c
end

function _oracle_underflow(
    c, min_rep, subnormal, ::Union{RoundingMode{:Nearest},RoundingMode{:NearestTiesAway}}
)
    abs(c) >= min_rep && return c
    thresh = min_rep / 2
    ok = subnormal ? abs(c) > thresh : abs(c) >= thresh
    return ok ? copysign(min_rep, c) : zero(c)
end
function _oracle_underflow(c, min_rep, subnormal, ::RoundingMode{:Up})
    abs(c) >= min_rep && return c
    (c > 0 && c < min_rep) && return min_rep
    return zero(c)
end
function _oracle_underflow(c, min_rep, subnormal, ::RoundingMode{:Down})
    abs(c) >= min_rep && return c
    (c < 0 && c > -min_rep) && return -min_rep
    return zero(c)
end
function _oracle_underflow(c, min_rep, subnormal, ::RoundingMode{:ToZero})
    abs(c) >= min_rep && return c
    return zero(c)
end

@testset "BigFloat oracle: deterministic modes agree exactly" begin
    rng = Xoshiro(424242)
    modes = (RoundNearest, RoundUp, RoundDown, RoundToZero, RoundNearestTiesAway)
    formats = [
        (4, 7),    # fp8-e4m3
        (3, 15),   # fp8-e5m2
        (11, 15),  # fp16
        (8, 127),  # bfloat16
        (24, 127), # fp32
    ]
    nchecked = 0
    for (t, emax) in formats,
        roundmode in modes, subnormal in (true, false),
        explim in (true, false)
        # Magnitudes spanning far-overflow, near-overflow, normal, near-underflow,
        # deep-subnormal, and exact zero-crossing regions.
        emin = 1 - emax
        magnitudes = Float64[
            0.0,
            ldexp(1.0, emin - 20),
            ldexp(1.0, emin - 1),
            ldexp(1.0, emin),
            ldexp(1.0, emin + 3),
            0.1,
            0.5,
            1.0,
            1.5,
            3.14159265358979,
            100.0,
            ldexp(1.0, emax - 1),
            ldexp(1.0, emax),
            ldexp(1.0, emax + 2),
        ]
        for _ in 1:40
            push!(magnitudes, ldexp(1.0 + rand(rng), rand(rng, (emin - 5):(emax + 3))))
        end
        for m in magnitudes, sgn in (1.0, -1.0)
            x = sgn * m
            got = _round_to_format(x, t, emax, roundmode, subnormal, explim)
            want = _oracle_round_to_format(x, t, emax, roundmode, subnormal, explim)
            nchecked += 1
            @test isequal(got, want)
        end
    end
    @test nchecked > 5000
end
