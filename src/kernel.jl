# Pure rounding kernel shared by `chop`/`chop!` and `LowFloat` arithmetic.
# Translated from higham/chop's `chop.m` and `roundit.m` (MATLAB, BSD
# 2-clause, Higham & Pranesh 2019) and cross-checked against Table 5.1 and
# §3.1/§5 of the accompanying paper. See docs/src/spec.md for the derivation
# and citations behind every branch below.

function _round_int(y::T, ::RoundingMode{:Nearest}, ::Random.AbstractRNG) where {T}
    return round(y, RoundNearest)
end
function _round_int(y::T, ::RoundingMode{:NearestTiesAway}, ::Random.AbstractRNG) where {T}
    return round(y, RoundNearestTiesAway)
end
_round_int(y::T, ::RoundingMode{:Up}, ::Random.AbstractRNG) where {T} = round(y, RoundUp)
function _round_int(y::T, ::RoundingMode{:Down}, ::Random.AbstractRNG) where {T}
    return round(y, RoundDown)
end
function _round_int(y::T, ::RoundingMode{:ToZero}, ::Random.AbstractRNG) where {T}
    return round(y, RoundToZero)
end

function _round_int(
    y::T, ::typeof(RoundStochasticProportional), rng::Random.AbstractRNG
) where {T}
    fl = floor(y)
    frac = y - fl
    iszero(frac) && return y
    u = rand(rng, T)
    return u <= frac ? fl + one(T) : fl
end

function _round_int(y::T, ::typeof(RoundStochasticEqual), rng::Random.AbstractRNG) where {T}
    fl = floor(y)
    frac = y - fl
    iszero(frac) && return y
    u = rand(rng, T)
    return u <= T(0.5) ? fl + one(T) : fl
end

# --- overflow (chop.m: `if fpopts.explim ... switch(fpopts.round)`, first block) ---
# Grouping is {1,6,7} -> nearest-to-boundary, {2} -> up, {3} -> down,
# {4,5} -> saturate-both. Mode 7 (RoundNearestTiesAway) has no reference
# behavior; grouped with the nearest family as a documented design choice
# (spec.md §3).

function _overflow(c::T, x::T, xmax::T, xboundary::T, ::RoundingMode{:Nearest}) where {T}
    return _overflow_nearest(c, x, xboundary)
end
function _overflow(
    c::T, x::T, xmax::T, xboundary::T, ::RoundingMode{:NearestTiesAway}
) where {T}
    return _overflow_nearest(c, x, xboundary)
end
function _overflow(
    c::T, x::T, xmax::T, xboundary::T, ::typeof(RoundStochasticEqual)
) where {T}
    return _overflow_nearest(c, x, xboundary)
end
function _overflow(c::T, x::T, xmax::T, xboundary::T, ::RoundingMode{:Up}) where {T}
    return _overflow_up(c, x, xmax)
end
function _overflow(c::T, x::T, xmax::T, xboundary::T, ::RoundingMode{:Down}) where {T}
    return _overflow_down(c, x, xmax)
end
function _overflow(c::T, x::T, xmax::T, xboundary::T, ::RoundingMode{:ToZero}) where {T}
    return _overflow_saturate(c, x, xmax)
end
function _overflow(
    c::T, x::T, xmax::T, xboundary::T, ::typeof(RoundStochasticProportional)
) where {T}
    return _overflow_saturate(c, x, xmax)
end

function _overflow_nearest(c::T, x::T, xboundary::T) where {T}
    x >= xboundary && return T(Inf)
    x <= -xboundary && return T(-Inf)
    return c
end
function _overflow_up(c::T, x::T, xmax::T) where {T}
    x > xmax && return T(Inf)
    (x < -xmax && x != T(-Inf)) && return -xmax
    return c
end
function _overflow_down(c::T, x::T, xmax::T) where {T}
    (x > xmax && x != T(Inf)) && return xmax
    x < -xmax && return T(-Inf)
    return c
end
function _overflow_saturate(c::T, x::T, xmax::T) where {T}
    (x > xmax && x != T(Inf)) && return xmax
    (x < -xmax && x != T(-Inf)) && return -xmax
    return c
end

# --- underflow / subnormal boundary (chop.m: second `if fpopts.explim` block) ---
# Grouping is {1,7} -> nearest tie rule, {2} -> up, {3} -> down, {4,5,6} ->
# flush to zero. Mode 7 grouped with mode 1 (design choice, spec.md §3).

function _underflow_nearest(c::T, min_rep::T, subnormal::Bool) where {T}
    thresh = min_rep / 2
    ok = subnormal ? abs(c) > thresh : abs(c) >= thresh
    return ok ? copysign(min_rep, c) : zero(T)
end
function _underflow_small(
    c::T, min_rep::T, subnormal::Bool, ::RoundingMode{:Nearest}
) where {T}
    return _underflow_nearest(c, min_rep, subnormal)
end
function _underflow_small(
    c::T, min_rep::T, subnormal::Bool, ::RoundingMode{:NearestTiesAway}
) where {T}
    return _underflow_nearest(c, min_rep, subnormal)
end
function _underflow_small(c::T, min_rep::T, ::Bool, ::RoundingMode{:Up}) where {T}
    (c > 0 && c < min_rep) && return min_rep
    return zero(T)
end
function _underflow_small(c::T, min_rep::T, ::Bool, ::RoundingMode{:Down}) where {T}
    (c < 0 && c > -min_rep) && return -min_rep
    return zero(T)
end
_underflow_small(::T, ::T, ::Bool, ::RoundingMode) where {T} = zero(T)

function _underflow(
    c::T, t::Integer, emax::Integer, roundmode::RoundingMode, subnormal::Bool
) where {T}
    emin = 1 - emax
    min_rep = subnormal ? ldexp(one(T), emin + 1 - t) : ldexp(one(T), emin)
    abs(c) >= min_rep && return c
    return _underflow_small(c, min_rep, subnormal, roundmode)
end

function _apply_range_limits(
    c::T, x::T, t::Integer, emax::Integer, roundmode::RoundingMode, subnormal::Bool
) where {T}
    xmax = ldexp(T(2) - ldexp(one(T), 1 - t), emax)
    xboundary = ldexp(T(2) - ldexp(one(T), -t), emax)
    c = _overflow(c, x, xmax, xboundary, roundmode)
    isfinite(c) || return c
    return _underflow(c, t, emax, roundmode, subnormal)
end

"""
    _round_to_format(x, t, emax, roundmode, subnormal, explim, rng=Random.default_rng())

Round `x::T` (`T <: Union{Float32,Float64}`) to a low-precision format with
`t` significand bits (including the hidden bit) and maximum unbiased
exponent `emax`, returning a value of the same type `T`. Pure function, no
mutation, no global state — the single kernel shared by `chop`/`chop!` and
`LowFloat` arithmetic. See docs/src/spec.md §§2–7 for the full derivation.
"""
function _round_to_format(
    x::T,
    t::Integer,
    emax::Integer,
    roundmode::RoundingMode,
    subnormal::Bool,
    explim::Bool,
    rng::Random.AbstractRNG=Random.default_rng(),
) where {T<:Union{Float32,Float64}}
    (isnan(x) || isinf(x) || iszero(x)) && return x

    emin = 1 - emax
    e = exponent(x)

    # Scale exponent for the round-to-integer step. Normal range (e >= emin,
    # or explim off): use the value's own local exponent, full t bits — the
    # standard floating-point grid. Below emin with explim on: use the FIXED
    # subnormal quantum 2^emins = 2^(emin+1-t), independent of x's own local
    # exponent, for every magnitude below emin (not only within
    # [emins, emin)). This is mathematically forced, not a simplification we
    # chose for convenience: chop.m's own k_sub formula (t1 = t-(emin-e),
    # scale = t1-1-e) collapses algebraically to the constant `-emins` for
    # every e in [emins, emin) — it never actually depended on e. Extending
    # that same fixed scale to e < emins too (instead of chop.m's literal
    # branch, which falls through to the *local*-exponent formula there) is
    # what makes the underflow/flush-to-zero decision exact: rounding at a
    # finer, magnitude-dependent grid first and only afterward comparing the
    # result to the coarser fixed cutoff is itself a double-rounding step,
    # and it silently flips the round-to-nearest answer at rare boundary
    # cases (found via exhaustive-vs-native-Float16 testing: two subnormal
    # fp16 operands whose exact product sits a few parts in 10^4 above
    # xmins/2 were incorrectly flushed to 0 by the literal local-exponent
    # translation, disagreeing with both native Float16 and the BigFloat
    # correctly-rounded value).
    scale = (explim && e < emin) ? -(emin + 1 - t) : (t - 1 - e)

    scaled = ldexp(x, scale)
    rounded = _round_int(scaled, roundmode, rng)
    c = ldexp(rounded, -scale)

    explim && (c = _apply_range_limits(c, x, t, emax, roundmode, subnormal))

    return c
end

"""
    _flipbit(x, t, rng, p=0.5)

Randomly flip one bit (chosen uniformly from bit positions `1` to `t-1`) in
the integer-scaled significand's magnitude, with probability `p`, matching
`roundit.m`'s post-rounding bit-flip step. Applied by the public API after
`_round_to_format`, not inside it — flip is an opt-in effect, not a format
property. `x` is expected to already be a value on the target-format grid
(the output of `_round_to_format`).
"""
function _flipbit(
    x::T, t::Integer, rng::Random.AbstractRNG, p::Real=0.5
) where {T<:Union{Float32,Float64}}
    (isnan(x) || isinf(x)) && return x
    rand(rng) > p && return x
    t <= 1 && return x
    e = iszero(x) ? 0 : exponent(x)
    scaled = ldexp(x, t - 1 - e)
    u = abs(round(Int, scaled))
    b = rand(rng, 1:(t - 1))
    u = xor(u, 1 << (b - 1))
    signed_u = signbit(x) ? -T(u) : T(u)
    return ldexp(signed_u, e - (t - 1))
end

"""
    issimulationexact(t, roundmode; containerprecision=53)

Report whether Float64-then-round simulation (`_round_to_format` applied to
an already-computed `Float64` result) is *provably* exact for `+, -, ×, ÷, √`
at target precision `t`, i.e. free of double-rounding error, per Higham &
Pranesh (2019), p. C588: round-to-nearest-family modes require
`2t + 2 ≤ containerprecision`; directed rounding modes (`RoundUp`,
`RoundDown`, `RoundToZero`) are unconditionally exact regardless of `t`
(paper: "this condition is needed for round to nearest but not for directed
rounding"). Stochastic modes and the LowPrecisionChop-only
`RoundNearestTiesAway` extension are conservatively treated as
nearest-family (not derived from the paper — see docs/src/spec.md §3 and §8,
and docs/src/accuracy.md, for why).

`containerprecision` is the precision (bits including hidden bit) of the
type the operation is actually carried out in before rounding to the target
format — `53` for `Float64` (the default, and the only value relevant to
`LowFloat`, which always stores as `Float64`), `24` for `Float32`.

# Examples
```jldoctest
julia> issimulationexact(11, RoundNearest)  # fp16: 2*11+2=24 <= 53
true

julia> issimulationexact(30, RoundNearest)  # 2*30+2=62 > 53
false
```
"""
function issimulationexact(
    t::Integer, roundmode::RoundingMode; containerprecision::Integer=53
)
    return if _isnearestfamily(roundmode)
        (2t + 2 <= containerprecision)
    else
        (t + 1 <= containerprecision)
    end
end
