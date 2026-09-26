"""
    LowFloat{t,emax,R,S} <: AbstractFloat

Parametric low-precision float. `t` significand bits (including hidden
bit), maximum unbiased exponent `emax`, rounding mode type `R<:RoundingMode`,
subnormal-support flag `S::Bool`. Stored as `Float64`; arithmetic computes
in `Float64` then applies the same `_round_to_format` kernel `chop` uses
(never a slower detour through `chop` itself). `explim` is always `true` for
`LowFloat` — see docs/src/spec.md §5 for why there is no type-parameter slot
for it.

**Double rounding caveat**: `LowFloat` arithmetic is exact (matches
correctly-rounded infinite-precision results) for `+,-,×,÷,√` only when
`issimulationexact(t, R())` holds against a 53-bit (`Float64`) container —
true for every named preset, false for custom formats with `t` too close to
53. Construct with `strict=true`-checked helpers or call
[`issimulationexact`](@ref) yourself before relying on a custom format.

**Stochastic-rounding reproducibility**: unlike [`LowPrecisionChop.chop`](@ref),
which takes an explicit `rng` keyword, `LowFloat`'s arithmetic operators
(`+`, `-`, `*`, `/`, ...) have no room for one — infix operator syntax only
takes the two operands. A `LowFloat{...,R,...}` with `R` a stochastic mode
(`RoundStochasticProportional`/`RoundStochasticEqual`) therefore draws from
the ambient `Random.default_rng()` for every arithmetic operation. To get
reproducible results, call `Random.seed!(seed)` before your computation —
not by trying to pass an RNG into an operator. (Found and fixed during the
Flux/OrdinaryDiffEq integration testing — see `docs/src/integrations.md`.)

# Examples
```jldoctest
julia> LowFloat16 = LowFloat{11,15,typeof(RoundNearest),true};

julia> Float64(LowFloat16(3.14159265358979))
3.140625
```
"""
struct LowFloat{t,emax,R<:RoundingMode,S} <: AbstractFloat
    val::Float64
    function LowFloat{t,emax,R,S}(val::Float64, ::Val{:raw}) where {t,emax,R,S}
        return new{t,emax,R,S}(val)
    end
end

function _roundctor(::Type{LowFloat{t,emax,R,S}}, v::Float64) where {t,emax,R,S}
    return LowFloat{t,emax,R,S}(_round_to_format(v, t, emax, R(), S, true), Val(:raw))
end

# Split by disjoint type domains, not a single `x::Real` catch-all: Base
# defines `(::Type{T})(::Rational{S}) where {S,T<:AbstractFloat}`, and a
# generic `LowFloat{t,emax,R,S}(x::Real)` method's signature still overlaps
# it (Julia's ambiguity detector flags this pairing even though the actual
# call resolves correctly once a third, Rational-specific method exists --
# a known false positive for parametric-type outer constructors). Keeping
# the domains disjoint avoids the overlap entirely rather than relying on
# resolution-by-specificity.
function LowFloat{t,emax,R,S}(
    x::Union{AbstractFloat,Integer,AbstractIrrational}
) where {t,emax,R,S}
    return _roundctor(LowFloat{t,emax,R,S}, Float64(x))
end
function LowFloat{t,emax,R,S}(x::Rational) where {t,emax,R,S}
    return _roundctor(LowFloat{t,emax,R,S}, Float64(x))
end
LowFloat{t,emax,R,S}(x::LowFloat{t,emax,R,S}) where {t,emax,R,S} = x

# --- conversion ---
# Narrow, concrete-type overloads only -- a blanket
# `(::Type{T})(x::LowFloat) where {T<:AbstractFloat}` is ambiguous against
# both `LowFloat`'s own type-parametric constructor and Base's
# `(::Type{T})(::Rational) where {T<:AbstractFloat}` (caught by Aqua).
Base.Float64(x::LowFloat) = x.val
Base.Float32(x::LowFloat) = Float32(x.val)
Base.Float16(x::LowFloat) = Float16(x.val)
Base.BigFloat(x::LowFloat) = BigFloat(x.val)
# Integer conversion (`Int(x::LowFloat)` etc.) -- needed for OrdinaryDiffEq's
# internal step-count bookkeeping (`Int(round(dt_related_value))`).
# `Union{Signed,Unsigned}` covers every
# standard integer type while cleanly excluding `Bool` (`Bool<:Integer` but
# `Bool <: Union{Signed,Unsigned}` is false) -- Base separately defines
# `Bool(::Real)`, which a blanket `T<:Integer` bound would ambiguously
# overlap (caught by Aqua).
(::Type{T})(x::LowFloat) where {T<:Union{Signed,Unsigned}} = T(x.val)
Base.Bool(x::LowFloat) = Bool(x.val)
Base.convert(::Type{LF}, x::Real) where {LF<:LowFloat} = LF(x)
Base.convert(::Type{LF}, x::LF) where {LF<:LowFloat} = x
Base.float(x::LowFloat) = x

# --- promotion: never resolve LowFloat/Real to the other Real type, per
# spec.md §9 -- reductions like `norm`/adaptive-step controllers must not
# silently upcast to Float64. ---
function Base.promote_rule(::Type{LowFloat{t,emax,R,S}}, ::Type{<:Real}) where {t,emax,R,S}
    return LowFloat{t,emax,R,S}
end
function Base.promote_rule(
    ::Type{LowFloat{t,emax,R,S}}, ::Type{LowFloat{t,emax,R,S}}
) where {t,emax,R,S}
    return LowFloat{t,emax,R,S}
end

# --- arithmetic (Simulation 3.1: compute in Float64, round once) ---
Base.:+(a::LF, b::LF) where {LF<:LowFloat} = _roundctor(LF, a.val + b.val)
Base.:-(a::LF, b::LF) where {LF<:LowFloat} = _roundctor(LF, a.val - b.val)
Base.:*(a::LF, b::LF) where {LF<:LowFloat} = _roundctor(LF, a.val * b.val)
Base.:/(a::LF, b::LF) where {LF<:LowFloat} = _roundctor(LF, a.val / b.val)
Base.:-(a::LF) where {LF<:LowFloat} = LF(-a.val, Val(:raw))
Base.sqrt(a::LF) where {LF<:LowFloat} = _roundctor(LF, sqrt(a.val))
# `Base.FastMath.sqrt_fast` has no generic `AbstractFloat` fallback (only
# Float16/32/64 -- base/fastmath.jl), which breaks OrdinaryDiffEq's default
# array-valued norm (`ODE_DEFAULT_NORM`, DiffEqBase.jl) for any `LowFloat`
# state vector; see docs/src/spec.md §9.
Base.FastMath.sqrt_fast(a::LowFloat) = sqrt(a)
Base.abs(a::LF) where {LF<:LowFloat} = LF(abs(a.val), Val(:raw))
Base.abs2(a::LF) where {LF<:LowFloat} = _roundctor(LF, a.val^2)
Base.inv(a::LF) where {LF<:LowFloat} = _roundctor(LF, inv(a.val))
# `^` is outside the (+,-,*,/,sqrt) double-rounding-safety derivation
# (spec.md §8) -- needed by Krylov.jl's `gmres!`, which uses it internally
# for Givens rotations. Same pattern as the other ops (compute in Float64,
# round once); no separate safety proof, flagged here rather than assumed
# exact.
Base.:^(a::LF, b::Integer) where {LF<:LowFloat} = _roundctor(LF, a.val^b)
Base.:^(a::LF, b::LF) where {LF<:LowFloat} = _roundctor(LF, a.val^b.val)
# floor/ceil/trunc/round: no generic `AbstractFloat` fallback exists in
# Base for these (they bottom out in type-specific bit tricks for
# Float16/32/64/BigFloat). Needed by OrdinaryDiffEq, which calls
# `round(::LowFloat, RoundUp)` internally for step-count bookkeeping even
# in fixed-step mode.
Base.floor(a::LF) where {LF<:LowFloat} = _roundctor(LF, floor(a.val))
Base.ceil(a::LF) where {LF<:LowFloat} = _roundctor(LF, ceil(a.val))
Base.trunc(a::LF) where {LF<:LowFloat} = _roundctor(LF, trunc(a.val))
Base.round(a::LF) where {LF<:LowFloat} = _roundctor(LF, round(a.val))
# Narrow to exactly the modes with no generic `AbstractFloat` fallback in
# Base (:Nearest/:Up/:Down/:ToZero bottom out in type-specific bit tricks
# per hardware float type). A blanket `m::RoundingMode` method would
# ambiguously overlap Base's *generic* `round(::AbstractFloat,
# RoundingMode{:NearestTiesAway/:NearestTiesUp/:FromZero})`, which are
# themselves built from floor/ceil and so already work correctly for
# `LowFloat` without any override here (caught by Aqua).
function Base.round(a::LF, ::RoundingMode{:Nearest}) where {LF<:LowFloat}
    return _roundctor(LF, round(a.val, RoundNearest))
end
function Base.round(a::LF, ::RoundingMode{:Up}) where {LF<:LowFloat}
    return _roundctor(LF, round(a.val, RoundUp))
end
function Base.round(a::LF, ::RoundingMode{:Down}) where {LF<:LowFloat}
    return _roundctor(LF, round(a.val, RoundDown))
end
function Base.round(a::LF, ::RoundingMode{:ToZero}) where {LF<:LowFloat}
    return _roundctor(LF, round(a.val, RoundToZero))
end

# --- comparisons ---
Base.:(==)(a::LF, b::LF) where {LF<:LowFloat} = a.val == b.val
Base.:<(a::LF, b::LF) where {LF<:LowFloat} = a.val < b.val
Base.isless(a::LF, b::LF) where {LF<:LowFloat} = isless(a.val, b.val)
Base.isequal(a::LF, b::LF) where {LF<:LowFloat} = isequal(a.val, b.val)
Base.iszero(a::LowFloat) = iszero(a.val)
Base.signbit(a::LowFloat) = signbit(a.val)
Base.sign(a::LF) where {LF<:LowFloat} = LF(sign(a.val), Val(:raw))
Base.hash(a::LowFloat, h::UInt) = hash(a.val, h)
Base.decompose(a::LowFloat) = Base.decompose(a.val)

# --- zero/one ---
Base.zero(::Type{LF}) where {LF<:LowFloat} = LF(0.0, Val(:raw))
Base.one(::Type{LF}) where {LF<:LowFloat} = LF(1.0, Val(:raw))

# --- format-derived constants ---
_xmax(t::Integer, emax::Integer) = ldexp(2.0 - ldexp(1.0, 1 - t), emax)
_xmin(emax::Integer) = ldexp(1.0, 1 - emax)
_xmins(t::Integer, emax::Integer) = ldexp(1.0, 1 - emax + 1 - t)

function Base.floatmax(::Type{LF}) where {t,emax,LF<:LowFloat{t,emax}}
    return LF(_xmax(t, emax), Val(:raw))
end
Base.floatmin(::Type{LF}) where {t,emax,LF<:LowFloat{t,emax}} = LF(_xmin(emax), Val(:raw))
Base.typemax(::Type{LF}) where {LF<:LowFloat} = LF(Inf, Val(:raw))
Base.typemin(::Type{LF}) where {LF<:LowFloat} = LF(-Inf, Val(:raw))
function Base.precision(::Type{LowFloat{t,emax,R,S}}; base::Integer=2) where {t,emax,R,S}
    return base == 2 ? t : throw(ArgumentError("only base=2 is supported"))
end
# Written with `t` destructured explicitly (not proxied through an opaque
# `LF<:LowFloat` type variable) so this resolves directly to the concrete
# `Type{LowFloat{t,emax,R,S}}` method above rather than risking a fallthrough
# to Base's generic `_precision`/`_precision_with_base_2` machinery, which
# JET flags as unreachable-but-statically-considered for our type (it has no
# method for `LowFloat`).
function Base.precision(::LowFloat{t,emax,R,S}; base::Integer=2) where {t,emax,R,S}
    return base == 2 ? t : throw(ArgumentError("only base=2 is supported"))
end
Base.eps(::Type{LF}) where {t,LF<:LowFloat{t}} = LF(ldexp(1.0, 1 - t), Val(:raw))
Base.eps(x::LowFloat) = eps(typeof(x))  # generic AbstractFloat fallback also works; explicit for clarity/speed

# --- exponent / ldexp / nextfloat / prevfloat: format grid, not Float64's ---
Base.exponent(x::LowFloat) = exponent(x.val)
Base.ldexp(x::LF, n::Integer) where {LF<:LowFloat} = _roundctor(LF, ldexp(x.val, n))

# Grid spacing at magnitude |v| (v != 0), for stepping AWAY from zero
# (v's own binade always applies -- magnitude only grows, so no boundary
# crossing into a finer grid is possible in this direction).
function _ulpaway(v::Float64, t::Integer, emax::Integer, subnormal::Bool)
    emin = 1 - emax
    e = exponent(v)
    if subnormal && e < emin
        return ldexp(1.0, emin + 1 - t)
    else
        return ldexp(1.0, max(e, emin) - (t - 1))
    end
end

# Grid spacing at magnitude |v| (v != 0), for stepping TOWARD zero. Needs
# special handling exactly when |v| sits at its own binade's lower boundary
# (a power of two): the representable point just toward zero from there
# belongs to the *next finer* binade, not v's own (coarser) one -- e.g. the
# representable value just below 2.0 is 1.75 (spacing 0.25, binade [1,2)),
# not 1.5 (spacing 0.5, binade [2,4), which is 2.0's own binade under
# `exponent`). At the smallest-normal boundary with subnormals unsupported,
# the step is a jump straight to zero rather than any finer grid at all.
function _ulptoward(v::Float64, t::Integer, emax::Integer, subnormal::Bool)
    emin = 1 - emax
    e = exponent(v)
    av = abs(v)
    if av == ldexp(1.0, e)
        if e <= emin && !subnormal
            return av  # v -/+ this step lands exactly on zero
        end
        e -= 1
    end
    if subnormal && e < emin
        return ldexp(1.0, emin + 1 - t)
    else
        return ldexp(1.0, max(e, emin) - (t - 1))
    end
end

function Base.nextfloat(x::LowFloat{t,emax,R,S}) where {t,emax,R,S}
    v = x.val
    LF = typeof(x)
    isnan(v) && return x
    v == Inf && return x
    xmax = _xmax(t, emax)
    v == -Inf && return LF(-xmax, Val(:raw))
    v >= xmax && return LF(Inf, Val(:raw))
    if v == 0
        step = S ? ldexp(1.0, (1 - emax) + 1 - t) : ldexp(1.0, 1 - emax)
        return LF(step, Val(:raw))
    end
    step = v > 0 ? _ulpaway(v, t, emax, S) : _ulptoward(v, t, emax, S)
    return LF(v + step, Val(:raw))
end

function Base.prevfloat(x::LowFloat{t,emax,R,S}) where {t,emax,R,S}
    v = x.val
    LF = typeof(x)
    isnan(v) && return x
    v == -Inf && return x
    xmax = _xmax(t, emax)
    v == Inf && return LF(xmax, Val(:raw))
    v <= -xmax && return LF(-Inf, Val(:raw))
    if v == 0
        step = S ? ldexp(1.0, (1 - emax) + 1 - t) : ldexp(1.0, 1 - emax)
        return LF(-step, Val(:raw))
    end
    step = v > 0 ? _ulptoward(v, t, emax, S) : _ulpaway(v, t, emax, S)
    return LF(v - step, Val(:raw))
end

# --- random ---
# Base/Random already dispatches `rand(rng, ::Type{T}) where {T<:AbstractFloat}`
# through `CloseOpen01`; overload the concrete `LowFloat` method directly
# (more specific, takes priority) rather than hooking `SamplerType`.
function Random.rand(rng::Random.AbstractRNG, ::Type{LF}) where {LF<:LowFloat}
    return _roundctor(LF, rand(rng))
end

# --- bitstring ---
"""
    bitstring(x::LowFloat)

The bit pattern `x` would have if stored natively in its own `(t, emax)`
format: 1 sign bit, `_nexpbits(emax)` exponent bits (IEEE-style bias
`emax`), `t-1` significand bits. Matches standard IEEE binary layout for
every named preset (whose `emax` is `2^e - 1` for integer `e`); for
arbitrary custom `emax` this is a well-defined but non-standard encoding
(see docs/src/spec.md and `_nexpbits`).
"""
function Base.bitstring(x::LowFloat{t,emax,R,S}) where {t,emax,R,S}
    ne = _nexpbits(emax)
    v = x.val
    s = signbit(v) ? "1" : "0"
    if isnan(v)
        return s * ("1"^ne) * "1" * ("0"^(t - 2))
    elseif isinf(v)
        return s * ("1"^ne) * ("0"^(t - 1))
    elseif iszero(v)
        return s * ("0"^ne) * ("0"^(t - 1))
    end
    emin = 1 - emax
    e = exponent(v)
    if e < emin
        expfield = 0
        mantissa = round(Int, abs(v) / ldexp(1.0, emin + 1 - t))
    else
        expfield = e + emax
        frac = abs(v) / ldexp(1.0, e) - 1.0
        mantissa = round(Int, frac * 2.0^(t - 1))
    end
    return s *
           lpad(string(expfield; base=2), ne, '0') *
           lpad(string(mantissa; base=2), t - 1, '0')
end

# --- show ---
function Base.show(io::IO, x::LowFloat)
    print(io, x.val)
    return nothing
end
