_containerprecision(::Type{Float64}) = 53
_containerprecision(::Type{Float32}) = 24

function _maybewarn_or_error(
    t::Integer,
    emax::Integer,
    roundmode::RoundingMode,
    containerprecision::Integer,
    strict::Bool,
)
    issimulationexact(t, roundmode; containerprecision) && return nothing
    msg =
        "Format (t=$t, emax=$emax) with round mode $roundmode is not provably " *
        "double-rounding-safe against a $containerprecision-bit container " *
        "(see `issimulationexact`, docs/src/spec.md §8): results may occasionally " *
        "disagree with the mathematically exact rounding at rare boundary cases."
    if strict
        throw(
            ArgumentError(
                msg *
                " Pass strict=false to proceed anyway, or choose a safer format/round mode.",
            ),
        )
    else
        @warn msg * " Shown once per distinct (t, emax, round mode, container)." maxlog = 1 _id = Symbol(
            :lowprecisionchop_unsafe_, t, :_, emax, :_, roundmode, :_, containerprecision
        )
    end
    return nothing
end

"""
    chop(x, fmt::Format; round=RoundNearest, subnormal=true, explim=true,
         flip=false, p=0.5, rng=Random.default_rng(), strict=false)

Round `x::Union{Float64,Float32}` to the low-precision format `fmt`,
returning a value of the same type as `x` with the unused significand bits
zeroed. MATLAB-`chop`-compatible semantics for `round` values 1-6 (see
docs/src/spec.md §3); `round=RoundNearestTiesAway` (MATLAB mode "7") is a
LowPrecisionChop-only extension with no MATLAB equivalent.

`Base.chop` (string truncation) is also exported — `using LowPrecisionChop`
will warn about the name clash. Use `using LowPrecisionChop: chop` (or
`LowPrecisionChop.chop(...)`) to disambiguate.

`strict=true` errors instead of warning when `(fmt, round)` is not provably
free of double-rounding error for `x`'s type — see [`issimulationexact`](@ref).

# Examples
```jldoctest
julia> using LowPrecisionChop: chop

julia> chop(3.14159265358979, Float16Format)
3.140625
```
"""
function chop(
    x::T,
    ::Format{t,emax};
    round::RoundingMode=RoundNearest,
    subnormal::Bool=true,
    explim::Bool=true,
    flip::Bool=false,
    p::Real=0.5,
    rng::Random.AbstractRNG=Random.default_rng(),
    strict::Bool=false,
) where {T<:Union{Float64,Float32},t,emax}
    _maybewarn_or_error(t, emax, round, _containerprecision(T), strict)
    c = _round_to_format(x, t, emax, round, subnormal, explim, rng)
    flip && (c = _flipbit(c, t, rng, p))
    return c
end

function chop(A::AbstractArray{T}, fmt::Format; kwargs...) where {T<:Union{Float64,Float32}}
    return chop.(A, fmt; kwargs...)
end

"""
    chop!(A::AbstractArray, fmt::Format; kwargs...)

In-place version of [`LowPrecisionChop.chop`](@ref): overwrites `A` with `chop.(A, fmt; kwargs...)`.

# Examples
```jldoctest
julia> using LowPrecisionChop: chop!

julia> A = [1.0, 2.5, 3.14159];

julia> chop!(A, Float16Format);

julia> A
3-element Vector{Float64}:
 1.0
 2.5
 3.140625
```
"""
function chop!(
    A::AbstractArray{T}, fmt::Format; kwargs...
) where {T<:Union{Float64,Float32}}
    A .= chop.(A, fmt; kwargs...)
    return A
end
