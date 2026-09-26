module LowPrecisionChop

using Random

include("roundingmodes.jl")
include("kernel.jl")
include("format.jl")
include("chop.jl")
include("lowfloat.jl")

"""
    lowfloattype(fmt::Format, round::RoundingMode=RoundNearest, subnormal::Bool=true)

Build a concrete `LowFloat{t,emax,R,S}` type from a `Format` preset (or
custom `Format(t,emax)`) plus a rounding mode and subnormal flag, e.g.
`lowfloattype(Float16Format)(3.14)`.

# Examples
```jldoctest
julia> LF16 = lowfloattype(Float16Format);

julia> Float64(LF16(3.14159265358979))
3.140625
```
"""
function lowfloattype(
    ::Format{t,emax}, round::RoundingMode=RoundNearest, subnormal::Bool=true
) where {t,emax}
    return LowFloat{t,emax,typeof(round),subnormal}
end

"""
    matchingfloatmutype(::Type{<:LowFloat})

The `MicroFloatingPoints.Floatmu{szE,szf}` type with the same `(t, emax)`
format as the given `LowFloat` type. Only has a method when
`MicroFloatingPoints` is loaded (package extension,
`LowPrecisionChopMicroFloatingPointsExt`) — `using MicroFloatingPoints`
before calling this, or it errors with "no method matching".

# Examples
```jldoctest
julia> using MicroFloatingPoints

julia> matchingfloatmutype(lowfloattype(Float16Format))
Floatmu{5, 10}
```
"""
function matchingfloatmutype end

export issimulationexact
export RoundStochasticProportional, RoundStochasticEqual
export Format,
    Float8E4M3Format,
    Float8E5M2Format,
    Float16Format,
    BFloat16Format,
    TF32Format,
    Float32Format,
    Float64Format
export chop, chop!
export LowFloat, lowfloattype, matchingfloatmutype

end # module
