module LowPrecisionChopMicroFloatingPointsExt

using LowPrecisionChop
using LowPrecisionChop: LowFloat, _nexpbits
using MicroFloatingPoints: Floatmu

# MicroFloatingPoints.Floatmu{szE,szf} uses exponent-bit-width/explicit-
# significand-bit-width parameters; LowPrecisionChop.Format/LowFloat use
# (t = significand bits INCLUDING hidden bit, emax = max unbiased exponent).
# Floatmu{szE,szf} <-> Format(szf+1, 2^(szE-1)-1) is the exact
# correspondence (confirmed against MicroFloatingPoints' own Emax/Emin:
# Floatmu{4,3} has Emax=7=2^(4-1)-1; Floatmu{5,10} reproduces Float16
# bit-for-bit, matching Format(11,15)).
_floatmu_szE(emax::Integer) = _nexpbits(emax)
_floatmu_szf(t::Integer) = t - 1

# LowFloat -> Floatmu: goes through Float64, same as every other
# LowFloat conversion (round-trip fidelity only, not a bit-exact
# requirement -- see docs/src/spec.md §1).
function Floatmu{szE,szf}(x::LowFloat) where {szE,szf}
    return Floatmu{szE,szf}(Float64(x))
end

# Floatmu -> LowFloat: LowFloat's own generic constructor already accepts
# any AbstractFloat (Floatmu <: AbstractFloat) via Float64(x), so no new
# method is needed for that direction -- this file exists only for the
# LowFloat -> Floatmu direction above, which MicroFloatingPoints itself
# does not provide (its own constructors are Real/AbstractFloat-generic in
# the *other* direction only).

"""
    matchingfloatmutype(::Type{<:LowFloat})

The `MicroFloatingPoints.Floatmu{szE,szf}` type with the same `(t, emax)`
format as the given `LowFloat` type.
"""
function LowPrecisionChop.matchingfloatmutype(
    ::Type{LowFloat{t,emax,R,S}}
) where {t,emax,R,S}
    return Floatmu{_floatmu_szE(emax),_floatmu_szf(t)}
end

end # module
