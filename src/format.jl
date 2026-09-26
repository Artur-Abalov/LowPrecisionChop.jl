"""
    Format{t,emax}

A low-precision floating-point format: `t` significand bits (including the
hidden bit), maximum unbiased exponent `emax`. Carries no runtime data — `t`
and `emax` are type parameters so they constant-fold in both `chop`/`chop!`
and `LowFloat` arithmetic. See docs/src/spec.md §1.

# Examples
```jldoctest
julia> Format(11, 15) == Float16Format
true
```
"""
struct Format{t,emax} end

Format(t::Integer, emax::Integer) = Format{Int(t),Int(emax)}()

Base.broadcastable(fmt::Format) = Ref(fmt)

_nexpbits(emax::Integer) = ceil(Int, log2(emax + 1)) + 1

"""
    Float8E4M3Format

fp8 format with 4 significand bits (3 explicit + hidden), 4 exponent bits
(`emax=7`). Higham & Pranesh (2019) Table 2.1 calls this "q43"; also the
IEEE P3109-adjacent E4M3 layout.

# Examples
```jldoctest
julia> using LowPrecisionChop: chop

julia> chop(3.7, Float8E4M3Format)
3.75
```
"""
const Float8E4M3Format = Format(4, 7)

"""
    Float8E5M2Format

fp8 format with 3 significand bits (2 explicit + hidden), 5 exponent bits
(`emax=15`). Higham & Pranesh (2019) Table 2.1 calls this "q52"; also the
E5M2 layout.

# Examples
```jldoctest
julia> using LowPrecisionChop: chop

julia> chop(3.7, Float8E5M2Format)
3.5
```
"""
const Float8E5M2Format = Format(3, 15)

"""
    Float16Format

IEEE binary16 (half precision): `t=11`, `emax=15`. Matches Julia's native
`Float16` exactly (see the native-type contract tests).

# Examples
```jldoctest
julia> using LowPrecisionChop: chop

julia> chop(3.14159265358979, Float16Format)
3.140625
```
"""
const Float16Format = Format(11, 15)

"""
    BFloat16Format

bfloat16: `t=8`, `emax=127` (same exponent range as `Float32`, reduced
significand). Matches BFloat16s.jl's `BFloat16` (weak-dependency extension
target, not a hard dependency).

# Examples
```jldoctest
julia> using LowPrecisionChop: chop

julia> chop(3.14159265358979, BFloat16Format)
3.140625
```
"""
const BFloat16Format = Format(8, 127)

"""
    TF32Format

NVIDIA TF32: `t=11`, `emax=127` (10 explicit mantissa bits + hidden bit,
8-bit exponent field). Not present in MATLAB `chop` or its paper — defined
here from NVIDIA's published TF32 layout, not golden-testable against
MATLAB. See docs/src/spec.md §1.

# Examples
```jldoctest
julia> using LowPrecisionChop: chop

julia> chop(3.14159265358979, TF32Format)
3.140625
```
"""
const TF32Format = Format(11, 127)

"""
    Float32Format

IEEE binary32 (single precision): `t=24`, `emax=127`. Matches Julia's
native `Float32` exactly.

# Examples
```jldoctest
julia> using LowPrecisionChop: chop

julia> chop(3.14159265358979, Float32Format)
3.1415927410125732
```
"""
const Float32Format = Format(24, 127)

"""
    Float64Format

IEEE binary64 (double precision): `t=53`, `emax=1023`. A no-op format —
`chop(x, Float64Format)` reproduces `x` for any finite `Float64` `x`.
Intended for exploring the effect of `subnormal`/`explim`/`round` in
isolation, per `chop.m`'s own advisory note; see docs/src/spec.md §1. Note:
`t=53` fails [`issimulationexact`](@ref) for `RoundNearest` against its own
53-bit container (`2*53+2 > 53`) — harmless here since `Float64Format` is a
literal identity map, not a lower-precision simulation, but `chop` will
still emit its one-time `strict`-mode warning for it.

# Examples
```jldoctest
julia> using LowPrecisionChop: chop

julia> using Logging

julia> with_logger(NullLogger()) do  # suppress the one-time strict-mode warning for this doctest
           chop(3.14159265358979, Float64Format; strict=false) == 3.14159265358979
       end
true
```
"""
const Float64Format = Format(53, 1023)
