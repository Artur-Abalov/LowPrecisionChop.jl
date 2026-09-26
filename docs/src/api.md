# API reference

Everything the package exports, in one place. For orientation, start with
[Getting started](getting_started.md); for the semantics behind these
signatures, see [Specification](spec.md).

```@meta
CurrentModule = LowPrecisionChop
```

## Index

```@index
```

## Value-level rounding

Note the import: `chop` collides with `Base.chop`, so use
`using LowPrecisionChop: chop` or the qualified name
([why](getting_started.md#The-chop-name-collision)). `chop!` is unaffected.

```@docs
LowPrecisionChop.chop
chop!
```

## The `LowFloat` type

```@docs
LowFloat
lowfloattype
```

Operations implemented for `LowFloat`: `+ - * / ^ sqrt abs abs2 inv`, unary
minus, `floor ceil trunc round` (including the `RoundingMode` forms),
`zero one`, `== < isless isequal iszero signbit sign`, `hash`,
`Base.decompose`, `eps floatmin floatmax typemin typemax precision`,
`exponent ldexp nextfloat prevfloat`, `rand`, `bitstring`, and conversion to
`Float64`/`Float32`/`Float16`/`BigFloat` and to integer types. `muladd`,
`oneunit`, `copysign`, `min`/`max`, `hypot` and `cbrt` come free from Base's
generic fallbacks.

Transcendental functions are deliberately absent, and `fma`/`rem`/`mod`/`div`
are simply not implemented — see
[the guide](lowfloat.md#Arithmetic) and
[Known limitations](known_limitations.md#Deliberate-scope-boundaries).

## Formats

```@docs
Format
Float8E4M3Format
Float8E5M2Format
Float16Format
BFloat16Format
TF32Format
Float32Format
Float64Format
```

## Rounding modes

Five of the seven modes are Julia's own `Base` rounding modes — `RoundNearest`
(the default), `RoundUp`, `RoundDown`, `RoundToZero`, and
`RoundNearestTiesAway` (which has no MATLAB `chop` equivalent; see
[Rounding modes](rounding.md#RoundNearestTiesAway-has-no-external-oracle)).
The two stochastic modes are defined here:

```@docs
RoundStochasticProportional
RoundStochasticEqual
```

## Correctness checking

```@docs
issimulationexact
```

## MicroFloatingPoints.jl extension

Available only once `MicroFloatingPoints` is loaded; without it, the call
raises a "no method matching" error.

```@docs
matchingfloatmutype
```
