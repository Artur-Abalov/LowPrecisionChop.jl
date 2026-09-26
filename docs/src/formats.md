# Formats

A format in this package is two integers: how many bits of significand, and
how large the exponent may get. Everything else — the smallest normal
number, the largest finite number, unit roundoff, the subnormal grid — is
derived from those two.

## The `(t, emax)` model

```julia
Format(t, emax)
```

- **`t`** — significand bits, *including the implicit leading bit*. fp16 has
  10 stored mantissa bits plus a hidden bit, so `t = 11`.
- **`emax`** — the largest unbiased exponent a normal number may have. fp16
  reaches `2^15 · 1.111...`, so `emax = 15`.

The exponent range is symmetric in the IEEE sense, `emin = 1 - emax`, which
fixes every other quantity:

| quantity | formula | meaning |
|---|---|---|
| `emin` | `1 - emax` | smallest normal exponent |
| `xmin` | `2^emin` | smallest positive normal number |
| `emins` | `emin + 1 - t` | smallest subnormal exponent |
| `xmins` | `2^emins` | smallest positive subnormal number |
| `xmax` | `2^emax · (2 - 2^(1-t))` | largest finite number |
| `u` | `2^-t` | unit roundoff (half an ulp at 1.0, round-to-nearest) |
| `eps` | `2^(1-t)` | gap between 1.0 and the next float up |

[`Format`](@ref) is a singleton type — `t` and `emax` are type parameters,
not fields, so they constant-fold into the rounding kernel. A `Format` value
carries no runtime data and costs nothing to pass around.

## Named presets

Seven presets are exported. The first five come from MATLAB `chop`'s own
format table; `TF32Format` and the derived columns below are documented
where they differ.

| preset | `t` | `emax` | exponent bits | `eps` | `xmin` | `xmax` |
|---|---|---|---|---|---|---|
| [`Float8E4M3Format`](@ref) | 4 | 7 | 4 | `0.125` | `1.5625e-2` | `240` |
| [`Float8E5M2Format`](@ref) | 3 | 15 | 5 | `0.25` | `6.1035e-5` | `57344` |
| [`Float16Format`](@ref) | 11 | 15 | 5 | `9.7656e-4` | `6.1035e-5` | `65504` |
| [`BFloat16Format`](@ref) | 8 | 127 | 8 | `7.8125e-3` | `1.1755e-38` | `3.3895e38` |
| [`TF32Format`](@ref) | 11 | 127 | 8 | `9.7656e-4` | `1.1755e-38` | `3.4012e38` |
| [`Float32Format`](@ref) | 24 | 127 | 8 | `1.1921e-7` | `1.1755e-38` | `3.4028e38` |
| [`Float64Format`](@ref) | 53 | 1023 | 11 | `2.2204e-16` | `2.2251e-308` | `1.7977e308` |

`Float16Format`, `Float32Format` and `Float64Format` reproduce Julia's
native `Float16`, `Float32` and `Float64` exactly; `BFloat16Format`
reproduces bfloat16. Those equivalences are asserted in the test suite
(`test/native_contracts.jl`).

Two presets deserve a note:

- **`TF32Format`** is not in MATLAB `chop` or in the paper. It is defined
  here from NVIDIA's published TF32 layout (10 explicit mantissa bits plus
  hidden bit, 8-bit exponent field), so it has no external golden-value
  oracle; it is verified against the package's own BigFloat oracle only.
- **`Float64Format`** is an identity map for any finite `Float64`. It exists
  so you can vary `round`, `subnormal` or `explim` in isolation without also
  changing precision. Because its precision equals its container's, it
  fails its own [`issimulationexact`](@ref) check and will emit the
  one-time `strict`-mode warning — harmless, but surprising if you don't
  expect it. See [Accuracy](accuracy.md#The-Float64Format-special-case).

Read the constants off a preset by building its `LowFloat` type:

```julia-repl
julia> using LowPrecisionChop

julia> T = lowfloattype(Float8E4M3Format);

julia> precision(T), Float64(eps(T)), Float64(floatmin(T)), Float64(floatmax(T))
(4, 0.125, 0.015625, 240.0)
```

## Custom formats

Any `(t, emax)` pair works:

```julia-repl
julia> using LowPrecisionChop: chop

julia> chop(0.1, Format(6, 20))      # 6 significand bits, emax = 20
0.099609375

julia> chop(0.1, Format(4, 10))      # a 4-bit significand
0.1015625
```

Two cautions.

**Check `issimulationexact` for wide significands.** A format whose
significand approaches `Float64`'s own 53 bits is no longer provably safe to
simulate by computing in `Float64` and rounding once:

```julia-repl
julia> issimulationexact(25, RoundNearest)   # safe
true

julia> issimulationexact(26, RoundNearest)   # not provably safe
false
```

`chop` warns once per offending `(t, emax, round, container)` combination;
`strict=true` turns that into an error. Full treatment in
[Accuracy and double rounding](accuracy.md).

**`Format` does not validate its arguments.** `t < 1` or `emax < 1` produce
undefined behavior rather than a clear error. Stay in `t ≥ 1`, `emax ≥ 1`.

Formats with `emax + 1` not a power of two are perfectly usable for
rounding, but their notional bit layout is non-standard — the exponent field
width is taken as `ceil(log2(emax + 1)) + 1`, which is only the IEEE width
when `emax = 2^(k-1) - 1`. This matters only for
[`bitstring`](lowfloat.md#Bit-level-inspection).

## Subnormals

`subnormal` controls whether the format has a gradual-underflow region
below `xmin`, or flushes straight to zero.

```julia-repl
julia> using LowPrecisionChop: chop

julia> xmin = 2.0^-14;                          # fp16's smallest normal

julia> chop(xmin / 2, Float16Format)            # subnormal=true (default)
3.0517578125e-5

julia> chop(xmin / 2, Float16Format; subnormal = false)
6.103515625e-5

julia> chop(xmin / 4, Float16Format; subnormal = false)
0.0
```

With subnormals off, values below `xmin/2` flush to zero and values in
`[xmin/2, xmin)` round up to `xmin` — note that an exact tie at `xmin/2`
rounds *up* here, which is asymmetric with the ties-to-even rule used in
the normal range. That asymmetry is inherited from the reference
implementation; see [Specification](spec.md#6.-Subnormals-and-the-underflow-boundary) for the exact
rule per rounding mode.

!!! note "Divergence from MATLAB `chop`'s bfloat16 default"
    MATLAB `chop` defaults `subnormal` to *off* for bfloat16 specifically,
    while defaulting it on elsewhere. This package defaults
    `subnormal = true` for **every** format, including `BFloat16Format`.
    Pass `subnormal = false` explicitly if you need MATLAB's bfloat16
    default. This is the one place where a default differs; the semantics
    of each setting are identical.

Whether subnormals are supported is a property of the *type* for
`LowFloat`, not a call option:

```julia-repl
julia> using LowPrecisionChop

julia> T = lowfloattype(Float16Format, RoundNearest, false);   # no subnormals

julia> Float64(nextfloat(zero(T)))     # first step off zero is a full xmin
6.103515625e-5
```

## Exponent limiting (`explim`)

`explim` (default `true`) controls whether the format's *exponent range* is
enforced at all. With `explim = false`, `emax` is ignored: the significand
is still rounded to `t` bits, but nothing overflows to `Inf`, nothing
saturates, and nothing underflows to zero except as the `Float64`/`Float32`
container itself requires.

```julia-repl
julia> using LowPrecisionChop: chop

julia> chop(1e-40, Float16Format)                     # below fp16's subnormals
0.0

julia> chop(1e-40, Float16Format; explim = false)     # significand only
9.999665841421895e-41

julia> chop(1e40, Float16Format; explim = false)      # no overflow to Inf
1.0001111440285707e40
```

This is a research knob for isolating "what does reducing the significand
alone do?" from "what does the limited exponent range do?". It is a keyword
argument on `chop`/`chop!` only — `LowFloat` values always behave as
`explim = true`, because a stored format that ignores its own `emax` is not
a well-defined format. See [Specification](spec.md#5.-explim-—-exponent-limiting).

## Interoperating with MicroFloatingPoints.jl

If [MicroFloatingPoints.jl](https://github.com/goualard-f/MicroFloatingPoints.jl)
is loaded, a package extension activates and
[`matchingfloatmutype`](@ref) maps a `LowFloat` type to the `Floatmu` type
with the same format:

```julia-repl
julia> using LowPrecisionChop, MicroFloatingPoints

julia> matchingfloatmutype(lowfloattype(Float16Format))
Floatmu{5, 10}
```

The correspondence is `Floatmu{szE,szf} ↔ Format(szf + 1, 2^(szE-1) - 1)`.
Conversion in the `LowFloat → Floatmu` direction is provided by the
extension; the reverse direction already works through `LowFloat`'s generic
`AbstractFloat` constructor.
