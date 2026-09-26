# The `LowFloat` type

[`LowFloat`](@ref) is a real number type that computes in low precision.
Unlike [`chop`](chop.md), which you call at chosen points, `LowFloat` rounds
after *every* arithmetic operation — so you can hand an existing generic
Julia algorithm a `Matrix{LowFloat}` and have the whole thing run at fp16,
bfloat16, or whatever format you picked, without editing the algorithm.

## Constructing a type

`LowFloat{t,emax,R,S}` has four parameters: significand bits `t`, maximum
exponent `emax`, rounding-mode type `R`, and a `Bool` `S` for subnormal
support. Build one with [`lowfloattype`](@ref) rather than writing the
parameters by hand:

```julia-repl
julia> using LowPrecisionChop

julia> lowfloattype(Float16Format)
LowFloat{11, 15, RoundingMode{:Nearest}, true}

julia> lowfloattype(BFloat16Format, RoundStochasticProportional)
LowFloat{8, 127, RoundingMode{:StochasticProportional}, true}

julia> lowfloattype(Float16Format, RoundNearest, false)     # subnormals off
LowFloat{11, 15, RoundingMode{:Nearest}, false}
```

The signature is `lowfloattype(fmt, round = RoundNearest, subnormal = true)`
— both extra arguments are positional.

Values are stored in a single `Float64` field, always on the target format's
grid. Constructing from anything convertible to `Float64` rounds on the way
in:

```julia-repl
julia> T = lowfloattype(Float16Format);

julia> T(3.14159265358979)
3.140625

julia> T(1//3)
0.333251953125

julia> T(π)
3.140625

julia> T(7)
7.0
```

There is no `explim` parameter. `LowFloat` values always behave as
`explim = true`: a stored format that ignores its own exponent range is not
a well-defined format, so that knob stays on `chop` only.

## Arithmetic

Every operation computes in `Float64` and rounds once, with the same kernel
`chop` uses:

```julia-repl
julia> using LowPrecisionChop

julia> BF16 = lowfloattype(BFloat16Format);

julia> a, b = BF16(1.0), BF16(3.0);

julia> a / b
0.333984375

julia> sqrt(BF16(2.0))
1.4140625

julia> BF16(1.0) + 2          # mixed with an Int -- stays BF16
3.0
```

Implemented directly: `+ - * / ^ sqrt abs abs2 inv`, unary minus,
`floor ceil trunc round` (including `round(x, RoundUp)` and friends),
`zero one`, comparisons (`== < isless isequal iszero signbit sign`), `hash`,
`Base.decompose`, `ldexp`, `exponent`, `nextfloat`, `prevfloat`, `rand`, and
`bitstring`.

Working through Base's generic `AbstractFloat` fallbacks: `muladd`, `oneunit`,
`copysign`, `min`, `max`, `isfinite`/`isnan`/`isinf`, `sum`, `prod`, `norm`,
`hypot`, `cbrt`, and `round` with `RoundNearestTiesAway`/`RoundNearestTiesUp`/
`RoundFromZero`.

**Not supported, deliberately:** `sin`, `cos`, `exp`, `log`, `tanh`, and every
other transcendental function. The exactness argument that justifies "compute
in `Float64`, round once" covers `+ - * / sqrt` and nothing else
([Accuracy](accuracy.md)), so rather than shipping unverified
implementations, `LowFloat` lets them `MethodError`. If you hit one, the answer
is to reformulate — not to work around it by converting to `Float64` and back,
which silently computes the function at full precision.

**Not supported, no decision behind it:** `fma`, `rem`, `mod` and `div` raise
`ErrorException("… not defined for LowFloat{…}")` from Base's own generic
fallback. Nothing needs them in the code paths exercised so far; they would be
straightforward to add in the same compute-then-round pattern.

Note that `hypot` and `cbrt` *do* work, because Base builds them from `sqrt`
and `^` — and `^` carries no separate exactness proof. Treat results from
those two as "same pattern, no proof", like `^` itself.

## Format constants

The standard `AbstractFloat` queries all report the *format's* values, not
`Float64`'s:

```julia-repl
julia> using LowPrecisionChop

julia> T = lowfloattype(Float16Format);

julia> precision(T)
11

julia> Float64(eps(T)), Float64(floatmin(T)), Float64(floatmax(T))
(0.0009765625, 6.103515625e-5, 65504.0)

julia> Float64(typemin(T)), Float64(typemax(T))
(-Inf, Inf)
```

`nextfloat` and `prevfloat` step along the format's grid, including across
binade boundaries and through the subnormal region:

```julia-repl
julia> nextfloat(T(1.0))
1.0009765625

julia> prevfloat(T(1.0))
0.99951171875

julia> Float64(nextfloat(zero(T)))     # first subnormal
5.960464477539063e-8

julia> nextfloat(floatmax(T))
Inf
```

Stepping toward zero from a power of two correctly uses the *finer* grid of
the binade below, which is the case a naive implementation gets wrong:
`prevfloat(T(2.0))` is `1.9990234375`, not `1.998046875`. On a tiny format
the whole grid is short enough to walk:

```julia-repl
julia> S = lowfloattype(Format(3, 3));       # t=3, emax=3

julia> v = zero(S); out = Float64[];

julia> for _ in 1:12
           v = nextfloat(v)
           push!(out, Float64(v))
       end

julia> out
12-element Vector{Float64}:
 0.0625
 0.125
 0.1875
 0.25
 0.3125
 0.375
 0.4375
 0.5
 0.625
 0.75
 0.875
 1.0
```

## Promotion never escapes to `Float64`

This is the single most important design decision in the type, and it is
intentionally *inconvenient*:

```julia-repl
julia> using LowPrecisionChop

julia> T = lowfloattype(Float16Format);

julia> promote_type(T, Float64)
LowFloat{11, 15, RoundingMode{:Nearest}, true}
```

`promote_type(LowFloat, Float64)` resolves to the `LowFloat`, never to
`Float64`. If it went the other way, `LinearAlgebra.generic_norm2` — which
accumulates in `promote_type(Float64, T)` — would quietly compute
`norm(::Vector{LowFloat})` at full double precision, and your "low
precision" experiment would be measuring nothing. The same risk exists in
OrdinaryDiffEq's adaptive-step controller.

So reductions stay in low precision:

```julia-repl
julia> sum(T.([1.0, 2.0, 3.0])) isa T
true
```

Relatedly, `LowFloat` is deliberately **not** a `LinearAlgebra.BlasFloat`
and does not match Krylov.jl's `FloatOrComplex` fast path:

```julia-repl
julia> using LinearAlgebra

julia> T <: LinearAlgebra.BLAS.BlasFloat
false
```

That single fact is what routes every stdlib call to its generic fallback
instead of a BLAS `ccall`, and it is asserted directly in the test suite. A
library that *would* bypass it will raise a `MethodError` rather than
silently produce a `Float64` result.

## Generic algorithms

Because the interface is complete enough for the generic paths in
`LinearAlgebra`, factorizations work directly:

```julia-repl
julia> using LowPrecisionChop, LinearAlgebra

julia> BF16 = lowfloattype(BFloat16Format);

julia> A = BF16.([4.0 1.0; 1.0 3.0]);

julia> lu(A).U
2×2 Matrix{LowFloat{8, 127, RoundingMode{:Nearest}, true}}:
 4.0  1.0
 0.0  2.75

julia> qr(A).R isa Matrix{BF16}
true
```

For fp16 specifically, a `LowFloat` LU factorization is **bitwise identical**
to a native `Float16` LU — not merely close. That is the expected outcome,
because fp16's precision is comfortably inside the double-rounding-safe
region, and it reproduces Higham & Pranesh's own Table 6.2 result. It is
checked over 20 random 5×5 systems in `test/integration/linearalgebra/`, with
zero mismatches.

Krylov.jl solvers, MultiPrecisionArrays.jl iterative refinement,
OrdinaryDiffEq.jl integrators, and Flux.jl linear layers with Zygote
autodiff all work too — with specific caveats per package documented in
[Integration findings](integrations.md).

## Stochastic rounding and the RNG

A `LowFloat` whose `R` parameter is one of the stochastic modes rounds
randomly on every operation. Since `a + b` has nowhere to put an `rng`
argument, it draws from the ambient `Random.default_rng()`:

```julia
using LowPrecisionChop, Random

S = lowfloattype(BFloat16Format, RoundStochasticProportional)

Random.seed!(1234)
r1 = my_computation(S)

Random.seed!(1234)
r2 = my_computation(S)      # identical to r1
```

Consequences worth internalising:

- Reproducibility comes from seeding the default RNG, not from passing an
  RNG object. A local `Xoshiro` handed to your own `randn` calls does
  nothing for the arithmetic.
- This is a global side effect on the task-local RNG stream. Unrelated code
  drawing from the same stream will interleave with it.
- If you need an isolated RNG, use [`chop`](chop.md) with an explicit `rng`
  keyword instead of a stochastic `LowFloat`.

`rand` on a `LowFloat` type is a normal `rand` and *does* take an RNG:

```julia-repl
julia> using LowPrecisionChop, Random

julia> T = lowfloattype(Float16Format);

julia> rand(Xoshiro(42), T) == rand(Xoshiro(42), T)
true
```

## Bit-level inspection

`bitstring` shows the bits the value *would* have if stored natively in its
own format — 1 sign bit, `ceil(log2(emax+1)) + 1` exponent bits with
IEEE-style bias `emax`, then `t-1` significand bits:

```julia-repl
julia> using LowPrecisionChop

julia> T = lowfloattype(Float16Format);

julia> bitstring(T(1.0))
"0011110000000000"

julia> bitstring(T(-3.14159265358979))
"1100001001001000"

julia> bitstring(lowfloattype(Float8E4M3Format)(1.5))
"00111100"
```

For every named preset this is exactly the standard IEEE layout — the fp16
case is checked against native `Float16` over 5000 random bit patterns. For a
custom `emax` where `emax + 1` is not a power of two, the encoding is
well-defined but non-standard, because no standard layout exists for such a
format.

`Base.decompose` and `hash` agree with the underlying `Float64`, so
`LowFloat` works as a `Dict` or `Set` key.

## Performance notes

`LowFloat` arithmetic is a `Float64` operation plus a handful of `ldexp`
and `round` calls — no allocation, no dispatch on runtime values (`t` and
`emax` are type parameters and constant-fold), and no global state outside
the stochastic modes' RNG draw. `benchmark/run.jl` records scalar and array
timings; they are not asserted against thresholds.

What you will *not* get is BLAS throughput. Every matrix operation on
`Matrix{LowFloat}` goes through Julia's generic loops, which is the point —
a BLAS call would compute in `Float64` and defeat the simulation.
