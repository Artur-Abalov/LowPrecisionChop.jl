# Accuracy and double rounding

This package simulates a low-precision operation by doing it in `Float64`
and then rounding the result to the target format. That is two roundings
where hardware would do one — and two roundings do not always agree with
one. This page explains when they do, how to check, and what the package
does when they might not.

Read this before trusting a custom [`Format`](@ref). Every named preset is
already safe.

## The problem

Suppose you want to simulate `a + b` in fp16. The honest answer is the
fp16-representable value nearest to the exact real sum. What you compute is:

1. `s = a + b` in `Float64` — rounds the exact sum to 53 bits;
2. `chop(s, Float16Format)` — rounds that to 11 bits.

Almost always these give the same answer as rounding the exact sum directly
to 11 bits. But if step 1 nudges the value across a tie-breaking boundary of
step 2, the final result is off by one ulp of the target format. That is
*double rounding* error.

The escape is a precision gap. If the container has enough extra bits, step
1's error is too small to ever move the value across a step-2 boundary.

## The condition

Let `p₁` be the container's precision (53 for `Float64`, 24 for `Float32`,
both counting the hidden bit) and `t` the target format's precision. Then
`Float64`-then-round is provably exact for `+ - * / sqrt` when:

- **round-to-nearest family:** `2t + 2 ≤ p₁`
- **directed rounding** (`RoundUp`, `RoundDown`, `RoundToZero`):
  unconditionally, for any `t < p₁`

For a `Float64` container that means `t ≤ 25` under round-to-nearest, and
`t ≤ 52` under directed rounding. The asymmetry is real and is stated
explicitly in Higham & Pranesh (2019), p. C588: the "roughly twice as many
digits" requirement "is needed for round to nearest but not for directed
rounding".

The reason nearest needs twice the bits rather than just one extra: with
directed rounding the first rounding can only ever move the value in a known
direction and never past the neighbour it would have selected anyway. With
round-to-nearest, a value sitting almost exactly on a midpoint of the target
grid can be pushed to the wrong side of it, and ties-to-even then resolves
to the wrong neighbour. Guarding against that requires the container's ulp
to be smaller than the *tie window* at the target precision, not merely
smaller than the target ulp.

## Checking a format

[`issimulationexact`](@ref) implements the condition:

```julia-repl
julia> using LowPrecisionChop

julia> issimulationexact(11, RoundNearest)          # fp16: 2*11+2 = 24 ≤ 53
true

julia> issimulationexact(24, RoundNearest)          # fp32 target: 50 ≤ 53
true

julia> issimulationexact(25, RoundNearest)          # 52 ≤ 53 -- the last safe t
true

julia> issimulationexact(26, RoundNearest)          # 54 > 53
false

julia> issimulationexact(30, RoundUp)               # directed: always safe
true
```

Pass `containerprecision` when the container is not `Float64`:

```julia-repl
julia> issimulationexact(11, RoundNearest; containerprecision = 24)
true

julia> issimulationexact(12, RoundNearest; containerprecision = 24)
false
```

The stochastic modes and `RoundNearestTiesAway` are treated conservatively
as nearest-family. For the stochastic modes that is a judgement call rather
than a derivation — the paper's analysis does not cover them — and the
conservative side is the right one to err on.

Every named preset except `Float64Format` passes:

| preset | `t` | nearest-safe? |
|---|---|---|
| `Float8E4M3Format` | 4 | yes |
| `Float8E5M2Format` | 3 | yes |
| `Float16Format` | 11 | yes |
| `BFloat16Format` | 8 | yes |
| `TF32Format` | 11 | yes |
| `Float32Format` | 24 | yes |
| `Float64Format` | 53 | no — see below |

## What the package does about it

`chop` and `chop!` check the format against the container type of `x` on
every call. By default an unsafe combination produces a warning, emitted
once per distinct `(t, emax, round, container)` tuple:

```julia-repl
julia> using LowPrecisionChop: chop

julia> chop(1.0, Format(30, 127));
┌ Warning: Format (t=30, emax=127) with round mode RoundingMode{:Nearest}() is
│ not provably double-rounding-safe against a 53-bit container ...
```

`strict = true` turns it into an error instead:

```julia-repl
julia> chop(1.0, Format(30, 127); strict = true)
ERROR: ArgumentError: Format (t=30, emax=127) with round mode
RoundingMode{:Nearest}() is not provably double-rounding-safe against a
53-bit container (see `issimulationexact`, docs/src/spec.md §8): results may
occasionally disagree with the mathematically exact rounding at rare
boundary cases. Pass strict=false to proceed anyway, or choose a safer
format/round mode.
```

This is a deliberate divergence from MATLAB `chop`, which refuses such
formats outright. The position taken here is that a wide custom format is a
legitimate thing to experiment with, and the right response is to tell you
it isn't provably exact — not to forbid it, and not to stay quiet.

`LowFloat` does not re-check on every operation (that would cost a branch in
the hot path). Check the format once, with `issimulationexact` or a
`strict = true` `chop` call, before building a type from a custom format.

## What "not provably exact" actually costs

It is worth being precise about the size of the risk, because "not provably
safe" reads worse than it is:

- The failure mode is a **one-ulp difference in the target format**, at
  inputs that land within a container-ulp of a target-format midpoint.
- It is **rare** — it needs the exact result to sit in a window roughly
  `2^-53` wide around a `2^-t`-spaced midpoint.
- It is **not** a crash, an `Inf`, or an accumulating drift; each operation
  is still within one ulp of correct.

So an unsafe format is usable for most experimental purposes. What you
cannot do is claim your simulation *is* the format's arithmetic, or compare
bit-for-bit against real hardware in that format.

Conversely, when the condition *does* hold, the guarantee is strong and
worth using: a `LowFloat` fp16 LU factorization is bitwise identical to a
native `Float16` LU, over every input tested. Any divergence there would be
a genuine bug, not expected numerical noise — which makes it a usable
regression test. Higham & Pranesh report the same bitwise identity for their
own MATLAB implementation (Table 6.2).

## The `Float64Format` special case

`Float64Format` has `t = 53`, so `2·53 + 2 = 108 > 53` and
`issimulationexact` says `false`. Meanwhile `chop(x, Float64Format)` is a
literal identity map on every finite `Float64`.

Both statements are correct. The theorem's hypothesis is a precision *gap*
between container and target, and there is none here — so the theorem
doesn't apply, and `issimulationexact` correctly declines to certify it. The
practical consequences:

```julia-repl
julia> using LowPrecisionChop: chop

julia> chop(3.14159265358979, Float64Format) == 3.14159265358979    # (warns once)
true

julia> chop(1.0, Float64Format; strict = true)
ERROR: ArgumentError: Format (t=53, emax=1023) ...
```

The warning is about precision, so no combination of `subnormal`/`explim`
silences it. Suppress it with `Logging.with_logger(NullLogger())` if it is in
your way — or simply don't route identity operations through `chop`.

## Scope of the guarantee

The exactness argument covers `+`, `-`, `*`, `/` and `sqrt`. It does **not**
cover:

- `^` — implemented for `LowFloat` because Krylov.jl's `gmres!` needs it for
  Givens rotations. Same compute-then-round pattern, no separate proof.
- `floor`, `ceil`, `trunc`, `round`, `abs2`, `inv`, `ldexp` — implemented for
  interface completeness, same pattern, no separate proof.
- transcendental functions — **not implemented at all**, precisely because
  there is no proof and no appetite for shipping an unverified answer. See
  [Known limitations](known_limitations.md#Deliberate-scope-boundaries).

The derivation, with the full quotation from the paper and the corresponding
MATLAB source, is in [Specification](spec.md#8.-Double-rounding-correctness-condition).
