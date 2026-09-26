# Comparison with related packages

This page names every package this project overlaps with or drew on, and says
plainly which to reach for when. None of them is worse than this one — they
make different trade-offs for different jobs.

## At a glance

| | arbitrary formats | stochastic rounding | bit flips | generic `AbstractFloat` | enumerate / plot the grid |
|---|---|---|---|---|---|
| **LowPrecisionChop** | ✅ up to `Float64` precision | ✅ | ✅ | ✅ `LowFloat` | ❌ by design |
| MicroFloatingPoints.jl | ✅ capped at `Float32` precision | ❌ | ❌ | ✅ `Floatmu` | ✅ |
| BFloat16s.jl | ❌ bfloat16 only | ❌ | ❌ | ✅ `BFloat16` | ❌ |
| StochasticRounding.jl | ❌ fp16/fp32/fp64 only | ✅ | ❌ | ✅ `Float16sr` etc. | ❌ |
| CPFloat (C, not Julia) | ✅ | ✅ | ✅ | N/A | ❌ |

## MicroFloatingPoints.jl

[goualard-f/MicroFloatingPoints.jl](https://github.com/goualard-f/MicroFloatingPoints.jl)
is the closest existing package. It also provides a parametric,
arbitrary-`(exponent, significand)` low-precision float type
(`Floatmu{szE,szf}`), and this package ships an extension converting to it
([`matchingfloatmutype`](@ref), active as soon as you
`using MicroFloatingPoints`).

The scope difference is deliberate on both sides. MicroFloatingPoints.jl
**caps precision at `Float32`** precisely so that its `Float64`-based
simulation can never suffer double-rounding error (citing Rump 2016), and its
real focus is exploring a format's float grid: enumerating every
representable value, plotting spacing on the real line, that kind of analysis.

LowPrecisionChop takes the opposite trade: **arbitrary formats are allowed**,
including ones where `Float64` simulation is *not* provably exact — with that
territory surfaced through [`issimulationexact`](@ref) and a `strict` mode
rather than hidden. And it deliberately does **no** grid enumeration or
plotting; that is MicroFloatingPoints.jl's territory.

**Reach for MicroFloatingPoints.jl** when you want to see or reason about a
format's actual grid (enumerate, plot, study spacing), or when you only need
precisions safely below `Float32` and want the strongest correctness guarantee
with zero configuration.

**Reach for LowPrecisionChop** when you want MATLAB-`chop`-compatible
semantics (seven rounding modes including stochastic, subnormals on/off,
mode-coupled overflow policy, bit flips), a type that participates in generic
Julia numerics with real third-party packages
([Integration findings](integrations.md)), or formats up to `Float64`'s own
precision.

## BFloat16s.jl

[JuliaMath/BFloat16s.jl](https://github.com/JuliaMath/BFloat16s.jl) provides
exactly one format — hardware bfloat16 — as a fast, minimal, self-contained
type with no configuration surface: round-to-nearest only, no stochastic
rounding, no subnormal switch, no `chop` semantics.

**Reach for BFloat16s.jl** when bfloat16 is all you need and you want the
fastest, simplest bfloat16 arithmetic available.

**Reach for LowPrecisionChop** when you need bfloat16 *alongside* other
formats, or need bfloat16 with stochastic rounding or bit-flip injection.

This package's [`BFloat16Format`](@ref) (`Format(8, 127)`) uses the same
format definition as BFloat16s.jl's `BFloat16` and reproduces its
round-to-nearest behaviour. Be aware, though, that the cross-check test for
this is a `@test_skip` — BFloat16s.jl has never actually been installed
alongside this package, so the claim rests on the shared `(8, 127)` definition
rather than a live comparison. See
[Known limitations](known_limitations.md#Test-and-verification-coverage-gaps).

## StochasticRounding.jl

[milankl/StochasticRounding.jl](https://github.com/milankl/StochasticRounding.jl)
provides stochastic-rounding variants of the three *hardware* IEEE formats
(`Float16sr`, `Float32sr`, `Float64sr`), each backed by genuine hardware
arithmetic at that precision with stochastic rounding applied to the result.

**Reach for StochasticRounding.jl** when you want stochastic rounding on a
real IEEE hardware format with hardware-native (not simulated) performance for
those exact three precisions.

**Reach for LowPrecisionChop** when you want stochastic rounding — or any of
the other five modes — on formats hardware doesn't provide (bfloat16, tf32,
fp8, arbitrary custom `(t, emax)`), or want to compare the *same* rounding
behaviour across many formats within one API.

## CPFloat

Fasi & Mikaitis, *"CPFloat: A C Library for Simulating Low-Precision
Arithmetic"*, ACM TOMS 49(2), 2023
([doi:10.1145/3585515](https://doi.org/10.1145/3585515)), is the fast C
implementation of the same simulation approach, using direct bit manipulation
of the IEEE representation rather than the scale/round/rescale arithmetic used
here.

This package's kernel design was informed by reading CPFloat's strategy but
**does not wrap or link against it** — no C dependency, no BinaryBuilder step,
pure Julia throughout, in keeping with the zero-non-stdlib-dependency
constraint.

**Reach for CPFloat directly** if you need C-level performance from a
non-Julia codebase, or specifically want the bit-manipulation implementation
strategy.

**Reach for LowPrecisionChop** if you are in Julia and want a native
`AbstractFloat` type that participates in Julia's own dispatch and generic
algorithms rather than sitting behind a C boundary. The usual reason to reach
for a C library from Julia — avoiding allocation and dispatch overhead —
largely doesn't apply: the scalar kernel here is verified non-allocating and
type-stable.

## The MATLAB original

[higham/chop](https://github.com/higham/chop) is the reference this package
implements. If you are working in MATLAB, use it — this package exists to make
the same semantics available in Julia, with the two behavioural differences
listed in [the `chop` guide](chop.md#Reproducing-MATLAB-chop): `subnormal`
defaults to `true` for every format here (MATLAB defaults it off for
bfloat16), and an over-wide custom format warns by default instead of erroring
(`strict = true` restores the error).
