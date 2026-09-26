# LowPrecisionChop.jl

Simulate low-precision floating-point arithmetic in Julia — fp8, fp16,
bfloat16, tf32, or any custom binary format you can describe with a
significand width and an exponent range — using hardware `Float64`
underneath.

The package is a Julia implementation of the method in Higham & Pranesh,
*"Simulating Low Precision Floating-Point Arithmetic"* (SIAM J. Sci.
Comput. 41(5), 2019), and of the semantics of their MATLAB `chop` function.

## Two APIs over one rounding kernel

Everything in the package is built from a single rounding kernel, exposed
two different ways. They are co-equal: neither is a wrapper around the
other, and both produce bit-identical results for the same format and
rounding mode.

| | [`chop`](@ref LowPrecisionChop.chop) / [`chop!`](@ref) | [`LowFloat`](@ref) |
|---|---|---|
| What it is | a function that rounds values | a `<: AbstractFloat` number type |
| Works on | `Float64`/`Float32` scalars and arrays | anything generic Julia code accepts |
| Storage | the input type, low bits zeroed | `Float64` field, values on the format's grid |
| Rounds | when you call it | after every arithmetic operation |
| Use it to | quantize data, reproduce MATLAB `chop` results, simulate at chosen points | run an existing algorithm end-to-end in low precision |

```julia-repl
julia> using LowPrecisionChop

julia> using LowPrecisionChop: chop   # Base exports `chop` too -- see below

julia> chop(3.14159265358979, BFloat16Format)      # value-level rounding
3.140625

julia> BF16 = lowfloattype(BFloat16Format);        # type-level simulation

julia> using LinearAlgebra

julia> A = BF16.([4.0 1.0; 1.0 3.0]);

julia> lu(A).U                                     # factorized entirely in bfloat16
2×2 Matrix{LowFloat{8, 127, RoundingMode{:Nearest}, true}}:
 4.0  1.0
 0.0  2.75
```

## Installation

```julia
using Pkg
Pkg.add("LowPrecisionChop")
```

Julia 1.10 or newer. The only dependency is the `Random` stdlib;
[MicroFloatingPoints.jl](https://github.com/goualard-f/MicroFloatingPoints.jl)
is an optional weak dependency that enables one conversion helper.

## Three things to know before you start

**1. `chop` collides with `Base.chop`.** `Base` exports `chop` for string
truncation. Plain `using LowPrecisionChop` therefore leaves the bare name
`chop` unusable — Julia reports a binding conflict, because the two are
different generic functions that happen to share a name. Write
`using LowPrecisionChop: chop`, or qualify as `LowPrecisionChop.chop(...)`.
[Details](getting_started.md#The-chop-name-collision).

**2. Simulation is exact, but only up to a precision limit.** Computing in
`Float64` and rounding once gives *exactly* the correctly-rounded
low-precision result for `+ - * / sqrt` — provided the target precision is
small enough that the two roundings can't compound. All seven named presets
satisfy this. Custom formats with a wide significand may not, and the
package tells you rather than guessing: [`issimulationexact`](@ref), plus a
`strict` mode that errors instead of warning. See
[Accuracy and double rounding](accuracy.md).

**3. Transcendental functions are deliberately absent.** `sin`, `exp`,
`tanh` and friends have no method for `LowFloat`, because the exactness
argument above covers only `+ - * / sqrt`. You get a `MethodError`, never a
silently unverified answer. See [Known limitations](known_limitations.md).

## Where to go next

**Start here**

- [Getting started](getting_started.md) — install, the name collision, your
  first `chop`, your first `LowFloat` computation, and how to pick between
  them.

**Guides**

- [Formats](formats.md) — the `(t, emax)` model, the named presets, custom
  formats, subnormals, and exponent limiting.
- [Rounding modes](rounding.md) — all seven modes, plus what each one does
  at overflow and underflow.
- [Value-level rounding with `chop`](chop.md) — every keyword argument,
  array handling, and bit-flip injection.
- [The `LowFloat` type](lowfloat.md) — construction, the supported
  interface, promotion rules, and the stochastic-RNG caveat.
- [Accuracy and double rounding](accuracy.md) — why simulation is exact,
  when it stops being exact, and what to do about it.

**Reference**

- [Specification](spec.md) — the format and rounding semantics, derived
  and cited line by line from the MATLAB reference and the paper.
- [Verification](verification.md) — the oracles and test suite that back
  the correctness claims.
- [Integration findings](integrations.md) — results from running `LowFloat`
  through `LinearAlgebra`, Krylov.jl, MultiPrecisionArrays.jl,
  OrdinaryDiffEq.jl, and Flux.jl.
- [Comparison with related packages](comparison.md) — when to reach for
  MicroFloatingPoints.jl, BFloat16s.jl, StochasticRounding.jl, or CPFloat
  instead.
- [Known limitations](known_limitations.md) — every gap, in one place.
- [API reference](api.md).

## Citing

This package implements the method described in:

```bibtex
@article{higham2019simulating,
  author  = {Higham, Nicholas J. and Pranesh, Srikara},
  title   = {Simulating Low Precision Floating-Point Arithmetic},
  journal = {SIAM Journal on Scientific Computing},
  volume  = {41},
  number  = {5},
  pages   = {C585--C602},
  year    = {2019},
  doi     = {10.1137/19M1251308}
}
```

To cite the package itself, see `CITATION.bib` in the repository root.

## License

MIT. The rounding kernel is a translation of
[higham/chop](https://github.com/higham/chop) (`chop.m`, `roundit.m`),
BSD 2-clause, © Higham & Pranesh.
