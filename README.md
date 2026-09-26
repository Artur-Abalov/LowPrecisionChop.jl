# LowPrecisionChop.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://Artur-Abalov.github.io/LowPrecisionChop.jl/)

Simulate low-precision floating-point arithmetic in Julia — fp8, fp16,
bfloat16, tf32, or any custom binary format — using hardware `Float64`
underneath. A Julia implementation of Higham & Pranesh's method and of the
semantics of their MATLAB `chop`.

Two APIs over one rounding kernel:

- **`chop`/`chop!`** — value-level rounding, MATLAB-`chop`-compatible. Round a
  `Float64`/`Float32` scalar or array to a low-precision format and get back
  the same storage type with the unused significand bits zeroed.
- **`LowFloat <: AbstractFloat`** — a real number type that computes in low
  precision. Hand it to generic Julia code and every operation rounds:
  `LinearAlgebra`, Krylov.jl, MultiPrecisionArrays.jl, OrdinaryDiffEq.jl and
  Flux.jl all work with it.

## Installation

```julia
using Pkg
Pkg.add("LowPrecisionChop")
```

Julia 1.10 or newer. The only dependency is the `Random` stdlib.

## Example

```julia
using LowPrecisionChop
using LowPrecisionChop: chop   # Base exports `chop` too -- see below

# Value-level rounding.
chop(3.14159265358979, BFloat16Format)                      # 3.140625
chop(3.14159265358979, Float16Format; round = RoundStochasticEqual)

# Type-level: a number that computes in bfloat16.
using LinearAlgebra
BF16 = lowfloattype(BFloat16Format)
lu(BF16.(randn(4, 4)))                                      # LU entirely in bfloat16

# Named presets, or roll your own.
chop(1.0, Format(6, 20))                                    # 6 significand bits, emax=20
```

Presets: `Float8E4M3Format`, `Float8E5M2Format`, `Float16Format`,
`BFloat16Format`, `TF32Format`, `Float32Format`, `Float64Format`.

## `Base.chop` name collision

`Base` already exports `chop` (string truncation), and this package exports a
different `chop` (float rounding). Plain `using LowPrecisionChop` therefore
leaves the unqualified name unusable — a hard binding conflict, not just
ambiguous dispatch. Use `using LowPrecisionChop: chop`, or qualify as
`LowPrecisionChop.chop(...)`. `chop!` is unaffected.

## Read this before using a custom format

Computing in `Float64` and rounding once is *exactly* the correctly-rounded
low-precision result for `+ - * / sqrt` — but only when the target precision
is small enough that the two roundings cannot compound (`2t + 2 ≤ 53` for
round-to-nearest; unconditional for directed rounding). Every named preset
satisfies this. A custom format may not, so the package tells you:
`issimulationexact(t, round)` reports it, and `strict = true` turns the
default warning into an error. Full treatment:
[Accuracy and double rounding](docs/src/accuracy.md).

## Comparison with related packages

| Package | Arbitrary formats | Stochastic rounding | Bit flips | Generic `AbstractFloat` | Enumerate/plot grid |
|---|---|---|---|---|---|
| **LowPrecisionChop** | ✅ up to `Float64` precision | ✅ | ✅ | ✅ `LowFloat` | ❌ by design |
| [MicroFloatingPoints.jl](https://github.com/goualard-f/MicroFloatingPoints.jl) | ✅ capped at `Float32` precision | ❌ | ❌ | ✅ `Floatmu` | ✅ |
| [BFloat16s.jl](https://github.com/JuliaMath/BFloat16s.jl) | ❌ bfloat16 only | ❌ | ❌ | ✅ `BFloat16` | ❌ |
| [StochasticRounding.jl](https://github.com/milankl/StochasticRounding.jl) | ❌ fp16/fp32/fp64 only | ✅ | ❌ | ✅ `Float16sr` etc. | ❌ |
| CPFloat (C, not Julia) | ✅ | ✅ | ✅ | N/A | ❌ |

When to prefer each one:
[comparison page](https://Artur-Abalov.github.io/LowPrecisionChop.jl/comparison/).

## Documentation

[Full documentation](https://Artur-Abalov.github.io/LowPrecisionChop.jl/)

- [Getting started](docs/src/getting_started.md) — install to first
  computation.
- Guides: [Formats](docs/src/formats.md) ·
  [Rounding modes](docs/src/rounding.md) ·
  [Value-level `chop`](docs/src/chop.md) ·
  [The `LowFloat` type](docs/src/lowfloat.md) ·
  [Accuracy](docs/src/accuracy.md)
- Reference: [Specification](docs/src/spec.md) ·
  [Verification](docs/src/verification.md) ·
  [Integration findings](docs/src/integrations.md) ·
  [Comparison](docs/src/comparison.md) ·
  [Known limitations](docs/src/known_limitations.md) ·
  [API](docs/src/api.md)

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

To cite the package itself, see [`CITATION.bib`](CITATION.bib).

## License

MIT — see [`LICENSE`](LICENSE). The rounding kernel is a translation of
[higham/chop](https://github.com/higham/chop) (BSD 2-clause, © Higham &
Pranesh).
