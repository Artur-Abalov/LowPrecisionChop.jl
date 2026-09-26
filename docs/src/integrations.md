# Integration findings

The claim that [`LowFloat`](@ref) "participates in generic Julia numerics" is
only worth something if it has been tried. This page records what happened
when it was run through five real numerical packages: what worked, what
needed adding, and where a third-party API simply doesn't accommodate a
custom float type.

Where a target package's API cannot take a `LowFloat`, that is recorded as a
limitation of that API — not silently worked around by converting to
`Float64`, which would defeat the point.

Each suite lives in its own isolated environment
(`test/integration/<name>/Project.toml`) and runs independently of
`Pkg.test()`, so a broken or uninstallable third-party dependency cannot break
the core test suite.

## Summary

| package | status |
|---|---|
| `LinearAlgebra` (stdlib) | works, no gaps; fp16 LU is bitwise identical to native `Float16` |
| Krylov.jl | works; one API-usage note about `atol`/`rtol` typing |
| MultiPrecisionArrays.jl | works via the lower-level API; convenience constructors are `Float64`/`Float32`-only |
| OrdinaryDiffEq.jl | works after three genuine interface gaps were found and fixed |
| Flux.jl | works for linear layers, including Zygote autodiff; `tanh` and other transcendentals are out of scope |

## LinearAlgebra (stdlib)

**Suite: `test/integration/linearalgebra/`. Fully working, no gaps.**

Generic `lu!` and `qr!` work directly on `Matrix{LowFloat}`. Because
`LowFloat` is deliberately not a `LinearAlgebra.BLAS.BlasFloat`, both dispatch
to the generic `generic_lufact!`/`qrfactUnblocked!` paths, and every scalar
operation inside the factorization is rounded to the target format.

The headline result: **an fp16 `LowFloat` LU factorization is bitwise
identical to a native `Float16` LU** — 20 random 5×5 trials, zero mismatches.
This is the expected outcome, not a lucky one: fp16's `t = 11` is comfortably
inside the double-rounding-safe region (`2·11 + 2 = 24 ≤ 53`, see
[Accuracy](accuracy.md)), so the guarantee is *zero* divergence rather than a
small tolerated one. Any future divergence here would indicate a real bug.
Higham & Pranesh report the same bitwise identity for their own
implementation (Table 6.2).

A hand-written, BLAS-free GMRES was also implemented directly against
`LowFloat`'s arithmetic and comparison interface, and converges on a
well-conditioned bfloat16 system — confirming the interface is complete enough
for iterative methods written from scratch, not just for stdlib code.

## Krylov.jl

**Suite: `test/integration/krylov/`. Fully working, one API-usage note.**

`LowFloat` matches Krylov's generic
`FloatOrComplex{T} where T<:AbstractFloat` dispatch branch rather than its
`BLAS.BlasFloat`-gated fast path, for every `k*` helper (`kdot`, `knorm`,
`kaxpy!`, …). There is no silent BLAS bypass.

`cg` (SPD, bfloat16) and `gmres` (nonsymmetric, fp16) both converge within
their iteration budgets on well-conditioned problems. A golden CG iteration
count is pinned as a regression guard (`GOLDEN_CG_NITER = 5` for the specific
seed, problem and tolerances in the test file).

!!! note "API-usage note, not a bug"
    Krylov's `atol`/`rtol` keyword arguments are typed to match the solve's
    element type exactly. Passing a bare `Float64` literal raises a
    `TypeError`:

    ```julia
    cg(A, b; rtol = 1e-2)              # TypeError
    cg(A, b; rtol = LowFloat16(1e-2))  # fine
    ```

This integration is also what found the missing `^` method — Krylov's
`gmres!` uses it internally for Givens rotations.

## MultiPrecisionArrays.jl

**Suite: `test/integration/mpa/`. The flagship use case works, but only
through the lower-level API.**

`mplu(A::Matrix{LowFloat}; ...)` fails with a `MethodError`. `MPArray`'s
convenience constructors are hardcoded to `AbstractMatrix{Float64}` and
`AbstractMatrix{Float32}`, even though the underlying `MPArray` struct is
fully generic over `{TW,TF,TR}<:AbstractFloat`. **This is a limitation of
MultiPrecisionArrays.jl's public API, not of `LowFloat`.**

**Workaround**: construct `MPArray` through its generic field constructor,
bypassing the convenience wrapper, then proceed normally:

```julia
MPA = MPArray(AH, AL, residual, sol, onthefly)
mplu!(MPA)
x = MPA \ b
```

With that, mixed-precision iterative refinement behaves exactly as it should:

- on a well-conditioned 20×20 system with an fp16 factorization, IR converges
  to a residual and solution error both below `1e-6` — i.e. full `Float64`
  accuracy — in well under 15 iterations;
- on a deliberately ill-conditioned system (`cond(A) ≈ 3×10⁹`), the same fp16
  factorization correctly **stagnates**: the final residual stays above `1e-3`
  and the solution error above `1e-2`. It does not silently converge to a
  falsely small residual, which is the failure mode that matters here.

## OrdinaryDiffEq.jl

**Suite: `test/integration/sciml/`. Works after three genuine `LowFloat`
interface gaps were found and fixed.**

Running real solvers surfaced three methods Base has no generic
`AbstractFloat` fallback for, which the package now implements:

1. `round(x, ::RoundingMode)`, `floor`, `ceil`, `trunc` — used internally for
   step-count bookkeeping *even in fixed-step (`adaptive=false`) mode*.
2. `(::Type{<:Integer})(x::LowFloat)` — needed for `Int(round(...))`-style
   conversions in the same bookkeeping path.
3. `Base.FastMath.sqrt_fast` — OrdinaryDiffEq's array-valued
   `ODE_DEFAULT_NORM` uses it, and Base defines it only for
   `Float16`/`Float32`/`Float64`. Fixed as
   `Base.FastMath.sqrt_fast(x::LowFloat) = sqrt(x)`.

None of these is covered by the double-rounding proof, which is scoped to
`+ - * / sqrt`. They follow the same compute-then-round pattern for interface
completeness, and that is stated rather than assumed.

With those three additions:

**Non-stiff, `y' = -y`, `Euler()`, fixed `h = 0.001`, bfloat16 — reproduces
the paper's stagnation phenomenon exactly.** For this ODE the relative Euler
update per step is exactly `h` regardless of `y`'s magnitude. Since
`h = 0.001` is below half of bfloat16's unit roundoff (`eps/2 ≈ 0.0039`),
round-to-nearest rounds the update to precisely zero at *every* step: the
trajectory freezes at `y = 1.0` for all 384 steps
(`length(unique(vals)) == 1`, verified exactly). Stochastic rounding under the
identical setup has no such failure mode — 373 of 594 steps produce distinct
values, and the trajectory reaches `y ≈ 0.073` (true value `≈ 0.050`) instead
of freezing.

**Stiff, `y' = -50(y - 1)`, `Rosenbrock23()`, bfloat16** — converges to within
0.05 of the true steady state for both rounding modes. This confirms
`LowFloat` works through an *implicit* solver too, which needs a linear solve
per step, routed through the same generic `lu!` path validated in the
LinearAlgebra suite.

The stiff test problem deliberately avoids `cos`/`sin`/`exp` of a `LowFloat`
argument. Transcendental functions have no double-rounding derivation here, so
the problem was chosen to need only `+ - * /` rather than adding unverified
methods to make a test pass.

**A reproducibility gap was found and fixed here too.** `LowFloat`'s
arithmetic operators have no slot for an `rng`, so stochastic-mode arithmetic
draws from the ambient `Random.default_rng()` — and a local `Xoshiro` handed to
a test's own `randn` calls does nothing to control it. The fix is
`Random.seed!(seed)` before each solve or training run, verified to give
byte-identical output across two runs. This is a permanent characteristic of
operator overloading, not a bug to fix in the type; it is documented in
`LowFloat`'s own docstring and in [the guide](lowfloat.md#Stochastic-rounding-and-the-RNG).

## Flux.jl

**Suite: `test/integration/sciml/` (shares the environment with
OrdinaryDiffEq). Works, with one scope boundary: linear layers only.**

`Dense` layers with a transcendental activation fail — `tanh(::LowFloat)` is a
`MethodError`. That is the intended behaviour, not a gap: no double-rounding
derivation exists for transcendentals, so they are not implemented.

Using `identity` activation sidesteps it and still exercises everything that
matters: `Dense` construction, the forward pass, **Zygote reverse-mode
autodiff tracing through `LowFloat`'s `+`/`-`/`*` methods**, `Flux.setup`, and
`Flux.update!`. A training loop with `Descent` runs with loss decreasing
monotonically.

**The stagnation-vs-stochastic-rounding phenomenon reproduces in a real
training loop**, not just in the ODE case. Training a one-parameter linear
model at `lr = 0.0001` — chosen so the weight update falls below bfloat16's
resolution relative to the weight — round-to-nearest **freezes the weight at
its initial value for all 200 steps** (`length(unique(weights)) == 1`), while
stochastic rounding keeps updating it (more than 100 distinct values over the
same 200 steps).

## Interface gaps found this way

Every item below was found by running real third-party code, not by static
inspection, and all are now implemented in `src/lowfloat.jl`:

| gap | found via | fix |
|---|---|---|
| `^(x, n)` | Krylov.jl `gmres!` (Givens rotations) | compute in `Float64`, round once |
| `round`, `floor`, `ceil`, `trunc` | OrdinaryDiffEq step-count bookkeeping | same pattern |
| `(::Type{<:Integer})(x::LowFloat)` | OrdinaryDiffEq's `Int(round(...))` | direct `T(x.val)` |
| `Base.FastMath.sqrt_fast` | OrdinaryDiffEq's array-valued default norm | `= sqrt(x)` |

None carries a separate double-rounding correctness proof. They follow the
same pattern as `+ - * / sqrt`, but that pattern's exactness argument
([Specification §8](spec.md#8.-Double-rounding-correctness-condition)) is
scoped to those five operations only.

Transcendental functions were deliberately **not** added. Both the stiff-ODE
test and the Flux test were designed around that boundary — a linear stiff
problem instead of a `cos` forcing term, `identity` activation instead of
`tanh` — rather than adding functions with no correctness derivation just to
make a test pass.
