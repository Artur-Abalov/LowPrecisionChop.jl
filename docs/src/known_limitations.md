# Known limitations

One place for every known gap, discrepancy and scope boundary. Nothing here is
buried in a source comment only: if it affects correctness, compatibility, or
what you can rely on, it is listed here.

## Unresolved discrepancies against the published paper

- **Table 6.4, harmonic series in fp16 with round-toward-+∞.** This package's
  reproduction terminates in 13911 terms; the paper reports 13912. A direct
  trace confirms the sum becomes `Inf` at term 13910 and is unchanged at term
  13911 — internally consistent, but off by one from the published table for
  this one cell. Every other deterministic cell in Tables 6.3 and 6.4 (13 of
  14) matches exactly. Traced detail in `test/paper_experiments.jl`.
- **Table 6.5, subnormal-vs-flush effect on a sum of squares in fp16.** The
  exact published digits could not be reproduced under either plausible
  reading of the per-operation rounding sequence (chopping the whole division
  once, versus chopping the squaring and the division separately — the latter
  collapses the entire subnormal-vs-flush distinction the experiment
  demonstrates, which is itself informative, but neither matches the table).
  The *qualitative* result does reproduce: subnormal support strictly improves
  accuracy, by a margin that grows with `n`. The original MATLAB code for this
  specific experiment is not available to disambiguate the operation order.

## Divergences from MATLAB `chop`

Both are deliberate, and both are the only places behaviour differs.

- **`subnormal` defaults to `true` for every format.** MATLAB `chop` defaults
  it to *off* for bfloat16 specifically. Pass `subnormal = false` explicitly
  to match. The semantics of each setting are identical.
- **An over-wide custom format warns rather than erroring by default.**
  MATLAB refuses any format whose precision exceeds its `maxfraction` cap.
  Here the default is a one-time warning, and `strict = true` restores the
  hard error. The position is that experimenting with a wide format is
  legitimate, and the right response is to say it isn't provably exact — not
  to forbid it, and not to stay quiet.

## Third-party API gaps (not bugs in this package)

- **MultiPrecisionArrays.jl**: `mplu()` and `MPArray()`'s convenience
  constructors are hardcoded to `Float64`/`Float32` and `MethodError` on
  `LowFloat`, even though the underlying `MPArray` struct is fully generic. A
  verified workaround exists — construct `MPArray` through its generic field
  constructor. See [Integration findings](integrations.md#MultiPrecisionArrays.jl).
- **Krylov.jl**: `atol`/`rtol` keyword arguments are typed to the solve's
  element type. A bare `Float64` literal raises a `TypeError`; pass a
  `LowFloat` value instead.

## Deliberate scope boundaries

- **No transcendental functions.** `sin`, `cos`, `exp`, `log`, `tanh` and the
  rest are not implemented for `LowFloat`, because the double-rounding
  exactness argument covers only `+ - * / sqrt`
  ([Accuracy](accuracy.md#Scope-of-the-guarantee)). Code that needs one gets a
  `MethodError` rather than a silently unverified answer. Confirmed in
  practice: Flux `Dense` layers with a `tanh` activation fail this way.
- **`fma`, `rem`, `mod` and `div` are not implemented for `LowFloat`.**
  Unlike the transcendentals, this is not a principled exclusion — nothing in
  the code paths exercised so far needs them, and they would be straightforward
  to add in the same compute-then-round pattern. They currently raise
  `ErrorException("… not defined for LowFloat{…}")` from Base's generic
  fallback rather than a `MethodError`.
- **`hypot` and `cbrt` work, but through `^`.** Base builds them from `sqrt`
  and `^`, and `^` has no separate exactness proof, so their results carry the
  same "same pattern, no proof" caveat as `^` itself. They are not blocked,
  but they are not covered by the `+ - * / sqrt` guarantee either.
- **No GPU/CUDA support.** The kernel is written to be GPU-friendly
  (branch-light, non-allocating, no global state), but this has never been
  tested on a GPU and no CUDA-specific code exists.
- **No fixed-point or integer quantization, posits, logarithmic or decimal
  formats.** Binary floating-point via `(t, emax)` only.
- **No multi-threading in the rounding kernel.**
- **No float enumeration, plotting, or real-line visualization** — that is
  [MicroFloatingPoints.jl](https://github.com/goualard-f/MicroFloatingPoints.jl)'s
  territory. See [Comparison](comparison.md).
- **No BLAS acceleration, ever.** `LowFloat` is deliberately not a
  `BlasFloat`, so matrix operations run Julia's generic loops. This is the
  mechanism that keeps the simulation honest, not an optimization oversight.

## Design characteristics worth knowing about

- **Stochastic-mode `LowFloat` arithmetic has no explicit RNG.**
  `chop(x, fmt; round = RoundStochasticEqual, rng = ...)` takes an `rng`
  keyword; the arithmetic operators cannot, since infix syntax has only two
  operand slots. A stochastic `LowFloat` therefore draws from the ambient
  `Random.default_rng()` on every operation. Reproducibility requires
  `Random.seed!(seed)` before the computation, and this is a **global side
  effect** — concurrent code drawing from the same task-local stream will
  interleave with it. Use `chop` with an explicit `rng` when you need
  isolation.
- **`Base.chop` name collision.** `Base` exports `chop` for string
  truncation. Plain `using LowPrecisionChop` leaves the unqualified name
  completely unusable — a hard binding conflict between two different generic
  functions, not merely ambiguous dispatch. Use `using LowPrecisionChop: chop`
  or qualify as `LowPrecisionChop.chop(...)`. `chop!` is unaffected.
- **`RoundNearestTiesAway` (mode 7) has no external oracle.** It is a
  package-only extension to MATLAB `chop`'s six modes. Its overflow,
  underflow and subnormal-boundary policy — grouped with the nearest-family
  modes — is a design choice made here by symmetry, not derived from any
  reference, and is verified only against the package's BigFloat oracle.
- **`TF32Format` is not in MATLAB `chop` or the paper.** Defined from NVIDIA's
  published layout; BigFloat-oracle-verified only.
- **`Float64Format` fails its own [`issimulationexact`](@ref) check**
  (`2·53 + 2 = 108 > 53`) even though it is mathematically an identity map, so
  `chop(x, Float64Format; strict = true)` throws and the default path emits a
  one-time warning. This is correct per the derivation — the theorem needs a
  precision gap, and there is none — but it surprises people who reach for
  `Float64Format` expecting it to always succeed. See
  [Accuracy](accuracy.md#The-Float64Format-special-case).
- **Custom `Format(t, emax)` does no input validation.** `t < 1` or
  `emax < 1` produce undefined or broken behaviour rather than a clear error.
- **`bitstring` uses a non-standard layout for exotic `emax`.** The exponent
  field width is `ceil(log2(emax+1)) + 1`, which coincides with IEEE only when
  `emax = 2^(k-1) - 1`. True for every named preset; not for an arbitrary
  custom `emax`, where no standard layout exists to match.

## Test and verification coverage gaps

- **`chop_golden.csv` covers 5 of the 7 named presets.** `TF32Format` and
  `Float64Format` have no pychop equivalents, so they are absent — both are
  BigFloat-oracle-only anyway.
- **The BFloat16s.jl native-type contract test has never actually run.** It is
  a `@test_skip` in every environment used during development, since
  BFloat16s.jl was never installed alongside this package. The claim in
  [Comparison](comparison.md#BFloat16s.jl) that the two agree rests on the
  shared `(8, 127)` format definition, not a live cross-check.
- **No benchmarking against alternative packages.** `benchmark/run.jl`
  measures this package's own operations in isolation; it does not compare
  against MicroFloatingPoints.jl, BFloat16s.jl, or CPFloat.
- **Extreme format edges are not specifically stress-tested.** Very small `t`
  (e.g. `t = 1`), or `emax` at or near 0, are covered only incidentally by
  whatever the property-based and BigFloat-oracle tests happen to sample. There
  is no dedicated suite for degenerate parameters.
- **`Base.round` for `RoundNearestTiesAway`, `RoundNearestTiesUp` and
  `RoundFromZero` relies on Base's generic `AbstractFloat` fallback** (built
  from `floor`/`ceil`) rather than a `LowFloat`-specific method. That is
  intentional — a dedicated method would ambiguously overlap Base's own
  generic one, which Aqua flags — but it means those three modes are less
  directly tested than `:Nearest`/`:Up`/`:Down`/`:ToZero`, which do have
  dedicated methods.
- **Array-valued `rand(rng, LowFloatType, dims...)` is not explicitly
  tested.** Only the scalar sampler is defined and tested; array sampling
  relies on Base's generic machinery built on top of it.

## Repository and release status

- **GitHub Pages must be enabled manually.** The `Documenter.yml` workflow
  builds the docs and pushes them to the `gh-pages` branch successfully, but
  publishing that branch is a repository setting, not something the workflow
  can do: **Settings → Pages → Source: "Deploy from a branch" → branch
  `gh-pages`, folder `/ (root)`**. Until that is set, every documentation URL
  returns 404 even though the content is on the branch.
- **There is no `stable` documentation URL yet.** Documenter only publishes
  `stable/` once a release tag exists; right now the site has only `dev/`, and
  the root URL redirects there. Links that hardcode `/stable/` will 404 until
  the first tagged release. README links point at `/dev/` for this reason.
- **The package is not registered** in the Julia General registry, and no
  release has been tagged. See `docs/RELEASING.md` in the repository root for
  the full checklist.
- **`TagBot.yml` needs a `DOCUMENTER_KEY` secret** if you want tagged releases
  to trigger a docs rebuild. The `Documenter.yml` push to `gh-pages` works
  without it (the workflow grants `contents: write` and passes
  `GITHUB_TOKEN`), so this only matters for the TagBot path.
