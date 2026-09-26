# Release checklist

Registration is performed by the package author. This document is the
checklist for what's already been verified and what still needs doing.
Every item states exactly how it was checked, so it can be re-verified
rather than taken on faith.

## Package identity

- [x] **Name available.** `LowPrecisionChop` does not appear in
  `JuliaRegistries/General`'s `Registry.toml` (checked directly — fetched
  the raw file, grepped for `Chop`/`LowPrecision`/near-neighbors). Nearest
  neighbors by substring: `ZChop`, `Microfloats`, `MicroFloatingPoints`,
  `BFloat16s`, `StochasticRounding` — none collide. **Re-check before
  registering** (the registry changes daily):
  `curl -s https://raw.githubusercontent.com/JuliaRegistries/General/master/Registry.toml | grep -i lowprecisionchop`
  should return nothing.
- [x] **Name passes AutoMerge naming rules**, confirmed against
  `JuliaRegistries/RegistryCI.jl`'s `AutoMerge/src/guidelines.jl`: valid
  Julia identifier, starts uppercase, ASCII alphanumeric only, contains a
  lowercase letter, ≥5 characters (17), doesn't contain "julia" / start
  with "Ju" / end in "jl". Name-similarity (Damerau–Levenshtein /
  visual-distance) check against the *live* registry contents can only be
  run again at actual registration time.
- [x] **UUID valid and non-colliding.** `b3eee944-c02d-431a-a26d-36d2d3c755d8`,
  generated via `UUIDs.uuid4()` (proper RFC 4122 v4 bit pattern, required
  for new-package registration), checked against `LinearAlgebra` and
  `Random`'s stdlib UUIDs and a registry snapshot — no collision found.

## `Project.toml`

- [x] `name`, `uuid`, `authors`, `version = "0.1.0"` all present.
  `version = "0.1.0"` is one of the AutoMerge-accepted initial versions
  (`0.0.1`/`0.1.0`/`1.0.0`/`X.0.0`).
- [x] `[compat]` entry for `julia = "1.10"` — acceptable as written (no
  separate upper bound needed; Pkg's caret-default parsing gives
  `[1.10.0, 2.0.0)`, which already satisfies AutoMerge's "does not reach
  2.0" check).
- [x] `[compat]` entries present for every direct dependency: `Random = "1"`,
  `MicroFloatingPoints = "2"` (weak dep — matches its current latest
  registered version 2.1.1; **re-check this bound hasn't gone stale** if
  registration happens much later).
- [x] Zero non-stdlib hard dependencies in `[deps]` (`Random` is stdlib).
  `MicroFloatingPoints` is a `[weakdeps]` extension target only.
- [x] No `[extras]`/`[targets]` — `test/Project.toml` is the single test
  environment (the modern convention).

## Repository files

- [x] `LICENSE` present, MIT. MIT is auto-detected as OSI-approved by the
  registry's `LicenseCheck.jl`-based check.
- [x] No `src/` filename collisions or reserved names.
- [x] `.github/workflows/CI.yml` — matrix `{1.10 (LTS), 1 (stable),
  nightly} × {ubuntu,macos,windows}-latest`, plus separate `aqua-jet` and
  `docs` jobs. `nightly` failures don't fail the build
  (`continue-on-error`).
- [x] `.github/workflows/Documenter.yml` — builds and deploys docs on push
  to `main`/tags; **needs the `DOCUMENTER_KEY` secret set up** (see below —
  requires a real GitHub repo).
- [x] `.github/workflows/TagBot.yml` — current recommended content, from
  `JuliaRegistries/TagBot`'s own `example.yml`.
- [x] `.github/workflows/CompatHelper.yml` — standard current template
  (daily cron, `Pkg.Registry.add("General")` + `CompatHelper.main()`).

## Tests, quality gates, docs

- [x] `julia --project=. -e 'using Pkg; Pkg.test()'` — exits 0. Full suite:
  kernel tests (BigFloat oracle, exhaustive Float16, invariants, MATLAB
  transcription), public API tests (native contracts, `LowFloat` interface,
  `chop`/`chop!`, `Base.chop` collision), MicroFloatingPoints extension
  round-trip test, MATLAB-parity golden CSV, paper-experiment reproductions.
- [x] `julia --project=test -e 'using Aqua, LowPrecisionChop; Aqua.test_all(LowPrecisionChop)'`
  — exits 0, all 8 Aqua categories pass (ambiguities, unbound type params,
  undefined exports, project/test-project comparison, stale deps, compat
  bounds, piracy, persistent tasks).
- [x] `julia --project=test -e 'using JET, LowPrecisionChop; ...'` —
  `JET.report_package` reports 0 possible errors.
- [x] `julia --project=docs docs/make.jl` — builds successfully, **all
  doctests pass** (every exported symbol has a `jldoctest` example).
  `remotes=nothing` is currently set in `docs/make.jl` because no commit
  exists yet in this repo — **remove that line once the repo has real git
  history** so source-link URLs resolve against GitHub.
- [x] Scalar kernel non-allocating/type-stable, asserted directly in the
  test suite (`@allocated == 0`, `@inferred`).
- [x] All four integration suites pass standalone, each in its own
  isolated environment (`test/integration/{linearalgebra,krylov,mpa,sciml}/`).
  Not part of the main `Pkg.test()` run (slow, heavy external deps); run
  individually per [Integration findings](src/integrations.md).
- [x] `julia --project=benchmark benchmark/run.jl` — runs and produces
  timing output (recorded, not asserted).
- [x] `test/data/chop_golden.csv` exists (2720 rows, generated from pychop
  v0.6.1, provenance documented in `test/data/README.md`), and the parity
  test in `test/chop_golden.jl` covers all seven rounding modes: modes 1-4
  by exact-value comparison against the CSV, mode 7 against an independent
  BigFloat oracle (no external reference implements this package's own
  extension mode), modes 5-6 by explicit pointer to the property-based
  tests that actually cover them (a fixed golden CSV is not meaningful for
  cross-language stochastic RNG).
- [x] At least three Higham–Pranesh paper experiments are reproduced as
  tests with expected values cited by table number, in
  `test/paper_experiments.jl` and `test/integration/linearalgebra/runtests.jl`
  — see [Known limitations](src/known_limitations.md) for the two cells
  that don't match the published tables exactly.

## Documentation content

- [x] `README.md` — installation, motivating example, comparison table,
  citation block.
- [x] `CITATION.bib` — both the package itself and the Higham–Pranesh paper.
- [x] `docs/src/comparison.md` — names MicroFloatingPoints.jl, BFloat16s.jl,
  StochasticRounding.jl, and CPFloat, states plainly when to prefer each
  over this package (and vice versa).
- [x] `docs/src/spec.md` — full derived specification with citations.
- [x] `docs/src/integrations.md` — every integration finding, including the
  one unresolved API gap (MultiPrecisionArrays.jl's convenience
  constructor) and how it was worked around.
- [x] `docs/src/known_limitations.md` — every known gap, in one place.
- [x] The `Base.chop` name collision is documented in `chop`'s docstring,
  `README.md`, and `docs/src/index.md` — and it's tested, not just
  documented (a subprocess-based test in `test/chop_api.jl` confirms the
  exact failure mode).
- [x] The double-rounding caveat is documented in `README.md`,
  `docs/src/index.md`, `LowFloat`'s own docstring, and
  [Specification](src/spec.md) §8.

## What still needs doing

1. **Create the GitHub repository** at `Artur-Abalov/LowPrecisionChop.jl`
   (confirm this is the right owner/name) and push this repository's
   contents to it.
2. **Generate and configure the `DOCUMENTER_KEY` secret**: run
   `julia -e 'using DocumenterTools; DocumenterTools.genkeys()'` from a
   checkout of the pushed repo, add the public half as a repo deploy key
   with write access, add the base64 private half as the `DOCUMENTER_KEY`
   repository secret. Needed by both `Documenter.yml` and `TagBot.yml`
   (which reuses it for `ssh:`).
3. **Remove `remotes=nothing` from `docs/make.jl`** once the repo has real
   git history, so Documenter can resolve source-link URLs automatically.
4. **Trigger registration**: comment `@JuliaRegistrator register` on a
   commit in the pushed repo (not on any PR against `General` — Registrator
   opens that PR for you). New-package registrations wait **3 days** after
   AutoMerge criteria are met before merging (vs. 15 minutes for new
   versions) — confirmed from `General/README.md`.
5. **Re-run the name-availability and name-similarity checks** at
   registration time — the registry changes daily.
6. A `paper/` directory could be added later for a JOSS submission; nothing
   in the current structure blocks adding one.
