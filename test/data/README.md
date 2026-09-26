# Golden test data provenance

## `chop_golden.csv`

Generated from **pychop v0.6.1** (`pychop.faultchop.FaultChop`, its
higham/chop-compatible port — not the separate `Chop`/`LightChop_` class,
which has different, non-MATLAB rounding-mode numbering), NumPy 2.0.2,
Python 3.9.13, via `test/data/generate_golden.py`. Covers rounding modes
**1–4 only** (deterministic, exact-value comparison — see below for why
modes 5–7 are handled differently), 5 formats shared with pychop's own
preset table (`fp8-e4m3`, `fp8-e5m2`, `fp16`, `bfloat16`, `fp32` — pychop
has no `tf32`/`Format64` presets), both `subnormal` settings, ~68
representative inputs per `(format, subnormal, mode)` combination (fixed
edge cases — zero, ±1, `xmin`/`xmins`/`xmax` and their neighbors,
`±1e±300` — plus 40 seeded-random values per combination). 2720 data rows
total.

MATLAB/Octave are not used as a source — neither was available when this
was generated; pychop is used instead (see [Specification](../../docs/src/spec.md)
§10 for the reasoning and a known caveat about pychop's own `flip`
implementation).

### Why not all seven rounding modes are in this CSV

- **Modes 1–4** (deterministic: nearest, +∞, −∞, toward-zero): in this
  file, exact-value comparison, tested in `test/chop_golden.jl`.
- **Mode 7** (`RoundNearestTiesAway`, this package's own MATLAB-parity
  extension — see [Specification](../../docs/src/spec.md) §3): **has no
  external oracle at all** — not MATLAB, not pychop, not any reference
  implementation, since it doesn't exist in any of them. Putting
  fabricated "expected" values in a file named `chop_golden.csv` would
  misrepresent them as having an external source when they don't. Verified
  instead directly against an independent BigFloat (256-bit)
  re-implementation in `test/chop_golden.jl` — the same oracle standard
  used in `test/kernel_bigfloat_oracle.jl`.
- **Modes 5–6** (stochastic): a fixed CSV of "expected" (input → output)
  pairs is fundamentally not meaningful across languages here — pychop's
  NumPy RNG and Julia's `Xoshiro` are different algorithms, so "the same
  seed" does not produce "the same draws," and there is no way to encode a
  stochastic *rule* (as opposed to one arbitrary realization of it) as a
  golden CSV row. Correctness for these modes is instead covered by
  property-based tests already in the suite: `test/kernel_invariants.jl`
  (reproducibility from a seed, within Julia's own RNG) and
  `test/kernel_matlab_transcribed.jl` (proportion/distance checks
  transcribed directly from `test_roundit.m`'s own executable assertions).

### Other sources

1. **`test_chop.m` / `test_roundit.m`** (github.com/higham/chop, BSD
   2-clause, Higham & Pranesh 2019) — every assertion in both files is
   closed-form (powers of two, `xmin`/`xmax`/`xmins`, `eps(single(1))`,
   etc.), computable directly in Julia rather than needing to run MATLAB.
   Ported directly as `@test` cases in `test/kernel_matlab_transcribed.jl`
   rather than as CSV rows, since they're symbolic, not tabular data.
   Covers rounding modes 1-4 (deterministic) plus the flip mechanism's
   proportion checks (`test_roundit.m` lines 90-113).
2. **pychop** — as above for `chop_golden.csv`'s modes 1-4. **Known
   caveat, do not use for `flip`/`p` coverage**: pychop's NumPy-backend
   `round_to_nearest` helper calls `np.random.randint(low=0, high=1, ...)`,
   which is half-open `[0,1)` and therefore always returns `0` — its `flip`
   fires 100% of the time regardless of `p`, diverging from MATLAB's
   continuous-uniform `≤ p` semantics.

## Regenerating

```
python3 -m venv /tmp/pychop_venv
/tmp/pychop_venv/bin/pip install pychop numpy
/tmp/pychop_venv/bin/python3 test/data/generate_golden.py
```

Overwrites `chop_golden.csv` in place. Re-pin the pychop/numpy versions in
this file if you regenerate with newer ones.
