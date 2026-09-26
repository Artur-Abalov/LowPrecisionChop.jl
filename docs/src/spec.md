# Specification

This page specifies the format and rounding semantics LowPrecisionChop.jl
implements, and cites the source of every rule.

The sources are the reference MATLAB implementation,
[github.com/higham/chop](https://github.com/higham/chop) (`chop.m`,
`roundit.m`, `float_params.m`, `test_chop.m`, `test_roundit.m`,
`demo_harmonic.m`; BSD 2-clause), and the accompanying paper: Higham &
Pranesh, *"Simulating Low Precision Floating-Point Arithmetic"*, SIAM J. Sci.
Comput. 41(5), 2019, C585–C602
([doi:10.1137/19M1251308](https://doi.org/10.1137/19M1251308)). Both are
cross-checked against [inEXASCALE/pychop](https://github.com/inEXASCALE/pychop)
and arXiv:2504.07835.

Every claim below is either a citation of one of those sources, or is marked
**derived** — computed here from cited material rather than stated verbatim
anywhere — or is marked as a **package-specific decision** where no reference
behaviour exists.

---

## 1. Format specification: `(t, emax)`

`t` is the number of significand bits **including the hidden bit**; `emax` is
the maximum **unbiased** exponent for normal numbers. `emin = 1 - emax`, the
IEEE symmetric-range convention. Source: `chop.m`'s doc comment and
`float_params.m`.

```
emin  = 1 - emax                     # smallest normal exponent
xmin  = 2^emin                       # smallest positive normal
emins = emin + 1 - t                 # smallest subnormal exponent
xmins = 2^emins                      # smallest positive subnormal
xmax  = 2^emax * (2 - 2^(1-t))       # largest finite
u     = 2^(-t)                       # unit roundoff (half-ulp at 1.0, RN)
```

### Named presets

From `chop.m`'s format-string table:

| preset | `t` | `emax` | source comment |
|---|---|---|---|
| `fp8-e4m3` → [`Float8E4M3Format`](@ref) | 4 | 7 | "3 bits plus 1 hidden; 4-bit exponent" |
| `fp8-e5m2` → [`Float8E5M2Format`](@ref) | 3 | 15 | "2 bits plus 1 hidden; 5-bit exponent" |
| `fp16` → [`Float16Format`](@ref) | 11 | 15 | "10 bits plus 1 hidden; 5-bit exponent" |
| `bfloat16` → [`BFloat16Format`](@ref) | 8 | 127 | "7 bits plus 1 hidden; 8-bit exponent" |
| `fp32` → [`Float32Format`](@ref) | 24 | 127 | "23 bits plus 1 hidden; 8-bit exponent" |
| `fp64` → [`Float64Format`](@ref) | 53 | 1023 | "52 bits plus 1 hidden; 11-bit exponent" |

Two additions:

- [`TF32Format`](@ref) (`t=11`, `emax=127`) — **package-specific**. Not in
  MATLAB `chop` or the paper; defined from NVIDIA's published TF32 layout
  (10 explicit mantissa bits plus hidden bit, 8-bit exponent field). No
  external oracle exists, so it is verified against the BigFloat oracle only
  (§10).
- Custom `Format(t, emax)` for any `t ≥ 1`, `emax ≥ 1`. No validation is
  performed; degenerate parameters produce undefined behaviour rather than an
  error.

`chop.m` notes of `format='d'` that it is "intended to be used only with
`options.subnormal = 0`". That is advisory, not enforced there or here; it is
surfaced as a docstring note on [`Float64Format`](@ref).

Notional bit layout for `bitstring` uses `ne = ceil(log2(emax + 1)) + 1`
exponent bits with bias `emax` — **derived**, and equal to the IEEE width
exactly when `emax = 2^(ne-1) - 1`, which holds for every named preset.

---

## 2. Kernel parameters

The reference holds these in a mutable `options` struct with state
persisting across calls: an omitted or empty `options` argument on a later
call reuses the previous call's resolved struct. This package has **no global
mutable state** — every field is an explicit argument, a `Format`/
`RoundingMode` value, or a `LowFloat` type parameter.

| MATLAB field | here | legal values | default here |
|---|---|---|---|
| `format` + `params` | `Format{t,emax}` | `t ≥ 1`, `emax ≥ 1` | none — always explicit |
| `subnormal` | `S` type param / `subnormal::Bool` | `true`/`false` | `true` for **every** format |
| `round` | `R` type param / `round::RoundingMode` | see §3 | `RoundNearest` |
| `flip`, `p` | `chop` keywords only, never a type param | `Bool`, `Real` | `false`, `0.5` |
| `explim` | `chop` keyword only (§5) | `true`/`false` | `true` |
| `randfunc` | `rng::AbstractRNG` keyword | any `AbstractRNG` | `Random.default_rng()` |
| `maxfraction` error | `strict::Bool` keyword (§8) | `true`/`false` | `false` (warn once) |

!!! warning "Known divergence: the bfloat16 `subnormal` default"
    MATLAB `chop` sets `subnormal = 0` by default for the `bfloat16` format
    specifically, while defaulting it on for other formats. **This package
    defaults `subnormal = true` uniformly, including for
    `BFloat16Format`.** The semantics of each setting are identical; only the
    default differs. Pass `subnormal = false` explicitly to reproduce
    MATLAB's bfloat16 default. Also listed in
    [Known limitations](known_limitations.md).

---

## 3. Rounding modes

MATLAB `chop` implements exactly **six** modes. From `roundit.m` and the
paper's Table 5.1 ("Rounding modes supported by MATLAB function `chop`"):

| `round` | MATLAB behaviour | exact rule | here |
|---|---|---|---|
| 1 | nearest, ties to **even** (default) | `round()`, then subtract 1 from ties whose floor is even | `RoundNearest` |
| 2 | toward +∞ | `ceil` | `RoundUp` |
| 3 | toward −∞ | `floor` | `RoundDown` |
| 4 | toward zero | sign-directed `floor`/`ceil` | `RoundToZero` |
| 5 | stochastic, probability ∝ distance | skips exact integers | [`RoundStochasticProportional`](@ref) |
| 6 | stochastic, equal probability | 50/50, skips exact integers | [`RoundStochasticEqual`](@ref) |

pychop's Higham-style port (`FaultChop`/`Chop_`) implements the same six and
nothing else in its numeric dispatch. (A separate, non-Higham class in
pychop — `Chop`/`LightChop_` — adds other modes under its own independent
10-mode numbering, unrelated to MATLAB `chop`'s.)

### Mode 7: `RoundNearestTiesAway` — package-specific

This package adds a seventh mode, round-to-nearest with **ties away from
zero**, exposed as Julia's own `RoundNearestTiesAway`. It has **no MATLAB or
paper equivalent**, and therefore:

- no external golden data exists for it — it is absent from
  `chop_golden.csv` by design, rather than being filled in with fabricated
  "expected" values;
- its overflow, underflow and subnormal-boundary behaviour could not be
  copied from a reference. The decision taken here is to group it with the
  nearest family: `{1, 7}` share a tie rule at the subnormal boundary,
  `{1, 6, 7}` share the overflow boundary, and it uses the stricter
  round-to-nearest double-rounding bound `2t + 2 ≤ p₁` (§8) rather than the
  looser directed-mode bound. The justification is symmetry — ties-away and
  ties-even are both nearest-rounding variants — not derivation.
- it is verified against the package's independent BigFloat oracle (§10).

---

## 4. Overflow is coupled to the rounding mode

There is no orthogonal overflow flag in the reference: `chop.m`'s overflow
`switch` is keyed directly on `round`, gated by `explim` (§5).

| `round` | `x` toward +overflow | `x` toward −overflow |
|---|---|---|
| 1, 6, **7** | `x ≥ xboundary → +Inf` | `x ≤ -xboundary → -Inf` |
| 2 | `x > xmax → +Inf` | `x < -xmax →` saturate to `-xmax` |
| 3 | `x > xmax →` saturate to `xmax` | `x < -xmax → -Inf` |
| 4, 5 | `x > xmax →` saturate to `xmax` | `x < -xmax →` saturate to `-xmax` |

with

```
xboundary = 2^emax * (2 - 2^(-t))
```

the round-to-nearest midpoint between `xmax` and `2^(emax+1)`; the paper
cites IEEE 754-2019 p. 16 for this threshold. Confirmed against `test_chop.m`
lines 273–308 ("IEEE 754-2019, page 27: rule for rounding to infinity"),
which hardcodes exactly this table for all four deterministic modes as
executable assertions. Mode 7's row is the package-specific grouping of §3.

Consequences: there is no separate `overflow` option, because fixing `R`
fixes overflow entirely. Directed modes (2, 3, 4, 5) never produce the
*opposite*-direction infinity — only nearest-family modes can overflow in
both directions. Note the split here puts mode 6 with nearest and mode 5 with
toward-zero, which is not the grouping used in §6 or §8; that asymmetry is
the reference's and is preserved.

---

## 5. `explim` — exponent limiting

`chop.m`: "If `explim = 0` then `emax` is ignored, so overflow, underflow, or
subnormal numbers will be produced only if necessary for the data type of X."

Two independently gated effects, both skipped when `explim = 0`:

1. **Subnormal routing.** With `explim = 1`, exponents below `emin` are
   rounded on the fixed subnormal grid rather than at their own local
   exponent (gradual underflow). With `explim = 0` this branch never
   triggers, and the full `t` significand bits apply uniformly at every
   magnitude.
2. **Range clamping.** The entire §4 overflow table and the §6 underflow
   table live inside `if fpopts.explim` in `chop.m`. With `explim = 0`,
   `chop` never emits `±Inf` from magnitude, never saturates, and never
   flushes to zero — the only remaining limits are the container type's own.

`explim` is a keyword argument on `chop`/`chop!` (default `true`). It is
deliberately **not** a `LowFloat` type parameter: unlike `t`, `emax`, `R` and
`S`, it changes whether range limiting happens at all rather than describing
a property of the stored values. `LowFloat` always behaves as `explim = true`
— a format that ignores its own `emax` is an exploration mode of `chop`, not
a well-defined stored format.

### Derived: the subnormal scale factor

`chop.m` computes the subnormal case's scale via `t1 = t - (emin - e)` and
`scale = t1 - 1 - e`. **Derived result:** those two expressions collapse
algebraically to the constant `-emins = -(emin + 1 - t)` for every `e` in
`[emins, emin)` — the formula never actually depended on `e`.

This package therefore uses that fixed scale for **every** magnitude below
`emin`, not only within `[emins, emin)`. `chop.m`'s literal branch falls
through to the *local*-exponent formula below `emins`, and that difference is
not cosmetic: rounding on a finer, magnitude-dependent grid first and only
then comparing against the coarser fixed cutoff is itself a double-rounding
step, and it flips the round-to-nearest answer at boundary cases. This was
found by exhaustive fp16 testing — two subnormal fp16 operands whose exact
product sits a few parts in 10⁴ above `xmins/2` were incorrectly flushed to
zero by the literal translation, disagreeing with both native `Float16` and
the correctly-rounded BigFloat value.

---

## 6. Subnormals and the underflow boundary

Let `min_rep = subnormal ? xmins : xmin`. For `|c| < min_rep`, behaviour is
mode-dependent:

| `round` | rule below `min_rep` |
|---|---|
| 1, **7** (nearest) | round to `min_rep` if `\|c\|` exceeds `min_rep/2`, else 0. With `subnormal = false` the comparison is `≥`, so an exact tie rounds **up** — asymmetric with the ties-to-even rule in the normal range. With `subnormal = true` it is strict `>`. |
| 2 (+∞) | `+min_rep` if `0 < c < min_rep`; negatives flush to 0 |
| 3 (−∞) | mirror of 2 |
| 4, 5, 6 | flush to 0 unconditionally |

Verified against `test_chop.m` lines 318–472 (underflow across all four
deterministic modes × both `subnormal` settings), ported directly into
`test/kernel_matlab_transcribed.jl`. Mode 7's row is the package-specific
grouping of §3.

---

## 7. Bit flip (`flip`, `p`)

Applied in `roundit.m` **after** rounding, to the integer-scaled significand
`y` rather than the final float: for each element draw `u ~ Uniform(0,1)`; if
`u ≤ p` (default 0.5), flip bit `b = randi(t-1)` (1-indexed, `1` to `t-1`) of
`|y|` by XOR, then reapply the sign. `t` is read from `options.params(1)` at
call time. Confirmed against `test_roundit.m` lines 90–113, including a "zero
can be flipped too" case.

Here it is a `chop`/`chop!` keyword applied after the kernel, not a format
property, and it is not available on `LowFloat`.

---

## 8. Double-rounding correctness condition

The reference enforces a precision cap as a hard error:

```matlab
if fpopts.round == 1
   maxfraction = isa(x,'single') * 11 + isa(x,'double') * 25;
else
   maxfraction = isa(x,'single') * 23 + isa(x,'double') * 52;
end
if (fpopts.params(1) > maxfraction)
   error('Precision of the custom format must be at most %d ...');
end
```

confirmed executable in `test_chop.m` lines 567–582: `t = 12` errors for a
`single` container under `round = 1` (`maxfraction` 11), and `t = 26` errors
for a `double` container under `round = 1` (`maxfraction` 25).

**Derived general form**, with `p₁` the container precision (24 for
`Float32`, 53 for `Float64`, both including the hidden bit):

- round-to-nearest (`round = 1`): safe iff `t ≤ ⌊(p₁ - 2)/2⌋`, i.e.
  **`2t + 2 ≤ p₁`**. For `p₁ = 53`: `t ≤ 25`.
- directed and stochastic modes: safe iff **`t + 1 ≤ p₁`**. For `p₁ = 53`:
  `t ≤ 52`.

[`issimulationexact`](@ref) implements exactly this, treating the stochastic
modes and mode 7 conservatively as nearest-family.

**Proof sketch** (the standard double-rounding argument, consistent with the
result attributed to Figueroa 1995 and restated in Higham–Pranesh): let `x`
be real, `fl₁` the container's round-to-nearest operator (unit roundoff
`2^(1-p₁)/2`), and `fl_t` the target's (unit roundoff `2^(1-t)/2`). Single
rounding gives `fl_t(x)`; double rounding gives `fl_t(fl₁(x))`. These
coincide for every `x` iff the first rounding's worst-case error can never
cross a tie-breaking boundary of the second — the first error, relative bound
`2^(1-p₁)/2`, must be strictly smaller than half the tie-interval width at
precision `t`. That yields `2^(1-p₁) ≤ 2^(-t)`, i.e. `p₁ ≥ t + 1` for
directed rounding, tightening to `p₁ ≥ 2t + 2` for ties-to-even because a
double-rounded tie can resolve to the wrong even neighbour.

The paper (p. C588, §3.1) states that `sqrt` shares the identical threshold:

> "for the operations of addition, subtraction, multiplication, division, and
> square root rounding first to fp32 or fp64 and then to bfloat16 or fp16
> gives the same result as rounding directly to bfloat16 or fp16. This is
> shown by results in [14], [47], [50], which essentially require that the
> format used for the first rounding has a little more than twice as many
> digits in the significand as the target format (this condition is needed
> for round to nearest but not for directed rounding)."

That parenthetical is the authority for the asymmetry above. It also shows
`chop.m`'s `maxfraction` cap for `round ≠ 1` is an implementation sanity
bound (target must have strictly fewer bits than container), not a
mathematically necessary double-rounding condition.

This compute-in-`Float64`-then-round design is the paper's **Simulation 3.1**
(§3.1, p. C588): "simulate every scalar operation by converting the operands
to fp32 or fp64, carrying out the operation in fp32 or fp64, then rounding
the result back to the target format".

For the user-facing treatment of all this, see
[Accuracy and double rounding](accuracy.md).

### Backward-error bounds (Simulation 3.2, §3.2, pp. C588–C591)

The paper also analyses a *kernel-level* strategy — round only the final
result of a whole matrix operation, not every scalar op:

- Matrix multiply `C = AB`, low precision with unit roundoff `u_ℓ`, high
  precision `u_h`. Computing entirely in low precision gives
  `|C - Ĉ| ≤ γ_ℓ(n)|A||B|` (eq. 3.5). Computing in high precision and
  rounding once gives the much tighter `|C - C̃| ≤ 2u_ℓ|A||B|` (eq. 3.3),
  **independent of `n`**, provided `n ≤ u_ℓ / (u_h(1 + 2u_ℓ))` (eq. 3.4) —
  for fp16 target in an fp64 container that bound is `4.3×10¹²`, "which
  covers all n of current practical interest" (p. C590).
- Triangular solve `Tx = b`: componentwise backward error `|ΔT_h| ≤ 2u_ℓ|T|`
  (eq. 3.8) under the same condition, and forward error
  `‖x - x̃‖∞/‖x‖∞ ≲ 2u_ℓ` (eq. 3.11) when `n·cond(T) ≲ u_ℓ/u_h` (eq. 3.10) —
  notably **independent of `cond(T)`** in the leading term, unlike genuine
  low-precision computation (eq. 3.9).

The implication: a `chop!`-based kernel-level simulation is generally *more
accurate* than real low-precision hardware, not merely similar to it.
`LowFloat` arithmetic therefore rounds element by element (Simulation 3.1)
rather than rounding only a whole `A*B` result, so that it faithfully
simulates hardware instead of silently simulating something better. Table 6.2
confirms this is also what MATLAB's own `chop`-based tooling does, which makes
it a valid comparison target.

---

## 9. `AbstractFloat` interface contract for `LowFloat`

Derived from an inventory of Julia's `LinearAlgebra` stdlib, Krylov.jl,
IterativeSolvers.jl, MultiPrecisionArrays.jl, and
OrdinaryDiffEq.jl/SciMLBase.jl.

### Must implement — no generic Base fallback exists

| method | why | citation |
|---|---|---|
| `Base.float(x) = x` | no `AbstractFloat(x::AbstractFloat)` fallback anywhere in Base | `base/float.jl:378` |
| `Base.decompose` | `hash` has no generic fallback for arbitrary `AbstractFloat`; without it, `Dict`/`Set` keyed on `LowFloat` `MethodError`s | `base/hashing.jl:139-175` |
| `eps`, `floatmin`, `floatmax`, `precision` | the generic `eps(x::AbstractFloat)` method is built from these | `base/float.jl:983,1013,1037,807` |
| `ldexp`, `exponent`, `nextfloat`, `copysign` | needed by the `eps(x)` fallback and by `LinearAlgebra`'s `reflector!` | `base/float.jl:983`, `generic.jl:1758` |
| `zero`, `one`, `oneunit`, `+ - * / muladd inv`, `abs`, `abs2`, `< == isless iszero signbit`, `sqrt` | called directly inside `generic_lufact!`, `qrfactUnblocked!`, `_generic_matmatmul!` | `lu.jl:152-235`, `qr.jl:229-240`, `matmul.jl:1028-1040` |

### Must actively avoid

- **`promote_rule(LowFloat, Float64)` must never resolve to `Float64`.**
  `LinearAlgebra.generic_norm2` accumulates in `promote_type(Float64, T)`
  (`generic.jl:558`); were that to resolve to `Float64`,
  `norm(::Vector{LowFloat})` would silently compute at full double precision
  — a silent-bypass failure that defeats the entire purpose of the type. The
  same risk exists in `OrdinaryDiffEqCore`'s adaptive-step controller
  datatype (`solve.jl:14-17`). This package resolves the pair *to the
  `LowFloat`*, so mixed expressions stay in low precision.
- **`LowFloat` must never be a `LinearAlgebra.BlasFloat`/`BLAS.BlasFloat`,
  nor match Krylov.jl's `FloatOrComplex{T} where T<:AbstractFloat` fast
  path.** As long as it isn't, every BLAS/LAPACK `ccall` path is gated away
  from it by the stdlib's own `where {T<:BlasFloat}` dispatch — confirmed at
  `lu.jl:89-93`, `qr.jl:291-293`, `matmul.jl:791-806`, `generic.jl:1690`, and
  Krylov's `krylov_utils.jl:309-345`. This is the single most important
  invariant behind the package's correctness claim, and it is asserted
  directly in the test suite.

### Hard-failure traps found in practice

- `Base.FastMath.sqrt_fast` has no generic `AbstractFloat` fallback
  (`base/fastmath.jl:163,310` — `FloatTypes` is `Union{Float16,Float32,Float64}`
  only), breaking OrdinaryDiffEq's array-valued `ODE_DEFAULT_NORM`
  (`DiffEqBase.jl/src/common_defaults.jl:54-60`) unless
  `Base.FastMath.sqrt_fast(x::LowFloat) = sqrt(x)` is defined. It is.
- `MultiPrecisionArrays.jl`'s public constructors (`MPArray`, `mplu`) are
  hardcoded to `Float64`/`Float32` input (`Structs4MP/MPArray.jl:24,47`),
  with no generic outer constructor for `AbstractArray{<:AbstractFloat}`
  despite the `MPArray` struct itself being fully generic. Workaround in
  [Integration findings](integrations.md#MultiPrecisionArrays.jl).

---

## 10. Verification oracles

Summarised here; the full inventory, including what each test file covers, is
in [Verification](verification.md).

- **`test_chop.m` (602 lines) and `test_roundit.m` (136 lines)** are a
  deterministic symbolic oracle: every assertion is closed-form (`xmin`,
  `xmax`, `xmins`, powers of two, `eps(single(y))`), so all of it is
  computable directly in Julia without running MATLAB. Ported as `@test`
  cases rather than as data rows.
- **`chop_golden.csv`** — 2720 rows generated from pychop v0.6.1's
  higham/chop-compatible port, covering modes 1–4 across the five formats
  pychop shares with this package, both `subnormal` settings.
- **BigFloat oracle** — an independent 256-bit re-implementation used for
  everything with no external oracle: `TF32Format`, `Float64Format`, and mode
  7.
- **Exhaustive fp16** — every `Float16` bit pattern compared against native
  `Float16` arithmetic.
- **pychop caveat**: pychop's NumPy-backend `round_to_nearest` helper draws
  `np.random.randint(low=0, high=1, ...)`, which is half-open `[0,1)` and so
  always returns 0 — its `flip` fires 100% of the time regardless of `p`.
  pychop is therefore **not** used as an oracle for `flip`/`p`;
  `test_roundit.m`'s own proportion checks are used instead.

---

## 11. Registry compliance

Verified against `Registry.toml` and RegistryCI.jl's AutoMerge guidelines:
the package name passes all `NewPackage` naming checks (17 characters, ASCII,
starts uppercase, contains a lowercase letter, no "julia"/"Ju"/"jl" collision
pattern); `julia = "1.10"` is a valid compat entry as written; MIT is
auto-detected as OSI-approved. The full checklist with citations is in
`docs/RELEASING.md` in the repository root.
