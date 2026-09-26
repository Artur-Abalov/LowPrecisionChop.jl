# Value-level rounding with `chop`

[`chop`](@ref LowPrecisionChop.chop) rounds a value — or every element of an
array — to a low-precision format, and gives you back the same Julia type
you passed in. It is the direct analogue of MATLAB `chop`, and the API to
use when you want to control exactly where rounding happens.

Remember the import: `using LowPrecisionChop: chop`
([why](getting_started.md#The-chop-name-collision)).

## Signature

```julia
chop(x, fmt::Format;
     round      = RoundNearest,
     subnormal  = true,
     explim     = true,
     flip       = false,
     p          = 0.5,
     rng        = Random.default_rng(),
     strict     = false)
```

`x` is a `Float64` or `Float32` scalar, or an `AbstractArray` of either.
There is no method for `Float16`, `BigFloat`, integers or rationals — the
kernel is defined for the two container types whose double-rounding
behaviour is characterized (see [Accuracy](accuracy.md)).

| keyword | meaning | see |
|---|---|---|
| `round` | which of the seven rounding modes to use | [Rounding modes](rounding.md) |
| `subnormal` | support gradual underflow, or flush to zero | [Formats](formats.md#Subnormals) |
| `explim` | enforce the format's exponent range at all | [Formats](formats.md#Exponent-limiting-(explim)) |
| `flip` | inject a random single-bit error after rounding | [Bit flips](rounding.md#Bit-flips) |
| `p` | probability that a flip actually fires | as above |
| `rng` | random source for stochastic rounding and flips | [Controlling the RNG](rounding.md#Controlling-the-RNG) |
| `strict` | error, rather than warn, on a format that is not provably double-rounding-safe | [Accuracy](accuracy.md) |

## Storage type is preserved

`chop` never changes the type of its argument. It returns a value of the
input type whose unused significand bits are zero:

```julia-repl
julia> using LowPrecisionChop

julia> using LowPrecisionChop: chop

julia> chop(3.14159265358979, Float16Format)
3.140625

julia> typeof(chop(3.14159265358979, Float16Format))
Float64

julia> chop(3.14159265f0, Float16Format)
3.140625f0

julia> typeof(chop(3.14159265f0, Float16Format))
Float32
```

This is what makes `chop` composable with ordinary `Float64` code: you can
drop it in around individual operations without changing any types.

The container type matters for correctness, though — rounding into a
`Float32` container leaves far less headroom for double rounding than a
`Float64` one. `issimulationexact(11, RoundNearest; containerprecision=24)`
is `true`, but `issimulationexact(12, RoundNearest; containerprecision=24)`
is `false`. `chop` checks against the actual container type of `x`.

## Arrays

Passing an array broadcasts elementwise and allocates a new array of the
same shape:

```julia-repl
julia> A = [1.0 2.5; 3.14159 1e-40];

julia> chop(A, Float16Format)
2×2 Matrix{Float64}:
 1.0      2.5
 3.14062  0.0
```

[`chop!`](@ref) does the same in place and returns the array it mutated:

```julia-repl
julia> B = copy(A);

julia> chop!(B, Float16Format) === B
true

julia> B
2×2 Matrix{Float64}:
 1.0      2.5
 3.14062  0.0
```

Because `Format` is broadcastable as a scalar, you can also write the
broadcast yourself when you want fusion with other operations:

```julia
C .= chop.(A .+ B, Float16Format)
```

The scalar kernel is non-allocating and type-stable (asserted in
`test/kernel_allocation.jl`), so elementwise rounding of a large array costs
one pass and one output allocation.

## Simulating an algorithm operation by operation

The paper's Simulation 3.1 is "convert operands to a wide format, do the
operation there, round the result back". With `chop` you write that
explicitly, which is what you want when only *some* operations should be in
low precision:

```julia
using LowPrecisionChop
using LowPrecisionChop: chop

# Inner product with low-precision products but a high-precision accumulator
# -- the classic mixed-precision pattern.
function dot_mixed(x, y, fmt)
    s = 0.0
    for i in eachindex(x, y)
        s += chop(x[i] * y[i], fmt)   # products rounded, sum in Float64
    end
    s
end

# Fully low-precision version: round the accumulator too.
function dot_low(x, y, fmt)
    s = 0.0
    for i in eachindex(x, y)
        s = chop(s + chop(x[i] * y[i], fmt), fmt)
    end
    s
end
```

If you want *every* operation rounded without writing it out, that is what
[`LowFloat`](lowfloat.md) is for.

!!! note "Rounding a whole kernel once is not the same simulation"
    Rounding only the final result of a matrix product — `chop(A * B, fmt)`
    — is the paper's Simulation 3.2, and it is *more accurate* than real
    low-precision hardware, not equivalent to it: the backward error bound
    becomes `2u_ℓ|A||B|`, independent of `n`, instead of `γ_ℓ(n)|A||B|`.
    That is a useful thing to measure, but it does not tell you what fp16
    hardware would produce. Round per operation (or use `LowFloat`) when you
    want hardware-faithful simulation. See
    [Specification](spec.md#Backward-error-bounds-(Simulation-3.2,-§3.2,-pp.-C588–C591)).

## Reproducing MATLAB `chop`

The mapping from MATLAB's `options` struct to this package's arguments is
one-to-one, with one structural difference: MATLAB keeps `options` as
persistent mutable state between calls, while here every setting is an
explicit argument with no global state.

| MATLAB | here |
|---|---|
| `options.format`, `options.params` | the `Format` argument |
| `options.round = 1..6` | `round = RoundNearest / RoundUp / RoundDown / RoundToZero / RoundStochasticProportional / RoundStochasticEqual` |
| `options.subnormal` | `subnormal = true/false` |
| `options.explim` | `explim = true/false` |
| `options.flip`, `options.p` | `flip = true/false`, `p` |
| `options.randfunc` | `rng = <any AbstractRNG>` |
| the `maxfraction` precision error | `strict = true` (see [Accuracy](accuracy.md)) |

Two deliberate differences to be aware of:

1. **`subnormal` defaults to `true` for every format**, including
   `BFloat16Format`. MATLAB `chop` defaults it to off for bfloat16
   specifically. Pass `subnormal = false` to match.
2. **`strict` defaults to `false`.** MATLAB raises a hard error when the
   target precision is too large for its container; here the default is a
   one-time warning, and `strict = true` restores the error. This is because
   the package deliberately permits formats MATLAB refuses — it just
   doesn't let them pass silently.

Golden-value parity for the four deterministic modes across five formats is
tested against 2720 rows of externally generated data; see
[Verification](verification.md).

## Full docstrings

See [`LowPrecisionChop.chop`](@ref) and [`chop!`](@ref) in the
[API reference](api.md).
