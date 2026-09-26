# Rounding modes

Seven rounding modes are available. Six of them are exactly MATLAB `chop`'s
modes 1–6; the seventh is an extension specific to this package.

The mode does more than pick a neighbour for an inexact value — in the
reference design it *also* determines what happens at overflow and at the
underflow boundary. There is no separate "overflow policy" setting, by
design; fixing the rounding mode fixes all three behaviours.

## The seven modes

| MATLAB `round` | this package | behaviour |
|---|---|---|
| 1 (default) | `RoundNearest` | nearest, ties to even |
| 2 | `RoundUp` | toward `+Inf` |
| 3 | `RoundDown` | toward `-Inf` |
| 4 | `RoundToZero` | truncate toward zero |
| 5 | [`RoundStochasticProportional`](@ref) | round up with probability equal to the distance to the lower neighbour |
| 6 | [`RoundStochasticEqual`](@ref) | round up or down with probability 1/2 |
| — | `RoundNearestTiesAway` | nearest, ties away from zero — **no MATLAB equivalent** |

`RoundNearest`, `RoundUp`, `RoundDown`, `RoundToZero` and
`RoundNearestTiesAway` are Julia's own `Base` rounding modes; the two
stochastic modes are defined by this package.

Pass one with the `round` keyword to `chop`, or as the third parameter of a
`LowFloat` type:

```julia-repl
julia> using LowPrecisionChop

julia> using LowPrecisionChop: chop

julia> x = 1.0 + 2.0^-12 + 2.0^-13    # sits between two fp16 neighbours
1.0003662109375

julia> chop(x, Float16Format; round = RoundNearest)
1.0

julia> chop(x, Float16Format; round = RoundUp)
1.0009765625

julia> chop(x, Float16Format; round = RoundDown)
1.0

julia> chop(x, Float16Format; round = RoundToZero)
1.0
```

```julia-repl
julia> T = lowfloattype(Float16Format, RoundUp)
LowFloat{11, 15, RoundingMode{:Up}, true}
```

!!! warning "`lowfloattype`'s third argument"
    `lowfloattype(fmt, round, subnormal)` takes `subnormal` *positionally*
    after `round`, and it defaults to `true`. Write
    `lowfloattype(Float16Format, RoundUp)` for subnormals on,
    `lowfloattype(Float16Format, RoundUp, false)` for off.

## `RoundNearestTiesAway` has no external oracle

Mode 7 is this package's own addition. MATLAB `chop` does not have it, the
paper does not describe it, and no reference implementation provides it — so
there is no external golden data for it, and its overflow and underflow
behaviour could not be copied from anywhere.

The choice made here is to group it with the nearest-family modes: it uses
the same overflow boundary as `RoundNearest`, the same tie rule at the
subnormal boundary, and the same (stricter) double-rounding threshold. That
is a design decision by symmetry — ties-away and ties-even are both
nearest-rounding variants — not a derivation from the reference. It is
verified only against the package's independent BigFloat oracle. See
[Specification](spec.md#3.-Rounding-modes).

## Stochastic rounding

The two stochastic modes round *up or down at random* instead of
deterministically. Exactly representable values are left alone in both
modes — only genuinely inexact results are randomized.

- [`RoundStochasticProportional`](@ref) (mode 5) rounds up with probability
  equal to how far the value sits from its lower neighbour. It is unbiased:
  the expected result equals the exact value.
- [`RoundStochasticEqual`](@ref) (mode 6) rounds up with probability 1/2
  regardless of distance. It is biased, but it is the cheaper rule and is
  sometimes what hardware implements.

### Why it matters: stagnation

Round-to-nearest has a failure mode that stochastic rounding does not: if
every increment is smaller than half an ulp of the accumulator, *every*
increment rounds away and the computation freezes. Euler's method on
`y' = -y` with `h = 0.001` in bfloat16 does exactly that — `h` is below half
of bfloat16's `eps`, so the update `y - h·y` rounds back to `y` at every
step:

```julia
using LowPrecisionChop, Random

nearest    = lowfloattype(BFloat16Format)
stochastic = lowfloattype(BFloat16Format, RoundStochasticProportional)

function euler(T; steps = 200, h = 0.001)
    y, hT = T(1.0), T(h)
    vals = [Float64(y)]
    for _ in 1:steps
        y = y - hT * y
        push!(vals, Float64(y))
    end
    vals
end

length(unique(euler(nearest)))              # 1   -- frozen at y = 1.0
Random.seed!(1)
v = euler(stochastic)
length(unique(v))                           # 42  -- still moving
v[end]                                      # 0.83984375
```

The same phenomenon reproduces in a Flux training loop, where
round-to-nearest freezes a weight for all 200 steps while stochastic
rounding keeps updating it. See [Integration findings](integrations.md#Flux.jl).

### Controlling the RNG

`chop` takes an explicit `rng` keyword, so stochastic rounding through the
function API is fully reproducible and thread-local:

```julia-repl
julia> using LowPrecisionChop, Random

julia> using LowPrecisionChop: chop

julia> chop(1.0003662109375, Float16Format; round = RoundStochasticEqual, rng = Xoshiro(1))
1.0009765625

julia> chop(1.0003662109375, Float16Format; round = RoundStochasticEqual, rng = Xoshiro(1))
1.0009765625
```

`LowFloat` **cannot** take an RNG: `a + b` has no argument slot for one.
Stochastic-mode `LowFloat` arithmetic therefore draws from the ambient
`Random.default_rng()`. To reproduce a run, seed the default RNG before it:

```julia
Random.seed!(1234)
result = my_computation(stochastic_lowfloat_inputs)
```

This is a permanent consequence of operator overloading, not a bug awaiting
a fix — and it means stochastic `LowFloat` arithmetic has a global side
effect on the task-local RNG stream. If you need isolation, use `chop` with
an explicit `rng` instead.

## Overflow behaviour is part of the mode

When a value exceeds the format's range and `explim = true`, what happens
depends on the rounding mode. Directed modes never produce an infinity in
the direction they don't round toward:

| mode | `x` above `+xmax` | `x` below `-xmax` |
|---|---|---|
| `RoundNearest`, `RoundNearestTiesAway`, `RoundStochasticEqual` | `+Inf` once `x ≥ xboundary` | `-Inf` once `x ≤ -xboundary` |
| `RoundUp` | `+Inf` | saturate to `-xmax` |
| `RoundDown` | saturate to `+xmax` | `-Inf` |
| `RoundToZero`, `RoundStochasticProportional` | saturate to `+xmax` | saturate to `-xmax` |

where `xboundary = 2^emax · (2 - 2^-t)`, the round-to-nearest midpoint
between `xmax` and `2^(emax+1)`, per IEEE 754-2019's rule for rounding to
infinity. In fp16 (`xmax = 65504`):

```julia-repl
julia> using LowPrecisionChop: chop

julia> chop(70000.0, Float16Format; round = RoundNearest)
Inf

julia> chop(70000.0, Float16Format; round = RoundUp), chop(-70000.0, Float16Format; round = RoundUp)
(Inf, -65504.0)

julia> chop(70000.0, Float16Format; round = RoundDown), chop(-70000.0, Float16Format; round = RoundDown)
(65504.0, -Inf)

julia> chop(70000.0, Float16Format; round = RoundToZero), chop(-70000.0, Float16Format; round = RoundToZero)
(65504.0, -65504.0)
```

Note the mode pairing here is not the same as in the nearest/directed split
elsewhere: `RoundStochasticEqual` overflows like `RoundNearest`, while
`RoundStochasticProportional` saturates like `RoundToZero`. That asymmetry
is in the reference implementation and is preserved deliberately.

## Underflow behaviour is part of the mode too

Below the smallest representable magnitude (`xmins` with subnormals, `xmin`
without), the mode decides again:

| mode | behaviour below the threshold |
|---|---|
| `RoundNearest`, `RoundNearestTiesAway` | round to the threshold if `abs` exceeds half of it, else zero (with subnormals off, an exact half rounds *up*) |
| `RoundUp` | positives become `+min`, negatives become zero |
| `RoundDown` | negatives become `-min`, positives become zero |
| `RoundToZero`, both stochastic modes | flush to zero |

```julia-repl
julia> using LowPrecisionChop: chop

julia> chop(2.0^-30, Float16Format; round = RoundUp)      # tiny positive
5.960464477539063e-8

julia> chop(2.0^-30, Float16Format; round = RoundDown)
0.0
```

The exact rules, with citations to the reference implementation's own test
assertions, are in [Specification](spec.md#6.-Subnormals-and-the-underflow-boundary).

## Bit flips

Independently of the rounding mode, `chop` can inject a random single-bit
error into the rounded significand — a crude model of a soft fault. Pass
`flip=true`, optionally with `p` (the per-value probability, default `0.5`):

```julia-repl
julia> using LowPrecisionChop, Random

julia> using LowPrecisionChop: chop

julia> x = chop(3.14159265358979, Float16Format)
3.140625

julia> rng = Xoshiro(9);

julia> [chop(x, Float16Format; flip = true, p = 1.0, rng = rng) for _ in 1:4]
4-element Vector{Float64}:
 3.142578125
 3.142578125
 2.140625
 3.171875
```

The flipped bit is chosen uniformly from positions `1` to `t-1` of the
integer-scaled significand's magnitude, matching the reference
implementation's `roundit.m`. The sign and exponent are never touched, so a
flip changes the value by at most one binade's worth of significand — as the
`2.140625` above shows, flipping the top explicit bit is a large but not
unbounded perturbation.

`flip` is applied *after* rounding, and only by `chop`/`chop!`. It is not a
property of a format and not available on `LowFloat` — a number type whose
values silently corrupt themselves is not a useful number type.
