"""
    RoundStochasticProportional

Stochastic rounding, mode 5 in MATLAB `chop`'s `options.round` numbering:
round up with probability equal to the fractional part of the scaled
significand (i.e. proportional to distance from the lower neighbor).
Exact integers are left unchanged. See `roundit.m` (higham/chop) and
Table 5.1 of Higham & Pranesh (2019).

# Examples
```jldoctest
julia> using Random, LowPrecisionChop

julia> using LowPrecisionChop: chop

julia> chop(1.5005, Float16Format; round=RoundStochasticProportional, rng=Xoshiro(1))
1.5009765625
```
"""
const RoundStochasticProportional = RoundingMode{:StochasticProportional}()

"""
    RoundStochasticEqual

Stochastic rounding, mode 6 in MATLAB `chop`'s `options.round` numbering:
round up or down with equal probability (1/2), regardless of distance.
Exact integers are left unchanged. See `roundit.m` (higham/chop) and
Table 5.1 of Higham & Pranesh (2019).

# Examples
```jldoctest
julia> using Random, LowPrecisionChop

julia> using LowPrecisionChop: chop

julia> chop(1.5005, Float16Format; round=RoundStochasticEqual, rng=Xoshiro(1))
1.5009765625
```
"""
const RoundStochasticEqual = RoundingMode{:StochasticEqual}()

"""
    LOWPRECISIONCHOP_ROUND_MODES

Maps MATLAB `chop`'s `options.round` integer values 1-6 to the `RoundingMode`
each corresponds to, per Table 5.1 of Higham & Pranesh (2019). Mode 7
(`RoundNearestTiesAway`) is a LowPrecisionChop-only extension not present in
MATLAB `chop` or its paper — see `docs/src/spec.md` §3.
"""
const LOWPRECISIONCHOP_ROUND_MODES = (
    RoundNearest,                  # 1: ties to even (default; matches MATLAB via Julia round())
    RoundUp,                       # 2: toward +Inf
    RoundDown,                     # 3: toward -Inf
    RoundToZero,                   # 4: toward zero
    RoundStochasticProportional,   # 5: stochastic, prob. proportional to distance
    RoundStochasticEqual,          # 6: stochastic, equal probability
    RoundNearestTiesAway,          # 7: LowPrecisionChop extension, NOT in MATLAB chop
)

_isnearestfamily(::RoundingMode{:Nearest}) = true
_isnearestfamily(::RoundingMode{:NearestTiesAway}) = true
_isnearestfamily(::typeof(RoundStochasticProportional)) = true
_isnearestfamily(::typeof(RoundStochasticEqual)) = true
_isnearestfamily(::RoundingMode) = false

_isstochastic(::typeof(RoundStochasticProportional)) = true
_isstochastic(::typeof(RoundStochasticEqual)) = true
_isstochastic(::RoundingMode) = false
