using Test
using Random
using .LowPrecisionChop: _round_to_format

# `@allocated` must be evaluated behind a function barrier. The loop below
# iterates a heterogeneous tuple of rounding-mode singletons, so `roundmode`
# is not a compile-time constant in the loop body: Julia 1.11+ happens to
# constant-propagate through it, but 1.10 emits a dynamic dispatch whose
# boxed `Float64` return shows up as 32 bytes and is charged to the kernel.
# Passing the mode as an argument makes it a concrete type parameter, so the
# reading reflects the kernel itself on every supported version.
function _kernel_allocs(x, t, emax, roundmode, subnormal, explim, rng)
    # Warm up (compilation must not be counted in `@allocated`).
    _round_to_format(x, t, emax, roundmode, subnormal, explim, rng)
    return @allocated _round_to_format(x, t, emax, roundmode, subnormal, explim, rng)
end

@testset "Kernel is non-allocating and type-stable" begin
    rng = Xoshiro(1234)
    x = 3.14159265
    for roundmode in (
        RoundNearest,
        RoundUp,
        RoundDown,
        RoundToZero,
        RoundNearestTiesAway,
        RoundStochasticProportional,
        RoundStochasticEqual,
    )
        @inferred _round_to_format(x, 11, 15, roundmode, true, true, rng)
        @test _kernel_allocs(x, 11, 15, roundmode, true, true, rng) == 0
    end

    xf32 = 3.14159265f0
    @inferred _round_to_format(xf32, 8, 127, RoundNearest, true, true, rng)
    @test _kernel_allocs(xf32, 8, 127, RoundNearest, true, true, rng) == 0
end
