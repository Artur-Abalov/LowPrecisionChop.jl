using Test
using Random
using .LowPrecisionChop: _round_to_format

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
        # Warm up (compilation must not be counted in @allocated).
        _round_to_format(x, 11, 15, roundmode, true, true, rng)
        allocs = @allocated _round_to_format(x, 11, 15, roundmode, true, true, rng)
        @test allocs == 0
    end

    xf32 = 3.14159265f0
    @inferred _round_to_format(xf32, 8, 127, RoundNearest, true, true, rng)
    _round_to_format(xf32, 8, 127, RoundNearest, true, true, rng)
    @test (@allocated _round_to_format(xf32, 8, 127, RoundNearest, true, true, rng)) == 0
end
