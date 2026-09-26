using Test
using .LowPrecisionChop: issimulationexact

@testset "issimulationexact matches chop.m's maxfraction check" begin
    # chop.m: round=1, single container -> maxfraction=11; double -> 25.
    @test issimulationexact(11, RoundNearest; containerprecision=24) == true
    @test issimulationexact(12, RoundNearest; containerprecision=24) == false
    @test issimulationexact(25, RoundNearest; containerprecision=53) == true
    @test issimulationexact(26, RoundNearest; containerprecision=53) == false

    # chop.m: round!=1, single -> maxfraction=23; double -> 52.
    for rm in (RoundUp, RoundDown, RoundToZero)
        @test issimulationexact(23, rm; containerprecision=24) == true
        @test issimulationexact(24, rm; containerprecision=24) == false
        @test issimulationexact(52, rm; containerprecision=53) == true
        @test issimulationexact(53, rm; containerprecision=53) == false
    end

    # Every named preset must be safe for a Float64 container (t <= 25 for
    # nearest-family, t <= 52 for directed) -- this is what makes LowFloat
    # arithmetic (always Float64-then-round) exact for all of them.
    presets = [4, 3, 11, 8, 24]  # fp8-e4m3, fp8-e5m2, fp16, bfloat16, fp32
    for t in presets
        @test issimulationexact(t, RoundNearest) == true
        @test issimulationexact(t, RoundStochasticProportional) == true
        @test issimulationexact(t, RoundNearestTiesAway) == true
    end
    # fp64 itself (t=53) is the container precision -- a no-op format, not
    # covered by (and not needing) the theorem.
    @test issimulationexact(53, RoundNearest) == false  # 2*53+2 > 53, as expected; fp64 is a degenerate boundary case
end
