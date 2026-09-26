# MicroFloatingPoints.jl weak-dependency extension test. Round-trip
# conversion only -- deeper conversion/arithmetic bugs would fail loudly
# and visibly rather than needing dedicated coverage here.

using Test
using MicroFloatingPoints
using LowPrecisionChop

@testset "MicroFloatingPoints extension: round-trip conversion" begin
    presets = [
        Float8E4M3Format, Float8E5M2Format, Float16Format, BFloat16Format, Float32Format
    ]
    for fmt in presets
        LF = lowfloattype(fmt)
        FM = matchingfloatmutype(LF)
        for x in (0.0, 1.0, -1.0, 3.14159265358979, 100.0, -0.001)
            lf = LF(x)
            fm = FM(lf)
            back = LF(fm)
            @test back == lf
        end
    end
    # Confirm the type-parameter correspondence directly against
    # MicroFloatingPoints' own Emax (not just round-trip values).
    @test matchingfloatmutype(lowfloattype(Float16Format)) == Floatmu{5,10}
    @test Emax(Floatmu{5,10}) == 15
end
