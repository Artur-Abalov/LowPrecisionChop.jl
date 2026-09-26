using Test
using Random
using LowPrecisionChop
using LowPrecisionChop: chop, chop!

@testset "chop: matches kernel exactly (thin wrapper check)" begin
    rng = Xoshiro(1)
    for _ in 1:2000
        x = (rand(rng) - 0.5) * exp(20 * (rand(rng) - 0.5))
        c1 = chop(x, Float16Format)
        c2 = LowPrecisionChop._round_to_format(x, 11, 15, RoundNearest, true, true)
        @test isequal(c1, c2)
    end
end

@testset "chop: preserves Float32 vs Float64 storage type" begin
    @test chop(3.14159265358979, Float16Format) isa Float64
    @test chop(3.14159265f0, Float16Format) isa Float32
end

@testset "chop!: in-place array rounding" begin
    A = [1.0, 2.5, 3.14159, 100.0]
    B = copy(A)
    ret = chop!(B, Float16Format)
    @test ret === B  # chop! mutates and returns the same array, not a copy
    @test B == chop(A, Float16Format)
end

@testset "chop: array broadcast is allocation-consistent with elementwise chop" begin
    A = randn(Xoshiro(2), 100)
    @test chop(A, Float16Format) == [chop(x, Float16Format) for x in A]
end

@testset "chop: strict mode errors on unsafe format, default warns once" begin
    unsafe_fmt = Format(30, 127)  # t=30: 2*30+2=62 > 53, unsafe for RoundNearest into Float64
    @test_throws ArgumentError chop(1.0, unsafe_fmt; strict=true)
    # default (strict=false) warns rather than throwing.
    @test chop(1.0, unsafe_fmt; strict=false) isa Float64
    # every named preset is safe by construction (spec.md §8) -- never warns/throws.
    for fmt in (
        Float8E4M3Format,
        Float8E5M2Format,
        Float16Format,
        BFloat16Format,
        TF32Format,
        Float32Format,
    )
        @test chop(1.0, fmt; strict=true) isa Float64
    end
end

@testset "chop: flip/p reproducible from seed, respects probability" begin
    x = chop(3.14159265358979, Float16Format)
    rngA = Xoshiro(9)
    rngB = Xoshiro(9)
    a = [chop(x, Float16Format; flip=true, p=0.5, rng=rngA) for _ in 1:500]
    b = [chop(x, Float16Format; flip=true, p=0.5, rng=rngB) for _ in 1:500]
    @test a == b
    none = [chop(x, Float16Format; flip=true, p=0.0, rng=Xoshiro(10)) for _ in 1:500]
    @test all(==(x), none)
end

@testset "chop: golden values reproduced from test_chop.m via the public API" begin
    # Cross-check against the same transcribed MATLAB values used at the
    # kernel level (test/kernel_matlab_transcribed.jl), through chop() this
    # time, confirming the public API doesn't diverge from the kernel.
    uh = 2.0^(-11)
    pi_h = 6432 * uh
    @test chop(Float64(pi), Float16Format) == pi_h
    @test chop(-Float64(pi), Float16Format) == -pi_h

    xmin = 2.0^(-14)
    @test chop(xmin, Float16Format) == xmin
    @test chop(xmin / 2, Float16Format; subnormal=false) == xmin
    @test chop(xmin / 4, Float16Format; subnormal=false) == 0.0
end

@testset "Base.chop name collision: documented, deliberate, and testably real" begin
    # `using LowPrecisionChop` alone leaves the unqualified name `chop`
    # ambiguous between Base.chop (string truncation) and
    # LowPrecisionChop.chop (rounding) -- confirmed by running a fresh
    # subprocess so this file's own `using LowPrecisionChop: chop` import
    # above can't mask the failure.
    project = Base.active_project()
    script = """
    using LowPrecisionChop
    try
        chop("hello world"; head=5)
        println("UNEXPECTED_SUCCESS")
    catch e
        println("EXPECTED_FAILURE: ", nameof(typeof(e)))
    end
    """
    out = read(`$(Base.julia_cmd()) --project=$project -e $script`, String)
    @test occursin("EXPECTED_FAILURE", out)

    script2 = """
    using LowPrecisionChop: chop, Float16Format
    println(chop(3.14159265358979, Float16Format))
    """
    out2 = read(`$(Base.julia_cmd()) --project=$project -e $script2`, String)
    @test occursin("3.140625", out2)
end
