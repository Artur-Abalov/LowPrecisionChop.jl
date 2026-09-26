# Benchmarks -- recorded, not asserted (no @test/pass-fail here). Run with:
#   julia --project=benchmark benchmark/run.jl

using BenchmarkTools
using Random
using LowPrecisionChop
using LowPrecisionChop: chop, _round_to_format

const SUITE = BenchmarkGroup()

rng = Xoshiro(1)
x = 3.14159265358979
A = randn(rng, 1000)
LF16 = lowfloattype(Float16Format)
LFBF16 = lowfloattype(BFloat16Format)

SUITE["kernel"] = BenchmarkGroup()
SUITE["kernel"]["scalar RoundNearest"] = @benchmarkable _round_to_format(
    $x, 11, 15, RoundNearest, true, true
)
SUITE["kernel"]["scalar RoundStochasticEqual"] = @benchmarkable _round_to_format(
    $x, 11, 15, RoundStochasticEqual, true, true, rng
) setup = (rng = Xoshiro(1))

SUITE["chop"] = BenchmarkGroup()
SUITE["chop"]["scalar"] = @benchmarkable chop($x, Float16Format)
SUITE["chop"]["array broadcast (n=1000)"] = @benchmarkable chop($A, Float16Format)
SUITE["chop"]["array in-place chop! (n=1000)"] = @benchmarkable chop!(B, Float16Format) setup = (
    B = copy($A)
) evals = 1

SUITE["lowfloat"] = BenchmarkGroup()
la, lb = LF16(1.5), LF16(0.5)
SUITE["lowfloat"]["+"] = @benchmarkable $la + $lb
SUITE["lowfloat"]["*"] = @benchmarkable $la * $lb
SUITE["lowfloat"]["sqrt"] = @benchmarkable sqrt($la)

la_vec = LFBF16.(A)
SUITE["lowfloat"]["sum(::Vector{LowFloat}, n=1000)"] = @benchmarkable sum($la_vec)

results = run(SUITE; verbose=true)
display(results)
println()
println("(Benchmarks recorded for reference; not asserted against a threshold.)")
