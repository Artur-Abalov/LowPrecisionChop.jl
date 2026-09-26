using Test
using Random
using LowPrecisionChop

# Kernel tests.
include("kernel_float16_exhaustive.jl")
include("kernel_bigfloat_oracle.jl")
include("kernel_invariants.jl")
include("kernel_matlab_transcribed.jl")
include("kernel_correctness_predicate.jl")
include("kernel_allocation.jl")

# Public API tests.
include("native_contracts.jl")
include("lowfloat_interface.jl")
include("chop_api.jl")

# Weak-dependency extension test, paper-experiment reproductions,
# MATLAB-parity golden CSV.
include("extension_microfloatingpoints.jl")
include("paper_experiments.jl")
include("chop_golden.jl")
