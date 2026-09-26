using Documenter
using LowPrecisionChop
using MicroFloatingPoints  # loads the weak-dep extension for the matchingfloatmutype doctest
using Random

DocMeta.setdocmeta!(
    LowPrecisionChop, :DocTestSetup, :(using LowPrecisionChop); recursive=true
)

makedocs(;
    sitename="LowPrecisionChop.jl",
    modules=[LowPrecisionChop],
    authors="Artur Abalov",
    checkdocs=:exports,
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Guides" => [
            "Formats" => "formats.md",
            "Rounding modes" => "rounding.md",
            "Value-level `chop`" => "chop.md",
            "The `LowFloat` type" => "lowfloat.md",
            "Accuracy and double rounding" => "accuracy.md",
        ],
        "Reference" => [
            "Specification" => "spec.md",
            "Verification" => "verification.md",
            "Integration findings" => "integrations.md",
            "Comparison with related packages" => "comparison.md",
            "Known limitations" => "known_limitations.md",
            "API reference" => "api.md",
        ],
    ],
)

deploydocs(; repo="github.com/Artur-Abalov/LowPrecisionChop.jl.git")
