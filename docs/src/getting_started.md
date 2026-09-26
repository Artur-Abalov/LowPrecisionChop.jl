# Getting started

This page takes you from an empty REPL to a working low-precision
computation, and explains the one piece of friction you will hit
immediately (the `chop` name collision).

## Install

```julia
using Pkg
Pkg.add("LowPrecisionChop")
```

## The `chop` name collision

`Base` exports a function called `chop` that truncates characters off the
end of a string:

```julia-repl
julia> chop("hello world")      # Base.chop: drops the last character
"hello worl"
```

This package also exports a `chop` — an unrelated generic function that
rounds floating-point values. Julia does not merge two exported bindings of
the same name from different modules; it marks the name as conflicting, and
*any* unqualified use fails:

```julia-repl
julia> using LowPrecisionChop

julia> chop(3.14, Float16Format)
ERROR: UndefVarError: `chop` not defined in `Main`
Hint: It looks like two or more modules export different bindings with this
name, resulting in ambiguity. Try explicitly importing it from a particular
module, or qualifying the name with the module it should come from.
```

Pick one of the two fixes and use it consistently:

```julia
using LowPrecisionChop        # brings in LowFloat, Format, chop!, presets, ...
using LowPrecisionChop: chop  # ...and resolves `chop` to the rounding one
```

or leave it qualified at every call site:

```julia
using LowPrecisionChop
LowPrecisionChop.chop(3.14, Float16Format)
```

Note that `chop!` is *not* affected — `Base` has no `chop!`, so it imports
cleanly. Only the non-mutating name collides.

The rest of this documentation assumes `using LowPrecisionChop: chop`.

## Your first `chop`

[`chop`](@ref LowPrecisionChop.chop) takes a value and a format, and
returns a value of the *same Julia type* whose significand has been rounded
to the format's width:

```julia-repl
julia> using LowPrecisionChop

julia> using LowPrecisionChop: chop

julia> chop(3.14159265358979, Float16Format)
3.140625

julia> chop(3.14159265358979, Float8E4M3Format)
3.25

julia> chop(3.14159265358979, Float32Format)
3.1415927410125732
```

The result of the first call is still a `Float64` — it is simply a `Float64`
that happens to be exactly representable in fp16. That is the whole idea:
low-precision *values* carried in a high-precision *container*.

```julia-repl
julia> typeof(chop(3.14159265358979, Float16Format))
Float64

julia> chop(3.14159265f0, Float16Format)   # Float32 in, Float32 out
3.140625f0
```

Arrays work too, elementwise, with an in-place variant:

```julia-repl
julia> A = [1.0 2.5; 3.14159 1e-40];

julia> chop(A, Float16Format)
2×2 Matrix{Float64}:
 1.0      2.5
 3.14062  0.0

julia> chop!(A, Float16Format);   # overwrites A
```

That `1e-40 → 0.0` is not a bug: `1e-40` is below the smallest fp16
subnormal, so it underflows to zero, exactly as fp16 hardware would.

## Your first `LowFloat`

The other half of the package is a number type. Build a concrete type from a
format with [`lowfloattype`](@ref), then use it like any other float:

```julia-repl
julia> BF16 = lowfloattype(BFloat16Format)
LowFloat{8, 127, RoundingMode{:Nearest}, true}

julia> x = BF16(1.0) / BF16(3.0)
0.333984375

julia> x + x + x
1.0
```

Each operation computes in `Float64` and then rounds the result back to the
format — so the arithmetic really is bfloat16 arithmetic, not `Float64`
arithmetic with a cosmetic wrapper.

Because `LowFloat <: AbstractFloat`, generic Julia algorithms accept it
without modification:

```julia-repl
julia> using LinearAlgebra

julia> A = BF16.([4.0 1.0; 1.0 3.0]);

julia> F = lu(A);

julia> F.U
2×2 Matrix{LowFloat{8, 127, RoundingMode{:Nearest}, true}}:
 4.0  1.0
 0.0  2.75
```

`LinearAlgebra` routes this to its generic `generic_lufact!` path rather
than to BLAS, because `LowFloat` is deliberately not a `BlasFloat`. Every
add, multiply and divide inside the factorization is rounded to bfloat16.
See [Integration findings](integrations.md) for the same exercise with
Krylov.jl, OrdinaryDiffEq.jl and Flux.jl.

## Which API should I use?

Use **`chop`/`chop!`** when the *data* is what you want to quantize, and you
control where rounding happens:

- quantizing inputs, weights, or intermediate results at chosen points;
- reproducing MATLAB `chop` results, or comparing against the paper;
- simulating a mixed-precision algorithm where you decide, per operation,
  which precision applies.

Use **`LowFloat`** when the *algorithm* is what you want to run in low
precision, and you would rather not touch its source:

- feeding an existing generic routine (`lu`, `qr`, a Krylov solver, an ODE
  integrator) and letting every internal operation round;
- studying how an algorithm degrades as precision drops;
- checking whether a library silently escapes to `Float64` or BLAS (it will
  `MethodError` rather than quietly upcast — that is by design).

They share a kernel, so results agree:

```julia-repl
julia> Float64(BF16(3.14159265358979)) == chop(3.14159265358979, BFloat16Format)
true
```

## A slightly bigger example: the harmonic series

Low precision makes `∑ 1/i` *stop growing* — once the running sum is large
enough, `1/i` is smaller than half an ulp of it, and every further term
rounds away. This is Table 6.3 of Higham & Pranesh (2019), and it
reproduces exactly:

```julia
using LowPrecisionChop
using LowPrecisionChop: chop

function harmonic(fmt; round = RoundNearest)
    s, n = 0.0, 0
    while true
        n += 1
        snew = chop(s + chop(1 / n, fmt; round), fmt; round)
        snew == s && return (sum = s, terms = n)
        s = snew
    end
end

harmonic(BFloat16Format)   # (sum = 5.0625, terms = 65)
harmonic(Float16Format)    # (sum = 7.0859375, terms = 513)
harmonic(Format(5, 3))     # (sum = 3.5, terms = 16)   -- Moler's informal fp8
```

The paper reports `5.0625`/65, `7.0859`/513 and `3.5`/16 for these three
formats. See [Verification](verification.md#Paper-experiment-reproductions)
for the full set of reproduced tables, including the two cells that do
*not* match.

## Next steps

- [Formats](formats.md) — what `(t, emax)` means and how to define your own.
- [Rounding modes](rounding.md) — including stochastic rounding, which
  fixes the stagnation you just saw.
- [Accuracy and double rounding](accuracy.md) — read this before trusting a
  custom format.
