# OrdinaryDiffEq.jl / Flux.jl integration suite (task's <integration_targets>,
# item 4).
# Run standalone: julia --project=test/integration/sciml
#                        test/integration/sciml/runtests.jl

using Test
using Random
using OrdinaryDiffEq
using OrdinaryDiffEqLowOrderRK
using Flux
using LowPrecisionChop

const LFBF16_RN = lowfloattype(BFloat16Format, RoundNearest)
const LFBF16_SE = lowfloattype(BFloat16Format, RoundStochasticEqual)

@testset "LowFloat needs round/floor/ceil/trunc/Int and sqrt_fast (found via this suite)" begin
    # These were all missing before this suite was written -- see
    # docs/src/integrations.md. Regression guards, not just documentation.
    x = LFBF16_RN(3.7)
    @test round(x) == LFBF16_RN(4.0)
    @test floor(x) == LFBF16_RN(3.0)
    @test ceil(x) == LFBF16_RN(4.0)
    @test Int(round(x)) == 4
    @test Base.FastMath.sqrt_fast(LFBF16_RN(4.0)) == LFBF16_RN(2.0)
end

@testset "Non-stiff: y'=-y stagnates under round-to-nearest when h << eps(format), not under stochastic rounding" begin
    # Direct reproduction of the mechanism Higham & Pranesh describe (§6,
    # Figure 6.1 discussion): for this ODE, the *relative* update per Euler
    # step is exactly h regardless of y's magnitude (dy/y = -h). If h is
    # below half the format's unit roundoff, round-to-nearest rounds the
    # update back to exactly 0 at *every* step -- permanent stagnation, not
    # just slow convergence. Stochastic rounding does not have this failure
    # mode: each step still has a nonzero probability of moving.
    f(y, p, t) = -y
    h = 0.001  # eps(bf16) = 0.0078125, so h is well below eps/2 = 0.0039

    y0_rn = LFBF16_RN(1.0)
    prob_rn = ODEProblem(f, y0_rn, (LFBF16_RN(0.0), LFBF16_RN(3.0)))
    sol_rn = solve(prob_rn, Euler(); dt=LFBF16_RN(h), adaptive=false, maxiters=1_000_000)
    vals_rn = Float64.(sol_rn.u)

    # `LFBF16_SE`'s Euler steps use `LowFloat`'s `+`/`-` arithmetic, which
    # for stochastic rounding modes draws from the ambient
    # `Random.default_rng()` (no room for an explicit rng kwarg in operator
    # syntax -- see the reproducibility note on the Flux stagnation test
    # below). `Random.seed!` before each solve, not an unused `rng` object,
    # is what actually makes this reproducible.
    Random.seed!(1)
    y0_se = LFBF16_SE(1.0)
    prob_se = ODEProblem(f, y0_se, (LFBF16_SE(0.0), LFBF16_SE(3.0)))
    sol_se = solve(prob_se, Euler(); dt=LFBF16_SE(h), adaptive=false, maxiters=1_000_000)
    vals_se = Float64.(sol_se.u)

    Random.seed!(1)
    sol_se2 = solve(prob_se, Euler(); dt=LFBF16_SE(h), adaptive=false, maxiters=1_000_000)
    @test Float64.(sol_se2.u) == vals_se  # same seed -> must reproduce exactly

    # Round-to-nearest: complete stagnation -- every step identical to y0.
    @test all(==(1.0), vals_rn)
    @test length(unique(vals_rn)) == 1

    # Stochastic rounding: genuinely progresses (robust bound, not a
    # fragile exact-count threshold -- unlike round-to-nearest's total
    # freeze, "more than one distinct value" is essentially certain here).
    @test length(unique(vals_se)) > 1
    @test vals_se[end] != vals_se[1]
end

@testset "Stiff: y' = -50*(y-1), Rosenbrock23, bf16, round-to-nearest vs stochastic" begin
    # Linear stiff problem (lambda=-50, requires an implicit method for
    # stability at reasonable step sizes); avoids any transcendental
    # function of a LowFloat argument (cos/sin/exp are out of this
    # package's scope -- see docs/src/spec.md, only +,-,*,/,sqrt have a
    # derived double-rounding safety proof).
    g(y, p, t) = -50.0 * (y - 1.0)
    for LF in (LFBF16_RN, LFBF16_SE)
        y0 = LF(0.0)
        prob = ODEProblem(g, y0, (LF(0.0), LF(1.0)))
        sol = solve(prob, Rosenbrock23(); dt=LF(0.01), adaptive=false, maxiters=10_000)
        finalv = Float64(sol.u[end])
        @test isfinite(finalv)
        @test abs(finalv - 1.0) < 0.05  # converges close to the true steady state
    end
end

@testset "Flux: forward pass, Zygote gradient, and a full training loop work with LowFloat" begin
    # `tanh`/other transcendental activations are out of scope (no double-
    # rounding safety derivation for them, per spec.md) and indeed
    # MethodError on LowFloat -- confirmed directly, not assumed. Use
    # `identity` activation (a linear layer) to stay within +,-,*,/,
    # which is enough to exercise Flux's actual machinery: Dense
    # construction, the forward pass, Zygote autodiff through it,
    # Flux.setup, and Flux.update!.
    rng = Xoshiro(1)
    W = LFBF16_RN.(0.1 .* randn(rng, 2, 4))
    b = LFBF16_RN.(zeros(2))
    m = Dense(W, b, identity)
    @test_throws MethodError Dense(W, b, tanh)(LFBF16_RN.(randn(rng, 4)))

    x = LFBF16_RN.(randn(rng, 4))
    target = LFBF16_RN.(randn(rng, 2))
    loss, grads = Flux.withgradient(m) do model
        sum(abs2, model(x) .- target)
    end
    @test isfinite(loss)
    @test eltype(grads[1].weight) == LFBF16_RN

    opt_state = Flux.setup(Descent(0.1), m)
    losses = Float64[]
    for _ in 1:5
        l, g = Flux.withgradient(m) do model
            sum(abs2, model(x) .- target)
        end
        Flux.update!(opt_state, m, g[1])
        push!(losses, Float64(l))
    end
    @test losses[end] < losses[1]  # loss decreases over the training loop
end

@testset "Flux training loop: round-to-nearest stagnates at a tiny learning rate, stochastic rounding does not" begin
    # Same mechanism as the ODE stagnation test above, reproduced in an
    # actual Flux training loop: at a small enough learning rate, the
    # weight update magnitude falls below the format's rounding
    # resolution, and round-to-nearest freezes the weight permanently.
    #
    # Reproducibility note (a real gap found by this test flaking on its
    # first run, fixed here rather than papered over with a looser bound):
    # `LowFloat`'s arithmetic operators (`+`,`-`,`*`,`/`) have no room for
    # an explicit `rng` argument -- infix operator syntax only takes the
    # two operands -- so stochastic-mode arithmetic necessarily draws from
    # the ambient `Random.default_rng()` (task-local RNG), unlike `chop`,
    # which *does* accept an explicit `rng` kwarg because it's a normal
    # function call with room for one. The reproducibility contract for
    # `LowFloat` stochastic arithmetic is therefore `Random.seed!(seed)`
    # before the computation, not passing an rng object into it -- verified
    # directly (`Random.seed!(42)` before two separate `LFBF16_SE(...)  +
    # LFBF16_SE(...)` runs gives identical sequences). Flagged here and in
    # docs/src/integrations.md so it isn't rediscovered as a bug later.
    function train_and_track(seed::Integer, LF; lr, nsteps=200)
        Random.seed!(seed)
        W = LF.(0.5 .* ones(1, 1))
        b = LF.(zeros(1))
        m = Dense(W, b, identity)
        x = LF.([1.0])
        target = LF.([2.0])
        opt_state = Flux.setup(Descent(lr), m)
        weights = Float64[]
        for _ in 1:nsteps
            _, g = Flux.withgradient(m) do model
                sum(abs2, model(x) .- target)
            end
            Flux.update!(opt_state, m, g[1])
            push!(weights, Float64(m.weight[1]))
        end
        return weights
    end

    lr = 0.0001  # updates land well below eps(bf16)/2 relative to the weight
    wr = train_and_track(1, LFBF16_RN; lr=lr)
    ws1 = train_and_track(1, LFBF16_SE; lr=lr)
    ws2 = train_and_track(1, LFBF16_SE; lr=lr)  # same seed -> must reproduce exactly

    @test ws1 == ws2  # the actual reproducibility requirement

    @test length(unique(wr)) == 1     # round-to-nearest: frozen at the initial weight
    @test wr[end] == wr[1]
    @test length(unique(ws1)) > 1     # stochastic: not frozen (robust bound, not a fragile threshold)
    @test ws1[end] != ws1[1]
end
