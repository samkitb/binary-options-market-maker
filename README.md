# Market Making Binary Options

An exact pricer, a parameter estimator, and a quoting strategy for binary
(digital) options on correlated underlyings — plus the exchange simulator used
to test them.

Built for **Akuna Capital's 2026 Virtual Quant Trading Challenge**. The challenge
itself is Akuna's, and none of their materials appear here: this is a clean-room
implementation of the same class of problem, written so it stands alone and runs
for anybody who clones it.

```bash
python3 tests/test_pricer.py            # closed form vs brute-force simulation
python3 tests/test_estimator.py         # parameter recovery and pricing error
python3 experiments/spread_study.py     # how wide should you quote?
python3 experiments/simulator_bias.py   # where my own simulator misled me
```

Python 3.11+, no dependencies.

---

## The problem

A binary option pays $1 if its event occurs and $0 otherwise, so **its fair price
is the probability of that event**. That single fact does most of the work.

Three observables evolve daily. A policy `RATE` walks a discrete grid, moving one
tick up, one tick down, or holding, with probabilities that tilt toward a target —
mean reversion on a lattice. Two firm valuations move multiplicatively on a log
return built from a drift, a sensitivity to the rate move, a **shared sector
shock**, and independent noise.

Contracts come in three shapes: `RATE >= k`, `FIRM >= k`, and the spread
`FIRM_A >= FIRM_B`. The market maker publishes two-sided quotes and is scored on
how many rival market makers it out-earns.

Two features shape everything downstream.

**Margin is charged at worst-case loss.** Buying `q` contracts at `p` costs
`q·p`; *selling* costs `q·(1−p)`. Selling a 5¢ contract earns five cents and
freezes ninety-five. Capital returns only at expiry, so capital — not conviction —
limits how much you can trade.

**Quote requests are blind two-sided auctions.** You submit a bid and an offer
without knowing which side the order is, and it routes to the best price. You win
exactly when you are the most aggressive in the room, which is to say when you
disagree most with everyone else. **Getting filled is evidence your price was
wrong.** Quoting fair value ± a penny loses money no matter how good your pricer
is.

---

## The pricer

Monte Carlo works here but is slow and noisy, and noise in your own fair value is
edge you hand to the counterparty. This model solves exactly. Two observations do
it.

**The rate path collapses.** Over N days a firm's log return accumulates
`rate_beta · Σ(daily rate change)`, and that sum telescopes:

```
(r₁ − r₀) + (r₂ − r₁) + … + (r_N − r_{N−1})  =  r_N − r₀
```

The firm does not care *how* the rate got where it got — up-up-down and
down-up-down are identical. Only the terminal rate matters, so instead of `3^N`
paths we need one distribution.

**Conditional on that terminal rate, the firm is lognormal.** Fix `r_N`; the
drift and rate terms become constants and the rest is a sum of independent
Gaussians. So every price is a short weighted sum of normal CDFs:

```
price = Σ over terminal rates r:  P(rate ends at r) · P(event | r)
```

The rate distribution itself is exact — the lattice recombines, so N days give at
most `2N+1` levels rather than `3^N` paths.

The spread contract needs one more step. `A ≥ B` is `log(A) − log(B) ≥ 0`, and
that difference is normal with variance `Var(A) + Var(B) − 2·Cov(A,B)`. When the
two firms share a sector beta the common shock **cancels exactly** and the spread
barely moves. This is why spreads cannot be priced by pricing each firm and
combining — the correlation *is* the question.
[`tests/test_pricer.py`](tests/test_pricer.py) checks this with a 33× variance
swing between matched and mismatched betas.

**Validation.** 45 contracts across randomised parameter sets, every shape and
edge — negative weights, non-unit weights, strikes at or below zero, same-day
expiry, the reflecting floor at rate zero — checked against brute-force
simulation of the model itself. Worst deviation **0.0048**, inside Monte Carlo
error.

---

## Estimating the parameters

You get a burn-in of history, not the parameters. Two things are worth stating
because both look like shortcuts and neither is.

**Sector beta, sector volatility and idiosyncratic volatility are not separately
identifiable.** Double every beta and halve the sector vol and you get an
identical world — [proven in the tests](tests/test_estimator.py), where the two
parameter sets price every contract the same to machine precision. Pricing needs
only `Var(A)`, `Var(B)` and `Cov(A,B)`, all of which fall straight out of the
residual covariance after regressing log returns on the rate change. The
estimator targets those and refuses to invent the rest.

**Drift is nearly unestimable.** Its standard error is roughly `σ/√n`, which for
realistic histories is the size of the drift itself. It is shrunk by the
empirical-Bayes factor `d²/(d² + se²)`, which collapses toward zero when the
estimate cannot be distinguished from it.

Measured pricing error:

| History | mean abs error | sd |
|---|---|---|
| 200 days | 0.0123 | 0.0243 |
| 800 days | 0.0052 | 0.0105 |
| 3200 days | 0.0025 | 0.0070 |

This error is **one draw per session**, not noise per trade. You estimate once
and quote off it all session, so it biases every quote the same direction. That
is what the spread has to cover.

---

## What the spread should be

Theory says the spread must exceed your own pricing error, or the blind auction
extracts it. That argues wide. But quoting wide wins nothing, and the score is
relative — you must out-earn rivals, not merely avoid losses. So there is an
interior optimum, and where it sits is an empirical question about the
competition, not a theoretical one about your model.

`experiments/spread_study.py` sweeps it. The spread between the best and worst
setting is **0.29 of score** — larger than any other single choice in the
strategy, and worth more than every refinement to pricing or inventory combined.

---

## Where I got it wrong

The most useful thing here is a documented failure, reproducible in
[`experiments/simulator_bias.py`](experiments/simulator_bias.py).

During the live challenge I built this simulator, fitted the rivals to what
little feedback I had, and used it to decide what to try. It said to quote wide.
I then spent a long session using it to *reject* ideas — concluding, with real
measurements behind me, that no further improvement was available.

The opposite was true. Tightening the half-spread from roughly 11¢ to roughly 4¢
was worth more than every other change combined, and it was the largest single
improvement of the whole effort.

The experiment reproduces the disagreement: this simulator prefers a wide spread
at **every** setting of rival skill and flow toxicity. A stable, confident, wrong
recommendation. Nothing in it is broken — the mechanics, margin rule and scoring
are faithful. It encodes one wrong assumption about how well the competition
prices, and that assumption is invisible from the inside.

What I would carry forward: **a simulator you calibrated yourself cannot tell you
where you sit relative to opponents you have never observed.** It is excellent
for correctness, for bankruptcy risk, and for how your own choices trade off
against each other. Its verdict on anything involving the competition is a
hypothesis, and real measurements are better spent testing that hypothesis than
refining the model that produced it.

---

## Layout

```
bomm/
  model.py          the dynamics: rate lattice, firms, shared sector shock
  pricer.py         exact closed form, plus Gauss-Hermite for exotic shapes
  estimator.py      moment estimation, identifiability, drift shrinkage
  market_maker.py   quoting, margin, inventory skew, fill-imbalance calibration
sim/
  exchange.py       blind RFQ auction, worst-case margin, rank scoring
tests/              closed form vs simulation; parameter recovery
experiments/        the spread study; the simulator-bias case study
```

## Notes on some choices

**Gauss-Hermite rather than Monte Carlo for the fallback.** Contracts with no
closed form are integrated on a deterministic grid, so repeated calls give
identical answers. A quoting loop that re-prices the same contract should not get
a different number each time.

**A quote of `0.00 / 1.00` is riskless and always legal.** Buying at zero and
selling at one both charge zero margin under the worst-case rule, so a side that
cannot be funded retreats to its extreme rather than going dark.

**Fill imbalance as a second price signal.** The other market makers' quotes are
an independent estimate of fair value, and the auction leaks it: winning mostly
buys means your price sits above the field's, so shade it down.
