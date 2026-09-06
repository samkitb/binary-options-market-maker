"""Validate the closed form against brute-force simulation of the model.

This is the load-bearing test in the project. The pricer claims to be exact;
the only honest way to support that is to simulate the actual dynamics many
times and check the analytic answer falls inside Monte Carlo error. Every
contract shape and every awkward edge is covered: negative weights, non-unit
weights, strikes at or below zero, same-day expiry, and the reflecting floor
where the rate piles up at zero.
"""
from __future__ import annotations

import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bomm.model import RATE, FIRM_A, FIRM_B, Leg, BinaryOption, Parameters
from bomm.pricer import Moments, price, terminal_rate_distribution

N_PATHS = 40_000


def monte_carlo(option, start, params, seed):
    """Brute force: roll the model forward and count."""
    rng = random.Random(seed)
    hits = 0
    for _ in range(N_PATHS):
        state = dict(start)
        for _ in range(option.days_to_expiry):
            state = params.advance(state, rng)
        hits += option.payoff(state)
    return hits / N_PATHS


def random_parameters(rng):
    return Parameters(
        drift_a=rng.uniform(-.003, .005), drift_b=rng.uniform(-.003, .005),
        rate_beta_a=rng.uniform(-1.2, .4), rate_beta_b=rng.uniform(-1.2, .4),
        sector_beta_a=rng.uniform(.3, 1.6), sector_beta_b=rng.uniform(.3, 1.6),
        sector_vol=rng.uniform(.005, .025),
        idio_vol_a=rng.uniform(.008, .035), idio_vol_b=rng.uniform(.008, .035),
        rate_up=rng.uniform(.10, .32), rate_down=rng.uniform(.10, .32),
        reversion=rng.uniform(0, .40), rate_target=rng.choice([1.5, 2.0, 2.5, 3.0]),
    )


def test_closed_form_matches_simulation():
    rng = random.Random(20260823)
    failures, checked, worst = [], 0, 0.0
    for trial in range(5):
        params = random_parameters(rng)
        start = {RATE: rng.choice([0.25, 1.5, 2.5, 3.25]),
                 FIRM_A: round(rng.uniform(400, 1200), 2),
                 FIRM_B: round(rng.uniform(400, 1200), 2)}
        m = Moments.from_parameters(params)
        days = rng.choice([1, 2, 3, 5, 8])
        cases = [
            ("rate above",     BinaryOption((Leg(RATE),), round(start[RATE] + .25, 2), days)),
            ("rate below",     BinaryOption((Leg(RATE),), round(max(start[RATE] - .5, 0), 2), days)),
            ("rate negated",   BinaryOption((Leg(RATE, -1.0),), round(-start[RATE] - .25, 2), days)),
            ("firm A up",      BinaryOption((Leg(FIRM_A),), round(start[FIRM_A] * 1.03, 2), days)),
            ("firm A down",    BinaryOption((Leg(FIRM_A),), round(start[FIRM_A] * .95, 2), days)),
            ("firm B up",      BinaryOption((Leg(FIRM_B),), round(start[FIRM_B] * 1.06, 2), days)),
            ("A beats B",      BinaryOption((Leg(FIRM_A), Leg(FIRM_B, -1.0)), 0.0, days)),
            ("B beats A",      BinaryOption((Leg(FIRM_B), Leg(FIRM_A, -1.0)), 0.0, days)),
            ("A beats 1.2*B",  BinaryOption((Leg(FIRM_A), Leg(FIRM_B, -1.2)), 0.0, days)),
        ]
        for label, option in cases:
            analytic = price(option, start, m)
            simulated = monte_carlo(option, start, params, seed=9000 + trial)
            se = math.sqrt(max(simulated * (1 - simulated), 1e-9) / N_PATHS)
            tol = max(4 * se, 0.004)
            checked += 1
            worst = max(worst, abs(analytic - simulated))
            if abs(analytic - simulated) > tol:
                failures.append(f"{label} d={days}: {analytic:.4f} vs {simulated:.4f} (tol {tol:.4f})")
    assert not failures, "\n".join(failures)
    print(f"  {checked} contracts checked, worst deviation {worst:.5f}")


def test_lattice_is_exact():
    """The rate distribution should match simulation to within sampling error."""
    rng = random.Random(11)
    params = random_parameters(rng)
    m = Moments.from_parameters(params)
    start = {RATE: 2.5, FIRM_A: 800.0, FIRM_B: 800.0}
    days = 6
    lattice = terminal_rate_distribution(m, start[RATE], days)
    assert abs(sum(lattice.values()) - 1.0) < 1e-12, "lattice must be a distribution"
    counts: dict[float, int] = {}
    sim_rng = random.Random(3)
    for _ in range(N_PATHS):
        state = dict(start)
        for _ in range(days):
            state = params.advance(state, sim_rng)
        counts[state[RATE]] = counts.get(state[RATE], 0) + 1
    for r, p in lattice.items():
        empirical = counts.get(r, 0) / N_PATHS
        se = math.sqrt(max(p * (1 - p), 1e-9) / N_PATHS)
        assert abs(p - empirical) <= max(4 * se, 0.004), f"rate {r}: {p} vs {empirical}"
    print(f"  lattice matched simulation across {len(lattice)} reachable levels")


def test_edge_cases():
    params = Parameters(drift_a=.0015, drift_b=.0009, rate_beta_a=-.6, rate_beta_b=-.85,
                        sector_beta_a=1.1, sector_beta_b=.75, sector_vol=.016,
                        idio_vol_a=.02, idio_vol_b=.026,
                        rate_up=.22, rate_down=.18, reversion=.12)
    m = Moments.from_parameters(params)
    state = {RATE: 2.5, FIRM_A: 840.0, FIRM_B: 910.0}
    checks = [
        ("expires today, in the money",  BinaryOption((Leg(RATE),), 2.0, 0), 1.0),
        ("expires today, out",           BinaryOption((Leg(RATE),), 9.0, 0), 0.0),
        ("strike 0 on a positive firm",  BinaryOption((Leg(FIRM_A),), 0.0, 3), 1.0),
        ("unreachable strike",           BinaryOption((Leg(FIRM_A),), 1e12, 3), 0.0),
        ("rate floor at zero",           BinaryOption((Leg(RATE),), 0.0, 9), 1.0),
        ("worthless firm",               BinaryOption((Leg(FIRM_A),), 50.0, 3), 0.0),
    ]
    for label, option, expected in checks:
        s = dict(state)
        if label == "worthless firm":
            s[FIRM_A] = 0.0
        got = price(option, s, m)
        assert abs(got - expected) < 1e-9, f"{label}: {got} != {expected}"
    long_dated = price(BinaryOption((Leg(RATE),), 3.0, 40), state, m)
    assert 0.0 <= long_dated <= 1.0
    print(f"  {len(checks)} edge cases correct; 40-day option priced {long_dated:.4f}")


def test_shared_shock_cancels_in_the_spread():
    """Identical sector betas should make the spread insensitive to the shared
    shock -- the mechanism that makes spread contracts a different problem."""
    base = dict(drift_a=.001, drift_b=.001, rate_beta_a=-.5, rate_beta_b=-.5,
                idio_vol_a=.01, idio_vol_b=.01, rate_up=.2, rate_down=.2, reversion=.1)
    state = {RATE: 2.5, FIRM_A: 800.0, FIRM_B: 800.0}
    spread = BinaryOption((Leg(FIRM_A), Leg(FIRM_B, -1.0)), 0.0, 5)
    same = Moments.from_parameters(Parameters(sector_beta_a=1.0, sector_beta_b=1.0,
                                              sector_vol=0.05, **base))
    diff = Moments.from_parameters(Parameters(sector_beta_a=1.8, sector_beta_b=0.2,
                                              sector_vol=0.05, **base))
    var_same = same.var_a + same.var_b - 2 * same.cov
    var_diff = diff.var_a + diff.var_b - 2 * diff.cov
    assert var_diff > 5 * var_same, (var_same, var_diff)
    print(f"  spread variance: matched betas {var_same:.2e}, mismatched {var_diff:.2e} "
          f"({var_diff / var_same:.0f}x)")


if __name__ == "__main__":
    for fn in (test_closed_form_matches_simulation, test_lattice_is_exact,
               test_edge_cases, test_shared_shock_cancels_in_the_spread):
        print(f"{fn.__name__}:")
        fn()
    print("\nAll pricer tests passed.")
