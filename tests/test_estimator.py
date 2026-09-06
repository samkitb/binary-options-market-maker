"""Can the parameters be recovered from history, and how well?

The answer matters more than it looks. You estimate once from a burn-in and then
quote off that estimate for the whole session, so the error does not average
away across trades -- it is one draw that biases every quote you make. Knowing
its size is what tells you how wide to quote.
"""
from __future__ import annotations

import math
import random
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bomm.model import RATE, FIRM_A, FIRM_B, Leg, BinaryOption, Parameters, simulate_path
from bomm.pricer import Moments, price
from bomm.estimator import estimate

TRUTH = Parameters(drift_a=.0015, drift_b=.0009, rate_beta_a=-.60, rate_beta_b=-.85,
                   sector_beta_a=1.10, sector_beta_b=0.75, sector_vol=.016,
                   idio_vol_a=.020, idio_vol_b=.026,
                   rate_up=.22, rate_down=.18, reversion=.12)
START = {RATE: 2.5, FIRM_A: 840.0, FIRM_B: 910.0}


def history_of(days, seed):
    path = simulate_path(TRUTH, START, days, random.Random(seed))
    return {k: [s[k] for s in path] for k in (RATE, FIRM_A, FIRM_B)}


def test_recovers_identifiable_quantities():
    """Variances, covariance and rate betas should converge. Drift should not --
    its standard error is the same size as the drift itself."""
    truth = Moments.from_parameters(TRUTH)
    est = estimate(history_of(4000, seed=1))
    for label, t, e, tol in (
        ("rate_beta_a", truth.beta_a, est.beta_a, 0.06),
        ("rate_beta_b", truth.beta_b, est.beta_b, 0.06),
        ("var_a", truth.var_a, est.var_a, truth.var_a * 0.12),
        ("var_b", truth.var_b, est.var_b, truth.var_b * 0.12),
        ("cov", truth.cov, est.cov, abs(truth.cov) * 0.20),
    ):
        assert abs(t - e) <= tol, f"{label}: true {t:.6f}, estimated {e:.6f}"
    print("  betas, variances and covariance all recovered from 4000 days")


def test_pricing_error_shrinks_with_history():
    """Quantify the error that actually matters: how far off are the prices?"""
    truth = Moments.from_parameters(TRUTH)
    options = [
        BinaryOption((Leg(RATE),), 2.75, 5),
        BinaryOption((Leg(FIRM_A),), 880.0, 5),
        BinaryOption((Leg(FIRM_A), Leg(FIRM_B, -1.0)), 0.0, 5),
    ]
    print(f"  {'history':>9} {'mean |error|':>13} {'sd of error':>12}")
    previous = None
    for days in (200, 800, 3200):
        errors = []
        for seed in range(24):
            hist = history_of(days, seed=100 + seed)
            est = estimate(hist)
            state = {RATE: hist[RATE][-1], FIRM_A: hist[FIRM_A][-1], FIRM_B: hist[FIRM_B][-1]}
            for o in options:
                errors.append(price(o, state, est) - price(o, state, truth))
        mean_abs = statistics.fmean(abs(e) for e in errors)
        sd = statistics.stdev(errors)
        print(f"  {days:>9} {mean_abs:>13.4f} {sd:>12.4f}")
        if previous is not None:
            assert mean_abs < previous * 1.25, "error should not grow with more data"
        previous = mean_abs
    assert previous < 0.10, f"error still {previous:.4f} with 3200 days"


def test_unidentifiability_is_real():
    """Two very different parameter sets that produce identical prices.

    This is why the estimator targets moments rather than the raw parameters:
    sector_beta and sector_vol are not separately knowable, and pretending
    otherwise would be fitting noise.
    """
    a = Parameters(drift_a=.001, drift_b=.001, rate_beta_a=-.5, rate_beta_b=-.5,
                   sector_beta_a=1.0, sector_beta_b=1.0, sector_vol=0.04,
                   idio_vol_a=.02, idio_vol_b=.02,
                   rate_up=.2, rate_down=.2, reversion=.1)
    b = Parameters(drift_a=.001, drift_b=.001, rate_beta_a=-.5, rate_beta_b=-.5,
                   sector_beta_a=2.0, sector_beta_b=2.0, sector_vol=0.02,
                   idio_vol_a=.02, idio_vol_b=.02,
                   rate_up=.2, rate_down=.2, reversion=.1)
    ma, mb = Moments.from_parameters(a), Moments.from_parameters(b)
    state = {RATE: 2.5, FIRM_A: 800.0, FIRM_B: 850.0}
    for o in (BinaryOption((Leg(FIRM_A),), 820.0, 5),
              BinaryOption((Leg(FIRM_A), Leg(FIRM_B, -1.0)), 0.0, 5)):
        pa, pb = price(o, state, ma), price(o, state, mb)
        assert abs(pa - pb) < 1e-12, f"{o}: {pa} vs {pb}"
    print("  betas doubled and sector vol halved -> identical prices, as expected")


def test_short_history_does_not_crash():
    for days in (1, 2, 5, 20):
        est = estimate(history_of(days, seed=3))
        p = price(BinaryOption((Leg(FIRM_A),), 850.0, 3), START, est)
        assert 0.0 <= p <= 1.0, (days, p)
    print("  histories of 1-20 days handled without error")


if __name__ == "__main__":
    for fn in (test_recovers_identifiable_quantities,
               test_pricing_error_shrinks_with_history,
               test_unidentifiability_is_real,
               test_short_history_does_not_crash):
        print(f"{fn.__name__}:")
        fn()
    print("\nAll estimator tests passed.")
