"""Recovering the model's parameters from observed history.

Pricing needs parameters. In practice you are handed a burn-in of daily
observations and must infer them, which introduces an error that does *not*
average away: you estimate once and then quote off that estimate all session,
so a biased estimate is biased on every quote you make.

Two design decisions are worth stating, because both look like shortcuts and
neither is.

**We fit P(up | rate) and P(down | rate) as lines rather than recovering the
underlying (base rate, target, reversion) triple.** Those three are not jointly
identified -- many combinations produce the same two lines -- and the lattice
only ever consumes the lines.

**We do not attempt to separate sector_beta, sector_vol and idio_vol.** They are
unidentifiable in principle. What pricing needs is Var(A), Var(B) and Cov(A, B),
all three of which fall directly out of the residual covariance matrix after
regressing each firm's log-returns on the rate change.

Drift is the hardest quantity here. Its standard error is roughly
sigma/sqrt(n), which for realistic histories is the same size as the drift
itself, so the raw estimate is mostly noise. We shrink it by the empirical-Bayes
factor d^2/(d^2 + se^2), which collapses toward zero when the estimate is
indistinguishable from zero and leaves it alone when it is well resolved.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .model import RATE, FIRM_A, FIRM_B
from .pricer import Moments


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least squares fit of y = intercept + slope*x."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-15:
        return my, 0.0
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    slope = sxy / sxx
    return my - slope * mx, slope


def _var(v: list[float], ddof: int = 1) -> float:
    n = len(v)
    if n <= ddof:
        return 1e-12
    m = sum(v) / n
    return max(sum((x - m) ** 2 for x in v) / (n - ddof), 1e-12)


def _cov(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    ma, mb = sum(a[:n]) / n, sum(b[:n]) / n
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (n - 1)


def _shrink(estimate: float, std_err: float) -> float:
    if std_err <= 0.0:
        return estimate
    e2 = estimate * estimate
    return estimate * (e2 / (e2 + std_err * std_err))


def estimate(history: dict[int, list[float]]) -> Moments:
    """Fit Moments to a history of daily observations."""
    rates = list(history[RATE])
    n = len(rates)

    # tick size: the modal non-zero absolute move
    moves = [round(abs(rates[i + 1] - rates[i]), 4) for i in range(n - 1)]
    nonzero = [m for m in moves if m > 1e-9]
    tick = 0.25
    if nonzero:
        counts: dict[float, int] = defaultdict(int)
        for m in nonzero:
            counts[m] += 1
        tick = max(counts.items(), key=lambda kv: kv[1])[0]

    # transition lines. Moves at the zero floor are censored -- a "down" draw
    # there leaves the rate unchanged and would be mislabelled -- so drop them.
    xs = rates[:-1]
    y_up = [1.0 if rates[i + 1] > rates[i] + 1e-9 else 0.0 for i in range(n - 1)]
    dn_x, dn_y = [], []
    for i in range(n - 1):
        if rates[i] <= 1e-9:
            continue
        dn_x.append(rates[i])
        dn_y.append(1.0 if rates[i + 1] < rates[i] - 1e-9 else 0.0)
    if len(dn_x) < 5:
        dn_x = xs
        dn_y = [1.0 if rates[i + 1] < rates[i] - 1e-9 else 0.0 for i in range(n - 1)]

    up_line = _ols(xs, y_up)
    down_line = _ols(dn_x, dn_y)

    # firm log-returns regressed on the rate change
    va, vb = list(history[FIRM_A]), list(history[FIRM_B])
    d_rate, la, lb = [], [], []
    for i in range(n - 1):
        if min(va[i], vb[i], va[i + 1], vb[i + 1]) > 0:
            d_rate.append(rates[i + 1] - rates[i])
            la.append(math.log(va[i + 1] / va[i]))
            lb.append(math.log(vb[i + 1] / vb[i]))

    drift_a, beta_a = _ols(d_rate, la)
    drift_b, beta_b = _ols(d_rate, lb)
    res_a = [la[i] - (drift_a + beta_a * d_rate[i]) for i in range(len(d_rate))]
    res_b = [lb[i] - (drift_b + beta_b * d_rate[i]) for i in range(len(d_rate))]
    var_a, var_b = _var(res_a, ddof=2), _var(res_b, ddof=2)
    cov = _cov(res_a, res_b)

    m = max(len(d_rate), 1)
    drift_a = _shrink(drift_a, math.sqrt(var_a / m))
    drift_b = _shrink(drift_b, math.sqrt(var_b / m))

    return Moments(tick, up_line, down_line,
                   drift_a, beta_a, var_a, drift_b, beta_b, var_b, cov)
