"""Exact closed-form pricing of binary options under the model.

The obvious approach is Monte Carlo. It works, but it is slow and noisy, and in
a market-making context noise in your own fair value is edge you hand away. This
model admits an exact solution, and the derivation rests on two observations.

**1. The rate path collapses.**

Stack up N days of log-returns for a firm:

    log(V_N / V_0) = N*drift + rate_beta * SUM(d_rate) + sector_beta * SUM(shock)
                                          + SUM(idio)

Each day's rate change is (today - yesterday), so the sum telescopes:

    (r_1 - r_0) + (r_2 - r_1) + ... + (r_N - r_{N-1})  =  r_N - r_0

The firm does not care *how* the rate got where it got -- up-up-down and
down-up-down have identical effects. Only the terminal rate matters, so instead
of 3^N paths we need only the distribution of the final rate.

**2. That distribution is a small exact computation.**

Because the rate lives on a discrete grid, probability mass can be pushed
forward day by day. The lattice recombines (up-then-down lands where
down-then-up lands), so after N days there are at most 2N+1 reachable levels,
not 3^N paths.

**3. Conditional on the terminal rate, the firm is lognormal.**

Fix r_N. Then N*drift and rate_beta*(r_N - r_0) are both constants, and the two
remaining terms are sums of independent Gaussians. A constant plus Gaussians is
Gaussian, so:

    log(V_N/V_0) | r_N  ~  Normal( N*drift + rate_beta*(r_N - r_0),
                                   N*(sector_beta^2*sector_vol^2 + idio_vol^2) )

Putting it together, every price is a short weighted sum of normal CDFs:

    price = SUM over terminal rates r:  P(rate ends at r) * P(event | r)

Exact, and microseconds rather than milliseconds. `tests/test_pricer.py` checks
every branch against brute-force simulation of the model itself.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .model import RATE, FIRM_A, FIRM_B, BinaryOption, Parameters

SQRT2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


# --------------------------------------------------------------------- moments
class Moments:
    """What pricing actually needs, which is less than Parameters contains.

    sector_beta, sector_vol and idio_vol are NOT separately identifiable from
    price history -- doubling every beta while halving the sector vol produces
    an identical world. Fortunately pricing never needs them apart, only three
    combinations, all of which are directly estimable. Working in this basis
    keeps the pricer honest about what is knowable.
    """

    __slots__ = ("rate_tick", "up_line", "down_line",
                 "drift_a", "beta_a", "var_a",
                 "drift_b", "beta_b", "var_b", "cov")

    def __init__(self, rate_tick, up_line, down_line,
                 drift_a, beta_a, var_a, drift_b, beta_b, var_b, cov):
        self.rate_tick = rate_tick
        self.up_line = up_line       # P(up|r)   = a + b*r
        self.down_line = down_line   # P(down|r) = a + b*r
        self.drift_a, self.beta_a, self.var_a = drift_a, beta_a, var_a
        self.drift_b, self.beta_b, self.var_b = drift_b, beta_b, var_b
        self.cov = cov               # daily covariance of the two log-returns

    @classmethod
    def from_parameters(cls, p: Parameters) -> "Moments":
        s2 = p.sector_vol ** 2
        return cls(
            rate_tick=p.rate_tick,
            up_line=(p.rate_up + p.reversion * p.rate_target, -p.reversion),
            down_line=(p.rate_down - p.reversion * p.rate_target, p.reversion),
            drift_a=p.drift_a, beta_a=p.rate_beta_a,
            var_a=p.sector_beta_a ** 2 * s2 + p.idio_vol_a ** 2,
            drift_b=p.drift_b, beta_b=p.rate_beta_b,
            var_b=p.sector_beta_b ** 2 * s2 + p.idio_vol_b ** 2,
            cov=p.sector_beta_a * p.sector_beta_b * s2,
        )

    def firm(self, observable: int):
        if observable == FIRM_A:
            return self.drift_a, self.beta_a, self.var_a
        return self.drift_b, self.beta_b, self.var_b


# ------------------------------------------------------------------- lattice
def terminal_rate_distribution(m: Moments, rate0: float, days: int) -> dict[float, float]:
    """Exact distribution of the rate after `days`, by forward induction.

    Recombining, so this is O(days^2) states rather than O(3^days) paths.
    """
    dist = {rate0: 1.0}
    for _ in range(days):
        nxt: dict[float, float] = defaultdict(float)
        for rate, p in dist.items():
            if p <= 0.0:
                continue
            up = min(max(m.up_line[0] + m.up_line[1] * rate, 0.0), 1.0)
            down = min(max(m.down_line[0] + m.down_line[1] * rate, 0.0), 1.0 - up)
            hold = 1.0 - up - down
            if up:
                nxt[max(round(rate + m.rate_tick, 2), 0.0)] += p * up
            if down:
                nxt[max(round(rate - m.rate_tick, 2), 0.0)] += p * down
            if hold:
                nxt[rate] += p * hold
        dist = dict(nxt)
    return dist


# -------------------------------------------------------------------- pricing
def price(option: BinaryOption, state: dict[int, float], m: Moments) -> float:
    """Probability the option expires in the money. Exact for every shape the
    model produces; falls back to numerical integration for anything exotic."""
    days = option.days_to_expiry
    if days <= 0:
        return option.payoff(state)

    legs = {l.observable: l.weight for l in option.legs}
    r0 = state[RATE]
    firms = [o for o in legs if o != RATE]
    dist = terminal_rate_distribution(m, r0, days)

    # --- rate only: read straight off the lattice
    if not firms:
        w = legs[RATE]
        return sum(p for r, p in dist.items() if w * r >= option.strike - 1e-12)

    # --- one firm, optionally alongside a rate leg: lognormal threshold
    if len(firms) == 1:
        obs = firms[0]
        w = legs[obs]
        w_rate = legs.get(RATE, 0.0)
        drift, beta, var = m.firm(obs)
        sd = math.sqrt(max(var * days, 1e-18))
        v0 = state[obs]
        if v0 <= 0.0:
            return 1.0 if w < 0.0 else 0.0
        total = 0.0
        for r, p in dist.items():
            threshold = (option.strike - w_rate * r) / w
            mu = days * drift + beta * (r - r0)
            if w > 0.0:
                if threshold <= 0.0:
                    total += p
                    continue
                total += p * norm_cdf((math.log(v0 / threshold) + mu) / sd)
            else:
                if threshold <= 0.0:
                    continue
                total += p * (1.0 - norm_cdf((math.log(v0 / threshold) + mu) / sd))
        return total

    # --- A vs B at strike 0: exact via the log-ratio
    #     w_p*V_p >= w_n*V_n  <=>  log(V_p/V_n) >= log(w_n/w_p), and the
    #     difference of the two log-returns is itself normal. Note the variance
    #     is var_a + var_b - 2*cov: if the firms share a sector beta the common
    #     shock cancels exactly and the spread barely moves.
    if (len(firms) == 2 and RATE not in legs and abs(option.strike) < 1e-12
            and legs[firms[0]] * legs[firms[1]] < 0.0):
        a, b = firms
        pos, neg = (a, b) if legs[a] > 0 else (b, a)
        w_pos, w_neg = abs(legs[pos]), abs(legs[neg])
        d_p, b_p, _ = m.firm(pos)
        d_n, b_n, _ = m.firm(neg)
        var = days * (m.var_a + m.var_b - 2.0 * m.cov)
        sd = math.sqrt(max(var, 1e-18))
        base = math.log(state[pos] / state[neg])
        threshold = math.log(w_neg / w_pos)
        total = 0.0
        for r, p in dist.items():
            mu = days * (d_p - d_n) + (b_p - b_n) * (r - r0)
            total += p * norm_cdf((base + mu - threshold) / sd)
        return total

    return _price_by_quadrature(option, state, m, dist, days)


def _price_by_quadrature(option, state, m, dist, days, nodes=48):
    """Two-dimensional Gauss-Hermite for shapes with no closed form.

    Deterministic, unlike Monte Carlo -- repeated calls give identical answers,
    which matters when the price feeds a quoting loop.
    """
    x, w = _hermite(nodes)
    sa = math.sqrt(max(m.var_a * days, 1e-18))
    sb = math.sqrt(max(m.var_b * days, 1e-18))
    rho = 0.0
    if sa > 0 and sb > 0:
        rho = max(-0.999999, min(0.999999, (m.cov * days) / (sa * sb)))
    rho_c = math.sqrt(max(1.0 - rho * rho, 0.0))
    r0 = state[RATE]
    total = 0.0
    for r, pr in dist.items():
        if pr <= 0.0:
            continue
        mu_a = days * m.drift_a + m.beta_a * (r - r0)
        mu_b = days * m.drift_b + m.beta_b * (r - r0)
        acc = 0.0
        for zi, wi in zip(x, w):
            va = state[FIRM_A] * math.exp(mu_a + sa * zi)
            inner = 0.0
            for zj, wj in zip(x, w):
                vb = state[FIRM_B] * math.exp(mu_b + sb * (rho * zi + rho_c * zj))
                inner += wj * option.payoff({RATE: r, FIRM_A: va, FIRM_B: vb})
            acc += wi * inner
        total += pr * acc
    return min(max(total, 0.0), 1.0)


_HERMITE_CACHE: dict[int, tuple] = {}


def _hermite(n: int):
    """Probabilists' Gauss-Hermite nodes/weights, normalised to a standard
    normal. Golub-Welsch on the symmetric tridiagonal Jacobi matrix."""
    if n in _HERMITE_CACHE:
        return _HERMITE_CACHE[n]
    beta = [math.sqrt(i) for i in range(1, n)]
    d = [0.0] * n
    e = beta + [0.0]
    # symmetric QL with implicit shifts
    z = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for l in range(n):
        it = 0
        while True:
            mm = l
            while mm < n - 1 and abs(e[mm]) > 1e-14 * (abs(d[mm]) + abs(d[mm + 1])):
                mm += 1
            if mm == l:
                break
            it += 1
            if it > 50:
                break
            g = (d[l + 1] - d[l]) / (2.0 * e[l])
            r = math.hypot(g, 1.0)
            g = d[mm] - d[l] + e[l] / (g + (r if g >= 0 else -r))
            s = c = 1.0
            p = 0.0
            for i in range(mm - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                r = math.hypot(f, g)
                e[i + 1] = r
                if r == 0.0:
                    d[i + 1] -= p
                    e[mm] = 0.0
                    break
                s, c = f / r, g / r
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
                for k in range(n):
                    f = z[k][i + 1]
                    z[k][i + 1] = s * z[k][i] + c * f
                    z[k][i] = c * z[k][i] - s * f
            else:
                d[l] -= p
                e[l] = g
                e[mm] = 0.0
    nodes = list(d)
    weights = [z[0][i] ** 2 for i in range(n)]
    order = sorted(range(n), key=lambda i: nodes[i])
    nodes = [nodes[i] for i in order]
    weights = [weights[i] for i in order]
    total = sum(weights)
    weights = [w / total for w in weights]
    _HERMITE_CACHE[n] = (nodes, weights)
    return nodes, weights
