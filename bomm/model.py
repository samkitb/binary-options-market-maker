"""The market model: a mean-reverting policy rate and two correlated firms.

A binary (digital) option pays 1 if its event occurs at expiry and 0 otherwise,
so its fair price *is* the probability of that event. Everything in this package
follows from that one fact.

Three observables evolve daily:

  RATE   a policy rate on a discrete grid, mean-reverting toward a target
  FIRM_A, FIRM_B   two firm valuations driven by a shared sector factor

The rate is a three-state lattice walk (up a tick, down a tick, or hold) whose
transition probabilities tilt toward the target. The firms move multiplicatively
on a log-return built from four pieces: a drift, a sensitivity to the rate move,
a *shared* sector shock, and independent idiosyncratic noise. The shared shock is
the only source of correlation between the firms, and it is what makes spread
contracts ("is A worth more than B?") a genuinely different problem from pricing
either firm alone.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

RATE = 1
FIRM_A = 2
FIRM_B = 3
NAMES = {RATE: "RATE", FIRM_A: "FIRM_A", FIRM_B: "FIRM_B"}


@dataclass(frozen=True)
class Leg:
    """One term of an option's payoff expression: weight * observable."""
    observable: int
    weight: float = 1.0


@dataclass(frozen=True)
class BinaryOption:
    """Pays 1.0 if sum(weight * observable) >= strike at expiry, else 0.0."""
    legs: tuple[Leg, ...]
    strike: float
    days_to_expiry: int
    option_id: int = 0

    def value_of(self, state: dict[int, float]) -> float:
        return sum(l.weight * state[l.observable] for l in self.legs)

    def payoff(self, state: dict[int, float]) -> float:
        return 1.0 if self.value_of(state) >= self.strike else 0.0

    def aged(self) -> "BinaryOption":
        from dataclasses import replace
        return self if self.days_to_expiry == 0 else replace(
            self, days_to_expiry=self.days_to_expiry - 1)

    def __str__(self) -> str:
        parts = []
        for i, l in enumerate(self.legs):
            sign = "" if i == 0 and l.weight > 0 else (" - " if l.weight < 0 else " + ")
            mag = "" if abs(l.weight) == 1 else f"{abs(l.weight):g}*"
            parts.append(f"{sign}{mag}{NAMES[l.observable]}")
        return f"{''.join(parts)} >= {self.strike:g}  ({self.days_to_expiry}d)"


@dataclass(frozen=True)
class Parameters:
    """The true data-generating process."""
    drift_a: float
    drift_b: float
    rate_beta_a: float
    rate_beta_b: float
    sector_beta_a: float
    sector_beta_b: float
    sector_vol: float
    idio_vol_a: float
    idio_vol_b: float
    rate_up: float
    rate_down: float
    reversion: float
    rate_target: float = 2.0
    rate_tick: float = 0.25

    def __post_init__(self) -> None:
        if self.rate_up <= 0 or self.rate_down <= 0:
            raise ValueError("rate move probabilities must be positive")
        if self.rate_up + self.rate_down > 1:
            raise ValueError("rate move probabilities must not exceed 1")
        if not 0 <= self.reversion <= 1:
            raise ValueError("reversion must lie in [0, 1]")

    # -- rate dynamics -----------------------------------------------------
    def transition(self, rate: float) -> tuple[float, float]:
        """P(up), P(down) at this rate. Below target, up is favoured."""
        tilt = self.reversion * (self.rate_target - rate)
        up = min(max(self.rate_up + tilt, 0.0), 1.0)
        down = min(max(self.rate_down - tilt, 0.0), 1.0 - up)
        return up, down

    def step_rate(self, rate: float, ticks: int) -> float:
        return max(round(rate + ticks * self.rate_tick, 2), 0.0)

    # -- one day forward ---------------------------------------------------
    def advance(self, state: dict[int, float],
                rng: random.Random | None = None) -> dict[int, float]:
        r = rng or random
        rate0 = state[RATE]
        up, down = self.transition(rate0)
        draw = r.random()
        if draw < up:
            rate = self.step_rate(rate0, 1)
        elif draw < up + down:
            rate = self.step_rate(rate0, -1)
        else:
            rate = rate0
        d_rate = round(rate - rate0, 2)
        sector = r.gauss(0.0, self.sector_vol)   # shared by BOTH firms
        return {
            RATE: rate,
            FIRM_A: self._advance_firm(state[FIRM_A], d_rate, sector, r,
                                       self.drift_a, self.rate_beta_a,
                                       self.sector_beta_a, self.idio_vol_a),
            FIRM_B: self._advance_firm(state[FIRM_B], d_rate, sector, r,
                                       self.drift_b, self.rate_beta_b,
                                       self.sector_beta_b, self.idio_vol_b),
        }

    @staticmethod
    def _advance_firm(value, d_rate, sector, r, drift, rbeta, sbeta, idio):
        log_return = drift + rbeta * d_rate + sbeta * sector + r.gauss(0.0, idio)
        return round(value * math.exp(log_return), 2)


def simulate_path(params: Parameters, start: dict[int, float], days: int,
                  rng: random.Random | None = None) -> list[dict[int, float]]:
    state, out = dict(start), [dict(start)]
    for _ in range(days):
        state = params.advance(state, rng)
        out.append(dict(state))
    return out
