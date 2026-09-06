"""A simulated exchange, so quoting strategies can be measured rather than argued about.

The mechanics that matter:

* **Requests for quote are blind.** Every market maker is asked for a two-sided
  quote and the order goes to the best price, but nobody is told which side the
  order is until it is filled. Winning therefore selects for being wrong.
* **Margin is charged at worst-case loss** and returned only at expiry.
* **Solvency is checked at end of day**, after expiring options are credited.
  Going below zero ends the session.
* **Scoring is by rank, not profit.** You are scored on the fraction of rival
  market makers you out-earn, so beating a rival by a dollar counts the same as
  beating them by a thousand.

Flow is a mix of noise traders (random side) and informed traders, who see the
winning quote and trade only when it is mispriced in their favour. The informed
fraction is the session's toxicity, and it is what makes some sessions
unprofitable for every market maker at once.
"""
from __future__ import annotations

import random
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bomm.model import RATE, FIRM_A, FIRM_B, Leg, BinaryOption, Parameters
from bomm.pricer import Moments, price


class Book:
    """One participant's margin account, mirroring the exchange's rule."""

    def __init__(self, cash):
        self.cash = self.start = float(cash)
        self.long, self.short = defaultdict(int), defaultdict(int)
        self.bankrupt = False
        self.trades = 0

    def can_afford(self, qty, px, is_buy):
        return self.cash - qty * (px if is_buy else 1.0 - px) >= 0.0

    def fill(self, oid, qty, px, is_buy):
        self.cash -= qty * (px if is_buy else 1.0 - px)
        (self.long if is_buy else self.short)[oid] += qty
        self.trades += 1

    def settle(self, oid, payoff):
        self.cash += self.long.pop(oid, 0) * payoff
        self.cash += self.short.pop(oid, 0) * (1.0 - payoff)

    @property
    def pnl(self):
        return self.cash - self.start


class Rival:
    """A competing market maker: quotes around fair value with a fixed spread
    and a per-session pricing bias."""

    def __init__(self, name, params, bias_sd, half, size, rng):
        self.name, self.params, self.half, self.size = name, params, half, size
        self.bias = rng.gauss(0.0, bias_sd)   # one draw per session
        self.moments = Moments.from_parameters(params)

    def quote(self, option, state):
        t = min(max(price(option, state, self.moments) + self.bias, 0.001), 0.999)
        bid = max(0.0, round(t - self.half, 2))
        ask = min(1.0, round(t + self.half, 2))
        if bid >= ask:
            bid, ask = max(0.0, ask - 0.01), min(1.0, bid + 0.01)
        return bid, ask, self.size


class Session:
    def __init__(self, seed, days=20, warmup=500, cash=1000.0,
                 n_rivals=2, orders_per_day=40, toxicity=None,
                 rival_bias_sd=0.008, rival_half=0.12):
        self.rng = random.Random(seed)
        r = self.rng
        self.params = Parameters(
            drift_a=r.uniform(-.003, .005), drift_b=r.uniform(-.003, .005),
            rate_beta_a=r.uniform(-1.2, .3), rate_beta_b=r.uniform(-1.2, .3),
            sector_beta_a=r.uniform(.3, 1.6), sector_beta_b=r.uniform(.3, 1.6),
            sector_vol=r.uniform(.005, .022),
            idio_vol_a=r.uniform(.008, .030), idio_vol_b=r.uniform(.008, .030),
            rate_up=r.uniform(.12, .30), rate_down=r.uniform(.12, .30),
            reversion=r.uniform(0, .35), rate_target=r.choice([1.5, 2.0, 2.5, 3.0]))
        self.state = {RATE: r.choice([1.0, 1.75, 2.5, 3.25]),
                      FIRM_A: round(r.uniform(400, 1200), 2),
                      FIRM_B: round(r.uniform(400, 1200), 2)}
        self.days, self.warmup, self.cash = days, warmup, cash
        self.n_rivals, self.orders_per_day = n_rivals, orders_per_day
        self.toxicity = r.uniform(0.0, 0.35) if toxicity is None else toxicity
        self.rival_bias_sd, self.rival_half = rival_bias_sd, rival_half
        self._next_id = 0
        self.truth = Moments.from_parameters(self.params)

    def _history(self):
        state, series = dict(self.state), {k: [v] for k, v in self.state.items()}
        for _ in range(self.warmup):
            state = self.params.advance(state, self.rng)
            for k in series:
                series[k].append(state[k])
        self.state = state
        return series

    def _new_options(self, n):
        out = []
        for _ in range(n):
            self._next_id += 1
            days = self.rng.choice([1, 2, 3, 5])
            k = self.rng.random()
            if k < 0.4:
                legs = (Leg(RATE),)
                strike = max(round(self.state[RATE] + self.rng.choice([-.5, -.25, 0, .25, .5]), 2), 0.0)
            elif k < 0.75:
                obs = self.rng.choice([FIRM_A, FIRM_B])
                legs, strike = (Leg(obs),), round(self.state[obs] * self.rng.uniform(.94, 1.07), 2)
            else:
                legs, strike = (Leg(FIRM_A), Leg(FIRM_B, -1.0)), 0.0
            out.append(BinaryOption(legs, strike, days, self._next_id))
        return out

    def run(self, maker):
        """Run one session. Returns (score, maker_book, all_books)."""
        maker.warm_up(self._history())
        books = {"maker": Book(self.cash)}
        rivals = []
        for i in range(self.n_rivals):
            rv = Rival(f"rival{i}", self.params, self.rival_bias_sd,
                       self.rival_half, 20, self.rng)
            rivals.append(rv)
            books[rv.name] = Book(self.cash)
        live = self._new_options(6)

        for _ in range(self.days):
            state = dict(self.state)
            for _ in range(self.orders_per_day):
                if not live or books["maker"].bankrupt:
                    break
                option = self.rng.choice(live)
                fair = price(option, state, self.truth)
                informed = self.rng.random() < self.toxicity
                qty = self.rng.randint(1, 10)

                quotes = {}
                try:
                    q = maker.quote(option, state)
                    quotes["maker"] = (q.bid, q.ask, q.bid_size, q.ask_size)
                except Exception:
                    quotes["maker"] = (0.0, 1.0, 1, 1)
                for rv in rivals:
                    b, a, s = rv.quote(option, state)
                    quotes[rv.name] = (b, a, s, s)

                best_bid = max(v[0] for v in quotes.values())
                best_ask = min(v[1] for v in quotes.values())
                if informed:
                    # trades only when the winning quote is wrong in their favour
                    gain_buy, gain_sell = fair - best_ask, best_bid - fair
                    if max(gain_buy, gain_sell) <= 0.0:
                        continue
                    buys = gain_buy >= gain_sell
                else:
                    buys = self.rng.random() < 0.5

                if buys:
                    winners = [k for k, v in quotes.items() if v[1] == best_ask]
                    who = self.rng.choice(winners)
                    px, cap, maker_buys = quotes[who][1], quotes[who][3], False
                else:
                    winners = [k for k, v in quotes.items() if v[0] == best_bid]
                    who = self.rng.choice(winners)
                    px, cap, maker_buys = quotes[who][0], quotes[who][2], True

                n = min(qty, cap)
                book = books[who]
                if n > 0 and book.can_afford(n, px, maker_buys):
                    book.fill(option.option_id, n, px, maker_buys)
                    if who == "maker":
                        maker.on_fill(option, px, n, maker_buys)

            self.state = self.params.advance(self.state, self.rng)
            surviving, expiring = [], []
            for o in live:
                (expiring if o.days_to_expiry <= 0 else surviving).append(o)
            for o in expiring:
                payoff = o.payoff(self.state)
                for b in books.values():
                    b.settle(o.option_id, payoff)
            live = [o.aged() for o in surviving] + self._new_options(self.rng.randint(1, 3))
            maker.on_day_end(self.state, {o.option_id for o in live})
            for b in books.values():
                if b.cash < 0:
                    b.bankrupt = True
            if books["maker"].bankrupt:
                break

        me = books["maker"]
        if me.bankrupt:
            return 0.0, me, books
        beaten = sum(1 for rv in rivals if me.pnl > books[rv.name].pnl)
        return 0.4 + 0.6 * (beaten / max(len(rivals), 1)), me, books
