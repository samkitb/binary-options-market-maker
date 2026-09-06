"""The market maker.

Quoting is not the same problem as pricing. Two features of this market dominate
the design, and neither is about fair value.

**Margin is charged at worst-case loss.** Buying q contracts at p costs q*p;
SELLING q at p costs q*(1-p). Selling a 5c contract earns you five cents and
freezes ninety-five. Capital is returned only at expiry, so the binding
constraint on how much you can trade is capital, not conviction.

**Requests for quote are blind two-sided auctions.** You are asked for a bid and
an offer without being told which side the order is, and the order is routed to
whoever quotes best. You therefore win exactly when you are the most aggressive
in the room -- which is to say, when you disagree most with everyone else. Being
filled is itself evidence your price was off. The consequence is that the spread
must be wide enough that even the winner makes money, and quoting theo +/- a
penny is a losing strategy no matter how good the pricer is.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .estimator import estimate
from .model import BinaryOption
from .pricer import Moments, price


class Quote:
    __slots__ = ("bid", "bid_size", "ask", "ask_size")

    def __init__(self, bid, bid_size, ask, ask_size):
        if not (0.0 <= bid < ask <= 1.0):
            raise ValueError(f"invalid quote {bid}/{ask}")
        if bid_size <= 0 or ask_size <= 0:
            raise ValueError("sizes must be positive")
        self.bid, self.bid_size, self.ask, self.ask_size = bid, bid_size, ask, ask_size

    def __repr__(self):
        return f"Quote({self.bid:.2f} x{self.bid_size} / {self.ask:.2f} x{self.ask_size})"


class MarketMaker:
    # Half-spread = HALF_BASE + HALF_UNC * (our own estimation error).
    # These are not arbitrary: see README, "What the spread should be".
    HALF_BASE = 0.015
    HALF_UNC = 0.35
    MAX_SIZE = 20
    SKEW_STRENGTH = 0.001    # price shift per net contract held
    SKEW_CAP = 0.06
    BUFFER = 0.25            # fraction of capital never committed
    CALIB_GAIN = 0.15        # how hard to lean on the fill-imbalance signal
    CALIB_MIN = 12

    def __init__(self, cash: float):
        self.cash = float(cash)
        self.start_cash = float(cash)
        self.moments: Moments | None = None
        self.uncertainty = 0.05
        self.long: dict[int, int] = defaultdict(int)
        self.short: dict[int, int] = defaultdict(int)
        self.last_quote: dict[int, tuple[float, float]] = {}
        self.option_ref: dict[int, BinaryOption] = {}
        self.n_buys = self.n_sells = 0

    # ---------------------------------------------------------------- setup
    def warm_up(self, history: dict[int, list[float]]) -> None:
        self.moments = estimate(history)
        # Our own pricing error, calibrated offline against ground truth:
        # roughly 7c at 400 days of history, 3c at 1200.
        n = max(len(history[next(iter(history))]), 1)
        self.uncertainty = min(0.12, max(0.015, 1.45 / math.sqrt(n)))

    # ------------------------------------------------------------- pricing
    def fair_value(self, option: BinaryOption, state: dict[int, float]) -> float:
        if self.moments is None:
            return 0.5
        return min(max(price(option, state, self.moments), 0.0), 1.0)

    def theo(self, option: BinaryOption, state: dict[int, float]) -> float:
        """Fair value, adjusted by what our own fills imply about the market.

        The other market makers' quotes are a second, independent estimate of
        fair value, and the auction leaks it: winning mostly BUYS means our
        price sits above the field's consensus, so shade it down.
        """
        t = self.fair_value(option, state)
        n = self.n_buys + self.n_sells
        if self.CALIB_GAIN and n >= self.CALIB_MIN:
            imbalance = (self.n_buys - self.n_sells) / n
            t -= self.CALIB_GAIN * imbalance
        return min(max(t, 0.0), 1.0)

    # ------------------------------------------------------------- capital
    @staticmethod
    def margin(qty: int, px: float, is_buy: bool) -> float:
        """Worst-case loss, which is what the exchange debits."""
        return qty * (px if is_buy else 1.0 - px)

    def headroom(self) -> float:
        return self.cash - self.BUFFER * self.start_cash

    def max_size(self, px: float, is_buy: bool) -> int:
        per = self.margin(1, px, is_buy)
        if per <= 1e-9:
            return self.MAX_SIZE          # riskless side; size is free
        return max(0, min(self.MAX_SIZE, int(self.headroom() / per)))

    # ------------------------------------------------------------ inventory
    def net(self, option_id: int) -> int:
        return self.long.get(option_id, 0) - self.short.get(option_id, 0)

    def skew(self, option: BinaryOption) -> float:
        s = self.net(option.option_id) * self.SKEW_STRENGTH
        return max(-self.SKEW_CAP, min(self.SKEW_CAP, s))

    def half_spread(self, option: BinaryOption) -> float:
        return self.HALF_BASE + self.HALF_UNC * self.uncertainty

    # --------------------------------------------------------------- quoting
    def quote(self, option: BinaryOption, state: dict[int, float]) -> Quote:
        t = self.theo(option, state)
        half = self.half_spread(option)
        sk = self.skew(option)
        bid = min(max(math.floor((t - half - sk) * 100) / 100, 0.0), 0.99)
        ask = min(max(math.ceil((t + half - sk) * 100) / 100, 0.01), 1.0)

        bq, aq = self.max_size(bid, True), self.max_size(ask, False)
        # A side we cannot fund retreats to its riskless extreme rather than
        # going dark: buying at 0.00 and selling at 1.00 both cost zero margin
        # and cannot lose, so they are always legal to show.
        if bq <= 0:
            bid, bq = 0.0, 1
        if aq <= 0:
            ask, aq = 1.0, 1
        if bid >= ask:
            bid = max(0.0, min(bid, ask - 0.01))
            if bid >= ask:
                bid, ask, bq, aq = 0.0, 1.0, 1, 1
        self.last_quote[option.option_id] = (bid, ask)
        return Quote(round(bid, 2), max(1, bq), round(ask, 2), max(1, aq))

    # ------------------------------------------------------------ accounting
    def on_fill(self, option: BinaryOption, px: float, qty: int, is_buy: bool) -> None:
        self.cash -= self.margin(qty, px, is_buy)
        self.option_ref[option.option_id] = option
        if is_buy:
            self.long[option.option_id] += qty
            self.n_buys += 1
        else:
            self.short[option.option_id] += qty
            self.n_sells += 1

    def on_day_end(self, state: dict[int, float], live_ids: set[int]) -> None:
        """Settle anything that has expired. Crediting can only raise cash."""
        for oid in list(set(self.long) | set(self.short)):
            if oid in live_ids:
                continue
            opt = self.option_ref.get(oid)
            if opt is None:
                continue
            payoff = opt.payoff(state)
            self.cash += self.long.pop(oid, 0) * payoff
            self.cash += self.short.pop(oid, 0) * (1.0 - payoff)
            self.last_quote.pop(oid, None)
            self.option_ref.pop(oid, None)

    @property
    def pnl(self) -> float:
        return self.cash - self.start_cash
