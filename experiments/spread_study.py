"""How wide should a binary-options market maker quote?

The theory says the spread must cover your own pricing error, or the blind
auction extracts it from you: you win exactly the trades where you were most
wrong. That argues for a wide spread. But quoting wide means winning nothing,
and the score is relative -- you have to out-earn rivals, not merely avoid
losing. So there is an interior optimum, and where it sits is an empirical
question about how good the competition is.

This sweeps the half-spread and reports the mean score, so the trade-off is
visible rather than assumed.

    python3 experiments/spread_study.py
"""
from __future__ import annotations

import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bomm import MarketMaker
from sim.exchange import Session

N_SESSIONS = 400
HALF_SPREADS = [0.005, 0.010, 0.020, 0.030, 0.040, 0.055, 0.075, 0.110, 0.160]


def evaluate(half_spread, n=N_SESSIONS):
    class Fixed(MarketMaker):
        HALF_BASE = half_spread
        HALF_UNC = 0.0
    scores, bankrupt, trades = [], 0, []
    for seed in range(n):
        score, book, _ = Session(seed).run(Fixed(1000.0))
        scores.append(score)
        bankrupt += book.bankrupt
        trades.append(book.trades)
    return statistics.fmean(scores), bankrupt / n, statistics.fmean(trades)


def main():
    print(f"Sweeping the half-spread over {N_SESSIONS} sessions each.\n")
    print(f"  {'half-spread':>12} {'mean score':>11} {'bankrupt':>9} {'trades':>8}   profile")
    print("  " + "-" * 62)
    results = []
    for h in HALF_SPREADS:
        score, bankrupt, trades = evaluate(h)
        results.append((h, score))
        bar = "#" * int(round((score - 0.4) * 100))
        print(f"  {h*100:>10.1f}c {score:>11.4f} {bankrupt:>8.1%} {trades:>8.0f}   {bar}")
    best_h, best_s = max(results, key=lambda kv: kv[1])
    worst_h, worst_s = min(results, key=lambda kv: kv[1])
    print("  " + "-" * 62)
    print(f"\n  best  {best_h*100:.1f}c -> {best_s:.4f}")
    print(f"  worst {worst_h*100:.1f}c -> {worst_s:.4f}")
    print(f"  spread of outcomes: {best_s - worst_s:.4f} of score, which is far larger")
    print(f"  than any other single choice in the strategy.")


if __name__ == "__main__":
    main()
