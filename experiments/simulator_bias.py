"""A documented case of a simulator confidently pointing the wrong way.

This experiment exists because I got the problem wrong, and the shape of the
error is more useful than the eventual answer.

Working the live challenge, I built the simulator in `sim/exchange.py`, fitted
its parameters to the small amount of feedback I had, and used it to decide what
to try. It told me to quote wide. I then spent a long session using it to
*reject* ideas -- concluding, with real measurements behind me, that no further
improvement was available.

Against the actual market the opposite was true. Tightening the half-spread from
roughly 11c to roughly 4c was worth more than every other change combined, and
it was the single largest improvement of the entire effort.

The sweep below reproduces the disagreement. Run it and you will see this
simulator prefer a wide spread across every setting of rival skill and flow
toxicity -- a stable, confident, wrong recommendation. Nothing is broken in it:
the mechanics, the margin rule and the scoring are all faithful. What is wrong
is a modelling assumption about how well the competition prices, and that
assumption is invisible from inside.

The lesson I would take to the next problem: a simulator you calibrated yourself
cannot tell you where you sit relative to opponents you have never observed. It
can tell you whether your code is correct, whether you can go bankrupt, and how
your own choices trade off against each other. Treat its verdict on anything
involving the competition as a hypothesis, and spend real measurements testing
that hypothesis rather than refining the model that produced it.

    python3 experiments/simulator_bias.py
"""
from __future__ import annotations

import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bomm import MarketMaker
from sim.exchange import Session

N_SESSIONS = 300
OUR_SPREADS = [0.02, 0.04, 0.06, 0.09, 0.13]


def evaluate(our_half, rival_half, toxicity, n=N_SESSIONS):
    class Fixed(MarketMaker):
        HALF_BASE = our_half
        HALF_UNC = 0.0
    return statistics.fmean(
        Session(seed, rival_half=rival_half, toxicity=toxicity).run(Fixed(1000.0))[0]
        for seed in range(n))


def main():
    print(__doc__.split("    python3")[0].strip()[:0] or "", end="")
    print("What this simulator recommends, across rival skill and flow toxicity.\n")
    for toxicity in (0.05, 0.20, 0.35):
        print(f"  toxicity = {toxicity:.2f}")
        print("    " + f"{'rivals':>8} " + "".join(f"{o*100:>8.0f}c" for o in OUR_SPREADS) + "     best")
        for rival in (0.03, 0.06, 0.09, 0.13):
            vals = {o: evaluate(o, rival, toxicity) for o in OUR_SPREADS}
            best = max(vals, key=vals.get)
            row = "".join(f"{vals[o]:>9.3f}" for o in OUR_SPREADS)
            print(f"    {rival*100:>7.0f}c {row}   {best*100:>5.0f}c")
        print()
    print("  Every row prefers a wide spread. The real market preferred ~4c.")
    print()
    print("  The simulator is not broken -- its mechanics, margin rule and")
    print("  scoring are faithful. It encodes one wrong assumption about how")
    print("  well the competition prices, and that assumption is invisible")
    print("  from the inside. No amount of tuning against it would have found")
    print("  the answer, because the answer was not in it.")


if __name__ == "__main__":
    main()
