"""Seed strategy: intraday mean reversion toward the session VWAP.

For each symbol, z = (close - vwap) / (close * sigma * sqrt(LOOKBACK)), where sigma is the
standard deviation of the last LOOKBACK one-minute log returns. Go against the move when
|z| > ENTRY_Z and exit when |z| < EXIT_Z. Each symbol gets an equal slice of GROSS.
Nothing is traded during the first WARMUP_BARS bars of the session.

Deliberately simple: it is the starting point the Darwin Gödel Machine improves on.
"""
import numpy as np

WARMUP_BARS = 15
LOOKBACK = 30
ENTRY_Z = 2.0
EXIT_Z = 0.5
GROSS = 0.8


class Strategy:
    def __init__(self, symbols):
        self.symbols = list(symbols)
        self.state = {s: 0 for s in self.symbols}

    def on_day_start(self, view):
        self.state = {s: 0 for s in self.symbols}

    def on_bar(self, view):
        if view.bars_today < max(WARMUP_BARS, LOOKBACK + 1):
            return None
        per_symbol = GROSS / len(self.symbols)
        out = {}
        for s in self.symbols:
            b = view.bars(s, today=True)
            vol = b[:, 4]
            if vol.sum() <= 0:
                continue
            typical = (b[:, 1] + b[:, 2] + b[:, 3]) / 3.0
            vwap = float((typical * vol).sum() / vol.sum())
            c = view.close(s, n=LOOKBACK + 1)
            r = np.diff(np.log(c))
            sigma = float(r.std())
            if sigma <= 0:
                continue
            z = (c[-1] - vwap) / (c[-1] * sigma * np.sqrt(LOOKBACK))
            st = self.state[s]
            if st == 0 and abs(z) > ENTRY_Z:
                st = -int(np.sign(z))
            elif st != 0 and abs(z) < EXIT_Z:
                st = 0
            self.state[s] = st
            out[s] = st * per_symbol
        return out
