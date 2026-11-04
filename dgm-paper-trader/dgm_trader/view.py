"""The MarketView object a Strategy receives on every bar (lives inside the sandboxed child).

It only ever contains bars that have already closed: the parent streams bars one at a time,
so there is no future data anywhere in the child's memory.
"""
from __future__ import annotations

import numpy as np

O, H, L, C, V = range(5)
_FIELD = {"open": O, "high": H, "low": L, "close": C, "volume": V}


class MarketView:
    """Read-only access to past minute bars, positions and session timing.

    Attributes
    ----------
    symbols : list[str]        tradable symbols, in a fixed order
    date : str                 current session date (YYYY-MM-DD, exchange-local)
    time : int                 epoch seconds (UTC) of the bar that just closed (bar start time)
    bar_index : int            0 for the first bar of the session
    minutes_to_close : int     bars left in the session after this one (0 on the last bar)
    session_bars : int         expected bars in this session (390 for a regular US day)
    equity : float             account equity marked at this bar's close
    positions : dict[str,float] current weights (signed notional / equity) at this bar's close
    """

    def __init__(self, symbols: list[str]):
        self.symbols = list(symbols)
        self.n = len(self.symbols)
        self._idx = {s: i for i, s in enumerate(self.symbols)}
        self._buf = np.zeros((4096, self.n, 5))
        self._times = np.zeros(4096, dtype=np.int64)
        self._len = 0
        self._day_start = 0
        self._prev_close = np.full(self.n, np.nan)
        self.date = ""
        self.time = 0
        self.bar_index = -1
        self.minutes_to_close = 0
        self.session_bars = 0
        self.equity = 0.0
        self._pos = np.zeros(self.n)
        self.sessions_seen = 0

    # ---------------------------------------------------------------- harness side
    def _start_day(self, date: str, session_bars: int) -> None:
        if self._len:
            self._prev_close = self._buf[self._len - 1, :, C].copy()
        self._day_start = self._len
        self.date = date
        self.session_bars = int(session_bars)
        self.bar_index = -1
        self.sessions_seen += 1

    def _push(self, t: int, bars, positions, equity: float, minutes_to_close: int, bar_index: int) -> None:
        if self._len >= len(self._buf):
            nb = np.zeros((len(self._buf) * 2, self.n, 5))
            nb[: self._len] = self._buf[: self._len]
            nt = np.zeros(len(self._times) * 2, dtype=np.int64)
            nt[: self._len] = self._times[: self._len]
            self._buf, self._times = nb, nt
        self._buf[self._len] = np.asarray(bars, dtype=np.float64)
        self._times[self._len] = t
        self._len += 1
        self.time = int(t)
        self.bar_index = int(bar_index)
        self.minutes_to_close = int(minutes_to_close)
        self.equity = float(equity)
        self._pos = np.asarray(positions, dtype=np.float64)

    # ---------------------------------------------------------------- strategy API
    def _series(self, sym, field: int, n: int | None, today: bool) -> np.ndarray:
        j = sym if isinstance(sym, int) else self._idx[sym]
        start = self._day_start if today else 0
        if n is not None:
            start = max(start, self._len - int(n))
        out = self._buf[start: self._len, j, field]
        out = out.view()
        out.flags.writeable = False
        return out

    def open(self, sym, n: int | None = None, today: bool = False) -> np.ndarray:
        return self._series(sym, O, n, today)

    def high(self, sym, n: int | None = None, today: bool = False) -> np.ndarray:
        return self._series(sym, H, n, today)

    def low(self, sym, n: int | None = None, today: bool = False) -> np.ndarray:
        return self._series(sym, L, n, today)

    def close(self, sym, n: int | None = None, today: bool = False) -> np.ndarray:
        return self._series(sym, C, n, today)

    def volume(self, sym, n: int | None = None, today: bool = False) -> np.ndarray:
        return self._series(sym, V, n, today)

    def bars(self, sym, n: int | None = None, today: bool = False) -> np.ndarray:
        """Array of shape (k, 5) with columns open, high, low, close, volume."""
        j = sym if isinstance(sym, int) else self._idx[sym]
        start = self._day_start if today else 0
        if n is not None:
            start = max(start, self._len - int(n))
        out = self._buf[start: self._len, j, :].view()
        out.flags.writeable = False
        return out

    def times(self, n: int | None = None, today: bool = False) -> np.ndarray:
        start = self._day_start if today else 0
        if n is not None:
            start = max(start, self._len - int(n))
        out = self._times[start: self._len].view()
        out.flags.writeable = False
        return out

    def all_closes(self, n: int | None = None, today: bool = False) -> np.ndarray:
        """Array of shape (k, n_symbols) of closes, columns in `symbols` order."""
        start = self._day_start if today else 0
        if n is not None:
            start = max(start, self._len - int(n))
        out = self._buf[start: self._len, :, C].view()
        out.flags.writeable = False
        return out

    def last(self, sym) -> dict:
        j = sym if isinstance(sym, int) else self._idx[sym]
        b = self._buf[self._len - 1, j]
        return dict(zip(_FIELD, map(float, b)))

    def prev_close(self, sym) -> float:
        """Last close of the previous session (nan on the first session)."""
        j = sym if isinstance(sym, int) else self._idx[sym]
        return float(self._prev_close[j])

    def position(self, sym) -> float:
        j = sym if isinstance(sym, int) else self._idx[sym]
        return float(self._pos[j])

    @property
    def positions(self) -> dict:
        return {s: float(self._pos[i]) for i, s in enumerate(self.symbols)}

    @property
    def bars_today(self) -> int:
        return self._len - self._day_start

    @property
    def total_bars(self) -> int:
        return self._len


API_DOC = """
STRATEGY CONTRACT (strategy.py)
-------------------------------
Define a top-level class `Strategy`:

    class Strategy:
        def __init__(self, symbols: list[str]): ...
        def on_day_start(self, view) -> None: ...          # optional
        def on_bar(self, view) -> dict[str, float] | list[float] | None: ...
        def on_day_end(self, view) -> None: ...            # optional

on_bar is called once per one-minute bar, AFTER the bar has closed. Return target portfolio
weights (signed notional / equity) per symbol; a dict may omit symbols (they keep their
current target) and None keeps every target unchanged. The harness then:
  * fills the change at the NEXT bar's open, paying spread + slippage + square-root impact,
    with each fill capped at a fraction of that bar's volume (the rest fills on later bars);
  * clips each weight to [-max_weight, max_weight] (or [0, max_weight] if shorting is off)
    and scales the whole vector so sum(|w|) <= max_gross;
  * forces all targets to 0 in the last `flatten_minutes` bars and closes anything left at
    the final close. Positions are never held overnight (this is a day-trading harness).
Fill timing, costs and PnL are computed by the harness, not by the strategy.

`view` (MarketView) gives only bars that have already closed:
  view.symbols, view.date, view.time (epoch s), view.bar_index (0 = first bar of session),
  view.minutes_to_close, view.session_bars, view.equity, view.positions (dict), view.position(sym)
  view.open/high/low/close/volume(sym, n=None, today=False) -> np.ndarray (oldest first)
  view.bars(sym, n=None, today=False) -> (k, 5) array [open, high, low, close, volume]
  view.all_closes(n=None, today=False) -> (k, n_symbols) array
  view.times(n=None, today=False), view.last(sym) -> dict, view.prev_close(sym)
  view.bars_today, view.total_bars, view.sessions_seen
History spans every earlier session of the current run, not just today. Missing bars are
forward-filled with volume 0 (no fills happen on zero-volume bars).

SANDBOX RULES (all agent code)
------------------------------
Code runs in a separate process with no network, no file access and no secrets.
Allowed imports: numpy, math, statistics, collections, itertools, functools, dataclasses,
typing, re, json, random, heapq, bisect, enum, abc, copy, operator, string, textwrap, difflib,
datetime, plus other .py files in the agent's own folder (`import helpers`).
Not allowed: open/eval/exec/compile/getattr/setattr/globals/__import__, dunder attributes,
numpy file I/O. Strategies must be deterministic (seed any randomness) and fast: there is a
per-bar time limit and a total time limit per evaluation.
""".strip()
