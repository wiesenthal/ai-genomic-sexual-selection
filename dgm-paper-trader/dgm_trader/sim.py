"""Execution simulator shared by backtests and the local paper broker.

Timing convention (no lookahead by construction):
  * the strategy sees bar t only after it has closed and returns target weights;
  * targets are filled at the OPEN of bar t+1, with spread, slippage, impact and a volume cap;
  * equity is marked at each bar's close;
  * the harness forces targets to 0 in the last `flatten_minutes` bars, and anything still
    open at the final bar is closed at that bar's close (with costs) so no position is held
    overnight.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import SimConfig

O, H, L, C, V = range(5)


def clip_targets(w, cfg: SimConfig, n: int) -> np.ndarray:
    """Sanitise strategy output into a feasible weight vector."""
    w = np.nan_to_num(np.asarray(w, dtype=np.float64).reshape(n), nan=0.0, posinf=0.0, neginf=0.0)
    lo = -cfg.max_weight if cfg.allow_short else 0.0
    w = np.clip(w, lo, cfg.max_weight)
    gross = np.abs(w).sum()
    if gross > cfg.max_gross > 0:
        w = w * (cfg.max_gross / gross)
    return w


@dataclass
class Fill:
    sym: int
    qty: float
    price: float
    cost: float          # dollars paid in costs relative to the reference price


@dataclass
class DayStats:
    date: str
    start_equity: float
    end_equity: float = 0.0
    costs: float = 0.0
    traded_notional: float = 0.0
    n_fills: int = 0
    gross_exposure_sum: float = 0.0
    n_bars: int = 0
    pnl_by_symbol: np.ndarray | None = None
    pnl_by_bucket: dict = field(default_factory=dict)   # 30-minute bucket index -> pnl

    @property
    def ret(self) -> float:
        return self.end_equity / self.start_equity - 1.0 if self.start_equity > 0 else 0.0


class Simulator:
    def __init__(self, n_symbols: int, cfg: SimConfig):
        self.n = n_symbols
        self.cfg = cfg
        self.cash = cfg.capital
        self.shares = np.zeros(n_symbols)
        self.target = np.zeros(n_symbols)
        self.dirty = np.zeros(n_symbols, dtype=bool)   # target changed and not yet fully executed
        self.last_close = None
        self.day: DayStats | None = None

    # ------------------------------------------------------------------ state
    def equity(self, prices: np.ndarray) -> float:
        return float(self.cash + np.dot(self.shares, prices))

    def weights(self, prices: np.ndarray) -> np.ndarray:
        eq = self.equity(prices)
        return self.shares * prices / eq if eq > 0 else np.zeros(self.n)

    def start_day(self, date: str, ref_prices: np.ndarray) -> None:
        self.target[:] = 0.0
        self.dirty[:] = False
        self.last_close = ref_prices.copy()
        self.day = DayStats(date=date, start_equity=self.equity(ref_prices),
                            pnl_by_symbol=np.zeros(self.n))

    def set_target(self, w: np.ndarray) -> None:
        changed = np.abs(w - self.target) > 1e-9
        self.target = w.copy()
        self.dirty |= changed

    # ------------------------------------------------------------------ fills
    def _fill(self, s: int, dq: float, ref_px: float, volume: float) -> Fill | None:
        c = self.cfg.costs
        if volume <= 0 or ref_px <= 0:
            return None
        cap = c.max_participation * volume
        if abs(dq) > cap:
            dq = math.copysign(cap, dq)
        participation = abs(dq) / volume
        bps = c.half_spread_bps + c.slippage_bps + c.commission_bps + c.impact_coef_bps * math.sqrt(participation)
        px = ref_px * (1.0 + math.copysign(bps, dq) / 1e4)
        self.cash -= dq * px
        self.shares[s] += dq
        cost = abs(dq) * ref_px * bps / 1e4
        if self.day is not None:
            self.day.costs += cost
            self.day.traded_notional += abs(dq) * ref_px
            self.day.n_fills += 1
        return Fill(s, dq, px, cost)

    def execute(self, ref_prices: np.ndarray, volumes: np.ndarray) -> list[Fill]:
        """Move dirty positions toward target at the given reference (open) prices."""
        fills: list[Fill] = []
        if not self.dirty.any():
            return fills
        eq = self.equity(ref_prices)
        for s in np.flatnonzero(self.dirty):
            want = self.target[s] * eq / ref_prices[s] if ref_prices[s] > 0 else 0.0
            dq = want - self.shares[s]
            if abs(dq * ref_prices[s]) < self.cfg.min_trade_frac * eq and self.target[s] != 0.0:
                self.dirty[s] = False
                continue
            if abs(dq) < 1e-12:
                self.dirty[s] = False
                continue
            f = self._fill(int(s), dq, float(ref_prices[s]), float(volumes[s]))
            if f is not None:
                fills.append(f)
                if abs(want - self.shares[s]) < 1e-9:
                    self.dirty[s] = False
        return fills

    def flatten(self, prices: np.ndarray, volumes: np.ndarray) -> list[Fill]:
        """Close every open position at `prices` (end of session). Ignores the volume cap."""
        fills = []
        for s in np.flatnonzero(np.abs(self.shares) > 1e-12):
            saved = self.cfg.costs.max_participation
            self.cfg.costs.max_participation = float("inf")
            try:
                f = self._fill(int(s), -self.shares[s], float(prices[s]), max(float(volumes[s]), 1.0))
            finally:
                self.cfg.costs.max_participation = saved
            if f is not None:
                fills.append(f)
        self.target[:] = 0.0
        self.dirty[:] = False
        return fills

    # ------------------------------------------------------------------ one bar
    def step(self, bar: np.ndarray, bar_index: int, shares_before: np.ndarray | None = None,
             fills: list[Fill] | None = None) -> float:
        """Mark to the bar close and attribute PnL. Call after `execute` for this bar."""
        close = bar[:, C]
        if self.day is not None:
            if shares_before is None:
                shares_before = self.shares
            trade_cash = np.zeros(self.n)
            for f in fills or ():
                trade_cash[f.sym] += f.qty * f.price
            pnl = self.shares * close - shares_before * self.last_close - trade_cash
            self.day.pnl_by_symbol += pnl
            b = bar_index // 30
            self.day.pnl_by_bucket[b] = self.day.pnl_by_bucket.get(b, 0.0) + float(pnl.sum())
            eq = self.equity(close)
            self.day.gross_exposure_sum += float(np.abs(self.shares * close).sum() / eq) if eq > 0 else 0.0
            self.day.n_bars += 1
        self.last_close = close.copy()
        return self.equity(close)

    def end_day(self, last_bar: np.ndarray) -> DayStats:
        close, vol = last_bar[:, C], last_bar[:, V]
        before = self.shares.copy()
        prev_close = self.last_close.copy()
        fills = self.flatten(close, vol)
        if self.day is not None and fills:
            trade_cash = np.zeros(self.n)
            for f in fills:
                trade_cash[f.sym] += f.qty * f.price
            pnl = self.shares * close - before * prev_close - trade_cash
            # prev_close == close here (already marked), so this is just the closing cost
            self.day.pnl_by_symbol += pnl
        day = self.day
        day.end_equity = self.equity(close)
        self.day = None
        return day
