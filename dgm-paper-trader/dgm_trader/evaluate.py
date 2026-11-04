"""Backtest an agent directory on a list of sessions by streaming bars to its sandboxed Strategy."""
from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import EvoConfig, SimConfig
from .data import DayData
from .metrics import summarize
from .sandbox import Child, ChildError
from .sim import C, O, V, Simulator, clip_targets


@dataclass
class EvalResult:
    ok: bool
    error: str | None = None
    metrics: dict = field(default_factory=dict)
    day_stats: list = field(default_factory=list)
    target_hash: str = ""
    seconds: float = 0.0
    targets: list = field(default_factory=list)   # (date, bar index, weights) when recorded

    def brief(self) -> dict:
        m = {k: v for k, v in self.metrics.items() if k != "diagnostics"}
        return {"ok": self.ok, "error": self.error, "seconds": round(self.seconds, 2), **m}


def run_backtest(agent_dir: Path, days: list[DayData], sim_cfg: SimConfig, evo: EvoConfig,
                 diagnostics: bool = True, record_targets: bool = False) -> EvalResult:
    t0 = time.monotonic()
    if not days:
        return EvalResult(ok=False, error="no days to evaluate")
    symbols = days[0].symbols
    S = len(symbols)
    sim = Simulator(S, copy.deepcopy(sim_cfg))
    h = hashlib.sha256()
    stats = []
    recorded: list = []
    deadline = t0 + evo.eval_timeout_s
    try:
        with Child("dgm_trader.strategy_host", agent_dir, mem_mb=evo.mem_mb,
                   cpu_s=int(evo.eval_timeout_s) + 30) as ch:
            ch.request({"op": "init", "symbols": symbols}, timeout=60)
            for day in days:
                T = day.n_bars
                sim.start_day(day.date, day.ohlcv[0, :, O])
                ch.request({"op": "day", "date": day.date, "n_bars": T}, timeout=30)
                for i in range(T):
                    bar = day.ohlcv[i]
                    before = sim.shares.copy()
                    fills = sim.execute(bar[:, O], bar[:, V])
                    eq = sim.step(bar, i, before, fills)
                    mtc = T - 1 - i
                    resp = ch.request({"op": "bar", "i": i, "t": int(day.times[i]),
                                       "b": bar.tolist(), "pos": sim.weights(bar[:, C]).tolist(),
                                       "eq": eq, "mtc": mtc}, timeout=evo.bar_timeout_s)
                    w = resp.get("w")
                    if mtc < sim_cfg.flatten_minutes:
                        w = [0.0] * S
                    if w is not None:
                        wv = clip_targets(w, sim_cfg, S)
                        sim.set_target(wv)
                        h.update(f"{day.date}:{i}:".encode() + np.round(wv, 9).tobytes())
                        if record_targets:
                            recorded.append((day.date, i, wv.tolist()))
                    if time.monotonic() > deadline:
                        raise ChildError(f"evaluation exceeded {evo.eval_timeout_s:.0f}s")
                ch.request({"op": "end_day"}, timeout=30)
                stats.append(sim.end_day(day.ohlcv[-1]))
    except ChildError as e:
        return EvalResult(ok=False, error=str(e)[-6000:], seconds=time.monotonic() - t0)
    metrics = summarize(stats, symbols, diagnostics=diagnostics)
    return EvalResult(ok=True, metrics=metrics, day_stats=stats, target_hash=h.hexdigest(),
                      seconds=time.monotonic() - t0, targets=recorded)
