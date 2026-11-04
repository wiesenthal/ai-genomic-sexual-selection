"""Child-process entry point: runs an agent's Strategy on bars streamed over the protocol.

Protocol (one JSON object per line):
  parent -> {"op": "init", "symbols": [...]}                         child -> {"ok": 1}
  parent -> {"op": "day", "date": "...", "n_bars": 390}              child -> {"ok": 1}
  parent -> {"op": "bar", "i", "t", "b": [[o,h,l,c,v],...], "pos", "eq", "mtc"}
                                                                     child -> {"w": [...] | null}
  parent -> {"op": "end_day"}                                        child -> {"ok": 1}
  parent -> {"op": "close"}
Any exception is reported as {"error": "<traceback>"} and the child exits.
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

from .sandbox import AgentLoader, protocol_streams, write_msg
from .view import MarketView


def _to_weights(out, symbols: list[str], current: np.ndarray):
    if out is None:
        return None
    n = len(symbols)
    if isinstance(out, dict):
        w = current.copy()
        idx = {s: i for i, s in enumerate(symbols)}
        for k, v in out.items():
            if k not in idx:
                raise ValueError(f"on_bar returned unknown symbol {k!r}")
            w[idx[k]] = float(v)
    else:
        w = np.asarray(out, dtype=np.float64).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(f"on_bar returned {w.shape[0]} weights for {n} symbols")
    return [0.0 if not math.isfinite(x) else float(x) for x in w]


def main() -> None:
    pin, pout = protocol_streams()
    strat = view = None
    targets = None
    try:
        loader = AgentLoader(Path.cwd())
        mod = loader.load("strategy")
        for line in pin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            op = msg.get("op")
            if op == "bar":
                view._push(msg["t"], msg["b"], msg["pos"], msg["eq"], msg["mtc"], msg["i"])
                out = strat.on_bar(view)
                w = _to_weights(out, view.symbols, targets)
                if w is not None:
                    targets = np.asarray(w)
                write_msg(pout, {"w": w})
            elif op == "init":
                view = MarketView(msg["symbols"])
                targets = np.zeros(len(msg["symbols"]))
                strat = mod.Strategy(list(msg["symbols"]))
                write_msg(pout, {"ok": 1})
            elif op == "day":
                view._start_day(msg["date"], msg.get("n_bars", 390))
                targets = np.zeros(view.n)
                if hasattr(strat, "on_day_start"):
                    strat.on_day_start(view)
                write_msg(pout, {"ok": 1})
            elif op == "end_day":
                if hasattr(strat, "on_day_end"):
                    strat.on_day_end(view)
                write_msg(pout, {"ok": 1})
            elif op == "close":
                break
            else:
                raise ValueError(f"unknown op {op!r}")
    except Exception:
        write_msg(pout, {"error": traceback.format_exc(limit=8)})
        sys.exit(1)


if __name__ == "__main__":
    main()
