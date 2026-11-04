"""Forward paper trading of archive agents on minute bars.

Feeds
  ReplayFeed   replays stored sessions (optionally in real time) - for testing the live loop
  AlpacaFeed   polls Alpaca's latest-bar endpoints once a minute (stocks or crypto)
Brokers
  LocalBroker  simulated fills with exactly the backtest's rules (next-bar open, costs, caps)
  AlpacaBroker Alpaca *paper* account via REST (market orders; flattens before the close)

Several agents can paper trade side by side with LocalBroker (a forward tournament).
AlpacaBroker supports one agent per account.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .config import EvoConfig, SimConfig
from .data import DayData
from .metrics import max_drawdown, psr
from .sandbox import Child, ChildError, protect_private
from .sim import C, O, V, Simulator, clip_targets

PAPER_TRADING_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"


@dataclass
class MinuteEvent:
    date: str
    i: int                 # bar index within the session
    t: int                 # bar start, epoch seconds UTC
    bars: np.ndarray       # (S, 5) open, high, low, close, volume
    mtc: int               # bars left in the session after this one
    session_bars: int
    new_day: bool
    last_of_day: bool


# =========================================================================== feeds

class ReplayFeed:
    def __init__(self, days: list[DayData], speed: float = 0.0):
        self.days, self.speed = days, speed
        self.symbols = days[0].symbols if days else []

    def events(self):
        for day in self.days:
            T = day.n_bars
            for i in range(T):
                yield MinuteEvent(day.date, i, int(day.times[i]), day.ohlcv[i], T - 1 - i, T,
                                  i == 0, i == T - 1)
                if self.speed > 0:
                    time.sleep(60.0 / self.speed)


def _alpaca_keys():
    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    return key, secret


def _headers():
    key, secret = _alpaca_keys()
    h = {"Accept": "application/json"}
    if key and secret:
        h.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    return h


class AlpacaFeed:
    """Live one-minute bars from Alpaca market data, emitted shortly after each minute closes.

    Stocks: session timing from the trading clock (/v2/clock). Needs API keys.
    Crypto: trades 24/7, so a daily session window in UTC is imposed (e.g. "13:30-20:00").
    """

    def __init__(self, symbols: list[str], crypto: bool = False, session_utc: str = "13:30-20:00",
                 delay_s: float = 4.0, feed: str = "iex"):
        import requests

        self.requests = requests
        self.symbols, self.crypto, self.delay_s, self.feed = list(symbols), crypto, delay_s, feed
        a, b = session_utc.split("-")
        self.s_min = int(a[:2]) * 60 + int(a[3:5])
        self.e_min = int(b[:2]) * 60 + int(b[3:5])
        self.last_t = 0
        self.last_close = None
        self.cur_date = None

    def _get(self, url, params=None):
        r = self.requests.get(url, params=params, headers=_headers(), timeout=20)
        r.raise_for_status()
        return r.json()

    def _session(self, now: datetime):
        """Return (date, open_dt, close_dt) of the session containing `now`, or None."""
        if self.crypto:
            day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            o = day + timedelta(minutes=self.s_min)
            c = day + timedelta(minutes=self.e_min)
            return (day.strftime("%Y-%m-%d"), o, c) if o <= now < c else None
        clk = self._get(f"{PAPER_TRADING_BASE}/v2/clock")
        if not clk.get("is_open"):
            return None
        close = datetime.fromisoformat(clk["next_close"].replace("Z", "+00:00")).astimezone(timezone.utc)
        cal = self._get(f"{PAPER_TRADING_BASE}/v2/calendar",
                        {"start": close.strftime("%Y-%m-%d"), "end": close.strftime("%Y-%m-%d")})
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
        d = cal[0]
        o = datetime.fromisoformat(f"{d['date']}T{d['open']}").replace(tzinfo=ny).astimezone(timezone.utc)
        return d["date"], o, close

    def _latest(self) -> dict:
        if self.crypto:
            js = self._get(f"{DATA_BASE}/v1beta3/crypto/us/latest/bars", {"symbols": ",".join(self.symbols)})
        else:
            js = self._get(f"{DATA_BASE}/v2/stocks/bars/latest", {"symbols": ",".join(self.symbols), "feed": self.feed})
        return js.get("bars") or {}

    def events(self):
        while True:
            now = datetime.now(timezone.utc)
            nxt = (now.replace(second=0, microsecond=0) + timedelta(minutes=1)).timestamp() + self.delay_s
            time.sleep(max(0.0, nxt - time.time()))
            now = datetime.now(timezone.utc)
            try:
                sess = self._session(now)
            except Exception as e:  # network hiccup: try again next minute
                print(f"[feed] session lookup failed: {e}")
                continue
            if sess is None:
                continue
            date, o, c = sess
            try:
                latest = self._latest()
            except Exception as e:
                print(f"[feed] bar fetch failed: {e}")
                continue
            minute = int(now.replace(second=0, microsecond=0).timestamp()) - 60  # bar that just closed
            if minute <= self.last_t:
                continue
            S = len(self.symbols)
            bars = np.zeros((S, 5))
            if self.last_close is None or self.cur_date != date:
                self.last_close = np.full(S, np.nan)
            for j, s in enumerate(self.symbols):
                b = latest.get(s)
                bt = int(datetime.fromisoformat(b["t"].replace("Z", "+00:00")).timestamp()) if b else 0
                if b and bt >= minute:
                    bars[j] = [b["o"], b["h"], b["l"], b["c"], b["v"]]
                else:  # no trade this minute: forward fill with zero volume
                    px = self.last_close[j] if np.isfinite(self.last_close[j]) else (b["c"] if b else np.nan)
                    bars[j] = [px, px, px, px, 0.0]
            if not np.all(np.isfinite(bars[:, C])):
                continue  # wait until every symbol has a price
            self.last_close = bars[:, C].copy()
            session_bars = int((c - o).total_seconds() // 60)
            i = int((minute - o.timestamp()) // 60)
            mtc = session_bars - 1 - i
            new_day = self.cur_date != date
            self.cur_date = date
            self.last_t = minute
            yield MinuteEvent(date, max(i, 0), minute, bars, max(mtc, 0), session_bars, new_day, mtc <= 0)


# =========================================================================== brokers

class LocalBroker:
    """Simulated paper account using the backtest's execution rules."""

    name = "local"

    def __init__(self, n_symbols: int, sim_cfg: SimConfig):
        self.sim = Simulator(n_symbols, copy.deepcopy(sim_cfg))

    def start_day(self, ev: MinuteEvent) -> None:
        self.sim.start_day(ev.date, ev.bars[:, O])

    def on_bar(self, ev: MinuteEvent):
        before = self.sim.shares.copy()
        fills = self.sim.execute(ev.bars[:, O], ev.bars[:, V])
        eq = self.sim.step(ev.bars, ev.i, before, fills)
        return eq, self.sim.weights(ev.bars[:, C]), [(f.sym, f.qty, f.price, f.cost) for f in fills]

    def set_targets(self, w: np.ndarray, ev: MinuteEvent) -> list:
        self.sim.set_target(w)
        return []

    def end_day(self, ev: MinuteEvent) -> dict:
        d = self.sim.end_day(ev.bars)
        return {"date": d.date, "ret": d.ret, "start_equity": d.start_equity, "end_equity": d.end_equity,
                "costs": d.costs, "fills": d.n_fills, "turnover": d.traded_notional / d.start_equity}


class AlpacaBroker:
    """Alpaca paper account. Orders are market orders sized from the latest close."""

    name = "alpaca"

    def __init__(self, symbols: list[str], sim_cfg: SimConfig, crypto: bool = False):
        import requests

        key, secret = _alpaca_keys()
        if not (key and secret):
            raise RuntimeError("set APCA_API_KEY_ID and APCA_API_SECRET_KEY for your Alpaca PAPER account")
        self.r = requests
        self.symbols, self.cfg, self.crypto = list(symbols), sim_cfg, crypto
        self.start_equity = None

    def _req(self, method, path, **kw):
        resp = self.r.request(method, f"{PAPER_TRADING_BASE}{path}", headers=_headers(), timeout=20, **kw)
        if resp.status_code >= 400:
            raise RuntimeError(f"Alpaca {method} {path}: {resp.status_code} {resp.text[:300]}")
        return resp.json() if resp.content else None

    def _state(self):
        acct = self._req("GET", "/v2/account")
        pos = {p["symbol"].replace("/", ""): float(p["qty"]) for p in self._req("GET", "/v2/positions")}
        return float(acct["equity"]), pos

    def start_day(self, ev: MinuteEvent) -> None:
        self._req("DELETE", "/v2/positions", params={"cancel_orders": "true"})
        self.start_equity, _ = self._state()

    def on_bar(self, ev: MinuteEvent):
        eq, pos = self._state()
        px = ev.bars[:, C]
        w = np.array([pos.get(s.replace("/", ""), 0.0) * px[j] / eq for j, s in enumerate(self.symbols)])
        return eq, w, []

    def set_targets(self, w: np.ndarray, ev: MinuteEvent) -> list:
        eq, pos = self._state()
        px = ev.bars[:, C]
        orders = []
        if self.crypto:
            w = np.clip(w, 0.0, None)  # Alpaca crypto is long-only
        self._req("DELETE", "/v2/orders")  # never stack stale orders
        for j, s in enumerate(self.symbols):
            cur = pos.get(s.replace("/", ""), 0.0)
            want = w[j] * eq / px[j] if px[j] > 0 else 0.0
            if not self.crypto:
                want = math.trunc(want)        # whole shares (fractional shorts are not allowed)
            dq = want - cur
            if abs(dq * px[j]) < max(self.cfg.min_trade_frac * eq, 1.0) and want != 0:
                continue
            if abs(dq) < (1e-6 if self.crypto else 1):
                continue
            # Crossing zero (long -> short) must be two orders on Alpaca.
            legs = [-cur, want] if (cur != 0 and want != 0 and np.sign(cur) != np.sign(want)) else [dq]
            for q in legs:
                body = {"symbol": s, "qty": f"{abs(q):.6f}" if self.crypto else str(int(abs(q))),
                        "side": "buy" if q > 0 else "sell", "type": "market",
                        "time_in_force": "gtc" if self.crypto else "day"}
                orders.append(self._req("POST", "/v2/orders", json=body).get("id"))
        return orders

    def end_day(self, ev: MinuteEvent) -> dict:
        self._req("DELETE", "/v2/positions", params={"cancel_orders": "true"})
        time.sleep(3)
        eq, _ = self._state()
        se = self.start_equity or eq
        return {"date": ev.date, "ret": eq / se - 1.0, "start_equity": se, "end_equity": eq}


# =========================================================================== runner

class PaperAgent:
    def __init__(self, agent_id: str, agent_dir: Path, broker, log_path: Path, evo: EvoConfig,
                 symbols: list[str], kill_dd: float | None):
        self.id, self.broker, self.log_path = agent_id, broker, log_path
        self.child = Child("dgm_trader.strategy_host", agent_dir, mem_mb=evo.mem_mb, cpu_s=7 * 24 * 3600)
        self.child.request({"op": "init", "symbols": symbols}, timeout=60)
        self.kill_dd = kill_dd
        self.daily: list[float] = []
        self.killed = False
        self.evo = evo

    def log(self, obj: dict) -> None:
        with self.log_path.open("a") as f:
            f.write(json.dumps(obj) + "\n")
        protect_private(self.log_path)


def run_paper(agents: list[tuple[str, Path]], feed, make_broker, sim_cfg: SimConfig, evo: EvoConfig,
              out_dir: Path, max_sessions: int | None = None, kill_dd: dict | None = None, log=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.chmod(0o700)
    symbols = feed.symbols
    S = len(symbols)
    pas = [PaperAgent(aid, adir, make_broker(), out_dir / f"{aid}.jsonl", evo, symbols,
                      (kill_dd or {}).get(aid)) for aid, adir in agents]
    sessions = 0
    try:
        for ev in feed.events():
            for pa in pas:
                if ev.new_day:
                    pa.broker.start_day(ev)
                    pa.child.request({"op": "day", "date": ev.date, "n_bars": ev.session_bars}, timeout=30)
                eq, wnow, fills = pa.broker.on_bar(ev)
                try:
                    resp = pa.child.request({"op": "bar", "i": ev.i, "t": ev.t, "b": ev.bars.tolist(),
                                             "pos": np.asarray(wnow).tolist(), "eq": eq, "mtc": ev.mtc},
                                            timeout=evo.bar_timeout_s)
                    w = resp.get("w")
                except ChildError as e:
                    pa.log({"t": ev.t, "error": str(e)[-2000:]})
                    log(f"[{pa.id}] strategy error, going flat: {str(e)[-300:]}")
                    pa.killed = True
                    w = None
                if pa.killed or ev.mtc < sim_cfg.flatten_minutes:
                    w = [0.0] * S
                orders = []
                if w is not None:
                    wv = clip_targets(w, sim_cfg, S)
                    orders = pa.broker.set_targets(wv, ev)
                    w = wv.tolist()
                pa.log({"t": ev.t, "date": ev.date, "i": ev.i, "eq": eq, "w_now": np.round(wnow, 5).tolist(),
                        "target": w, "fills": fills, "orders": orders})
                if ev.last_of_day:
                    pa.child.request({"op": "end_day"}, timeout=30)
                    summary = pa.broker.end_day(ev)
                    pa.daily.append(summary["ret"])
                    dd = max_drawdown(pa.daily)
                    summary.update(type="day", cum_return=float(np.prod(1 + np.array(pa.daily)) - 1),
                                   max_drawdown=dd)
                    if pa.kill_dd and dd > pa.kill_dd and not pa.killed:
                        pa.killed = True
                        summary["killed"] = f"drawdown {dd:.2%} exceeded kill threshold {pa.kill_dd:.2%}"
                    pa.log(summary)
                    log(f"[{pa.id}] {ev.date} ret={summary['ret']*100:+.3f}% cum={summary['cum_return']*100:+.2f}% "
                        f"dd={dd*100:.2f}%{' KILLED' if summary.get('killed') else ''}")
            if ev.last_of_day:
                sessions += 1
                if max_sessions and sessions >= max_sessions:
                    break
    finally:
        for pa in pas:
            pa.child.close()
    return {pa.id: pa.daily for pa in pas}


def paper_report(out_dir: Path) -> dict:
    out = {}
    for p in sorted(Path(out_dir).glob("*.jsonl")):
        days = [json.loads(l) for l in p.read_text().splitlines() if '"type": "day"' in l]
        rets = np.array([d["ret"] for d in days])
        out[p.stem] = {
            "sessions": len(rets),
            "cum_return": float(np.prod(1 + rets) - 1) if len(rets) else 0.0,
            "mean_daily_bps": float(rets.mean() * 1e4) if len(rets) else 0.0,
            "sharpe_ann": float(rets.mean() / rets.std(ddof=1) * math.sqrt(252)) if len(rets) > 1 and rets.std(ddof=1) > 0 else 0.0,
            "psr": psr(rets) if len(rets) else 0.5,
            "max_drawdown": max_drawdown(rets) if len(rets) else 0.0,
            "killed": any(d.get("killed") for d in days),
        }
    return out
