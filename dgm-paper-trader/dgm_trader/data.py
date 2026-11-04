"""Minute-bar data: storage format, session splitting, fetchers and a synthetic generator.

Storage format: CSV (optionally .gz) with columns
    timestamp (UTC, ISO-8601), symbol, open, high, low, close, volume
Each row is one completed one-minute bar labelled by its *start* time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class DayData:
    date: str                  # session date in exchange-local time, YYYY-MM-DD
    times: np.ndarray          # int64 epoch seconds (bar start, UTC), shape (T,)
    symbols: list[str]
    ohlcv: np.ndarray          # float64, shape (T, S, 5); missing bars forward-filled with volume 0

    @property
    def n_bars(self) -> int:
        return len(self.times)


# --------------------------------------------------------------------------- loading

def load_bars(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"timestamp", "symbol", *FIELDS} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0)
    return df.sort_values(["timestamp", "symbol"]).drop_duplicates(["timestamp", "symbol"], keep="last")


def save_bars(df: pd.DataFrame, path: str | Path) -> None:
    out = df[["timestamp", "symbol", *FIELDS]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def to_days(df: pd.DataFrame, tz: str = "America/New_York", session: str = "09:30-16:00",
            min_bars: int = 30) -> list[DayData]:
    """Split a bar table into per-session DayData aligned on a common minute grid."""
    symbols = sorted(df["symbol"].unique().tolist())
    local = df["timestamp"].dt.tz_convert(tz)
    df = df.assign(_date=local.dt.strftime("%Y-%m-%d"), _hm=local.dt.hour * 60 + local.dt.minute)
    if session:
        start, end = session.split("-")
        s_min = int(start[:2]) * 60 + int(start[3:5])
        e_min = int(end[:2]) * 60 + int(end[3:5])
        df = df[(df["_hm"] >= s_min) & (df["_hm"] < e_min)]
    days: list[DayData] = []
    for date, g in df.groupby("_date", sort=True):
        if set(g["symbol"].unique()) != set(symbols):
            continue  # a symbol did not trade at all this session: skip the day
        wide = {f: g.pivot(index="timestamp", columns="symbol", values=f).reindex(columns=symbols)
                for f in FIELDS}
        idx = wide["close"].index
        if len(idx) < min_bars:
            continue
        close = wide["close"].ffill().bfill()  # bfill only fills leading gaps with the first real price
        arr = np.empty((len(idx), len(symbols), 5), dtype=np.float64)
        for k, f in enumerate(FIELDS[:4]):
            arr[:, :, k] = wide[f].fillna(close).to_numpy()
        arr[:, :, 4] = wide["volume"].fillna(0.0).to_numpy()
        times = ((idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy()
        days.append(DayData(date=str(date), times=np.asarray(times, dtype=np.int64),
                            symbols=symbols, ohlcv=arr))
    return days


def split_days(days: list[DayData], train_frac: float, val_frac: float):
    n = len(days)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    return days[:n_train], days[n_train:n_train + n_val], days[n_train + n_val:]


# --------------------------------------------------------------------------- fetchers

ALPACA_DATA = "https://data.alpaca.markets"


def _alpaca_headers(key: str | None, secret: str | None) -> dict:
    h = {"Accept": "application/json"}
    if key and secret:
        h.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    return h


def fetch_alpaca(symbols: list[str], start: str, end: str, crypto: bool = False,
                 key: str | None = None, secret: str | None = None, feed: str = "iex") -> pd.DataFrame:
    """Fetch 1-minute bars from Alpaca market data.

    Stocks need API keys (free account; feed='iex' on the free plan). Crypto bars need no keys.
    """
    import requests

    url = (f"{ALPACA_DATA}/v1beta3/crypto/us/bars" if crypto else f"{ALPACA_DATA}/v2/stocks/bars")
    params = {"symbols": ",".join(symbols), "timeframe": "1Min", "start": start, "end": end,
              "limit": 10000, "sort": "asc"}
    if not crypto:
        params["feed"] = feed
        params["adjustment"] = "all"
    rows = []
    while True:
        r = requests.get(url, params=params, headers=_alpaca_headers(key, secret), timeout=30)
        r.raise_for_status()
        js = r.json()
        for sym, bars in (js.get("bars") or {}).items():
            for b in bars:
                rows.append((b["t"], sym, b["o"], b["h"], b["l"], b["c"], b["v"]))
        token = js.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
        time.sleep(0.3)  # stay well under the free-tier rate limit
    df = pd.DataFrame(rows, columns=["timestamp", "symbol", *FIELDS])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def fetch_yahoo(symbols: list[str], days_back: int = 7) -> pd.DataFrame:
    """Fetch 1-minute bars from Yahoo's chart endpoint (only the last ~30 days, 7 days per request).

    Unofficial endpoint: fine for quick experiments, not for anything you rely on.
    """
    import requests

    rows = []
    now = datetime.now(timezone.utc)
    span_start = now - timedelta(days=min(days_back, 29))
    for sym in symbols:
        chunk_start = span_start
        while chunk_start < now:
            chunk_end = min(chunk_start + timedelta(days=7), now)
            r = requests.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                params={"interval": "1m", "period1": int(chunk_start.timestamp()),
                        "period2": int(chunk_end.timestamp()), "includePrePost": "false"},
                headers={"User-Agent": "Mozilla/5.0 (dgm-trader research script)"}, timeout=30)
            r.raise_for_status()
            res = (r.json().get("chart", {}).get("result") or [None])[0]
            if res and res.get("timestamp"):
                q = res["indicators"]["quote"][0]
                for i, ts in enumerate(res["timestamp"]):
                    vals = [q[f][i] for f in FIELDS]
                    if any(v is None for v in vals[:4]):
                        continue
                    rows.append((pd.Timestamp(ts, unit="s", tz="UTC"), sym, *vals))
            chunk_start = chunk_end
            time.sleep(0.5)
    return pd.DataFrame(rows, columns=["timestamp", "symbol", *FIELDS])


# --------------------------------------------------------------------------- synthetic (tests only)

def synthetic_bars(n_symbols: int = 4, n_days: int = 40, seed: int = 0,
                   start: str = "2024-01-02", tz: str = "America/New_York") -> pd.DataFrame:
    """Random-walk minute bars with a U-shaped intraday volatility profile.

    FOR TESTING THE MACHINERY ONLY. There is no exploitable signal in this data beyond noise
    (prices are a driftless random walk), so any "alpha" found on it is a pipeline artefact.
    """
    rng = np.random.default_rng(seed)
    symbols = [f"SYN{i}" for i in range(n_symbols)]
    sessions = pd.bdate_range(start=start, periods=n_days)
    minutes = np.arange(390)
    u_shape = 1.0 + 1.5 * ((minutes - 195) / 195.0) ** 2
    rows = []
    price = 100.0 * np.exp(rng.normal(0, 0.2, n_symbols))
    for d in sessions:
        open_local = pd.Timestamp(d.strftime("%Y-%m-%d") + " 09:30", tz=tz)
        sig = 0.0006 * u_shape[:, None] * np.exp(rng.normal(0, 0.2, n_symbols))[None, :]
        rets = rng.standard_t(5, size=(390, n_symbols)) * sig / np.sqrt(5 / 3)
        price = price * np.exp(rng.normal(0, 0.004, n_symbols))  # overnight gap
        for t in range(390):
            o = price
            c = price * np.exp(rets[t])
            spread = np.abs(rng.normal(0, sig[t], n_symbols)) * price
            h = np.maximum(o, c) + spread
            lo = np.minimum(o, c) - spread
            v = np.round(rng.lognormal(8 + 0.5 * u_shape[t], 0.5, n_symbols))
            ts = (open_local + pd.Timedelta(minutes=t)).tz_convert("UTC")
            for s in range(n_symbols):
                rows.append((ts, symbols[s], o[s], h[s], lo[s], c[s], v[s]))
            price = c
    return pd.DataFrame(rows, columns=["timestamp", "symbol", *FIELDS])
