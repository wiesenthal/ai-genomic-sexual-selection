"""Performance statistics: Sharpe, Probabilistic and Deflated Sharpe, drawdown, diagnostics."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329
TRADING_DAYS = 252


def _moments(r: np.ndarray):
    mu, sd = float(r.mean()), float(r.std(ddof=1)) if len(r) > 1 else 0.0
    if sd <= 0:
        return mu, sd, 0.0, 3.0
    z = (r - mu) / sd
    return mu, sd, float((z ** 3).mean()), float((z ** 4).mean())


def psr(returns, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012), per-period Sharpe units.

    Probability that the true Sharpe exceeds `sr_benchmark` given the sample, adjusting for
    sample length, skewness and kurtosis.
    """
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    if n < 3:
        return 0.5 if sr_benchmark == 0 else 0.0
    mu, sd, g3, g4 = _moments(r)
    if sd <= 0:
        return 0.5 if mu == 0 else (1.0 if mu > 0 else 0.0)
    sr = mu / sd
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    if denom <= 1e-12:
        denom = 1e-12
    return float(_N.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)))


def expected_max_sharpe(trial_sharpes, n_trials: int | None = None) -> float:
    """SR0 of the Deflated Sharpe Ratio: expected max Sharpe of N skill-less trials."""
    s = np.asarray(trial_sharpes, dtype=np.float64)
    N = int(n_trials or len(s))
    if N < 2 or len(s) < 2:
        return 0.0
    v = float(s.var(ddof=1))
    return math.sqrt(max(v, 0.0)) * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1.0 / N)
                                     + EULER_GAMMA * _N.inv_cdf(1 - 1.0 / (N * math.e)))


def dsr(returns, trial_sharpes, n_trials: int | None = None) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014): PSR against SR0."""
    return psr(returns, expected_max_sharpe(trial_sharpes, n_trials))


def max_drawdown(returns) -> float:
    eq = np.cumprod(1.0 + np.asarray(returns, dtype=np.float64))
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float((1.0 - eq / peak).max())


def summarize(days, symbols: list[str], diagnostics: bool = True) -> dict:
    """Metrics from a list of sim.DayStats."""
    if not days:
        return {"n_days": 0}
    rets = np.array([d.ret for d in days])
    start = np.array([d.start_equity for d in days])
    costs = np.array([d.costs for d in days])
    turnover = np.array([d.traded_notional for d in days]) / start
    gross = rets + costs / start
    mu, sd, g3, g4 = _moments(rets)
    sr_d = mu / sd if sd > 0 else 0.0
    exposure = np.array([d.gross_exposure_sum / d.n_bars if d.n_bars else 0.0 for d in days])
    m = {
        "n_days": int(len(rets)),
        "mean_daily_ret": mu,
        "std_daily_ret": sd,
        "sharpe_daily": sr_d,
        "sharpe_ann": sr_d * math.sqrt(TRADING_DAYS),
        "psr": psr(rets),
        "total_return": float(np.prod(1 + rets) - 1),
        "max_drawdown": max_drawdown(rets),
        "win_rate": float((rets > 0).mean()),
        "mean_daily_turnover": float(turnover.mean()),
        "mean_daily_cost": float((costs / start).mean()),
        "mean_daily_gross_ret": float(gross.mean()),
        "fills_per_day": float(np.mean([d.n_fills for d in days])),
        "avg_gross_exposure": float(exposure.mean()),
        "skew": g3,
        "kurtosis": g4,
    }
    if diagnostics:
        cap = float(start.mean())
        by_sym = np.sum([d.pnl_by_symbol for d in days], axis=0) / cap
        buckets: dict[int, float] = {}
        for d in days:
            for b, v in d.pnl_by_bucket.items():
                buckets[b] = buckets.get(b, 0.0) + v / cap
        order = np.argsort(rets)
        pos_pnl = np.clip(rets, 0, None)
        top5 = float(np.sort(pos_pnl)[-5:].sum() / pos_pnl.sum()) if pos_pnl.sum() > 0 else 0.0

        def _day(i):
            d = days[i]
            return {"date": d.date, "ret": float(rets[i]), "turnover": float(turnover[i]),
                    "cost": float(costs[i] / start[i]), "fills": d.n_fills}

        m["diagnostics"] = {
            "pnl_by_symbol_pct_of_capital": {s: round(float(v) * 100, 4) for s, v in zip(symbols, by_sym)},
            "pnl_by_30min_bucket_pct_of_capital": {
                f"{b * 30:03d}-{b * 30 + 29:03d}min": round(v * 100, 4) for b, v in sorted(buckets.items())},
            "worst_days": [_day(i) for i in order[:3]],
            "best_days": [_day(i) for i in order[-3:][::-1]],
            "share_of_positive_pnl_from_top5_days": round(top5, 3),
            "gross_vs_net_daily_mean_bps": {"gross": round(float(gross.mean()) * 1e4, 3),
                                             "net": round(mu * 1e4, 3)},
            "daily_returns_bps": [round(float(r) * 1e4, 2) for r in rets],
        }
    return m


def dgm_score(val_metrics: dict, min_turnover: float) -> float:
    """Archive fitness in [0, 1]: PSR of validation daily returns; inactive agents score 0."""
    if not val_metrics or val_metrics.get("n_days", 0) < 2:
        return 0.0
    if val_metrics.get("mean_daily_turnover", 0.0) < min_turnover:
        return 0.0
    return float(val_metrics["psr"])
