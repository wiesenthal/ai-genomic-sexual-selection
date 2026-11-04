"""Command line: python -m dgm_trader <command> ...

  synth         write synthetic minute bars (testing the machinery only)
  fetch         download minute bars (Alpaca stocks/crypto, Yahoo)
  init          create a run directory from a bar file
  evolve        run Darwin Gödel Machine iterations
  report        archive, lineage and deflated-Sharpe summary
  holdout       one-shot evaluation of chosen agents on the holdout sessions (logged)
  paper         forward paper trading (replay or live Alpaca feed; local or Alpaca broker)
  paper-report  summarise paper-trading logs
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import DataConfig, EvoConfig, RunConfig, SimConfig
from .data import fetch_alpaca, fetch_yahoo, save_bars, synthetic_bars


def _evo_overrides(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("evolution settings (saved to the run's config.json)")
    g.add_argument("--llm", choices=["anthropic", "claude-cli", "mock"])
    g.add_argument("--model")
    g.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    g.add_argument("--iterations", type=int)
    g.add_argument("--children", type=int, dest="children_per_iter")
    g.add_argument("--workers", type=int)
    g.add_argument("--ablation", choices=["none", "no_self_improve", "no_open_ended", "best_of_n"])
    g.add_argument("--token-budget", type=int)
    g.add_argument("--seed", type=int)


def _apply_evo(cfg: RunConfig, a) -> None:
    for k in ("llm", "model", "effort", "iterations", "children_per_iter", "workers", "ablation",
              "token_budget", "seed"):
        v = getattr(a, k, None)
        if v is not None:
            setattr(cfg.evo, k, v)


def cmd_synth(a):
    df = synthetic_bars(a.symbols, a.days, a.seed)
    save_bars(df, a.out)
    print(f"wrote {len(df)} synthetic bars ({a.symbols} symbols x {a.days} sessions) to {a.out}")
    print("NOTE: synthetic data is a driftless random walk; use it to test the pipeline, not to find alpha.")


def cmd_fetch(a):
    syms = [s.strip() for s in a.symbols.split(",") if s.strip()]
    import os
    if a.source in ("alpaca", "alpaca-crypto"):
        df = fetch_alpaca(syms, a.start, a.end, crypto=a.source == "alpaca-crypto",
                          key=os.environ.get("APCA_API_KEY_ID"), secret=os.environ.get("APCA_API_SECRET_KEY"),
                          feed=a.feed)
    else:
        df = fetch_yahoo(syms, a.days_back)
    if df.empty:
        sys.exit("no bars returned")
    save_bars(df, a.out)
    print(f"wrote {len(df)} bars for {sorted(df['symbol'].unique())} to {a.out}")


def cmd_init(a):
    run_dir = Path(a.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_copy = run_dir / ("data.csv.gz" if str(a.data).endswith(".gz") else "data.csv")
    shutil.copyfile(a.data, data_copy)
    cfg = RunConfig(
        data=DataConfig(path=str(data_copy.resolve()), tz=a.tz, session=a.session,
                        train_frac=a.train, val_frac=a.val),
        sim=SimConfig(capital=a.capital, allow_short=not a.long_only, flatten_minutes=a.flatten_minutes),
        evo=EvoConfig(),
    )
    cfg.sim.costs.half_spread_bps = a.half_spread_bps
    cfg.sim.costs.slippage_bps = a.slippage_bps
    _apply_evo(cfg, a)
    from .dgm import Run
    run = Run.create(run_dir, cfg)
    train, val, hold = run.days()
    print(f"run created at {run_dir}: {len(train)} train / {len(val)} validation / {len(hold)} holdout sessions")
    print(f"  train {train[0].date}..{train[-1].date}  val {val[0].date}..{val[-1].date}"
          + (f"  holdout {hold[0].date}..{hold[-1].date}" if hold else "  (no holdout sessions)"))
    print(f"  symbols: {train[0].symbols}")


def cmd_evolve(a):
    from .dgm import DGM, Run
    run = Run(Path(a.run))
    _apply_evo(run.cfg, a)
    run.cfg.save(run.path / "config.json")
    DGM(run).evolve(iterations=a.iterations)


def _archive_rows(run):
    rows = run.all_meta()
    return sorted(rows, key=lambda m: m["id"])


def cmd_report(a):
    from .dgm import Run
    from .metrics import dsr, expected_max_sharpe
    run = Run(Path(a.run))
    rows = _archive_rows(run)
    valid = [m for m in rows if m.get("status") == "valid"]
    tried = [m for m in rows if m.get("status") in ("valid", "rejected")]
    tok = sum((m.get("tokens") or {}).get("input_tokens", 0) + (m.get("tokens") or {}).get("output_tokens", 0) for m in rows)
    cost = sum((m.get("tokens") or {}).get("cost_usd", 0.0) for m in rows)
    status_counts = {}
    for m in rows:
        status_counts[m.get("status")] = status_counts.get(m.get("status"), 0) + 1
    print(f"run {run.path}: {len(rows)} agents {status_counts}; LLM tokens {tok:,}; est. cost ${cost:.2f}")
    print(f"{'id':6} {'parent':6} {'status':9} {'score':>6} {'trainSR':>8} {'valSR':>7} {'valRet%':>8} {'turn/d':>6}  summary")
    for m in sorted(rows, key=lambda m: -m.get("score", 0))[: a.top]:
        tr, va = m.get("train", {}), m.get("val", {})
        print(f"{m['id']:6} {str(m.get('parent') or '-'):6} {m.get('status',''):9} {m.get('score',0):6.3f} "
              f"{tr.get('sharpe_ann', float('nan')):8.2f} {va.get('sharpe_ann', float('nan')):7.2f} "
              f"{va.get('total_return', float('nan'))*100:8.3f} {va.get('mean_daily_turnover', float('nan')):6.2f}  "
              f"{(m.get('summary') or m.get('error') or '')[:70].replace(chr(10), ' ')}")
    if len(tried) >= 2:
        best = max(valid, key=lambda m: m["score"])
        trial_srs = [m["val"]["sharpe_daily"] for m in tried if m.get("val")]
        sr0 = expected_max_sharpe(trial_srs, len(tried))
        d = dsr(best["val_daily"], trial_srs, len(tried))
        print(f"\nbest by validation PSR: {best['id']} (PSR {best['score']:.3f}, val Sharpe {best['val']['sharpe_ann']:.2f})")
        print(f"deflated for {len(tried)} evaluated agents: expected max daily Sharpe under no skill = {sr0:.3f} "
              f"(ann. {sr0*math.sqrt(252):.2f}); DSR = {d:.3f}")
        print("A DSR well below 0.95 means the best validation result is consistent with luck across trials.")
    if a.lineage:
        from .dgm import DGM
        chain = DGM.__new__(DGM)
        chain.run = run
        for step in DGM.lineage(chain, a.lineage):
            print(f"  {step['id']}: {step['summary'][:200]}")


def cmd_holdout(a):
    from .dgm import Run
    from .evaluate import run_backtest
    from .sandbox import protect_private
    run = Run(Path(a.run))
    _, _, hold = run.days()
    if not hold:
        sys.exit("this run has no holdout sessions")
    log = run.path / "holdout_log.jsonl"
    prior = len(log.read_text().splitlines()) if log.exists() else 0
    ids = _resolve_agents(run, a.agents)
    print(f"holdout: {len(hold)} sessions {hold[0].date}..{hold[-1].date}; previous holdout openings: {prior}")
    if prior:
        print("WARNING: the holdout has been used before; every extra look weakens it as an unbiased test.")
    for aid in ids:
        r = run_backtest(run.agent_dir(aid), hold, run.cfg.sim, run.cfg.evo, diagnostics=False)
        row = {"time": datetime.now(timezone.utc).isoformat(), "agent": aid, **r.brief(),
               "daily": [d.ret for d in r.day_stats]}
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        protect_private(log)
        print(json.dumps({k: v for k, v in row.items() if k != "daily"}, indent=1))


def _resolve_agents(run, spec: str) -> list[str]:
    valid = [m for m in run.all_meta() if m.get("status") == "valid"]
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok == "best":
            out.append(max(valid, key=lambda m: m["score"])["id"])
        elif tok.startswith("top"):
            out += [m["id"] for m in sorted(valid, key=lambda m: -m["score"])[: int(tok[3:])]]
        elif tok == "seed":
            out.append("a0000")
        else:
            out.append(tok)
    seen = []
    for x in out:
        if x not in seen:
            seen.append(x)
    return seen


def cmd_paper(a):
    from .dgm import Run
    from .paper import AlpacaBroker, AlpacaFeed, LocalBroker, ReplayFeed, run_paper
    run = Run(Path(a.run))
    ids = _resolve_agents(run, a.agents)
    train, val, hold = run.days()
    symbols = train[0].symbols
    if a.feed == "replay":
        days = {"holdout": hold, "val": val, "train": train, "all": train + val + hold}[a.replay_days]
        if not days:
            sys.exit(f"no {a.replay_days} sessions to replay")
        feed = ReplayFeed(days, speed=a.speed)
    else:
        feed = AlpacaFeed(symbols, crypto=a.feed == "alpaca-crypto", session_utc=a.session_utc)
    if a.broker == "alpaca":
        if len(ids) != 1:
            sys.exit("the Alpaca broker trades one agent per paper account; pass a single agent")
        make_broker = lambda: AlpacaBroker(symbols, run.cfg.sim, crypto=a.feed == "alpaca-crypto")
    else:
        make_broker = lambda: LocalBroker(len(symbols), run.cfg.sim)
    kill = {}
    if a.kill_dd_mult:
        for aid in ids:
            m = run.meta(aid)
            base = max((m.get("val") or {}).get("max_drawdown", 0.0), 0.02)
            kill[aid] = a.kill_dd_mult * base
    name = a.name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{a.feed}-{a.broker}"
    out_dir = run.path / "paper" / name
    print(f"paper trading {ids} | feed={a.feed} broker={a.broker} | symbols={symbols} | logs: {out_dir}")
    if kill:
        print("kill rule (drawdown): " + ", ".join(f"{k} > {v:.2%}" for k, v in kill.items()))
    run_paper([(aid, run.agent_dir(aid)) for aid in ids], feed, make_broker, run.cfg.sim, run.cfg.evo,
              out_dir, max_sessions=a.max_sessions, kill_dd=kill)
    cmd_paper_report(argparse.Namespace(run=a.run, name=name))


def cmd_paper_report(a):
    from .paper import paper_report
    base = Path(a.run) / "paper"
    names = [a.name] if a.name else sorted(p.name for p in base.iterdir()) if base.exists() else []
    for n in names:
        print(f"== {n}")
        for aid, s in paper_report(base / n).items():
            print(f"  {aid}: sessions={s['sessions']} cum={s['cum_return']*100:+.3f}% "
                  f"mean={s['mean_daily_bps']:+.2f}bps/day sharpe={s['sharpe_ann']:.2f} psr={s['psr']:.3f} "
                  f"maxDD={s['max_drawdown']*100:.2f}%{' KILLED' if s['killed'] else ''}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dgm_trader", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="synthetic bars for testing")
    p.add_argument("--out", required=True)
    p.add_argument("--symbols", type=int, default=4)
    p.add_argument("--days", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("fetch", help="download minute bars")
    p.add_argument("--source", choices=["alpaca", "alpaca-crypto", "yahoo"], required=True)
    p.add_argument("--symbols", required=True, help="comma separated, e.g. SPY,QQQ or BTC/USD,ETH/USD")
    p.add_argument("--start", help="ISO date (Alpaca)")
    p.add_argument("--end", help="ISO date (Alpaca)")
    p.add_argument("--feed", default="iex", help="Alpaca stock feed: iex (free) or sip")
    p.add_argument("--days-back", type=int, default=29, help="Yahoo: days of history (max ~29)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("init", help="create a run")
    p.add_argument("--run", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--tz", default="America/New_York")
    p.add_argument("--session", default="09:30-16:00", help='local session window; "" for 24h data')
    p.add_argument("--train", type=float, default=0.6)
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--flatten-minutes", type=int, default=5)
    p.add_argument("--half-spread-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=0.5)
    _evo_overrides(p)
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("evolve", help="run DGM iterations")
    p.add_argument("--run", required=True)
    _evo_overrides(p)
    p.set_defaults(fn=cmd_evolve)

    p = sub.add_parser("report", help="archive summary")
    p.add_argument("--run", required=True)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--lineage", help="print the lineage of this agent id")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("holdout", help="one-shot holdout evaluation (logged)")
    p.add_argument("--run", required=True)
    p.add_argument("--agents", default="best", help="best, top3, seed, or ids like a0004,a0011")
    p.set_defaults(fn=cmd_holdout)

    p = sub.add_parser("paper", help="forward paper trading")
    p.add_argument("--run", required=True)
    p.add_argument("--agents", default="best", help="best, top3, seed, or ids like a0004,a0011")
    p.add_argument("--feed", choices=["replay", "alpaca", "alpaca-crypto"], default="replay")
    p.add_argument("--broker", choices=["local", "alpaca"], default="local")
    p.add_argument("--replay-days", choices=["holdout", "val", "train", "all"], default="holdout")
    p.add_argument("--speed", type=float, default=0.0, help="replay speed-up (0 = as fast as possible, 1 = real time)")
    p.add_argument("--session-utc", default="13:30-20:00", help="crypto session window in UTC")
    p.add_argument("--max-sessions", type=int)
    p.add_argument("--kill-dd-mult", type=float, default=1.5,
                   help="stop an agent if paper drawdown exceeds this multiple of its validation max drawdown (0 = off)")
    p.add_argument("--name", help="name of this paper session (default: timestamp)")
    p.set_defaults(fn=cmd_paper)

    p = sub.add_parser("paper-report", help="summarise paper logs")
    p.add_argument("--run", required=True)
    p.add_argument("--name")
    p.set_defaults(fn=cmd_paper_report)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
