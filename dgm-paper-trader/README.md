# DGM paper trader

A Darwin Gödel Machine (Zhang, Hu, Lu, Lange & Clune, 2025, arXiv 2505.22954) applied to
minute-bar day-trading agents, plus a paper-trading loop to run the evolved agents forward.

Each **agent** is a small folder of Python:

- `strategy.py` holds the trading logic. It sees one closed minute bar at a time and returns
  target weights.
- `self_improve.py` holds the agent's *own* self-modification procedure. It reads the agent's
  evaluation logs, asks Claude for one improvement and rewrites the agent's files, which can
  include `self_improve.py` itself.

The harness around them is fixed, and the agents cannot edit it. It covers data, execution,
costs, scoring, the archive and the sandbox.

| DGM paper | Here |
|---|---|
| Coding agent that edits its own repo | Trading agent that edits its own `strategy.py` *and* `self_improve.py` |
| SWE-bench / Polyglot score | Probabilistic Sharpe Ratio of daily **net** returns on validation sessions |
| Agent reads its benchmark logs | Agent reads its **train**-session diagnostics (PnL by symbol / time of day, costs, worst days) |
| Archive of all valid agents, open-ended | Same (`agents/`, `meta/`) |
| Parent ∝ sigmoid(λ(α−α₀)) / (1 + children) | Same, λ=10, α₀=0.5 (PSR 0.5 = zero Sharpe) |
| Staged evaluation | Lint → 2× smoke test on a few sessions (crash and determinism check) → full train + validation |
| Ablations: w/o self-improve, w/o open-endedness | `--ablation no_self_improve`, `no_open_ended`, plus `best_of_n` |

## Safeguards built in

**No lookahead, by construction.** Bars are streamed to the strategy process one at a time,
so future bars are never in its memory. A test replaces every bar after bar *k* with noise
and checks that the decisions up to bar *k* do not change.

**Execution the agent can't game.** The harness, not the strategy, computes fills and PnL:
- orders fill at the next bar's open;
- each fill pays spread, slippage and square-root impact;
- each fill is capped at 5% of that bar's volume;
- targets are forced to zero in the last 5 minutes;
- every position is closed at the session close.

**Sandbox.** Agent code is written by the LLM, so it is never imported into the harness. It runs
in a child process with:
- an AST lint (allowed imports only; no `open`, `eval`, `getattr` or dunder attributes);
- restricted builtins;
- no API keys or broker credentials in its environment;
- CPU, memory and file limits.

When the harness runs as root (e.g. in Docker) it also:
- drops the child to user `nobody`, working on a private copy of the agent's files;
- runs the child in its own network namespace (no network at all).

Data, ledger, validation and holdout results are written mode 0600, so the child cannot read
them. If you run as a normal user you get only the lint and restricted builtins. For code you
don't trust, run the whole thing in a container with `--network none` apart from the LLM proxy.

**Information discipline.** Results are split three ways:
- **Train:** agents see full diagnostics, but only for train sessions.
- **Validation:** used only for archive selection. Agents never see these scores.
- **Holdout:** touched only by the `holdout` command, and every opening is logged.

**Multiple-testing honesty.** Every child, including failed ones, goes into `ledger.jsonl`.
`report` prints the **Deflated Sharpe Ratio** of the best agent given how many agents were
tried.

**Kill rule for paper trading.** An agent is stopped if its paper drawdown exceeds 1.5× its
validation max drawdown.

## Quick start

```bash
cd dgm-paper-trader
pip install -r requirements.txt pytest

# 1. Test the machinery offline: synthetic data + a mock LLM (no keys, no network).
python -m dgm_trader synth --out data/synth.csv.gz --symbols 4 --days 30
python -m dgm_trader init  --run runs/smoke --data data/synth.csv.gz --llm mock
python -m dgm_trader evolve --run runs/smoke --iterations 3
python -m dgm_trader report --run runs/smoke
python -m dgm_trader paper  --run runs/smoke --agents top3 --feed replay --replay-days holdout
python -m pytest -q tests
```

Synthetic data is a driftless random walk, and the mock LLM just rescales constants. Runs like
this only show that the pipeline works; their "results" mean nothing.

### Real run (US stocks, minute bars)

You need free [Alpaca](https://alpaca.markets) **paper** keys and an Anthropic API key:

```bash
export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...   # Alpaca paper account
export ANTHROPIC_API_KEY=...

python -m dgm_trader fetch --source alpaca --symbols SPY,QQQ,IWM,XLF,XLE,XLK \
       --start 2025-01-01 --end 2026-10-31 --out data/etf_1m.csv.gz
python -m dgm_trader init --run runs/etf --data data/etf_1m.csv.gz \
       --llm anthropic --model claude-opus-5-5 --effort high --children 4 --workers 4
python -m dgm_trader evolve --run runs/etf --iterations 20 --token-budget 20000000
python -m dgm_trader report --run runs/etf --lineage a0042
python -m dgm_trader holdout --run runs/etf --agents best              # once, at the end
python -m dgm_trader paper --run runs/etf --agents best --feed alpaca --broker alpaca
```

- **Crypto** (24/7, no keys needed for data): use `--source alpaca-crypto --symbols BTC/USD,ETH/USD`.
  Pass `--session ""` to `init`, which keeps every bar of the day. When paper trading, use
  `--feed alpaca-crypto --session-utc 13:30-20:00`; this window defines a "day" so the agent
  still flattens daily. Alpaca crypto is long-only.
- **Quick experiments without Alpaca:** `--source yahoo` gives the last ~29 days of 1-minute
  bars. It is an unofficial endpoint.
- **Claude Code instead of an API key:** `--llm claude-cli` uses a logged-in `claude` CLI on
  this machine, with tools disabled.
- **Several agents side by side:** `--broker local` simulates fills with exactly the backtest's
  rules, so you can paper trade `--agents top5` together. With `--broker alpaca`, only one agent
  can use each paper account.

Network hosts used: `data.alpaca.markets`, `paper-api.alpaca.markets`, `api.anthropic.com`
(and `query1.finance.yahoo.com` for Yahoo).

### Is the self-improvement doing anything?

Run the same data and token budget with the ablations, then compare the reports:

```bash
python -m dgm_trader init --run runs/etf_bon  --data data/etf_1m.csv.gz --ablation best_of_n
python -m dgm_trader init --run runs/etf_nosi --data data/etf_1m.csv.gz --ablation no_self_improve
python -m dgm_trader evolve --run runs/etf_bon  --token-budget 20000000
```

- `best_of_n` always edits the seed agent with the fixed improver. It is the token-matched
  "just sample many times" baseline.
- `no_self_improve` keeps the archive but freezes `self_improve.py`.
- `no_open_ended` hill-climbs from the current best agent.

If full DGM doesn't beat these on validation *and* on the holdout, the evolution isn't adding
anything.

## How a generation works

1. Choose parents from the archive, weighted by validation PSR and by how few children they
   already have.
2. Copy each parent to a new folder. Run the parent's own `improve(ctx)` in the sandbox. It can:
   - `ctx.llm(...)`, limited to 6 calls by default;
   - `ctx.read_file` and `ctx.write_file` in its own folder;
   - `ctx.backtest(max_days)` on train sessions, limited to 3;
   - read `ctx.report` (train diagnostics), `ctx.lineage` and `ctx.siblings` (earlier attempts
     from the same parent).
3. Lint the child. Smoke test it twice, which catches crashes and nondeterminism. Then backtest
   it on all train and validation sessions.
4. A valid child joins the archive. Everything is written to `meta/<id>.json` and
   `ledger.jsonl`.

The default improver uses three prompts: DIAGNOSE (one change, as JSON), IMPLEMENT (full new
files) and REPAIR (if the self-check fails). It's a starting point that descendants are
expected to rewrite.

## Layout

```
dgm_trader/
  dgm.py            archive, parent selection, child creation, staged evaluation
  evaluate.py       backtest: streams bars to the sandboxed strategy, runs the simulator
  sim.py            fills, costs, volume caps, flatten-at-close, PnL attribution
  sandbox.py        lint, restricted loader, child processes (privilege drop, netns, rlimits)
  strategy_host.py  child entry point for strategies
  improver_host.py  child entry point for self_improve.improve(ctx)
  view.py           MarketView API given to strategies (+ API_DOC shown to the LLM)
  metrics.py        Sharpe, PSR, DSR, drawdown, diagnostics
  llm.py            Anthropic SDK / Claude CLI / mock backends, token and cost accounting
  paper.py          replay + Alpaca feeds, local + Alpaca paper brokers, kill rule
  data.py           storage format, session splitting, Alpaca/Yahoo fetchers, synthetic bars
initial_agent/      seed agent: VWAP mean reversion + default self_improve.py
tests/              32 tests (sandbox, lookahead, costs, paper/backtest parity, DGM loop)
```

## Caveats

- Minute-bar day trading with retail-level costs is a hard place to find edge. Expect most
  agents to lose money after costs. On the synthetic random walk the seed strategy loses ~7
  Sharpe on train, almost all of it costs.
- Default costs are 1 bp half-spread + 0.5 bp slippage + impact. That suits liquid ETFs and
  large caps; set `--half-spread-bps` higher for anything else.
- A few weeks of validation sessions gives a very noisy PSR, and selecting across many agents
  inflates it. Trust the DSR in `report`, the holdout and paper results, not the best
  validation number.
- Alpaca's free IEX feed covers only a few percent of US volume, so bar volumes understate the
  real market. Fills in the paper account are Alpaca's simulation, not real fills.
- This is research software. It does not place live-money orders, and nothing here is
  investment advice.
