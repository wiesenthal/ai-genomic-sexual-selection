"""Tests for the harness. Synthetic data only: these check mechanics, not trading skill."""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dgm_trader.config import EvoConfig, RunConfig, SimConfig  # noqa: E402
from dgm_trader.data import DayData, save_bars, synthetic_bars, to_days  # noqa: E402
from dgm_trader.evaluate import run_backtest  # noqa: E402
from dgm_trader.llm import MockLLM  # noqa: E402
from dgm_trader.metrics import dsr, expected_max_sharpe, psr  # noqa: E402
from dgm_trader.sandbox import AgentLoader, lint_source  # noqa: E402
from dgm_trader.sim import C, O, V, Simulator, clip_targets  # noqa: E402

SEED = ROOT / "initial_agent"
SELF_IMPROVE_SRC = (SEED / "self_improve.py").read_text()


@pytest.fixture(scope="module")
def days():
    return to_days(synthetic_bars(n_symbols=3, n_days=8, seed=11))


def make_agent(tmp_path: Path, strategy_src: str, improver_src: str = SELF_IMPROVE_SRC) -> Path:
    d = tmp_path / f"agent{len(list(tmp_path.iterdir()))}"
    d.mkdir()
    (d / "strategy.py").write_text(strategy_src)
    (d / "self_improve.py").write_text(improver_src)
    d.chmod(0o755)
    for p in d.iterdir():
        p.chmod(0o644)
    return d


# ----------------------------------------------------------------------------- lint / sandbox

@pytest.mark.parametrize("src", [
    "import os",
    "import subprocess",
    "from pathlib import Path",
    "x = open('f')",
    "x = eval('1')",
    "x = getattr(object, 'a')",
    "x = ().__class__.__bases__",
    "import numpy as np\nnp.load('x.npy')",
    "import numpy as np\nx = np.lib",
    "x = __builtins__",
    "from . import y",
    "import socket",
])
def test_lint_rejects(src):
    assert lint_source(src), src


def test_lint_accepts_seed_agent():
    local = {"strategy", "self_improve"}
    for f in ("strategy.py", "self_improve.py"):
        assert lint_source((SEED / f).read_text(), local) == []


def test_restricted_builtins_block_open_even_without_lint(tmp_path):
    (tmp_path / "evil.py").write_text("def f():\n    return open('/etc/hostname').read()\n")
    mod = AgentLoader(tmp_path).load("evil")
    with pytest.raises(NameError):
        mod.f()


def test_restricted_import_blocks_os(tmp_path):
    (tmp_path / "evil.py").write_text("import os\n")
    with pytest.raises(ImportError):
        AgentLoader(tmp_path).load("evil")


@pytest.mark.skipif(os.getuid() != 0, reason="privilege drop only happens when the harness runs as root")
def test_child_cannot_read_private_files(tmp_path, days):
    secret = tmp_path / "secret.txt"
    secret.write_text("future prices")
    secret.chmod(0o600)
    # Smuggle a file read past lint via numpy's genfromtxt-like path is banned; use the loader's
    # only remaining route, which is a harness bug if it works: read through pathlib.
    strat = (
        "import numpy as np\n"
        "class Strategy:\n"
        "    def __init__(self, symbols):\n"
        "        self.symbols = symbols\n"
        "    def on_bar(self, view):\n"
        "        return None\n"
    )
    agent = make_agent(tmp_path, strat)
    r = run_backtest(agent, days[:1], SimConfig(), EvoConfig())
    assert r.ok
    # The sandbox user is not root: the private file is unreadable to it.
    import subprocess
    from dgm_trader.sandbox import _limits
    out = subprocess.run([sys.executable, "-c", f"open({str(secret)!r}).read()"],
                         preexec_fn=_limits(1024, 10), capture_output=True, text=True)
    assert out.returncode != 0 and "Permission denied" in out.stderr


# ----------------------------------------------------------------------------- simulator

def test_round_trip_costs_match_cost_model():
    cfg = SimConfig(capital=100_000, flatten_minutes=0)
    cfg.costs.impact_coef_bps = 0.0
    sim = Simulator(1, cfg)
    px = np.array([100.0])
    vol = np.array([1e9])
    sim.start_day("d", px)
    sim.set_target(np.array([0.5]))
    sim.execute(px, vol)
    sim.set_target(np.array([0.0]))
    sim.execute(px, vol)
    bps = cfg.costs.half_spread_bps + cfg.costs.slippage_bps
    expected_cost = 2 * 50_000 * bps / 1e4
    assert sim.equity(px) == pytest.approx(100_000 - expected_cost, rel=1e-9)


def test_volume_cap_spreads_fill_over_bars():
    cfg = SimConfig(capital=1_000_000)
    cfg.costs.max_participation = 0.1
    sim = Simulator(1, cfg)
    px, vol = np.array([10.0]), np.array([1000.0])     # 100 shares per bar max
    sim.start_day("d", px)
    sim.set_target(np.array([0.01]))                     # wants 1000 shares
    for _ in range(3):
        sim.execute(px, vol)
    assert sim.shares[0] == pytest.approx(300.0)
    assert sim.dirty[0]


def test_clip_targets():
    cfg = SimConfig(max_gross=1.0, allow_short=False)
    w = clip_targets([0.8, -0.5, float("nan")], cfg, 3)
    assert list(w) == [0.8, 0.0, 0.0]
    cfg = SimConfig(max_gross=1.0)
    w = clip_targets([1.0, -1.0], cfg, 2)
    assert np.abs(w).sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------------- backtest

def test_seed_agent_runs_and_flattens(days):
    r = run_backtest(SEED, days[:2], SimConfig(), EvoConfig(), record_targets=True)
    assert r.ok, r.error
    assert r.metrics["n_days"] == 2
    for date, i, w in r.targets:
        if i >= days[0].n_bars - SimConfig().flatten_minutes:
            assert all(x == 0.0 for x in w)


def test_no_lookahead_future_perturbation(tmp_path, days):
    """Targets up to bar k must not change when every bar after k is replaced by noise."""
    day = days[0]
    k = 200
    rng = np.random.default_rng(0)
    pert = day.ohlcv.copy()
    pert[k + 1:, :, :4] *= np.exp(rng.normal(0, 0.05, pert[k + 1:, :, :4].shape))
    pert[k + 1:, :, 4] *= 10
    day2 = DayData(day.date, day.times, day.symbols, pert)
    a = run_backtest(SEED, [day], SimConfig(), EvoConfig(), record_targets=True)
    b = run_backtest(SEED, [day2], SimConfig(), EvoConfig(), record_targets=True)
    pa = [t for t in a.targets if t[1] <= k]
    pb = [t for t in b.targets if t[1] <= k]
    assert pa == pb and len(pa) > 0


def test_crashing_strategy_is_reported(tmp_path, days):
    agent = make_agent(tmp_path, "class Strategy:\n    def __init__(self, s):\n        pass\n"
                                 "    def on_bar(self, view):\n        return 1 / 0\n")
    r = run_backtest(agent, days[:1], SimConfig(), EvoConfig())
    assert not r.ok and "ZeroDivisionError" in r.error


def test_slow_strategy_times_out(tmp_path, days):
    agent = make_agent(tmp_path, "class Strategy:\n    def __init__(self, s):\n        pass\n"
                                 "    def on_bar(self, view):\n        while True:\n            pass\n")
    evo = EvoConfig(bar_timeout_s=0.5)
    r = run_backtest(agent, days[:1], SimConfig(), evo)
    assert not r.ok and "timed out" in r.error


def test_print_in_strategy_does_not_break_protocol(tmp_path, days):
    agent = make_agent(tmp_path, "class Strategy:\n    def __init__(self, s):\n        pass\n"
                                 "    def on_bar(self, view):\n        print('hello')\n        return None\n")
    assert run_backtest(agent, days[:1], SimConfig(), EvoConfig()).ok


# ----------------------------------------------------------------------------- paper parity

def test_paper_replay_matches_backtest(tmp_path, days):
    from dgm_trader.paper import LocalBroker, ReplayFeed, paper_report, run_paper
    cfg = SimConfig()
    bt = run_backtest(SEED, days[:3], cfg, EvoConfig())
    out = tmp_path / "paper"
    run_paper([("seed", SEED)], ReplayFeed(days[:3]), lambda: LocalBroker(3, cfg), cfg, EvoConfig(), out,
              log=lambda *_: None)
    rep = paper_report(out)["seed"]
    assert rep["sessions"] == 3
    assert rep["cum_return"] == pytest.approx(bt.metrics["total_return"], abs=1e-12)


# ----------------------------------------------------------------------------- metrics

def test_psr_properties():
    rng = np.random.default_rng(1)
    good = rng.normal(0.002, 0.01, 250)
    bad = -good
    assert psr(good) > 0.9 and psr(bad) < 0.1
    assert psr(np.zeros(10)) == 0.5


def test_dsr_penalises_many_trials():
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.01, 120)
    trials_few = rng.normal(0, 0.05, 5)
    trials_many = rng.normal(0, 0.05, 500)
    assert expected_max_sharpe(trials_many) > expected_max_sharpe(trials_few)
    assert dsr(r, trials_many) < dsr(r, trials_few) <= psr(r) + 1e-12


# ----------------------------------------------------------------------------- DGM loop

def _new_run(tmp_path, seed_agent=SEED, **evo):
    from dgm_trader.config import DataConfig
    from dgm_trader.dgm import Run
    data = tmp_path / "bars.csv"
    save_bars(synthetic_bars(n_symbols=3, n_days=8, seed=5), data)
    cfg = RunConfig(data=DataConfig(path=str(data)), evo=EvoConfig(llm="mock", children_per_iter=2,
                                                                   workers=2, stage1_days=1, **evo))
    return Run.create(tmp_path / "run", cfg, seed_agent=seed_agent)


def test_dgm_loop_with_mock_llm(tmp_path):
    from dgm_trader.dgm import DGM
    run = _new_run(tmp_path)
    dgm = DGM(run, llm=MockLLM(seed=3, break_rate=0.0))
    dgm.evolve(iterations=2, log=lambda *_: None)
    metas = run.all_meta()
    assert len(metas) == 5                      # seed + 2 iterations x 2 children
    assert any(m["status"] == "valid" and m["id"] != "a0000" for m in metas)
    ledger = run.ledger()
    assert len(ledger) == 5 and all("val_daily" in r for r in ledger)
    # harness-private files are not readable by others
    for p in (run.ledger_path, run.path / "config.json", Path(run.cfg.data.path)):
        assert (p.stat().st_mode & 0o077) == 0, p
    # agents never see validation results: the improver payload holds only train metrics
    lineage = dgm.lineage(metas[-1]["id"])
    assert all("val" not in step for step in lineage)


SELF_REWRITING_IMPROVER = '''
def improve(ctx):
    src = ctx.read_file("strategy.py")
    ctx.write_file("strategy.py", src.replace("ENTRY_Z = 2.0", "ENTRY_Z = 2.5"))
    me = ctx.read_file("self_improve.py")
    gen = len([l for l in me.splitlines() if l == "#" + "GEN"])
    ctx.write_file("self_improve.py", me + "\\n#" + "GEN\\n")
    ctx.set_summary("rewrote itself, generation " + str(gen + 1))
    return "ok"
'''


def test_agent_can_rewrite_its_own_improver(tmp_path):
    """The Gödel part: a child's improver is whatever its parent wrote, and it runs next time."""
    from dgm_trader.dgm import DGM
    seed = tmp_path / "seed"
    seed.mkdir()
    shutil.copyfile(SEED / "strategy.py", seed / "strategy.py")
    (seed / "self_improve.py").write_text(SELF_REWRITING_IMPROVER)
    run = _new_run(tmp_path, seed_agent=seed)
    run.cfg.evo.children_per_iter = 1
    dgm = DGM(run, llm=MockLLM())
    dgm.ensure_seed_evaluated()
    child = dgm.make_child(run.meta("a0000"), 1)
    assert child["status"] == "valid", child.get("error")
    assert "self_improve.py" in child["files_changed"]
    grandchild = dgm.make_child(child, 2)
    assert grandchild["summary"] == "rewrote itself, generation 2"
    lines = (run.agent_dir(grandchild["id"]) / "self_improve.py").read_text().splitlines()
    assert lines.count("#GEN") == 2


def test_no_self_improve_ablation_restores_fixed_improver(tmp_path):
    from dgm_trader.dgm import DGM
    seed = tmp_path / "seed"
    seed.mkdir()
    shutil.copyfile(SEED / "strategy.py", seed / "strategy.py")
    (seed / "self_improve.py").write_text(SELF_REWRITING_IMPROVER)
    run = _new_run(tmp_path, seed_agent=seed, ablation="no_self_improve")
    dgm = DGM(run, llm=MockLLM())
    dgm.ensure_seed_evaluated()
    child = dgm.make_child(run.meta("a0000"), 1)
    assert "self_improve.py" not in child["files_changed"]


def test_parent_selection_prefers_high_score_and_few_children(tmp_path):
    from dgm_trader.dgm import DGM
    run = _new_run(tmp_path)
    dgm = DGM(run, llm=MockLLM())
    fake = [{"id": "a", "score": 0.9, "parent": None, "status": "valid"},
            {"id": "b", "score": 0.3, "parent": None, "status": "valid"},
            {"id": "c", "score": 0.9, "parent": None, "status": "valid"},
            {"id": "c1", "score": 0.1, "parent": "c", "status": "valid"},
            {"id": "c2", "score": 0.1, "parent": "c", "status": "valid"}]
    dgm.archive = lambda: fake
    picks = [p["id"] for p in dgm.select_parents(4000)]
    assert picks.count("a") > picks.count("c") > picks.count("b")


def test_improver_cannot_escape_workspace(tmp_path):
    from dgm_trader.dgm import ImproveSession
    run = _new_run(tmp_path)
    sess = ImproveSession(run, run.agent_dir("a0000"), MockLLM())
    for bad in ("../x.py", "/etc/passwd", "sub/x.py", "x.sh", ".hidden.py"):
        with pytest.raises(ValueError):
            sess.handle({"op": "write_file", "name": bad, "text": "x"})
    with pytest.raises(ValueError):
        sess.handle({"op": "delete_file", "name": "strategy.py"})
