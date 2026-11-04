"""Darwin Gödel Machine loop over self-modifying day-trading agents.

Mirrors Zhang, Hu, Lu, Lange & Clune, "Darwin Gödel Machine" (arXiv 2505.22954):
  * an ARCHIVE of agents, starting from one seed agent, that only ever grows (open-ended);
  * each iteration samples parents with probability ∝ sigmoid(λ(α − α0)) / (1 + n_children);
  * the parent's OWN self-improvement code (self_improve.py) reads its evaluation logs,
    proposes a change and edits a copy of its own codebase, which may include
    self_improve.py itself;
  * the child is evaluated in stages (cheap smoke test, then the full benchmark) and added
    to the archive if it still works.

Trading-specific choices:
  * benchmark = minute-bar day-trading backtest through a fixed harness the agent cannot
    edit (fills at next open, costs, volume caps, forced flat at the close);
  * the agent sees full diagnostics for TRAIN sessions only; the archive score α is the
    Probabilistic Sharpe Ratio on VALIDATION sessions, which the agent never sees directly;
  * HOLDOUT sessions are touched only by the `holdout` command, and every opening is logged;
  * every child (valid or not) is written to ledger.jsonl with its validation daily returns
    so the final pick can be deflated for the number of trials (Deflated Sharpe Ratio).
"""
from __future__ import annotations

import json
import math
import random
import re
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import RunConfig
from .data import DayData, load_bars, split_days, to_days
from .evaluate import run_backtest
from .llm import LLMError, Usage, make_llm
from .metrics import dgm_score
from .sandbox import (ALLOWED_AGENT_FILE_SUFFIXES, MAX_AGENT_FILE_BYTES, Child, ChildError,
                      check_agent_files, lint_agent_dir, protect_private)
from .view import API_DOC

SEED_AGENT = Path(__file__).resolve().parent.parent / "initial_agent"
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.(py|md|txt|json)$")
PROTECTED = {"strategy.py", "self_improve.py"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_private(path: Path, text: str) -> None:
    path.write_text(text)
    protect_private(path)


def _brief(m: dict) -> dict:
    keys = ("n_days", "sharpe_ann", "psr", "mean_daily_ret", "total_return", "max_drawdown",
            "mean_daily_turnover", "mean_daily_cost", "fills_per_day", "avg_gross_exposure", "win_rate")
    return {k: m[k] for k in keys if k in m}


class Run:
    """A run directory: config, agent code, private metadata, ledger."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.cfg = RunConfig.load(self.path / "config.json")
        self.agents_dir = self.path / "agents"
        self.meta_dir = self.path / "meta"
        self.logs_dir = self.path / "logs"
        self.ledger_path = self.path / "ledger.jsonl"
        self._lock = threading.Lock()
        self._days = None

    # ---------------------------------------------------------------- layout
    @staticmethod
    def create(path: Path, cfg: RunConfig, seed_agent: Path = SEED_AGENT) -> "Run":
        path = Path(path)
        if (path / "config.json").exists():
            raise FileExistsError(f"{path} already contains a run")
        for d in ("agents", "meta", "logs"):
            (path / d).mkdir(parents=True, exist_ok=True)
        path.chmod(0o711)            # sandbox user may traverse into agents/, not list the run
        (path / "agents").chmod(0o755)
        (path / "meta").chmod(0o700)
        (path / "logs").chmod(0o700)
        cfg.save(path / "config.json")
        protect_private(path / "config.json")
        protect_private(Path(cfg.data.path))
        run = Run(path)
        run._install_agent(seed_agent, "a0000")
        return run

    def _install_agent(self, src: Path, agent_id: str) -> Path:
        dst = self.agents_dir / agent_id
        dst.mkdir(mode=0o755)
        for p in sorted(Path(src).iterdir()):
            if p.is_file() and p.name.endswith(ALLOWED_AGENT_FILE_SUFFIXES):
                shutil.copyfile(p, dst / p.name)
                (dst / p.name).chmod(0o644)
        dst.chmod(0o755)
        return dst

    def agent_dir(self, agent_id: str) -> Path:
        return self.agents_dir / agent_id

    # ---------------------------------------------------------------- data
    def days(self):
        if self._days is None:
            d = self.cfg.data
            all_days = to_days(load_bars(d.path), tz=d.tz, session=d.session)
            if len(all_days) < 3:
                raise ValueError(f"need at least 3 sessions of data, found {len(all_days)}")
            self._days = split_days(all_days, d.train_frac, d.val_frac)
        return self._days

    # ---------------------------------------------------------------- metadata
    def meta(self, agent_id: str) -> dict:
        p = self.meta_dir / f"{agent_id}.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def save_meta(self, agent_id: str, meta: dict) -> None:
        _write_private(self.meta_dir / f"{agent_id}.json", json.dumps(meta, indent=1))

    def all_meta(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.meta_dir.glob("a*.json"))]

    def next_id(self) -> str:
        with self._lock:
            existing = [int(p.name[1:]) for p in self.agents_dir.iterdir() if p.name.startswith("a")]
            n = max(existing, default=-1) + 1
            d = self.agents_dir / f"a{n:04d}"
            d.mkdir(mode=0o755)
            return d.name

    def append_ledger(self, row: dict) -> None:
        with self._lock:
            with self.ledger_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            protect_private(self.ledger_path)

    def ledger(self) -> list[dict]:
        if not self.ledger_path.exists():
            return []
        return [json.loads(l) for l in self.ledger_path.read_text().splitlines() if l.strip()]


# ===================================================================== evaluation

def evaluate_agent(run: Run, agent_dir: Path) -> dict:
    """Staged evaluation. Returns a dict with status, score, train/val metrics and errors."""
    cfg = run.cfg
    train, val, _ = run.days()
    out = {"status": "invalid", "score": 0.0}
    errs = check_agent_files(agent_dir)
    lint = lint_agent_dir(agent_dir)
    if errs or lint:
        out["error"] = "; ".join(errs) + (f" lint: {json.dumps(lint)[:3000]}" if lint else "")
        return out
    # Stage 1: smoke test on a few sessions, twice, to check it runs and is deterministic.
    k = max(1, min(cfg.evo.stage1_days, len(train)))
    r1 = run_backtest(agent_dir, train[:k], cfg.sim, cfg.evo, diagnostics=False)
    if not r1.ok:
        out["error"] = f"stage1: {r1.error}"
        return out
    r2 = run_backtest(agent_dir, train[:k], cfg.sim, cfg.evo, diagnostics=False)
    if not r2.ok or r2.target_hash != r1.target_hash:
        out["error"] = "stage1: strategy is not deterministic (two identical runs gave different targets)"
        return out
    # Stage 2: full train (diagnostics for the agent) and validation (score only).
    rt = run_backtest(agent_dir, train, cfg.sim, cfg.evo, diagnostics=True)
    if not rt.ok:
        out["error"] = f"train: {rt.error}"
        return out
    rv = run_backtest(agent_dir, val, cfg.sim, cfg.evo, diagnostics=False)
    if not rv.ok:
        out["error"] = f"val: {rv.error}"
        return out
    out.update(status="valid", train=rt.metrics, val=rv.metrics,
               val_daily=[d.ret for d in rv.day_stats],
               score=dgm_score(rv.metrics, cfg.evo.min_daily_turnover),
               eval_seconds=round(r1.seconds + r2.seconds + rt.seconds + rv.seconds, 1))
    if out["score"] == 0.0 and rv.metrics.get("mean_daily_turnover", 0) < cfg.evo.min_daily_turnover:
        out["note"] = "inactive on validation (turnover below threshold): score 0"
    return out


# ===================================================================== self-improvement

class ImproveSession:
    """Serves one improver child's requests: LLM calls, file edits, train backtests."""

    def __init__(self, run: Run, workspace: Path, llm):
        self.run, self.ws, self.llm = run, workspace, llm
        evo = run.cfg.evo
        self.llm_calls_left = evo.max_llm_calls_per_child
        self.backtests_left = evo.max_backtests_per_child
        self.usage = Usage()
        self.log: list[str] = []
        self.summary = ""

    def budget(self) -> dict:
        return {"llm_calls_left": self.llm_calls_left, "backtests_left": self.backtests_left}

    def _path(self, name: str) -> Path:
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise ValueError(f"invalid file name {name!r}: use a flat name like helpers.py")
        return self.ws / name

    def handle(self, msg: dict):
        op = msg.get("op")
        if op == "llm":
            if self.llm_calls_left <= 0:
                raise RuntimeError("LLM call budget exhausted")
            prompt = str(msg.get("prompt", ""))
            if len(prompt) > 600_000:
                raise ValueError("prompt too long (600k characters max)")
            self.llm_calls_left -= 1
            before = self.usage.tokens
            text = self.llm.complete(prompt, msg.get("system"), msg.get("max_tokens"), self.usage)
            self.log.append(f"llm call: +{self.usage.tokens - before} tokens")
            return text
        if op == "list_files":
            return sorted(p.name for p in self.ws.iterdir() if p.is_file())
        if op == "read_file":
            p = self._path(msg.get("name"))
            if not p.exists():
                raise FileNotFoundError(msg.get("name"))
            return p.read_text()
        if op == "write_file":
            p = self._path(msg.get("name"))
            text = str(msg.get("text", ""))
            if len(text.encode()) > MAX_AGENT_FILE_BYTES:
                raise ValueError(f"file larger than {MAX_AGENT_FILE_BYTES} bytes")
            p.write_text(text)
            p.chmod(0o644)
            return None
        if op == "delete_file":
            p = self._path(msg.get("name"))
            if p.name in PROTECTED:
                raise ValueError(f"{p.name} cannot be deleted")
            p.unlink(missing_ok=True)
            return None
        if op == "lint":
            return {"files": check_agent_files(self.ws), "lint": lint_agent_dir(self.ws)}
        if op == "backtest":
            if self.backtests_left <= 0:
                raise RuntimeError("backtest budget exhausted")
            self.backtests_left -= 1
            problems = check_agent_files(self.ws)
            lint = lint_agent_dir(self.ws)
            if problems or lint:
                return {"ok": False, "error": {"files": problems, "lint": lint}}
            train = self.run.days()[0]
            n = msg.get("max_days")
            days = train[: int(n)] if n else train
            r = run_backtest(self.ws, days, self.run.cfg.sim, self.run.cfg.evo, diagnostics=True)
            return {"ok": r.ok, "error": r.error, "metrics": r.metrics}
        if op == "log":
            self.log.append(str(msg.get("text", "")))
            return None
        if op == "set_summary":
            self.summary = str(msg.get("text", ""))
            return None
        raise ValueError(f"unknown op {op!r}")


def run_improver(run: Run, workspace: Path, payload: dict, llm) -> ImproveSession:
    evo = run.cfg.evo
    sess = ImproveSession(run, workspace, llm)
    deadline = time.monotonic() + evo.improve_timeout_s
    with Child("dgm_trader.improver_host", workspace, mem_mb=evo.mem_mb, cpu_s=evo.improve_timeout_s) as ch:
        ch.send({"op": "init", "payload": {**payload, "budget": sess.budget()}})
        while True:
            msg = ch.recv(timeout=max(1.0, deadline - time.monotonic()))
            if msg.get("op") == "done":
                sess.log.append(f"improve() returned: {msg.get('result')}")
                break
            try:
                value = sess.handle(msg)
                ch.send({"value": value, "budget": sess.budget()})
            except Exception as e:  # LLM, budget or file errors are reported to the agent
                ch.send({"err": f"{type(e).__name__}: {e}", "budget": sess.budget()})
    return sess


# ===================================================================== the DGM loop

class DGM:
    def __init__(self, run: Run, llm=None):
        self.run = run
        evo = run.cfg.evo
        self.llm = llm or make_llm(evo.llm, evo.model, evo.effort, evo.max_tokens, evo.seed)
        self.usage = Usage()
        self.rng = random.Random(evo.seed)
        for m in run.all_meta():          # resume token accounting
            t = m.get("tokens") or {}
            self.usage.input_tokens += t.get("input_tokens", 0)
            self.usage.output_tokens += t.get("output_tokens", 0)
            self.usage.cache_read_tokens += t.get("cache_read_tokens", 0)
            self.usage.cost_usd += t.get("cost_usd", 0.0)
            self.usage.calls += t.get("calls", 0)

    # ---------------------------------------------------------------- archive
    def archive(self) -> list[dict]:
        return [m for m in self.run.all_meta() if m.get("status") == "valid"]

    def select_parents(self, k: int) -> list[dict]:
        evo = self.run.cfg.evo
        arch = self.archive()
        if not arch:
            raise RuntimeError("archive is empty")
        if evo.ablation == "best_of_n":
            seed = [m for m in arch if m["id"] == "a0000"]
            return seed * k
        if evo.ablation == "no_open_ended":
            best = max(arch, key=lambda m: (m["score"], m["id"]))
            return [best] * k
        children = {m["id"]: 0 for m in arch}
        for m in arch:
            if m.get("parent") in children:
                children[m["parent"]] += 1
        weights = []
        for m in arch:
            s = 1.0 / (1.0 + math.exp(-evo.lam * (m["score"] - evo.alpha0)))
            h = 1.0 / (1.0 + children[m["id"]])
            weights.append(s * h)
        return self.rng.choices(arch, weights=weights, k=k)

    def lineage(self, agent_id: str) -> list[dict]:
        chain = []
        cur = self.run.meta(agent_id)
        while cur:
            chain.append({"id": cur["id"], "summary": cur.get("summary", ""),
                          "train": _brief(cur.get("train", {}))})
            cur = self.run.meta(cur["parent"]) if cur.get("parent") else None
        return chain[::-1]

    def sibling_attempts(self, parent_id: str) -> list[dict]:
        out = []
        for m in self.run.all_meta():
            if m.get("parent") == parent_id:
                out.append({"id": m["id"], "status": m["status"], "summary": m.get("summary", "")[:600],
                            "error": (m.get("error") or "")[:600], "train": _brief(m.get("train", {}))})
        return out[-8:]

    # ---------------------------------------------------------------- bootstrap
    def ensure_seed_evaluated(self) -> dict:
        m = self.run.meta("a0000")
        if m.get("status") == "valid":
            return m
        res = evaluate_agent(self.run, self.run.agent_dir("a0000"))
        meta = {"id": "a0000", "parent": None, "iteration": 0, "created": _now(), "summary": "seed agent",
                "files_changed": [], "tokens": Usage().as_dict(), **res}
        self.run.save_meta("a0000", meta)
        self._ledger(meta)
        if meta["status"] != "valid":
            raise RuntimeError(f"seed agent failed evaluation: {meta.get('error')}")
        return meta

    def _ledger(self, meta: dict) -> None:
        self.run.append_ledger({k: meta.get(k) for k in (
            "id", "parent", "iteration", "status", "score", "summary", "error", "files_changed",
            "tokens", "created")} | {"train": _brief(meta.get("train", {})),
                                     "val": _brief(meta.get("val", {})),
                                     "val_daily": meta.get("val_daily", [])})

    # ---------------------------------------------------------------- one child
    def make_child(self, parent: dict, iteration: int) -> dict:
        run, evo = self.run, self.run.cfg.evo
        cid = run.next_id()
        ws = run.agent_dir(cid)
        pdir = run.agent_dir(parent["id"])
        for p in sorted(pdir.iterdir()):
            shutil.copyfile(p, ws / p.name)
            (ws / p.name).chmod(0o644)
        fixed_improver = evo.ablation in ("no_self_improve", "best_of_n")
        if fixed_improver:
            shutil.copyfile(run.agent_dir("a0000") / "self_improve.py", ws / "self_improve.py")
        before = {p.name: p.read_bytes() for p in ws.iterdir()}
        meta = {"id": cid, "parent": parent["id"], "iteration": iteration, "created": _now(),
                "status": "invalid", "score": 0.0, "summary": "", "files_changed": []}
        payload = {"agent_id": cid, "parent_id": parent["id"], "report": parent.get("train", {}),
                   "lineage": self.lineage(parent["id"]), "siblings": self.sibling_attempts(parent["id"]),
                   "api_doc": API_DOC}
        sess = None
        try:
            sess = run_improver(run, ws, payload, self.llm)
            meta["summary"] = sess.summary
        except ChildError as e:
            meta["error"] = f"self-improvement step failed: {str(e)[-3000:]}"
        except Exception as e:
            meta["error"] = f"harness error during self-improvement: {e}\n{traceback.format_exc()[-2000:]}"
        if sess is not None:
            meta["tokens"] = sess.usage.as_dict()
            with self.usage._lock:
                self.usage.calls += sess.usage.calls
                self.usage.input_tokens += sess.usage.input_tokens
                self.usage.output_tokens += sess.usage.output_tokens
                self.usage.cache_read_tokens += sess.usage.cache_read_tokens
                self.usage.cost_usd += sess.usage.cost_usd
            _write_private(run.logs_dir / f"{cid}.improve.log", "\n".join(sess.log))
        if fixed_improver:
            shutil.copyfile(run.agent_dir("a0000") / "self_improve.py", ws / "self_improve.py")
        after = {p.name: p.read_bytes() for p in ws.iterdir() if p.is_file()}
        changed = sorted(n for n in set(before) | set(after) if before.get(n) != after.get(n))
        meta["files_changed"] = changed
        if "error" not in meta:
            if not changed:
                meta["status"] = "no_change"
                meta["error"] = "self-improvement produced no file changes"
            else:
                res = evaluate_agent(run, ws)
                meta.update(res)
                if evo.ablation == "no_open_ended" and meta["status"] == "valid":
                    best = max(m["score"] for m in self.archive())
                    if meta["score"] <= best:
                        meta["status"] = "rejected"
                        meta["note"] = "not better than current best (no_open_ended ablation)"
        run.save_meta(cid, meta)
        self._ledger(meta)
        return meta

    # ---------------------------------------------------------------- loop
    def evolve(self, iterations: int | None = None, log=print) -> None:
        evo = self.run.cfg.evo
        seed = self.ensure_seed_evaluated()
        log(f"seed a0000: score={seed['score']:.3f} train_sharpe={seed['train'].get('sharpe_ann', 0):.2f}")
        start_it = 1 + max((m.get("iteration", 0) for m in self.run.all_meta()), default=0)
        n_it = iterations if iterations is not None else evo.iterations
        for it in range(start_it, start_it + n_it):
            if evo.token_budget and self.usage.tokens >= evo.token_budget:
                log(f"token budget reached ({self.usage.tokens} >= {evo.token_budget}); stopping")
                break
            parents = self.select_parents(evo.children_per_iter)
            log(f"iteration {it}: parents {[p['id'] for p in parents]}")
            with ThreadPoolExecutor(max_workers=max(1, evo.workers)) as ex:
                results = list(ex.map(lambda p: self.make_child(p, it), parents))
            for m in results:
                tr = m.get("train", {})
                log(f"  {m['id']} <- {m['parent']}: {m['status']:9s} score={m['score']:.3f} "
                    f"train_sharpe={tr.get('sharpe_ann', float('nan')):.2f} "
                    f"files={m['files_changed']} {(m.get('error') or '')[:160]}")
            log(f"  archive size={len(self.archive())} tokens={self.usage.tokens} "
                f"cost=${self.usage.cost_usd:.2f}")
