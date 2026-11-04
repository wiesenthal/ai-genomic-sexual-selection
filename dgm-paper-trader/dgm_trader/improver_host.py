"""Child-process entry point: runs an agent's own self_improve.improve(ctx) on its workspace.

This is the self-referential part of the Darwin Gödel Machine: the code that proposes and
writes the next version of the agent is itself part of the agent, and can rewrite itself.

All side effects go through `ctx`, which forwards requests to the parent over the protocol:
the parent holds the API key, enforces budgets, and is the only process that touches disk.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from .sandbox import AgentLoader, protocol_streams, write_msg


class ImproveContext:
    """Everything self_improve.improve(ctx) may use.

    ctx.report        dict: this agent's evaluation on the TRAIN days (metrics + diagnostics)
    ctx.lineage       list[dict]: ancestors, oldest first (id, summary of change, train metrics)
    ctx.siblings      list[dict]: earlier children of the same parent (status, summary, train metrics)
    ctx.api_doc       str: the strategy contract and sandbox rules
    ctx.budget        dict: remaining LLM calls / backtests
    ctx.llm(prompt, system=None, max_tokens=None) -> str
    ctx.list_files() -> list[str]
    ctx.read_file(name) -> str
    ctx.write_file(name, text)
    ctx.delete_file(name)
    ctx.backtest(max_days=None) -> dict   # evaluates the CURRENT workspace on train days
    ctx.lint() -> dict                    # sandbox lint results for the current workspace
    ctx.log(text)
    ctx.set_summary(text)                 # one-paragraph description of the change made
    """

    def __init__(self, pin, pout, payload: dict):
        self._in, self._out = pin, pout
        self.report = payload.get("report", {})
        self.lineage = payload.get("lineage", [])
        self.siblings = payload.get("siblings", [])
        self.api_doc = payload.get("api_doc", "")
        self.budget = payload.get("budget", {})
        self.agent_id = payload.get("agent_id", "")
        self.parent_id = payload.get("parent_id", "")

    def _call(self, op: str, **kw):
        write_msg(self._out, {"op": op, **kw})
        line = self._in.readline()
        if not line:
            raise RuntimeError("parent closed the channel")
        resp = json.loads(line)
        if "err" in resp:
            raise RuntimeError(resp["err"])
        if "budget" in resp:
            self.budget = resp["budget"]
        return resp.get("value")

    def llm(self, prompt: str, system: str | None = None, max_tokens: int | None = None) -> str:
        return self._call("llm", prompt=prompt, system=system, max_tokens=max_tokens)

    def list_files(self) -> list:
        return self._call("list_files")

    def read_file(self, name: str) -> str:
        return self._call("read_file", name=name)

    def write_file(self, name: str, text: str) -> None:
        self._call("write_file", name=name, text=text)

    def delete_file(self, name: str) -> None:
        self._call("delete_file", name=name)

    def backtest(self, max_days: int | None = None) -> dict:
        return self._call("backtest", max_days=max_days)

    def lint(self) -> dict:
        return self._call("lint")

    def log(self, text: str) -> None:
        self._call("log", text=str(text)[:4000])

    def set_summary(self, text: str) -> None:
        self._call("set_summary", text=str(text)[:4000])


def main() -> None:
    pin, pout = protocol_streams()
    try:
        first = json.loads(pin.readline())
        if first.get("op") != "init":
            raise ValueError("expected init")
        ctx = ImproveContext(pin, pout, first.get("payload", {}))
        loader = AgentLoader(Path.cwd())
        mod = loader.load("self_improve")
        result = mod.improve(ctx)
        write_msg(pout, {"op": "done", "result": str(result)[:2000] if result is not None else None})
    except Exception:
        write_msg(pout, {"error": traceback.format_exc(limit=8)})
        sys.exit(1)


if __name__ == "__main__":
    main()
