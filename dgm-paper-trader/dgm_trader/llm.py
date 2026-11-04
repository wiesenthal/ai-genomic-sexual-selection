"""LLM backends used by the parent process on behalf of an agent's self_improve module.

Only the parent process holds credentials; agent code reaches the model through ctx.llm().
"""
from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field

# List prices in USD per million tokens (input, output), for cost tracking only.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, model: str, inp: int, out: int, cache_read: int = 0, cost: float | None = None):
        with self._lock:
            self.calls += 1
            self.input_tokens += inp
            self.output_tokens += out
            self.cache_read_tokens += cache_read
            if cost is None:
                pi, po = PRICES.get(model, (0.0, 0.0))
                cost = (inp * pi + cache_read * pi * 0.1 + out * po) / 1e6
            self.cost_usd += cost

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens

    def as_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "cache_read_tokens": self.cache_read_tokens,
                "cost_usd": round(self.cost_usd, 4)}


class LLMError(RuntimeError):
    pass


class AnthropicLLM:
    """Claude via the official Anthropic SDK (reads ANTHROPIC_API_KEY or an `ant auth` profile)."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", max_tokens: int = 64000):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=4)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(self, prompt: str, system: str | None, max_tokens: int | None, usage: Usage) -> str:
        anthropic = self._anthropic
        kwargs = dict(
            model=self.model,
            max_tokens=int(max_tokens or self.max_tokens),
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        )
        if system:
            # The system prompt (contract + rules) is stable across calls: cache it.
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if self.model in FALLBACK_MODELS:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        try:
            with self.client.beta.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
        except anthropic.AuthenticationError as e:
            raise LLMError("Anthropic authentication failed: set ANTHROPIC_API_KEY") from e
        except anthropic.RateLimitError as e:
            raise LLMError(f"rate limited: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"connection error: {e}") from e
        u = msg.usage
        usage.add(self.model, int(u.input_tokens or 0) + int(getattr(u, "cache_creation_input_tokens", 0) or 0),
                  int(u.output_tokens or 0), int(getattr(u, "cache_read_input_tokens", 0) or 0))
        if msg.stop_reason == "refusal":
            raise LLMError("model declined the request (stop_reason=refusal)")
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        if msg.stop_reason == "max_tokens":
            text += "\n[TRUNCATED: max_tokens reached]"
        return text


class ClaudeCLILLM:
    """Claude through a locally logged-in Claude Code CLI (`claude -p`), tools disabled."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", timeout_s: int = 1200):
        self.exe = shutil.which("claude")
        if not self.exe:
            raise LLMError("`claude` CLI not found on PATH")
        self.model, self.effort, self.timeout_s = model, effort, timeout_s

    def complete(self, prompt: str, system: str | None, max_tokens: int | None, usage: Usage) -> str:
        cmd = [self.exe, "-p", "--output-format", "json", "--tools", "", "--model", self.model,
               "--effort", self.effort]
        if system:
            cmd += ["--system-prompt", system]
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout_s)
        if r.returncode != 0:
            raise LLMError(f"claude CLI failed ({r.returncode}): {r.stderr[-2000:]}")
        try:
            js = json.loads(r.stdout)
        except json.JSONDecodeError as e:
            raise LLMError(f"unexpected claude CLI output: {r.stdout[:500]}") from e
        u = js.get("usage") or {}
        usage.add(self.model, int(u.get("input_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0)),
                  int(u.get("output_tokens", 0)), int(u.get("cache_read_input_tokens", 0)),
                  cost=js.get("total_cost_usd"))
        if js.get("is_error"):
            raise LLMError(f"claude CLI error: {js.get('result')}")
        return js.get("result", "")


class MockLLM:
    """Deterministic offline stand-in for TESTING THE MACHINERY ONLY.

    It answers the default improver's prompts by randomly perturbing numeric constants in
    strategy.py (occasionally emitting a broken file, to exercise the invalid-child path).
    Results produced with it say nothing about what a real model would do.
    """

    def __init__(self, seed: int = 0, break_rate: float = 0.1):
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.break_rate = break_rate

    def complete(self, prompt: str, system: str | None, max_tokens: int | None, usage: Usage) -> str:
        with self.lock:
            r = self.rng.random()
            seed = self.rng.randrange(1 << 30)
        usage.add("mock", len(prompt) // 4, 400)
        if prompt.lstrip().startswith("## TASK: DIAGNOSE"):
            return json.dumps({"diagnosis": "mock diagnosis", "proposal": "retune thresholds",
                               "files": ["strategy.py"]})
        body = prompt.split("## CURRENT FILES", 1)[-1]
        m = re.search(r"<file name=\"strategy\.py\">\n(.*?)\n</file>", body, re.S)
        if not m:
            return "<summary>no change</summary>"
        src = m.group(1)
        if r < self.break_rate:
            return '<file name="strategy.py">\nclass Strategy(:\n    pass\n</file>\n<summary>broken</summary>'
        rng = random.Random(seed)

        def bump(mo):
            name, val = mo.group(1), float(mo.group(2))
            new = val * rng.choice([0.5, 0.8, 1.25, 2.0])
            new = int(round(new)) if "." not in mo.group(2) else round(new, 4)
            return f"{name} = {max(new, 1) if isinstance(new, int) else new}"

        new_src = re.sub(r"^([A-Z_]{3,}) = ([0-9]+(?:\.[0-9]+)?)\s*$", bump, src, count=0, flags=re.M)
        return f'<file name="strategy.py">\n{new_src}\n</file>\n<summary>mock: rescaled constants</summary>'


def make_llm(kind: str, model: str, effort: str, max_tokens: int, seed: int = 0):
    if kind == "anthropic":
        return AnthropicLLM(model, effort, max_tokens)
    if kind == "claude-cli":
        return ClaudeCLILLM(model, effort)
    if kind == "mock":
        return MockLLM(seed)
    raise ValueError(f"unknown llm backend {kind!r}")
