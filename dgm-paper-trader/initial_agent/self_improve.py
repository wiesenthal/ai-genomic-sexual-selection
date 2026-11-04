"""Self-improvement module. Part of the agent, and editable by the agent.

The harness calls improve(ctx) on a fresh copy of this agent's folder. Whatever this
function writes becomes the child agent, including a new version of this very file: that
is what makes the system a (Darwin) Gödel Machine rather than a fixed optimiser.

Default procedure (after the DGM paper):
  1. DIAGNOSE  - show the model this agent's own code, its TRAIN evaluation log, its lineage
                 and earlier attempts from the same parent; ask for ONE concrete change.
  2. IMPLEMENT - ask for complete new contents of every file that changes.
  3. SELF-CHECK- lint + short train backtest; on failure, one repair round.

Everything goes through ctx (see the harness's improver_host.ImproveContext):
ctx.llm, ctx.list_files, ctx.read_file, ctx.write_file, ctx.delete_file, ctx.backtest,
ctx.lint, ctx.log, ctx.set_summary, ctx.report, ctx.lineage, ctx.siblings, ctx.api_doc, ctx.budget.
"""
import json
import re

SYSTEM = """You are the self-improvement module of an autonomous intraday (minute-bar) trading agent.
The agent is a small folder of Python files. You are improving a COPY of it; the copy becomes a
new agent in an open-ended archive (Darwin Gödel Machine). Agents are scored on trading sessions
they never see, by the Probabilistic Sharpe Ratio of daily NET returns (after spread, slippage,
market impact and volume limits). You only ever see results on TRAIN sessions.

What makes a good change:
- Aim for edges that should generalise to unseen days: simple, economically motivated ideas
  (volatility scaling, time-of-day effects, cross-sectional relative value, regime filters,
  smarter entries/exits, turnover control). Costs are real: churn kills intraday strategies.
- Do NOT tune constants to fit the specific train days shown; few free parameters, robust defaults.
- Doing nothing scores PSR 0.5 only if the agent still trades: inactive agents score 0.
- Make ONE coherent change per generation, so its effect can be attributed.
- You may also improve self_improve.py (this module): e.g. better use of the diagnostics, asking
  for several candidate ideas and choosing, ablating with ctx.backtest, or fixing failure modes you
  see in the lineage. Changes to it only pay off through better future descendants, and it must
  keep working: it must define improve(ctx) and obey the sandbox rules.

{api_doc}
"""

FILE_RE = re.compile(r'<file name="([A-Za-z_][A-Za-z0-9_]*\.(?:py|md|txt|json))">\n(.*?)\n</file>', re.S)


def _files_block(sources):
    return "\n\n".join(f'<file name="{n}">\n{t}\n</file>' for n, t in sorted(sources.items()))


def _parse_files(text):
    return {m.group(1): m.group(2) + "\n" for m in FILE_RE.finditer(text or "")}


def _tag(text, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", re.S)
    return m.group(1).strip() if m else ""


def _json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


def _compact_report(report):
    rep = dict(report or {})
    diag = dict(rep.pop("diagnostics", {}) or {})
    return json.dumps({"metrics": rep, "diagnostics": diag}, indent=1, default=str)


def improve(ctx):
    system = SYSTEM.replace("{api_doc}", ctx.api_doc)
    names = [n for n in ctx.list_files() if n.endswith((".py", ".md", ".txt", ".json"))]
    sources = {n: ctx.read_file(n) for n in names}

    history = json.dumps({"lineage": ctx.lineage, "earlier_attempts_from_this_parent": ctx.siblings},
                         indent=1, default=str)

    diagnose = (
        "## TASK: DIAGNOSE\n"
        "Below are this agent's files, its evaluation on the TRAIN sessions, and its history.\n"
        "Study where the PnL comes from and where it is lost (costs, time of day, symbols, worst days).\n"
        "Propose exactly ONE improvement. Reply with a JSON object:\n"
        '{"diagnosis": "...", "proposal": "precise description of the change", '
        '"files": ["strategy.py", ...], "expected_effect": "..."}\n\n'
        f"## TRAIN EVALUATION\n{_compact_report(ctx.report)}\n\n"
        f"## HISTORY\n{history}\n\n"
        f"## AGENT FILES\n{_files_block(sources)}\n"
    )
    plan_text = ctx.llm(diagnose, system=system)
    plan = _json(plan_text) or {"proposal": plan_text.strip()[:4000]}
    ctx.log("plan: " + json.dumps(plan)[:3000])

    implement = (
        "## TASK: IMPLEMENT\n"
        f"Implement this change to the agent:\n{json.dumps(plan, indent=1)}\n\n"
        "Return the COMPLETE new contents of every file you change or add, each as\n"
        '<file name="NAME">\n...full file...\n</file>\n'
        "Files you do not return stay unchanged. End with <summary>one paragraph: what changed "
        "and why</summary>.\n\n"
        f"## CURRENT FILES\n{_files_block(sources)}\n"
    )
    impl_text = ctx.llm(implement, system=system)
    new_files = _parse_files(impl_text)
    if not new_files:
        ctx.set_summary("no change: the implementation step returned no files")
        return "no files"
    for n, t in new_files.items():
        ctx.write_file(n, t)
    summary = _tag(impl_text, "summary") or str(plan.get("proposal", ""))[:2000]

    # Self-check: does the new agent pass the sandbox lint and run on a few train sessions?
    check = ctx.backtest(max_days=2) if ctx.budget.get("backtests_left", 0) > 0 else {"ok": True}
    if not check.get("ok") and ctx.budget.get("llm_calls_left", 0) > 0:
        current = {n: ctx.read_file(n) for n in ctx.list_files() if n.endswith(".py")}
        repair = (
            "## TASK: REPAIR\n"
            "The new agent failed its self-check. Fix the error without changing the idea.\n"
            f"Error:\n{json.dumps(check.get('error'), default=str)[:6000]}\n\n"
            "Return complete corrected files as <file name=\"NAME\">...</file> blocks.\n\n"
            f"## CURRENT FILES\n{_files_block(current)}\n"
        )
        for n, t in _parse_files(ctx.llm(repair, system=system)).items():
            ctx.write_file(n, t)
        summary += " (repaired after failed self-check)"
    ctx.set_summary(summary)
    return "ok"

