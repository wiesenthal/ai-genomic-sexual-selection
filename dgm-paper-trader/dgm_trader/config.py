"""Configuration dataclasses shared by the harness, the evolver and the paper trader."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CostModel:
    """Per-fill cost model, in basis points of traded notional.

    total_bps = half_spread_bps + slippage_bps + commission_bps
                + impact_coef_bps * sqrt(participation)
    where participation = |shares traded| / bar volume.
    """

    half_spread_bps: float = 1.0
    slippage_bps: float = 0.5
    commission_bps: float = 0.0
    impact_coef_bps: float = 10.0
    max_participation: float = 0.05  # max fraction of a bar's volume one fill may take


@dataclass
class SimConfig:
    capital: float = 100_000.0
    max_gross: float = 1.0          # sum(|weights|) cap
    max_weight: float = 1.0         # per-symbol |weight| cap
    allow_short: bool = True
    flatten_minutes: int = 5        # targets forced to 0 in the last N bars of a session
    min_trade_frac: float = 0.001   # skip fills smaller than this fraction of equity
    costs: CostModel = field(default_factory=CostModel)

    @staticmethod
    def from_dict(d: dict) -> "SimConfig":
        d = dict(d)
        costs = CostModel(**d.pop("costs", {}))
        return SimConfig(costs=costs, **d)


@dataclass
class DataConfig:
    path: str = ""
    tz: str = "America/New_York"    # exchange timezone used to assign bars to sessions
    session: str = "09:30-16:00"    # local session window; "" keeps every bar of the day
    train_frac: float = 0.6
    val_frac: float = 0.2           # the remainder is the holdout


@dataclass
class EvoConfig:
    iterations: int = 20
    children_per_iter: int = 4
    workers: int = 4
    ablation: str = "none"          # none | no_self_improve | no_open_ended | best_of_n
    llm: str = "anthropic"          # anthropic | claude-cli | mock
    model: str = "claude-opus-5-5"
    effort: str = "high"
    max_tokens: int = 64000
    token_budget: int = 0           # 0 = unlimited; stops the run once input+output tokens exceed it
    max_llm_calls_per_child: int = 6
    max_backtests_per_child: int = 3
    improve_timeout_s: int = 1800
    stage1_days: int = 3
    min_daily_turnover: float = 0.05   # below this an agent counts as inactive (score 0)
    lam: float = 10.0               # DGM parent-selection sigmoid sharpness
    alpha0: float = 0.5             # DGM sigmoid midpoint; PSR 0.5 == zero Sharpe
    seed: int = 0
    bar_timeout_s: float = 5.0
    eval_timeout_s: float = 1800.0
    mem_mb: int = 4096


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    evo: EvoConfig = field(default_factory=EvoConfig)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path: Path) -> "RunConfig":
        d = json.loads(path.read_text())
        return RunConfig(
            data=DataConfig(**d.get("data", {})),
            sim=SimConfig.from_dict(d.get("sim", {})),
            evo=EvoConfig(**d.get("evo", {})),
        )
