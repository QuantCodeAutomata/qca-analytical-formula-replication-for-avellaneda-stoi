"""Baseline parameters for Avellaneda-Stoikov Monte Carlo experiments."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple, Literal


@dataclass(frozen=True)
class BaselineParams:
    """Paper's baseline parameter set (Section 3.3)."""
    s0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0
    dt: float = 0.005
    q0: int = 0
    x0: float = 0.0
    A: float = 140.0
    k: float = 1.5
    n_paths: int = 1000
    gammas: Tuple[float, ...] = (0.1, 0.01, 0.5)
    seed: int = 42

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))


@dataclass(frozen=True)
class SimConfig:
    """Per-run simulation configuration with explicit ambiguity flags."""
    gamma: float
    strategy: Literal["inventory", "symmetric"]
    spread_mode: Literal["table_faithful", "equation_faithful"] = "table_faithful"
    price_model: Literal["binomial", "gaussian"] = "binomial"
    fill_model: Literal["bernoulli", "exact_event", "poisson_capped"] = "bernoulli"
    prob_handling: Literal["raw", "clipped", "one_minus_exp"] = "raw"
    simultaneous_fills: bool = True
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
