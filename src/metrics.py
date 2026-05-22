"""Aggregation helpers for the Avellaneda-Stoikov Monte Carlo experiments."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .simulator import SimulationResult


def summarize(result: SimulationResult) -> Dict[str, float]:
    """Per-paper summary: spread, profit mean/std, final-q mean/std."""
    # Representative spread = midpoint of the trajectory (constant in table-faithful mode).
    spread = float(result.spread_grid.mean())
    return {
        "gamma": result.config.gamma,
        "strategy": result.config.strategy,
        "spread_mode": result.config.spread_mode,
        "Spread": spread,
        "Profit": result.mean_profit,
        "std(Profit)": result.std_profit,
        "Final q": result.mean_final_q,
        "std(Final q)": result.std_final_q,
    }


def results_to_dataframe(results: List[SimulationResult]) -> pd.DataFrame:
    return pd.DataFrame([summarize(r) for r in results])


def diagnostics_to_dataframe(
    results: List[SimulationResult], extra_label: str = ""
) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "label": r.config.label or extra_label,
            "gamma": r.config.gamma,
            "strategy": r.config.strategy,
            "spread_mode": r.config.spread_mode,
            "price_model": r.config.price_model,
            "fill_model": r.config.fill_model,
            "prob_handling": r.config.prob_handling,
            "simultaneous_fills": r.config.simultaneous_fills,
            **r.diagnostics,
            "mean_profit": r.mean_profit,
            "std_profit": r.std_profit,
            "mean_final_q": r.mean_final_q,
            "std_final_q": r.std_final_q,
        }
        rows.append(row)
    return pd.DataFrame(rows)
