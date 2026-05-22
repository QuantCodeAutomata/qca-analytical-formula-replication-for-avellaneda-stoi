"""Experiment 4: Robustness diagnostics for the AS Monte Carlo replication.

Sweeps fill model, probability handling, simultaneous-fill rule and price model
under the table-faithful spread baseline, then collects the same statistics.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import List

import pandas as pd

from src.config import BaselineParams, SimConfig
from src.metrics import diagnostics_to_dataframe
from src.simulator import SimulationResult, run_simulation

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def _label(cfg: SimConfig) -> str:
    return (
        f"price={cfg.price_model}|fill={cfg.fill_model}|"
        f"prob={cfg.prob_handling}|simu={cfg.simultaneous_fills}|"
        f"strat={cfg.strategy}|g={cfg.gamma}"
    )


def baseline_configs() -> List[SimConfig]:
    """Baseline + each robustness sweep variant."""
    configs: List[SimConfig] = []
    gammas = (0.1, 0.01, 0.5)
    strategies = ("inventory", "symmetric")

    # Baseline (A row)
    for g, s in product(gammas, strategies):
        configs.append(SimConfig(gamma=g, strategy=s, spread_mode="table_faithful",
                                 fill_model="bernoulli", prob_handling="raw",
                                 simultaneous_fills=True, price_model="binomial",
                                 label="A_baseline"))

    # Test A: Bernoulli vs. exact-event vs. Poisson-capped
    for g, s in product(gammas, strategies):
        configs.append(SimConfig(gamma=g, strategy=s, spread_mode="table_faithful",
                                 fill_model="exact_event", prob_handling="raw",
                                 simultaneous_fills=True, price_model="binomial",
                                 label="A_exact_event"))
        configs.append(SimConfig(gamma=g, strategy=s, spread_mode="table_faithful",
                                 fill_model="poisson_capped", prob_handling="raw",
                                 simultaneous_fills=True, price_model="binomial",
                                 label="A_poisson_capped"))

    # Test B: lambda dt > 1 handling
    for g, s in product(gammas, strategies):
        configs.append(SimConfig(gamma=g, strategy=s, spread_mode="table_faithful",
                                 fill_model="bernoulli", prob_handling="clipped",
                                 simultaneous_fills=True, price_model="binomial",
                                 label="B_clipped"))
        configs.append(SimConfig(gamma=g, strategy=s, spread_mode="table_faithful",
                                 fill_model="bernoulli", prob_handling="one_minus_exp",
                                 simultaneous_fills=True, price_model="binomial",
                                 label="B_one_minus_exp"))

    # Test C: simultaneous fills forbidden
    for g, s in product(gammas, strategies):
        configs.append(SimConfig(gamma=g, strategy=s, spread_mode="table_faithful",
                                 fill_model="bernoulli", prob_handling="raw",
                                 simultaneous_fills=False, price_model="binomial",
                                 label="C_no_simultaneous"))

    # Test D: Gaussian price increments
    for g, s in product(gammas, strategies):
        configs.append(SimConfig(gamma=g, strategy=s, spread_mode="table_faithful",
                                 fill_model="bernoulli", prob_handling="raw",
                                 simultaneous_fills=True, price_model="gaussian",
                                 label="D_gaussian"))

    return configs


def run() -> pd.DataFrame:
    params = BaselineParams()
    cfgs = baseline_configs()
    all_results: List[SimulationResult] = []

    for i, cfg in enumerate(cfgs):
        res = run_simulation(cfg, params, record_sample_path=False,
                             seed=params.seed + i)
        all_results.append(res)

    diag = diagnostics_to_dataframe(all_results, extra_label="exp4")
    diag.to_csv(RESULTS / "exp4_robustness.csv", index=False)

    # Pivoted summary: profile per (label, gamma, strategy)
    pivot = diag.pivot_table(
        index=["label", "gamma", "strategy"],
        values=["mean_profit", "std_profit", "mean_final_q", "std_final_q",
                "viol_ask_pct", "viol_bid_pct", "simultaneous_pct"],
    )
    pivot.to_csv(RESULTS / "exp4_robustness_pivot.csv")

    print("\n=== Experiment 4: Robustness sweep ===")
    print(pivot.round(3).to_string())
    return diag


if __name__ == "__main__":
    run()
