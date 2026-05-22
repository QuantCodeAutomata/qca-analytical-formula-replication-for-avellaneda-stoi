"""Experiment 2: Monte Carlo replication of Section 3.3 with the table-faithful
constant spread Spread = (2/gamma) ln(1 + gamma/k).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BaselineParams, SimConfig
from src.metrics import results_to_dataframe, diagnostics_to_dataframe
from src.plots import (
    plot_inventory_histograms,
    plot_inventory_path,
    plot_profit_histograms,
    plot_sample_path,
)
from src.simulator import build_shared_rng, run_simulation

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def run() -> pd.DataFrame:
    params = BaselineParams()
    all_results = []

    for gamma in params.gammas:
        # Common random numbers across strategies (price moves + fill uniforms)
        shared = build_shared_rng(params, price_model="binomial",
                                  seed=params.seed + int(gamma * 1000))

        inv_cfg = SimConfig(
            gamma=gamma, strategy="inventory", spread_mode="table_faithful",
            label=f"exp2_inventory_g{gamma}",
        )
        sym_cfg = SimConfig(
            gamma=gamma, strategy="symmetric", spread_mode="table_faithful",
            label=f"exp2_symmetric_g{gamma}",
        )

        # Sample-path runs share seed but record the first path
        inv_res = run_simulation(inv_cfg, params, record_sample_path=True,
                                 shared_rng_state=shared)
        sym_res = run_simulation(sym_cfg, params, record_sample_path=True,
                                 shared_rng_state=shared)
        all_results.extend([inv_res, sym_res])

        # Plots per gamma
        plot_sample_path(
            inv_res.sample_path,
            f"Inventory strategy sample path (gamma={gamma})",
            RESULTS / f"exp2_sample_path_inventory_g{gamma}.png",
        )
        plot_sample_path(
            sym_res.sample_path,
            f"Symmetric strategy sample path (gamma={gamma})",
            RESULTS / f"exp2_sample_path_symmetric_g{gamma}.png",
        )
        plot_inventory_path(
            inv_res.sample_path,
            f"Inventory trajectory (inventory strategy, gamma={gamma})",
            RESULTS / f"exp2_inv_path_inventory_g{gamma}.png",
        )
        plot_profit_histograms(
            inv_res, sym_res,
            f"Terminal P&L histogram (gamma={gamma}, table-faithful spread)",
            RESULTS / f"exp2_profit_hist_g{gamma}.png",
        )
        plot_inventory_histograms(
            inv_res, sym_res,
            f"Terminal inventory histogram (gamma={gamma})",
            RESULTS / f"exp2_inventory_hist_g{gamma}.png",
        )

    summary = results_to_dataframe(all_results)
    summary.to_csv(RESULTS / "exp2_summary.csv", index=False)

    diag = diagnostics_to_dataframe(all_results, extra_label="exp2_table_faithful")
    diag.to_csv(RESULTS / "exp2_diagnostics.csv", index=False)

    print("\n=== Experiment 2: Table-faithful constant spread ===")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
