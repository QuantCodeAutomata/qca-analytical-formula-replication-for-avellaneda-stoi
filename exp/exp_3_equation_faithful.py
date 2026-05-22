"""Experiment 3: Monte Carlo with the equation-faithful time-varying spread
Spread(t) = gamma sigma^2 (T-t) + (2/gamma) ln(1 + gamma/k).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BaselineParams, SimConfig
from src.metrics import diagnostics_to_dataframe, results_to_dataframe
from src.plots import (
    plot_inventory_histograms,
    plot_profit_histograms,
    plot_sample_path,
    plot_spread_trajectories,
)
from src.simulator import build_shared_rng, run_simulation

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def run() -> pd.DataFrame:
    params = BaselineParams()
    all_results = []

    for gamma in params.gammas:
        shared = build_shared_rng(params, price_model="binomial",
                                  seed=params.seed + int(gamma * 1000))

        inv_cfg = SimConfig(
            gamma=gamma, strategy="inventory", spread_mode="equation_faithful",
            label=f"exp3_inventory_g{gamma}",
        )
        sym_cfg = SimConfig(
            gamma=gamma, strategy="symmetric", spread_mode="equation_faithful",
            label=f"exp3_symmetric_g{gamma}",
        )
        inv_res = run_simulation(inv_cfg, params, record_sample_path=True,
                                 shared_rng_state=shared)
        sym_res = run_simulation(sym_cfg, params, record_sample_path=True,
                                 shared_rng_state=shared)
        all_results.extend([inv_res, sym_res])

        plot_sample_path(
            inv_res.sample_path,
            f"Inventory strategy sample path (gamma={gamma}, eq-faithful)",
            RESULTS / f"exp3_sample_path_inventory_g{gamma}.png",
        )
        plot_profit_histograms(
            inv_res, sym_res,
            f"Terminal P&L histogram (gamma={gamma}, eq-faithful spread)",
            RESULTS / f"exp3_profit_hist_g{gamma}.png",
        )
        plot_inventory_histograms(
            inv_res, sym_res,
            f"Terminal inventory histogram (gamma={gamma}, eq-faithful)",
            RESULTS / f"exp3_inventory_hist_g{gamma}.png",
        )

    # Spread trajectory comparison plot (eq vs table) at gamma=0.1
    p = BaselineParams()
    t_grid = np.arange(p.n_steps) * p.dt
    eq_curves = []
    labels = []
    for gamma in params.gammas:
        eq_curve = gamma * p.sigma**2 * (p.T - t_grid) + (2.0 / gamma) * np.log(1.0 + gamma / p.k)
        tab_const = (2.0 / gamma) * np.log(1.0 + gamma / p.k)
        eq_curves.append(eq_curve)
        labels.append(f"eq-faithful, gamma={gamma}")
        eq_curves.append(np.full_like(t_grid, tab_const))
        labels.append(f"table-faithful (const), gamma={gamma}")
    plot_spread_trajectories(
        t_grid, eq_curves, labels,
        "Spread trajectories: equation-faithful vs. table-faithful",
        RESULTS / "exp3_spread_trajectories.png",
    )

    summary = results_to_dataframe(all_results)
    summary.to_csv(RESULTS / "exp3_summary.csv", index=False)

    diag = diagnostics_to_dataframe(all_results, extra_label="exp3_equation_faithful")
    diag.to_csv(RESULTS / "exp3_diagnostics.csv", index=False)

    print("\n=== Experiment 3: Equation-faithful time-varying spread ===")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
