"""Plotting helpers (matplotlib + seaborn) for the AS replication."""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .simulator import PathRecord, SimulationResult

sns.set_theme(style="whitegrid")


def plot_sample_path(
    path: PathRecord, title: str, out: Path
) -> None:
    """One-path diagnostic: mid-price, reservation/center, bid, ask."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(path.t, path.S, label="Mid price $S_t$", color="black", lw=1.2)
    ax.plot(path.t, path.r, label="Quote center $r_t$", color="tab:blue", lw=1.2)
    ax.plot(path.t, path.ask, label="Ask quote", color="tab:red", lw=0.8, linestyle="--")
    ax.plot(path.t, path.bid, label="Bid quote", color="tab:green", lw=0.8, linestyle="--")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_inventory_path(path: PathRecord, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(path.t, path.q, where="post", color="tab:purple")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Inventory $q_t$")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_profit_histograms(
    inventory_res: SimulationResult,
    symmetric_res: SimulationResult,
    title: str,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(
        min(inventory_res.terminal_profit.min(), symmetric_res.terminal_profit.min()),
        max(inventory_res.terminal_profit.max(), symmetric_res.terminal_profit.max()),
        50,
    )
    sns.histplot(
        inventory_res.terminal_profit,
        bins=bins,
        ax=ax,
        label="Inventory strategy",
        color="tab:blue",
        alpha=0.55,
        stat="density",
    )
    sns.histplot(
        symmetric_res.terminal_profit,
        bins=bins,
        ax=ax,
        label="Symmetric strategy",
        color="tab:orange",
        alpha=0.55,
        stat="density",
    )
    ax.set_xlabel("Terminal profit $\\Pi$")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_inventory_histograms(
    inventory_res: SimulationResult,
    symmetric_res: SimulationResult,
    title: str,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(
        min(inventory_res.terminal_inventory.min(), symmetric_res.terminal_inventory.min()) - 0.5,
        max(inventory_res.terminal_inventory.max(), symmetric_res.terminal_inventory.max()) + 1.5,
        1,
    )
    sns.histplot(inventory_res.terminal_inventory, bins=bins, ax=ax,
                 label="Inventory strategy", color="tab:blue", alpha=0.55, stat="density")
    sns.histplot(symmetric_res.terminal_inventory, bins=bins, ax=ax,
                 label="Symmetric strategy", color="tab:orange", alpha=0.55, stat="density")
    ax.set_xlabel("Final inventory $q_T$")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_spread_trajectories(
    t_grid: np.ndarray, spread_paths: List, labels: List[str], title: str, out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for sp, lab in zip(spread_paths, labels):
        ax.plot(t_grid, sp, label=lab)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Total spread $\\delta^a + \\delta^b$")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
