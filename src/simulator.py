"""Discrete-time Monte Carlo simulator for the Avellaneda-Stoikov model.

# Custom — Context7 confirmed no library equivalent (paper Section 3.3 plus
# the implementation-ambiguity diagnostics defined in the experiment plan).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .config import BaselineParams, SimConfig
from .formulas import (
    total_spread_equation_faithful,
    total_spread_table_faithful,
)
from .strategies import (
    inventory_quote_center,
    make_quotes,
    symmetric_quote_center,
)


@dataclass
class PathRecord:
    """Time-series record for a single representative path."""
    t: np.ndarray
    S: np.ndarray
    r: np.ndarray  # quote center (reservation price for inventory strategy)
    bid: np.ndarray
    ask: np.ndarray
    q: np.ndarray
    x: np.ndarray


@dataclass
class SimulationResult:
    """Aggregated output of a Monte Carlo run."""
    config: SimConfig
    terminal_profit: np.ndarray   # shape (n_paths,)
    terminal_inventory: np.ndarray  # shape (n_paths,)
    spread_grid: np.ndarray       # shape (n_steps,) — spread at each step
    diagnostics: Dict[str, float]
    sample_path: Optional[PathRecord] = None

    @property
    def mean_profit(self) -> float:
        return float(np.mean(self.terminal_profit))

    @property
    def std_profit(self) -> float:
        return float(np.std(self.terminal_profit, ddof=0))

    @property
    def mean_final_q(self) -> float:
        return float(np.mean(self.terminal_inventory))

    @property
    def std_final_q(self) -> float:
        return float(np.std(self.terminal_inventory, ddof=0))


def _spread_grid(
    cfg: SimConfig, params: BaselineParams, t_grid: np.ndarray
) -> np.ndarray:
    """Build total-spread array on the time grid per cfg.spread_mode."""
    if cfg.spread_mode == "table_faithful":
        s = total_spread_table_faithful(cfg.gamma, params.k)
        return np.full_like(t_grid, s, dtype=np.float64)
    if cfg.spread_mode == "equation_faithful":
        return np.array(
            [
                total_spread_equation_faithful(
                    t, cfg.gamma, params.sigma, params.T, params.k
                )
                for t in t_grid
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unknown spread_mode={cfg.spread_mode}")


def _fill_probability(
    lam_dt: np.ndarray,
    fill_model: str,
    prob_handling: str,
) -> np.ndarray:
    """Convert lambda*dt into a per-step fill probability per cfg flags."""
    if fill_model == "exact_event" or prob_handling == "one_minus_exp":
        return 1.0 - np.exp(-np.clip(lam_dt, 0.0, None))
    if fill_model == "poisson_capped":
        # P(K >= 1) for K ~ Poisson(lam_dt)
        return 1.0 - np.exp(-np.clip(lam_dt, 0.0, None))
    # bernoulli
    if prob_handling == "clipped":
        return np.clip(lam_dt, 0.0, 1.0)
    # 'raw' Bernoulli — let probability exceed 1 occasionally; np.random.uniform
    # comparison still treats it as guaranteed fill. We log violations separately.
    return lam_dt


def run_simulation(
    cfg: SimConfig,
    params: BaselineParams,
    record_sample_path: bool = False,
    seed: Optional[int] = None,
    shared_rng_state: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> SimulationResult:
    """Run a Monte Carlo simulation under the given configuration.

    Parameters
    ----------
    cfg : SimConfig
        Per-run flags (strategy, spread, fill model, etc.).
    params : BaselineParams
        Shared baseline parameters.
    record_sample_path : bool
        If True, also record full state trajectory of path index 0.
    seed : Optional[int]
        RNG seed; falls back to params.seed if None.
    shared_rng_state : Optional tuple
        Pre-generated arrays (U_a, U_b, xi) of shape (n_steps, n_paths) used to
        couple Monte Carlo experiments via common random numbers.
    """
    n_paths = params.n_paths
    n_steps = params.n_steps
    dt = params.dt
    sigma = params.sigma
    gamma = cfg.gamma
    A = params.A
    k = params.k
    sqrt_dt = np.sqrt(dt)

    t_grid = np.arange(n_steps, dtype=np.float64) * dt
    spread_grid = _spread_grid(cfg, params, t_grid)
    half_grid = 0.5 * spread_grid

    rng = np.random.default_rng(params.seed if seed is None else seed)
    if shared_rng_state is None:
        U_a = rng.random((n_steps, n_paths))
        U_b = rng.random((n_steps, n_paths))
        if cfg.price_model == "binomial":
            xi = rng.choice([-1.0, 1.0], size=(n_steps, n_paths))
        elif cfg.price_model == "gaussian":
            xi = rng.standard_normal((n_steps, n_paths))
        else:
            raise ValueError(f"Unknown price_model={cfg.price_model}")
    else:
        U_a, U_b, xi = shared_rng_state

    # Independent stream for tie-breaking when both bid and ask fire in the
    # same step under simultaneous_fills=False. This MUST be independent of
    # U_a, U_b because (U_a, U_b) are conditioned on the "both fired" event
    # (both lie in [0,p)), which biases any deterministic function of them.
    tie_rng = np.random.default_rng(
        (params.seed if seed is None else seed) ^ 0xA51057C0FFEE
    )
    U_tie = tie_rng.random((n_steps, n_paths))

    S = np.full(n_paths, params.s0, dtype=np.float64)
    q = np.full(n_paths, params.q0, dtype=np.float64)
    x = np.full(n_paths, params.x0, dtype=np.float64)

    # Diagnostics
    violations_a = 0
    violations_b = 0
    neg_delta_a = 0
    neg_delta_b = 0
    simultaneous = 0
    total_fills_a = 0
    total_fills_b = 0

    if record_sample_path:
        rec_S = np.empty(n_steps + 1)
        rec_r = np.empty(n_steps + 1)
        rec_bid = np.empty(n_steps + 1)
        rec_ask = np.empty(n_steps + 1)
        rec_q = np.empty(n_steps + 1)
        rec_x = np.empty(n_steps + 1)
        rec_S[0] = params.s0
        rec_q[0] = params.q0
        rec_x[0] = params.x0
    else:
        rec_S = rec_r = rec_bid = rec_ask = rec_q = rec_x = None

    for n in range(n_steps):
        tau = params.T - t_grid[n]
        h = half_grid[n]

        if cfg.strategy == "inventory":
            center = inventory_quote_center(S, q, tau, gamma, sigma)
        elif cfg.strategy == "symmetric":
            center = symmetric_quote_center(S)
        else:
            raise ValueError(f"Unknown strategy {cfg.strategy}")

        bid, ask = make_quotes(center, h)
        delta_a = ask - S
        delta_b = S - bid

        neg_delta_a += int(np.sum(delta_a < 0))
        neg_delta_b += int(np.sum(delta_b < 0))

        lam_a = A * np.exp(-k * delta_a)
        lam_b = A * np.exp(-k * delta_b)
        lam_a_dt = lam_a * dt
        lam_b_dt = lam_b * dt

        violations_a += int(np.sum(lam_a_dt > 1.0))
        violations_b += int(np.sum(lam_b_dt > 1.0))

        p_a = _fill_probability(lam_a_dt, cfg.fill_model, cfg.prob_handling)
        p_b = _fill_probability(lam_b_dt, cfg.fill_model, cfg.prob_handling)

        fill_a = U_a[n] < p_a
        fill_b = U_b[n] < p_b

        if not cfg.simultaneous_fills:
            both = fill_a & fill_b
            simultaneous += int(np.sum(both))
            # Hazard-based competition: when both fire, pick ask with
            # probability lam_a / (lam_a + lam_b). The tie-break uniform
            # MUST come from an independent stream (U_tie) because U_a, U_b
            # are conditioned on the "both fired" event and are therefore
            # not uniform on [0,1] in that subpopulation.
            prob_keep_ask = lam_a / (lam_a + lam_b + 1e-300)
            keep_ask = U_tie[n] < prob_keep_ask
            fill_a_new = fill_a & np.where(both, keep_ask, True)
            fill_b_new = fill_b & np.where(both, ~keep_ask, True)
            fill_a, fill_b = fill_a_new, fill_b_new
        else:
            simultaneous += int(np.sum(fill_a & fill_b))

        # Cash & inventory updates (use quoted prices)
        x = x + np.where(fill_a, ask, 0.0) - np.where(fill_b, bid, 0.0)
        q = q + np.where(fill_b, 1.0, 0.0) - np.where(fill_a, 1.0, 0.0)
        total_fills_a += int(np.sum(fill_a))
        total_fills_b += int(np.sum(fill_b))

        if record_sample_path:
            rec_r[n] = center[0]
            rec_bid[n] = bid[0]
            rec_ask[n] = ask[0]

        # Price update
        if cfg.price_model == "binomial":
            S = S + xi[n] * sigma * sqrt_dt
        else:
            S = S + sigma * sqrt_dt * xi[n]

        if record_sample_path:
            rec_S[n + 1] = S[0]
            rec_q[n + 1] = q[0]
            rec_x[n + 1] = x[0]

    if record_sample_path:
        # Fill final-step quote with last-known center
        tau_end = 0.0
        if cfg.strategy == "inventory":
            final_center = float(S[0] - q[0] * gamma * sigma**2 * tau_end)
        else:
            final_center = float(S[0])
        rec_r[-1] = final_center
        rec_bid[-1] = final_center - 0.5 * spread_grid[-1]
        rec_ask[-1] = final_center + 0.5 * spread_grid[-1]
        sample_path = PathRecord(
            t=np.concatenate([t_grid, [params.T]]),
            S=rec_S,
            r=rec_r,
            bid=rec_bid,
            ask=rec_ask,
            q=rec_q,
            x=rec_x,
        )
    else:
        sample_path = None

    terminal_profit = x + q * S
    n_step_paths = n_steps * n_paths
    diagnostics = {
        "viol_ask_pct": 100.0 * violations_a / n_step_paths,
        "viol_bid_pct": 100.0 * violations_b / n_step_paths,
        "neg_delta_ask_pct": 100.0 * neg_delta_a / n_step_paths,
        "neg_delta_bid_pct": 100.0 * neg_delta_b / n_step_paths,
        "simultaneous_pct": 100.0 * simultaneous / n_step_paths,
        "avg_fills_ask_per_path": total_fills_a / n_paths,
        "avg_fills_bid_per_path": total_fills_b / n_paths,
    }

    return SimulationResult(
        config=cfg,
        terminal_profit=terminal_profit,
        terminal_inventory=q,
        spread_grid=spread_grid,
        diagnostics=diagnostics,
        sample_path=sample_path,
    )


def build_shared_rng(
    params: BaselineParams, price_model: str = "binomial", seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-generate (U_a, U_b, xi) for common-random-number coupling."""
    rng = np.random.default_rng(params.seed if seed is None else seed)
    U_a = rng.random((params.n_steps, params.n_paths))
    U_b = rng.random((params.n_steps, params.n_paths))
    if price_model == "binomial":
        xi = rng.choice([-1.0, 1.0], size=(params.n_steps, params.n_paths))
    elif price_model == "gaussian":
        xi = rng.standard_normal((params.n_steps, params.n_paths))
    else:
        raise ValueError(f"Unknown price_model={price_model}")
    return U_a, U_b, xi
