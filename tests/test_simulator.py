"""Integration-style tests for the Avellaneda-Stoikov Monte Carlo simulator."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.config import BaselineParams, SimConfig
from src.simulator import build_shared_rng, run_simulation


@pytest.fixture
def small_params() -> BaselineParams:
    """Smaller-scale params for fast tests."""
    return BaselineParams(n_paths=200)


def test_terminal_arrays_shape(small_params):
    cfg = SimConfig(gamma=0.1, strategy="inventory")
    res = run_simulation(cfg, small_params)
    assert res.terminal_profit.shape == (small_params.n_paths,)
    assert res.terminal_inventory.shape == (small_params.n_paths,)
    assert np.all(np.isfinite(res.terminal_profit))


def test_inventory_strategy_reduces_inventory_dispersion(small_params):
    """Core paper claim: inventory strategy lowers std(final q)."""
    shared = build_shared_rng(small_params)
    inv = run_simulation(SimConfig(gamma=0.1, strategy="inventory"),
                         small_params, shared_rng_state=shared)
    sym = run_simulation(SimConfig(gamma=0.1, strategy="symmetric"),
                         small_params, shared_rng_state=shared)
    assert inv.std_final_q < sym.std_final_q


def test_inventory_strategy_reduces_profit_dispersion(small_params):
    """Core paper claim: inventory strategy lowers std(profit)."""
    shared = build_shared_rng(small_params)
    inv = run_simulation(SimConfig(gamma=0.1, strategy="inventory"),
                         small_params, shared_rng_state=shared)
    sym = run_simulation(SimConfig(gamma=0.1, strategy="symmetric"),
                         small_params, shared_rng_state=shared)
    assert inv.std_profit < sym.std_profit


def test_table_faithful_spread_constant(small_params):
    cfg = SimConfig(gamma=0.1, strategy="inventory", spread_mode="table_faithful")
    res = run_simulation(cfg, small_params)
    assert np.allclose(res.spread_grid, res.spread_grid[0])


def test_equation_faithful_spread_monotonic_decrease(small_params):
    cfg = SimConfig(gamma=0.1, strategy="inventory", spread_mode="equation_faithful")
    res = run_simulation(cfg, small_params)
    assert np.all(np.diff(res.spread_grid) <= 1e-12)
    # The grid ends at t = T - dt; as t -> T the spread approaches the
    # table-faithful constant from above.
    table_const = (2.0 / 0.1) * np.log(1.0 + 0.1 / 1.5)
    assert res.spread_grid[-1] >= table_const
    assert (res.spread_grid[-1] - table_const) < 0.1


def test_sample_path_record_present_when_requested(small_params):
    cfg = SimConfig(gamma=0.1, strategy="inventory")
    res = run_simulation(cfg, small_params, record_sample_path=True)
    assert res.sample_path is not None
    assert len(res.sample_path.t) == small_params.n_steps + 1


def test_zero_gamma_invalid():
    with pytest.raises(ValueError):
        SimConfig(gamma=0.0, strategy="inventory")
        # SimConfig itself doesn't validate; the simulator does at first use.
        run_simulation(SimConfig(gamma=0.0, strategy="inventory"), BaselineParams(n_paths=10))


def test_seed_reproducibility(small_params):
    cfg = SimConfig(gamma=0.1, strategy="inventory")
    res1 = run_simulation(cfg, small_params, seed=123)
    res2 = run_simulation(cfg, small_params, seed=123)
    np.testing.assert_array_equal(res1.terminal_profit, res2.terminal_profit)
    np.testing.assert_array_equal(res1.terminal_inventory, res2.terminal_inventory)


def test_robustness_flags_run_without_error(small_params):
    """All cfg flag combinations should execute and produce finite output."""
    base = SimConfig(gamma=0.1, strategy="inventory")
    for fill in ("bernoulli", "exact_event", "poisson_capped"):
        for prob in ("raw", "clipped", "one_minus_exp"):
            for simu in (True, False):
                for price in ("binomial", "gaussian"):
                    cfg = replace(base, fill_model=fill, prob_handling=prob,
                                  simultaneous_fills=simu, price_model=price)
                    res = run_simulation(cfg, small_params, seed=7)
                    assert np.all(np.isfinite(res.terminal_profit))


def test_inventory_mean_close_to_zero(small_params):
    """Inventory strategy should keep mean final q close to zero."""
    res = run_simulation(SimConfig(gamma=0.5, strategy="inventory"),
                         small_params, seed=1)
    assert abs(res.mean_final_q) < 1.0


def test_initial_state_correct(small_params):
    cfg = SimConfig(gamma=0.1, strategy="inventory")
    res = run_simulation(cfg, small_params, record_sample_path=True)
    sp = res.sample_path
    assert sp.S[0] == small_params.s0
    assert sp.q[0] == small_params.q0
    assert sp.x[0] == small_params.x0
