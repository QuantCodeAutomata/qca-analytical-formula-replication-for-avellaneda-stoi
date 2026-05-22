"""Unit tests for the Avellaneda-Stoikov analytical formulas (exp_1)."""
from __future__ import annotations

import numpy as np
import pytest

from src.formulas import (
    frozen_inventory_value,
    implicit_quote_general,
    infinite_horizon_admissible,
    infinite_horizon_reservation_prices,
    optimal_delta_exp_intensity,
    reservation_ask,
    reservation_bid,
    reservation_price,
    solve_implicit_quote,
    total_spread_equation_faithful,
    total_spread_table_faithful,
)


PARAMS = dict(s=100.0, gamma=0.1, sigma=2.0, T=1.0, k=1.5, A=140.0)


def test_frozen_value_q0():
    """For q=0 the frozen-inventory value equals -exp(-gamma x)."""
    v = frozen_inventory_value(10.0, 100.0, 0, 0.5, 0.1, 2.0, 1.0)
    assert np.isclose(v, -np.exp(-0.1 * 10.0))


def test_frozen_value_terminal():
    """At t=T the value collapses to -exp(-gamma (x + q s))."""
    v = frozen_inventory_value(5.0, 100.0, 3, 1.0, 0.1, 2.0, 1.0)
    assert np.isclose(v, -np.exp(-0.1 * (5.0 + 3 * 100.0)))


def test_reservation_price_midpoint():
    """r = (r^a + r^b)/2."""
    for q in (-5, -1, 0, 1, 5):
        for t in (0.0, 0.5, 0.99):
            r = reservation_price(100.0, q, t, 0.1, 2.0, 1.0)
            r_a = reservation_ask(100.0, q, t, 0.1, 2.0, 1.0)
            r_b = reservation_bid(100.0, q, t, 0.1, 2.0, 1.0)
            assert np.isclose(r, 0.5 * (r_a + r_b))


def test_reservation_price_q0_equals_s():
    assert np.isclose(reservation_price(100.0, 0, 0.5, 0.1, 2.0, 1.0), 100.0)


def test_reservation_price_at_terminal():
    """At t=T, r(s, q, T) = s for all q."""
    for q in (-3, 0, 5):
        assert np.isclose(reservation_price(100.0, q, 1.0, 0.1, 2.0, 1.0), 100.0)


def test_reservation_price_linear_in_q():
    """r decreases linearly in q with slope -gamma sigma^2 (T - t)."""
    g, sig, T, t = 0.1, 2.0, 1.0, 0.5
    slope = -g * sig**2 * (T - t)
    r0 = reservation_price(100.0, 0, t, g, sig, T)
    r1 = reservation_price(100.0, 1, t, g, sig, T)
    assert np.isclose(r1 - r0, slope)


def test_optimal_delta_exp_intensity_total_spread():
    """delta^a + delta^b == gamma sigma^2 tau + (2/gamma) ln(1 + gamma/k)."""
    s, T, sigma, gamma, k = 100.0, 1.0, 2.0, 0.1, 1.5
    for q in (-3, 0, 4):
        for t in (0.0, 0.5):
            r_a = reservation_ask(s, q, t, gamma, sigma, T)
            r_b = reservation_bid(s, q, t, gamma, sigma, T)
            d_a = optimal_delta_exp_intensity("ask", s, r_a, gamma, k)
            d_b = optimal_delta_exp_intensity("bid", s, r_b, gamma, k)
            expected = total_spread_equation_faithful(t, gamma, sigma, T, k)
            assert np.isclose(d_a + d_b, expected)


def test_implicit_general_root_matches_closed_form():
    """Solving the general implicit equation reproduces the closed form."""
    s, gamma, k, A = 100.0, 0.1, 1.5, 140.0
    lam = lambda d: A * np.exp(-k * d)
    lam_p = lambda d: -k * A * np.exp(-k * d)
    r_a = 100.5
    d_a_closed = optimal_delta_exp_intensity("ask", s, r_a, gamma, k)
    d_a_num = solve_implicit_quote("ask", s, r_a, gamma, lam, lam_p)
    assert np.isclose(d_a_closed, d_a_num, atol=1e-8)


def test_implicit_residual_zero_at_solution():
    s, gamma, k, A = 100.0, 0.5, 1.5, 140.0
    lam = lambda d: A * np.exp(-k * d)
    lam_p = lambda d: -k * A * np.exp(-k * d)
    r_a = 100.7
    d_a = optimal_delta_exp_intensity("ask", s, r_a, gamma, k)
    res = implicit_quote_general("ask", s, r_a, d_a, gamma, lam, lam_p)
    assert abs(res) < 1e-12


def test_total_spread_table_vs_eq_at_terminal():
    """Eq-faithful spread collapses to table-faithful at t=T."""
    gamma, sigma, T, k = 0.1, 2.0, 1.0, 1.5
    eq = total_spread_equation_faithful(T, gamma, sigma, T, k)
    tab = total_spread_table_faithful(gamma, k)
    assert np.isclose(eq, tab)


def test_table_faithful_spread_values_match_paper():
    """Cross-check the three reported spreads (≈1.33, 1.29, 1.15)."""
    k = 1.5
    assert np.isclose(total_spread_table_faithful(0.01, k), (2 / 0.01) * np.log(1 + 0.01 / 1.5))
    assert abs(total_spread_table_faithful(0.1, k) - 1.29) < 0.02
    assert abs(total_spread_table_faithful(0.5, k) - 1.15) < 0.05


def test_infinite_horizon_admissibility():
    sigma, gamma = 2.0, 0.1
    # Large q + low omega should violate admissibility
    assert not infinite_horizon_admissible(10, gamma, sigma, 0.001)
    assert infinite_horizon_admissible(1, gamma, sigma, 1.0)


def test_infinite_horizon_raises_on_violation():
    with pytest.raises(ValueError):
        infinite_horizon_reservation_prices(100.0, 10, 0.1, 2.0, 0.001)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        reservation_price(100.0, 0, 0.5, -0.1, 2.0, 1.0)
    with pytest.raises(ValueError):
        total_spread_table_faithful(0.0, 1.5)
