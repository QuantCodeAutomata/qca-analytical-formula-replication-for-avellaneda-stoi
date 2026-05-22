"""Experiment 1: Analytical formula replication for Avellaneda-Stoikov.

Produces a reference table of closed-form values and verifies the algebraic
identities used by the simulation experiments.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def reference_table() -> pd.DataFrame:
    """Reference table of finite-horizon reservation prices and quotes."""
    s, T, sigma, A, k = 100.0, 1.0, 2.0, 140.0, 1.5
    rows = []
    for gamma in (0.01, 0.1, 0.5):
        for q in (-5, -1, 0, 1, 5):
            for t in (0.0, 0.25, 0.5, 0.75, 0.99):
                r = reservation_price(s, q, t, gamma, sigma, T)
                r_a = reservation_ask(s, q, t, gamma, sigma, T)
                r_b = reservation_bid(s, q, t, gamma, sigma, T)
                d_a = optimal_delta_exp_intensity("ask", s, r_a, gamma, k)
                d_b = optimal_delta_exp_intensity("bid", s, r_b, gamma, k)
                spread_eq = total_spread_equation_faithful(t, gamma, sigma, T, k)
                spread_tab = total_spread_table_faithful(gamma, k)
                rows.append(
                    dict(
                        gamma=gamma, q=q, t=t,
                        r=r, r_a=r_a, r_b=r_b,
                        delta_a=d_a, delta_b=d_b,
                        spread_eq_faithful=spread_eq,
                        spread_tab_faithful=spread_tab,
                    )
                )
    return pd.DataFrame(rows)


def verify_identities() -> dict:
    """Numeric verification of paper identities."""
    s, T, sigma, A, k = 100.0, 1.0, 2.0, 140.0, 1.5
    checks = {}

    # 1. r = (r^a + r^b)/2
    err = 0.0
    for gamma in (0.01, 0.1, 0.5):
        for q in (-5, -1, 0, 1, 5):
            for t in (0.0, 0.5, 0.99):
                r = reservation_price(s, q, t, gamma, sigma, T)
                r_a = reservation_ask(s, q, t, gamma, sigma, T)
                r_b = reservation_bid(s, q, t, gamma, sigma, T)
                err = max(err, abs(r - 0.5 * (r_a + r_b)))
    checks["max_err_r_eq_midpoint"] = err

    # 2. q=0 implies r = s
    checks["q0_r_equals_s"] = abs(reservation_price(s, 0, 0.0, 0.1, sigma, T) - s)

    # 3. t -> T implies r -> s
    checks["t_to_T_r_to_s"] = abs(reservation_price(s, 3, T, 0.1, sigma, T) - s)

    # 4. Exponential-intensity total spread equation:
    #    delta^a + delta^b = gamma sigma^2 tau + (2/gamma) ln(1 + gamma/k)
    err_spread = 0.0
    for gamma in (0.01, 0.1, 0.5):
        for q in (-3, 0, 3):
            for t in (0.0, 0.5, 0.99):
                r_a = reservation_ask(s, q, t, gamma, sigma, T)
                r_b = reservation_bid(s, q, t, gamma, sigma, T)
                d_a = optimal_delta_exp_intensity("ask", s, r_a, gamma, k)
                d_b = optimal_delta_exp_intensity("bid", s, r_b, gamma, k)
                exp_spread = total_spread_equation_faithful(t, gamma, sigma, T, k)
                err_spread = max(err_spread, abs((d_a + d_b) - exp_spread))
    checks["max_err_total_spread_identity"] = err_spread

    # 5. General implicit equation -> same delta as closed form (with lambda exp)
    A_, k_ = A, k
    lam = lambda d: A_ * np.exp(-k_ * d)
    lam_p = lambda d: -k_ * A_ * np.exp(-k_ * d)
    err_root = 0.0
    for gamma in (0.01, 0.1, 0.5):
        for q in (-2, 0, 2):
            r_a = reservation_ask(s, q, 0.5, gamma, sigma, T)
            r_b = reservation_bid(s, q, 0.5, gamma, sigma, T)
            d_a_closed = optimal_delta_exp_intensity("ask", s, r_a, gamma, k_)
            d_b_closed = optimal_delta_exp_intensity("bid", s, r_b, gamma, k_)
            d_a_num = solve_implicit_quote("ask", s, r_a, gamma, lam, lam_p)
            d_b_num = solve_implicit_quote("bid", s, r_b, gamma, lam, lam_p)
            err_root = max(err_root, abs(d_a_closed - d_a_num), abs(d_b_closed - d_b_num))
    checks["max_err_general_vs_closed_form"] = err_root

    # 6. Infinite-horizon admissibility
    g, q, om = 0.1, 4, 0.05
    checks["infinite_horizon_admissible_low_omega"] = infinite_horizon_admissible(q, g, sigma, om)
    checks["infinite_horizon_admissible_high_omega"] = infinite_horizon_admissible(q, g, sigma, 5.0)

    # 7. Frozen-inventory value: q=0 -> v = -exp(-gamma x)
    v0 = frozen_inventory_value(10.0, s, 0, 0.5, 0.1, sigma, T)
    checks["frozen_value_q0_err"] = abs(v0 - (-np.exp(-0.1 * 10.0)))

    return checks


def infinite_horizon_table() -> pd.DataFrame:
    """Reference table of infinite-horizon reservation prices."""
    s, sigma = 100.0, 2.0
    rows = []
    for gamma in (0.01, 0.1, 0.5):
        for omega in (1.0, 5.0):
            for q in (-3, -1, 0, 1, 3):
                if not infinite_horizon_admissible(q, gamma, sigma, omega):
                    rows.append(dict(gamma=gamma, omega=omega, q=q,
                                     r_bar_a=np.nan, r_bar_b=np.nan, admissible=False))
                    continue
                r_a, r_b = infinite_horizon_reservation_prices(s, q, gamma, sigma, omega)
                rows.append(dict(gamma=gamma, omega=omega, q=q,
                                 r_bar_a=r_a, r_bar_b=r_b, admissible=True))
    return pd.DataFrame(rows)


def main() -> dict:
    tab = reference_table()
    tab.to_csv(RESULTS / "exp1_reference_table.csv", index=False)

    inf_tab = infinite_horizon_table()
    inf_tab.to_csv(RESULTS / "exp1_infinite_horizon_table.csv", index=False)

    checks = verify_identities()
    with open(RESULTS / "exp1_identity_checks.txt", "w") as fh:
        for k_, v in checks.items():
            fh.write(f"{k_}: {v}\n")

    print("Experiment 1: identity checks")
    for k_, v in checks.items():
        print(f"  {k_}: {v}")
    return checks


if __name__ == "__main__":
    main()
