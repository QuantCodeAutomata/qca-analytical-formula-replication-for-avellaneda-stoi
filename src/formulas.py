"""Analytical formulas for the Avellaneda-Stoikov market-making model.

Reference: Avellaneda, M. & Stoikov, S. (2008) "High-frequency trading in a
limit order book", Quantitative Finance, 8(3), 217-224.

# Custom — Context7 confirmed no library equivalent for these HJB closed-form
# expressions; implemented faithfully from the paper.
"""
from __future__ import annotations

from typing import Callable, Literal, Tuple

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Section 3.1 - Frozen-inventory value function and reservation prices
# ---------------------------------------------------------------------------
def frozen_inventory_value(
    x: float, s: float, q: float, t: float, gamma: float, sigma: float, T: float
) -> float:
    r"""Closed form for the frozen-inventory exponential-utility value function.

    v(x, s, q, t) = -exp(-gamma x) * exp(-gamma q s) * exp(gamma^2 q^2 sigma^2 (T-t) / 2)

    Paper Eq. (Section 3.1) — derived from E_t[-exp(-gamma (x + q S_T))] under
    dS = sigma dW with S_t = s.
    """
    _validate_positive(gamma=gamma, sigma=sigma, T=T)
    if t > T + 1e-12:
        raise ValueError("t must be <= T")
    tau = T - t
    return -np.exp(-gamma * x) * np.exp(-gamma * q * s) * np.exp(
        0.5 * gamma**2 * q**2 * sigma**2 * tau
    )


def reservation_ask(
    s: float, q: float, t: float, gamma: float, sigma: float, T: float
) -> float:
    r"""Reservation ask price (finite-horizon, frozen inventory).

    r^a(s, q, t) = s + ((1 - 2 q) / 2) * gamma * sigma^2 * (T - t)
    """
    _validate_positive(gamma=gamma, sigma=sigma, T=T)
    tau = T - t
    return s + 0.5 * (1.0 - 2.0 * q) * gamma * sigma**2 * tau


def reservation_bid(
    s: float, q: float, t: float, gamma: float, sigma: float, T: float
) -> float:
    r"""Reservation bid price (finite-horizon, frozen inventory).

    r^b(s, q, t) = s + ((-1 - 2 q) / 2) * gamma * sigma^2 * (T - t)
    """
    _validate_positive(gamma=gamma, sigma=sigma, T=T)
    tau = T - t
    return s + 0.5 * (-1.0 - 2.0 * q) * gamma * sigma**2 * tau


def reservation_price(
    s: float, q: float, t: float, gamma: float, sigma: float, T: float
) -> float:
    r"""Reservation (indifference) mid price.

    r(s, q, t) = (r^a + r^b) / 2 = s - q * gamma * sigma^2 * (T - t)
    """
    _validate_positive(gamma=gamma, sigma=sigma, T=T)
    tau = T - t
    return s - q * gamma * sigma**2 * tau


# ---------------------------------------------------------------------------
# Section 3.2 - HJB / implicit optimal quote relations
# ---------------------------------------------------------------------------
def implicit_quote_general(
    side: Literal["ask", "bid"],
    s: float,
    r: float,
    delta: float,
    gamma: float,
    lambda_func: Callable[[float], float],
    lambda_prime_func: Callable[[float], float],
) -> float:
    r"""Residual of the general implicit optimal-quote equation.

    For arbitrary differentiable intensity lambda(delta):
        ask:  r^a - s = delta - (1/gamma) ln(1 - gamma * lambda(delta) / lambda'(delta))
        bid:  s - r^b = delta - (1/gamma) ln(1 - gamma * lambda(delta) / lambda'(delta))

    Returns LHS - RHS; root finder seeks zero in `delta`.
    """
    _validate_positive(gamma=gamma)
    lam = lambda_func(delta)
    lam_p = lambda_prime_func(delta)
    if lam_p == 0:
        raise ZeroDivisionError("lambda_prime_func(delta) == 0")
    inside = 1.0 - gamma * lam / lam_p
    if inside <= 0:
        raise ValueError(f"log argument {inside} not positive; adjust delta range")
    rhs = delta - (1.0 / gamma) * np.log(inside)
    if side == "ask":
        lhs = r - s
    elif side == "bid":
        lhs = s - r
    else:
        raise ValueError("side must be 'ask' or 'bid'")
    return lhs - rhs


def solve_implicit_quote(
    side: Literal["ask", "bid"],
    s: float,
    r: float,
    gamma: float,
    lambda_func: Callable[[float], float],
    lambda_prime_func: Callable[[float], float],
    bracket: Tuple[float, float] = (-10.0, 10.0),
) -> float:
    """Numerically solve the general implicit quote equation for delta."""

    def f(d: float) -> float:
        return implicit_quote_general(
            side, s, r, d, gamma, lambda_func, lambda_prime_func
        )

    return brentq(f, bracket[0], bracket[1])


def optimal_delta_exp_intensity(
    side: Literal["ask", "bid"],
    s: float,
    r: float,
    gamma: float,
    k: float,
) -> float:
    r"""Optimal half-distance under exponential intensity lambda = A e^{-k delta}.

    Substituting lambda/lambda' = -1/k into the implicit equation yields:
        delta^a = (r^a - s) + (1/gamma) ln(1 + gamma/k)
        delta^b = (s - r^b) + (1/gamma) ln(1 + gamma/k)
    Here `r` is the side-specific reservation price.
    """
    _validate_positive(gamma=gamma, k=k)
    add = (1.0 / gamma) * np.log(1.0 + gamma / k)
    if side == "ask":
        return (r - s) + add
    if side == "bid":
        return (s - r) + add
    raise ValueError("side must be 'ask' or 'bid'")


def total_spread_equation_faithful(
    t: float, gamma: float, sigma: float, T: float, k: float
) -> float:
    r"""Equation-faithful total spread (paper Section 3.2).

    delta^a + delta^b = gamma * sigma^2 * (T - t) + (2/gamma) ln(1 + gamma/k)
    """
    _validate_positive(gamma=gamma, sigma=sigma, T=T, k=k)
    tau = T - t
    return gamma * sigma**2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)


def total_spread_table_faithful(gamma: float, k: float) -> float:
    r"""Table-faithful constant total spread used to reproduce Tables 1-3.

    Spread = (2/gamma) ln(1 + gamma/k)
    """
    _validate_positive(gamma=gamma, k=k)
    return (2.0 / gamma) * np.log(1.0 + gamma / k)


# ---------------------------------------------------------------------------
# Section 4 - Infinite-horizon reservation prices
# ---------------------------------------------------------------------------
def infinite_horizon_reservation_prices(
    s: float, q: float, gamma: float, sigma: float, omega: float
) -> Tuple[float, float]:
    r"""Infinite-horizon reservation bid and ask.

    bar r^a(s, q) = s + (1/gamma) ln(1 + ((1 - 2 q) gamma^2 sigma^2)
                                          / (2 omega - gamma^2 q^2 sigma^2))
    bar r^b(s, q) = s + (1/gamma) ln(1 + ((-1 - 2 q) gamma^2 sigma^2)
                                          / (2 omega - gamma^2 q^2 sigma^2))

    Admissibility: omega > 0.5 * gamma^2 * sigma^2 * q^2.
    """
    _validate_positive(gamma=gamma, sigma=sigma)
    admissibility = 2.0 * omega - gamma**2 * q**2 * sigma**2
    if admissibility <= 0:
        raise ValueError(
            f"Admissibility violated: 2*omega ({2*omega}) must exceed "
            f"gamma^2 q^2 sigma^2 ({gamma**2 * q**2 * sigma**2})"
        )
    num_a = (1.0 - 2.0 * q) * gamma**2 * sigma**2
    num_b = (-1.0 - 2.0 * q) * gamma**2 * sigma**2
    r_a = s + (1.0 / gamma) * np.log(1.0 + num_a / admissibility)
    r_b = s + (1.0 / gamma) * np.log(1.0 + num_b / admissibility)
    return r_a, r_b


def infinite_horizon_admissible(
    q: float, gamma: float, sigma: float, omega: float
) -> bool:
    """Check the infinite-horizon admissibility condition omega > 0.5 gamma^2 sigma^2 q^2."""
    return omega > 0.5 * gamma**2 * sigma**2 * q**2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_positive(**kwargs: float) -> None:
    for name, val in kwargs.items():
        if not np.isfinite(val) or val <= 0:
            raise ValueError(f"Parameter '{name}' must be positive and finite (got {val}).")
