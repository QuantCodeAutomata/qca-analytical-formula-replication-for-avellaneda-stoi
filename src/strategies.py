"""Quoting strategies for Avellaneda-Stoikov simulation.

# Custom — Context7 found no library equivalent (paper Section 3.3).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def inventory_quote_center(
    S: np.ndarray, q: np.ndarray, tau: float, gamma: float, sigma: float
) -> np.ndarray:
    """r_n = S_n - q_n * gamma * sigma^2 * tau (paper Section 3.3)."""
    return S - q * gamma * sigma**2 * tau


def symmetric_quote_center(S: np.ndarray) -> np.ndarray:
    """Benchmark center c_n = S_n."""
    return S.copy()


def make_quotes(
    center: np.ndarray, half_spread: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct (bid, ask) quotes from a center and a half-spread."""
    ask = center + half_spread
    bid = center - half_spread
    return bid, ask
