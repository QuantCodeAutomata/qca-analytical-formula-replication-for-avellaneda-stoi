"""Smoke tests for experiment entry points."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_exp1_module_imports_and_runs_identities():
    mod = importlib.import_module("exp.exp_1_formulas")
    checks = mod.verify_identities()
    # Tolerances reflect symbolic identities computed in float64
    assert checks["max_err_r_eq_midpoint"] < 1e-10
    assert checks["q0_r_equals_s"] < 1e-12
    assert checks["t_to_T_r_to_s"] < 1e-12
    assert checks["max_err_total_spread_identity"] < 1e-10
    assert checks["max_err_general_vs_closed_form"] < 1e-8


def test_exp1_reference_table_has_rows():
    mod = importlib.import_module("exp.exp_1_formulas")
    tab = mod.reference_table()
    assert isinstance(tab, pd.DataFrame)
    assert len(tab) > 0
    for col in ("r", "r_a", "r_b", "delta_a", "delta_b"):
        assert col in tab.columns
