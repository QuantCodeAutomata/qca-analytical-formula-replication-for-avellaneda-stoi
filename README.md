# Avellaneda–Stoikov Market-Making Replication

Reference Python implementation and Monte Carlo replication of
*Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"*.

Four experiments are delivered:

| ID  | Title                                                  | Folder                                |
|-----|--------------------------------------------------------|---------------------------------------|
| 1   | Analytical formula replication                         | `exp/exp_1_formulas.py`               |
| 2   | Monte Carlo with table-faithful constant spread        | `exp/exp_2_table_faithful.py`         |
| 3   | Monte Carlo with equation-faithful time-varying spread | `exp/exp_3_equation_faithful.py`      |
| 4   | Robustness sweep over implementation ambiguities       | `exp/exp_4_robustness.py`             |

## Project layout

```
src/        # closed-form formulas, strategies, simulator, metrics, plots
exp/        # experiment driver scripts (depend only on src/)
tests/      # pytest unit / integration tests
results/    # generated tables (CSV), plots (PNG) and RESULTS.md
data/       # empty — experiments are fully synthetic
```

## Reproducing the experiments

```bash
pip install numpy pandas scipy sympy matplotlib seaborn pytest tabulate
pytest -q                                  # 27 tests, ~2s
PYTHONPATH=. python exp/run_all.py         # writes everything in results/
```

The runner regenerates every CSV / PNG under `results/` and rewrites
`results/RESULTS.md` with the headline metrics.

## Key implementation choices

* **Closed-form formulas** (`src/formulas.py`) — implemented from scratch from
  the paper (Eqs. for `r^a`, `r^b`, frozen-inventory value function, general
  HJB-implicit optimal quotes, exponential-intensity specialisation and the
  infinite-horizon variant). Context7 confirmed no library equivalent.
* **Monte Carlo** (`src/simulator.py`) — vectorised over paths with a
  per-step Python loop because cash/inventory are path-state dependent.
* **Tie-break under `simultaneous_fills=False`** — uses an independent
  uniform stream `U_tie`; building the tie-break from the fill uniforms
  themselves is biased because, conditioned on both fires, both lie in
  `[0, p)` and are therefore not uniform on `[0, 1]`.
* **Both spread conventions** — `spread_mode="table_faithful"` reproduces the
  paper's tables (constant spread `(2/γ) ln(1+γ/k)`);
  `spread_mode="equation_faithful"` uses the theoretical formula
  `γσ²(T−t) + (2/γ) ln(1+γ/k)` from the HJB derivation.
* **Robustness sweep** — every implementation rule is exposed as a flag on
  `SimConfig` (`fill_model`, `prob_handling`, `simultaneous_fills`,
  `price_model`) so any ambiguity in the paper can be flipped independently.

See `results/RESULTS.md` for headline numbers and `src/` for the documented
implementations.
