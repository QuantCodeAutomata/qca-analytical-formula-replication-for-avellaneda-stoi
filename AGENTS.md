# Repository skill / persistent memory

## Project intent
Replicate Avellaneda–Stoikov (2008) market-making model: analytical formulas + Monte Carlo.

## Conventions
* `src/formulas.py` is the analytical reference (Section 2 of the paper).
* `src/simulator.py` is the discrete-time Monte Carlo engine. It exposes
  `SimConfig` flags `fill_model`, `prob_handling`, `simultaneous_fills`,
  `price_model` so every implementation ambiguity is selectable.
* Spread mode is selected via `SimConfig.spread_mode in {"table_faithful",
  "equation_faithful"}`; the time-varying half-spread is precomputed in
  `h_grid` to keep the per-step loop cheap.
* Use `numpy.random.default_rng(seed)` for reproducibility — seeds are
  fixed in `src/config.py`.
* Tests live in `tests/` and run in <2 s; experiment scripts in `exp/`
  call only `src/`.

## Gotchas encountered
* `simultaneous_fills=False` cannot reuse the fill uniforms `U_a`, `U_b`
  to choose which side wins on a tie — conditioned on both firing, both
  uniforms are confined to `[0, p)` and are therefore not uniform on
  `[0, 1]`. This biased inventory drift to `mean_q ≈ -13` in early runs.
  Fix: independent `U_tie` stream in `simulator.run_simulation`.
* When `γ=0.5`, the table-faithful constant spread is small enough that
  the reservation-price drift `qγσ²(T−t)` can push one quote into
  negative-`δ` territory, so the diagnostic `viol_*_pct` is non-zero (≈ 9 %).
  This is expected and matches the paper's discussion.

## How to run everything
```
pip install numpy pandas scipy sympy matplotlib seaborn pytest tabulate
pytest -q
PYTHONPATH=. python exp/run_all.py
```
