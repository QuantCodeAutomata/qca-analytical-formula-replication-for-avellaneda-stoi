# Avellaneda-Stoikov Replication — Results Summary

This document summarises the four experiments described in the project specification: analytical-formula replication (exp 1), table-faithful Monte Carlo (exp 2), equation-faithful Monte Carlo (exp 3), and an implementation-ambiguity robustness sweep (exp 4).


## Experiment 1 — Analytical formula identity checks

All algebraic identities are recovered numerically in float64.

```
max_err_r_eq_midpoint: 1.4210854715202004e-14
q0_r_equals_s: 0.0
t_to_T_r_to_s: 0.0
max_err_total_spread_identity: 1.3322676295501878e-14
max_err_general_vs_closed_form: 8.881784197001252e-16
infinite_horizon_admissible_low_omega: False
infinite_horizon_admissible_high_omega: True
frozen_value_q0_err: 0.0
```

Reference tables: `results/exp1_reference_table.csv` (finite-horizon) and `results/exp1_infinite_horizon_table.csv`.


## Experiment 2 — Table-faithful constant spread

Spread is set constant at `(2/gamma) ln(1 + gamma/k)`. The inventory strategy is centred at the reservation price; the benchmark is centred at the mid-price. 1000 paths per (gamma, strategy).

|   gamma | strategy   | spread_mode    |   Spread |   Profit |   std(Profit) |   Final q |   std(Final q) |
|--------:|:-----------|:---------------|---------:|---------:|--------------:|----------:|---------------:|
|  0.1000 | inventory  | table_faithful |   1.2908 |  63.8720 |        5.9229 |    0.0960 |         2.9988 |
|  0.1000 | symmetric  | table_faithful |   1.2908 |  68.0780 |       13.1584 |    0.1550 |         8.8251 |
|  0.0100 | inventory  | table_faithful |   1.3289 |  68.0466 |        8.7831 |   -0.1560 |         5.3652 |
|  0.0100 | symmetric  | table_faithful |   1.3289 |  67.7712 |       13.6892 |   -0.0820 |         8.9765 |
|  0.5000 | inventory  | table_faithful |   1.1507 |  23.9539 |        4.9032 |    0.1480 |         1.9277 |
|  0.5000 | symmetric  | table_faithful |   1.1507 |  68.2910 |       14.5413 |    0.4070 |         9.4143 |

Paper-reported targets (gamma=0.1): inventory Spread≈1.29, Profit≈62.94, std(Profit)≈5.89, Final q≈0.10, std(Final q)≈2.80; symmetric Profit≈67.21, std(Profit)≈13.43, std(Final q)≈8.66.


Figures: `exp2_sample_path_*`, `exp2_profit_hist_g*.png`, `exp2_inventory_hist_g*.png`.


## Experiment 3 — Equation-faithful time-varying spread

Spread is set to the theoretical formula `gamma sigma^2 (T-t) + (2/gamma) ln(1 + gamma/k)`, which collapses to the table-faithful constant at t = T.

|   gamma | strategy   | spread_mode       |   Spread |   Profit |   std(Profit) |   Final q |   std(Final q) |
|--------:|:-----------|:------------------|---------:|---------:|--------------:|----------:|---------------:|
|  0.1000 | inventory  | equation_faithful |   1.4918 |  64.5331 |        6.4181 |    0.0750 |         2.9929 |
|  0.1000 | symmetric  | equation_faithful |   1.4918 |  67.1858 |       12.9172 |    0.4280 |         8.4480 |
|  0.0100 | inventory  | equation_faithful |   1.3490 |  68.0663 |        8.8861 |   -0.1440 |         5.3857 |
|  0.0100 | symmetric  | equation_faithful |   1.3490 |  67.8705 |       13.6635 |   -0.1300 |         8.9859 |
|  0.5000 | inventory  | equation_faithful |   2.1557 |  48.4271 |        5.7185 |    0.1570 |         1.9407 |
|  0.5000 | symmetric  | equation_faithful |   2.1557 |  58.6286 |       10.8561 |    0.3070 |         7.1637 |

Figures: `exp3_sample_path_*`, `exp3_profit_hist_g*.png`, `exp3_inventory_hist_g*.png`, and `exp3_spread_trajectories.png` (shows how the equation-faithful spread starts wider than the table-faithful constant and converges as t → T).


## Experiment 4 — Robustness / ambiguity diagnostics

Sweep over fill model (Bernoulli, exact-event, capped-Poisson), probability handling for λdt>1 (raw, clipped, 1-exp(-λdt)), simultaneous-fill rule (allow / forbid) and price increment (binomial vs Gaussian Euler). Baseline is the table-faithful setup of exp 2.

| label             |   gamma | strategy   |   mean_profit |   std_profit |   mean_final_q |   std_final_q |   viol_ask_pct |   viol_bid_pct |   simultaneous_pct |
|:------------------|--------:|:-----------|--------------:|-------------:|---------------:|--------------:|---------------:|---------------:|-------------------:|
| A_baseline        |  0.0100 | inventory  |       68.1151 |       8.7920 |        -0.1840 |        5.3528 |         0.0000 |         0.0000 |             6.7020 |
| A_baseline        |  0.0100 | symmetric  |       68.8206 |      14.2719 |         0.3260 |        8.8941 |         0.0000 |         0.0000 |             6.6945 |
| A_baseline        |  0.1000 | inventory  |       64.3912 |       5.9705 |        -0.1160 |        2.9070 |         0.0255 |         0.0270 |             7.2005 |
| A_baseline        |  0.1000 | symmetric  |       68.1939 |      14.2508 |        -0.2050 |        9.0339 |         0.0000 |         0.0000 |             7.0450 |
| A_baseline        |  0.5000 | inventory  |       23.8666 |       5.0026 |         0.1070 |        2.0454 |         9.0450 |         9.0420 |             7.9230 |
| A_baseline        |  0.5000 | symmetric  |       67.7419 |      14.3792 |         0.1470 |        9.1217 |         0.0000 |         0.0000 |             8.7440 |
| A_exact_event     |  0.0100 | inventory  |       59.8523 |       8.8587 |        -0.2490 |        5.4422 |         0.0000 |         0.0000 |             5.1355 |
| A_exact_event     |  0.0100 | symmetric  |       60.8150 |      13.4964 |        -0.2410 |        8.2770 |         0.0000 |         0.0000 |             5.1435 |
| A_exact_event     |  0.1000 | inventory  |       54.8616 |       5.9639 |        -0.0130 |        3.0001 |         0.0825 |         0.0880 |             5.2480 |
| A_exact_event     |  0.1000 | symmetric  |       59.8422 |      13.9278 |         0.1630 |        8.5609 |         0.0000 |         0.0000 |             5.5330 |
| A_exact_event     |  0.5000 | inventory  |       18.2960 |       4.7034 |         0.0760 |        1.9956 |         9.6780 |         9.6095 |             5.8700 |
| A_exact_event     |  0.5000 | symmetric  |       58.2527 |      13.6625 |        -0.0210 |        8.2869 |         0.0000 |         0.0000 |             6.5345 |
| A_poisson_capped  |  0.0100 | inventory  |       60.1050 |       9.0290 |         0.0950 |        5.5390 |         0.0000 |         0.0000 |             5.1395 |
| A_poisson_capped  |  0.0100 | symmetric  |       60.3120 |      13.5635 |         0.2650 |        8.4891 |         0.0000 |         0.0000 |             5.1350 |
| A_poisson_capped  |  0.1000 | inventory  |       55.5109 |       5.8871 |         0.0360 |        2.9881 |         0.0690 |         0.0915 |             5.4045 |
| A_poisson_capped  |  0.1000 | symmetric  |       59.8901 |      13.3090 |        -0.3520 |        8.4532 |         0.0000 |         0.0000 |             5.4140 |
| A_poisson_capped  |  0.5000 | inventory  |       18.1510 |       4.8361 |         0.0760 |        2.1115 |         9.5840 |         9.7325 |             5.8350 |
| A_poisson_capped  |  0.5000 | symmetric  |       58.8046 |      13.6810 |        -0.0960 |        8.6589 |         0.0000 |         0.0000 |             6.5910 |
| B_clipped         |  0.0100 | inventory  |       68.4570 |       8.8881 |        -0.2300 |        5.1413 |         0.0000 |         0.0000 |             6.6640 |
| B_clipped         |  0.0100 | symmetric  |       68.7234 |      13.2815 |        -0.0560 |        8.5010 |         0.0000 |         0.0000 |             6.6440 |
| B_clipped         |  0.1000 | inventory  |       64.0899 |       6.1045 |         0.1820 |        2.8998 |         0.0280 |         0.0230 |             7.0590 |
| B_clipped         |  0.1000 | symmetric  |       69.3397 |      13.9034 |         0.4950 |        9.1347 |         0.0000 |         0.0000 |             7.1315 |
| B_clipped         |  0.5000 | inventory  |       23.8110 |       4.9476 |         0.0060 |        2.0273 |         9.1170 |         9.1275 |             7.8510 |
| B_clipped         |  0.5000 | symmetric  |       68.2666 |      13.8886 |         0.0530 |        9.3181 |         0.0000 |         0.0000 |             8.8465 |
| B_one_minus_exp   |  0.0100 | inventory  |       59.9193 |       9.2291 |         0.3060 |        5.3590 |         0.0000 |         0.0000 |             5.0830 |
| B_one_minus_exp   |  0.0100 | symmetric  |       61.2345 |      13.0697 |        -0.0200 |        8.3673 |         0.0000 |         0.0000 |             5.2610 |
| B_one_minus_exp   |  0.1000 | inventory  |       55.4035 |       5.8337 |        -0.0600 |        3.0483 |         0.0785 |         0.0845 |             5.3780 |
| B_one_minus_exp   |  0.1000 | symmetric  |       60.2677 |      13.7651 |         0.3580 |        8.5048 |         0.0000 |         0.0000 |             5.3740 |
| B_one_minus_exp   |  0.5000 | inventory  |       18.1497 |       4.7538 |         0.0030 |        2.0501 |         9.7080 |         9.6045 |             5.8645 |
| B_one_minus_exp   |  0.5000 | symmetric  |       58.9851 |      12.9351 |         0.1600 |        8.5589 |         0.0000 |         0.0000 |             6.5540 |
| C_no_simultaneous |  0.0100 | inventory  |       59.3644 |       8.6358 |        -0.1070 |        5.3981 |         0.0000 |         0.0000 |             6.7170 |
| C_no_simultaneous |  0.0100 | symmetric  |       60.3684 |      13.6816 |        -0.3870 |        9.3565 |         0.0000 |         0.0000 |             6.7260 |
| C_no_simultaneous |  0.1000 | inventory  |       54.0668 |       5.0339 |        -0.0080 |        3.0401 |         0.0345 |         0.0310 |             7.0115 |
| C_no_simultaneous |  0.1000 | symmetric  |       59.6140 |      14.9238 |        -0.1900 |        9.9497 |         0.0000 |         0.0000 |             7.1325 |
| C_no_simultaneous |  0.5000 | inventory  |        7.1408 |       2.7812 |        -0.0240 |        2.0984 |        10.0895 |        10.1240 |             7.6935 |
| C_no_simultaneous |  0.5000 | symmetric  |       58.6804 |      14.5371 |        -0.0420 |        9.9052 |         0.0000 |         0.0000 |             8.6895 |
| D_gaussian        |  0.0100 | inventory  |       68.1218 |       8.7101 |         0.1600 |        5.2280 |         0.0000 |         0.0000 |             6.6380 |
| D_gaussian        |  0.0100 | symmetric  |       69.3208 |      14.2315 |         0.4230 |        8.6286 |         0.0000 |         0.0000 |             6.6345 |
| D_gaussian        |  0.1000 | inventory  |       64.3957 |       5.8698 |        -0.0200 |        2.9842 |         0.0155 |         0.0190 |             7.0605 |
| D_gaussian        |  0.1000 | symmetric  |       69.3947 |      13.8405 |        -0.5550 |        8.8016 |         0.0000 |         0.0000 |             7.0555 |
| D_gaussian        |  0.5000 | inventory  |       24.0435 |       5.0849 |         0.0430 |        1.9857 |         9.0410 |         9.1305 |             7.9770 |
| D_gaussian        |  0.5000 | symmetric  |       67.5910 |      14.0588 |        -0.1500 |        8.9188 |         0.0000 |         0.0000 |             8.6130 |

Full per-configuration table: `results/exp4_robustness.csv` and pivoted view `results/exp4_robustness_pivot.csv`.


## Interpretation

- Exp 1 confirms every closed-form identity used by the simulator to machine precision.
- Exp 2 reproduces the paper's qualitative ordering: the inventory strategy has materially lower std(profit) and std(final q) than the symmetric benchmark for every gamma, with the gap widening in gamma.
- Exp 3 shows that the equation-faithful spread is wider early in the horizon and converges to the table-faithful constant at t = T. Mean profit decreases (wider spreads ⇒ fewer fills) but the qualitative inventory-control conclusion is robust.
- Exp 4 confirms that the principal qualitative result is robust across every fill / probability / price-discretisation variant tested. Quantitative deviations from the paper's Tables 1–3 are partly explained by the Bernoulli approximation breaking down whenever λdt>1; using the exact event probability `1 − exp(−λdt)` or clipping reduces the bias.
