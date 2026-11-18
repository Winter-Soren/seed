# E112: EVAL v3 saved-genome diagnostics

Measured on 19 November 2026. No evolution or fitting was performed.

**Scope:** eight saved-genome panels, each on the same 256 fresh held-out worlds (seed 2026111901; 128 worlds per initial food rule; four agents per world). Each panel includes stationary rules and an unannounced reversal before eating at step 150. The three A genomes come from separate evolution seeds; B0 is a single-seed diagnostic. Blank 0–2 are the saved E100 starting genomes. The biology prior is one deterministic reference.

**Uncertainty:** every ± value below is one standard error, not a 95% interval. Fitness SE uses the sample standard deviation of 256 world-family means divided by √256. Accuracy and fitness/greedy ratios use paired delta-method residuals clustered by world; the four agents within a world are not treated as independent. These errors measure held-out-world uncertainty for each fixed genome, not uncertainty across evolution seeds.

**Reference policies:** random walk samples the same five movement actions uniformly, with zero communication. Greedy sees the same local visual patch and also knows which item is currently food, including immediately after reversal. It approaches visible food while avoiding immediate poison when possible; ties are random. It is a privileged local reference, not an optimal-policy bound. Common policy outputs are identical across the eight panels and are shown only once; the repeated runs are not independent replications.

## Main observations

- **A0:** stationary fitness 15.346 ± 0.377; meal accuracy 0.470 ± 0.016 at meal 1 and 0.794 ± 0.014 at meal 20. Among the same 798 agents reaching meal 20, the change is 26.82 ± 2.42 percentage points. After reversal, all-agent 50-step accuracy is 0.256 ± 0.009, 0.355 ± 0.010, 0.462 ± 0.014.
- **A1:** stationary fitness 15.187 ± 0.385; meal accuracy 0.489 ± 0.016 at meal 1 and 0.768 ± 0.017 at meal 20. Among the same 790 agents reaching meal 20, the change is 21.14 ± 2.44 percentage points. After reversal, all-agent 50-step accuracy is 0.280 ± 0.009, 0.377 ± 0.011, 0.493 ± 0.012.
- **A2:** stationary fitness 14.984 ± 0.351; meal accuracy 0.490 ± 0.014 at meal 1 and 0.794 ± 0.014 at meal 20. Among the same 780 agents reaching meal 20, the change is 24.87 ± 2.24 percentage points. After reversal, all-agent 50-step accuracy is 0.267 ± 0.009, 0.381 ± 0.010, 0.481 ± 0.013.

The matched-meal changes test experience-related improvement within the eligible cohort. They still condition on reaching meal 20 and do not establish that every agent learns. The reversal rows measure adaptation to one specified switch; they do not demonstrate general learning, long-term retention or an advantage over the fair nonplastic baseline, which belongs to Phase 2. Historical C is not that baseline.

**Connection growth:** these diagnostics enable lifetime weight updates on the existing wiring; structural plasticity is off. They do not demonstrate formation of new connections. The earlier growth/pruning prototype lowered fitness in all three fixed-genome comparisons and remains outside this measurement. Better learning here is not evidence that its connection-growth rule helps.

## Fitness and fraction of greedy

Fitness is food minus poison per initial agent over the whole lifetime, including agents that die. The fraction is the ratio of controller mean fitness to the matched greedy mean. A fraction of one is the reference itself, not a universal performance ceiling.

| Condition | Controller | Fitness ± SE | Fraction of greedy ± SE | Survival fraction ± SE | Meals |
| --- | --- | --- | --- | --- | --- |
| stationary | A0 | 15.346 ± 0.377 | 0.1865 ± 0.0056 | 0.841 ± 0.012 | 30566 |
| stationary | A1 | 15.187 ± 0.385 | 0.1846 ± 0.0056 | 0.847 ± 0.012 | 30821 |
| stationary | A2 | 14.984 ± 0.351 | 0.1821 ± 0.0047 | 0.851 ± 0.012 | 29978 |
| stationary | B0 | 14.970 ± 0.376 | 0.1819 ± 0.0054 | 0.830 ± 0.012 | 31141 |
| stationary | Biology prior | 5.793 ± 0.266 | 0.0704 ± 0.0033 | 0.647 ± 0.015 | 24258 |
| stationary | Blank 0 | 0.080 ± 0.139 | 0.0010 ± 0.0017 | 0.379 ± 0.017 | 16012 |
| stationary | Blank 1 | 0.597 ± 0.155 | 0.0073 ± 0.0019 | 0.392 ± 0.016 | 16745 |
| stationary | Blank 2 | 0.257 ± 0.139 | 0.0031 ± 0.0017 | 0.391 ± 0.017 | 16633 |
| stationary | Random walk | -0.093 ± 0.128 | -0.0011 ± 0.0016 | 0.369 ± 0.015 | 14885 |
| stationary | Local rule-knowing greedy | 82.278 ± 1.364 | 1.0000 ± 0.0000 | 1.000 ± 0.000 | 84253 |
| reversal | A0 | 3.127 ± 0.183 | 0.0453 ± 0.0027 | 0.378 ± 0.016 | 25302 |
| reversal | A1 | 3.529 ± 0.183 | 0.0511 ± 0.0027 | 0.422 ± 0.015 | 25920 |
| reversal | A2 | 3.576 ± 0.166 | 0.0518 ± 0.0024 | 0.421 ± 0.015 | 25512 |
| reversal | B0 | 3.161 ± 0.173 | 0.0458 ± 0.0025 | 0.394 ± 0.016 | 26065 |
| reversal | Biology prior | 1.179 ± 0.154 | 0.0171 ± 0.0022 | 0.390 ± 0.016 | 22043 |
| reversal | Blank 0 | -0.015 ± 0.127 | -0.0002 ± 0.0018 | 0.382 ± 0.016 | 15969 |
| reversal | Blank 1 | 0.161 ± 0.136 | 0.0023 ± 0.0020 | 0.380 ± 0.016 | 16991 |
| reversal | Blank 2 | 0.058 ± 0.130 | 0.0008 ± 0.0019 | 0.367 ± 0.014 | 16553 |
| reversal | Random walk | 0.011 ± 0.120 | 0.0002 ± 0.0017 | 0.397 ± 0.015 | 15083 |
| reversal | Local rule-knowing greedy | 69.024 ± 0.700 | 1.0000 ± 0.0000 | 1.000 ± 0.000 | 70681 |

## Accuracy by each agent’s meal number

Each cell is **accuracy ± world-cluster SE (eligible agents)**. Each agent contributes at most one binary food/poison outcome at each ordinal. Every panel starts with 1,024 agents. Agents that never reach a meal are absent from that column, not counted as incorrect. Later columns therefore describe a selected cohort. Food counts and contributing-world counts are retained in each linked summary.

### Stationary

| Controller | Meal 1 | Meal 2 | Meal 3 | Meal 5 | Meal 10 | Meal 20 |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | 0.470 ± 0.016 (1024) | 0.541 ± 0.016 (1024) | 0.573 ± 0.015 (1024) | 0.633 ± 0.016 (966) | 0.738 ± 0.014 (888) | 0.794 ± 0.014 (798) |
| A1 | 0.489 ± 0.016 (1024) | 0.536 ± 0.017 (1024) | 0.589 ± 0.016 (1024) | 0.634 ± 0.017 (971) | 0.710 ± 0.015 (896) | 0.768 ± 0.017 (790) |
| A2 | 0.490 ± 0.014 (1024) | 0.521 ± 0.016 (1024) | 0.592 ± 0.015 (1022) | 0.653 ± 0.015 (968) | 0.706 ± 0.016 (899) | 0.794 ± 0.014 (780) |
| B0 | 0.512 ± 0.017 (1024) | 0.521 ± 0.016 (1024) | 0.528 ± 0.015 (1024) | 0.641 ± 0.015 (971) | 0.688 ± 0.016 (900) | 0.811 ± 0.014 (792) |
| Biology prior | 0.505 ± 0.015 (1024) | 0.515 ± 0.016 (1024) | 0.544 ± 0.015 (1024) | 0.529 ± 0.016 (966) | 0.619 ± 0.017 (803) | 0.667 ± 0.019 (636) |
| Blank 0 | 0.513 ± 0.016 (1024) | 0.486 ± 0.015 (1024) | 0.529 ± 0.015 (1024) | 0.511 ± 0.017 (962) | 0.478 ± 0.018 (735) | 0.474 ± 0.026 (329) |
| Blank 1 | 0.492 ± 0.017 (1024) | 0.498 ± 0.017 (1024) | 0.507 ± 0.016 (1024) | 0.490 ± 0.016 (961) | 0.519 ± 0.018 (727) | 0.538 ± 0.025 (383) |
| Blank 2 | 0.495 ± 0.016 (1024) | 0.486 ± 0.016 (1024) | 0.511 ± 0.015 (1024) | 0.514 ± 0.017 (967) | 0.501 ± 0.019 (740) | 0.503 ± 0.028 (370) |
| Random walk | 0.481 ± 0.016 (1024) | 0.481 ± 0.017 (1024) | 0.498 ± 0.015 (1023) | 0.502 ± 0.015 (943) | 0.470 ± 0.018 (698) | 0.493 ± 0.030 (272) |
| Local rule-knowing greedy | 1.000 ± 0.000 (1024) | 1.000 ± 0.000 (1024) | 1.000 ± 0.000 (1022) | 1.000 ± 0.000 (1022) | 1.000 ± 0.000 (1019) | 1.000 ± 0.000 (1015) |

### Reversal

| Controller | Meal 1 | Meal 2 | Meal 3 | Meal 5 | Meal 10 | Meal 20 |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | 0.470 ± 0.016 (1024) | 0.540 ± 0.016 (1024) | 0.575 ± 0.015 (1023) | 0.634 ± 0.016 (967) | 0.705 ± 0.015 (882) | 0.529 ± 0.018 (722) |
| A1 | 0.489 ± 0.016 (1024) | 0.536 ± 0.017 (1024) | 0.587 ± 0.016 (1024) | 0.634 ± 0.017 (971) | 0.693 ± 0.015 (888) | 0.566 ± 0.018 (731) |
| A2 | 0.490 ± 0.014 (1024) | 0.521 ± 0.016 (1024) | 0.591 ± 0.015 (1022) | 0.652 ± 0.015 (965) | 0.684 ± 0.016 (891) | 0.543 ± 0.018 (726) |
| B0 | 0.512 ± 0.017 (1024) | 0.521 ± 0.016 (1024) | 0.528 ± 0.015 (1024) | 0.640 ± 0.015 (971) | 0.666 ± 0.016 (896) | 0.573 ± 0.018 (735) |
| Biology prior | 0.505 ± 0.015 (1024) | 0.515 ± 0.016 (1024) | 0.545 ± 0.015 (1024) | 0.526 ± 0.016 (967) | 0.601 ± 0.017 (799) | 0.485 ± 0.020 (582) |
| Blank 0 | 0.513 ± 0.016 (1024) | 0.486 ± 0.015 (1024) | 0.529 ± 0.015 (1024) | 0.515 ± 0.017 (963) | 0.476 ± 0.018 (731) | 0.486 ± 0.027 (313) |
| Blank 1 | 0.492 ± 0.017 (1024) | 0.498 ± 0.017 (1024) | 0.507 ± 0.016 (1024) | 0.493 ± 0.016 (960) | 0.521 ± 0.019 (731) | 0.488 ± 0.026 (383) |
| Blank 2 | 0.495 ± 0.016 (1024) | 0.486 ± 0.016 (1024) | 0.511 ± 0.015 (1024) | 0.514 ± 0.017 (967) | 0.476 ± 0.018 (739) | 0.467 ± 0.026 (347) |
| Random walk | 0.481 ± 0.016 (1024) | 0.481 ± 0.017 (1024) | 0.497 ± 0.015 (1023) | 0.506 ± 0.015 (943) | 0.458 ± 0.018 (720) | 0.479 ± 0.029 (284) |
| Local rule-knowing greedy | 1.000 ± 0.000 (1024) | 1.000 ± 0.000 (1024) | 1.000 ± 0.000 (1024) | 1.000 ± 0.000 (1023) | 1.000 ± 0.000 (1022) | 1.000 ± 0.000 (1020) |

### Matched first-to-twentieth-meal change

Both observations come from the same agents that reach meal 20. The numerator is the world-level sum of twentieth-meal food indicator minus first-meal food indicator; the denominator is the number of eligible agents. The result is reported in percentage points. This removes changes in the marginal cohort between those two meals, while retaining conditioning on meal-20 eligibility. In reversal worlds, a meal may occur before or after the switch, so this contrast alone is not a reversal-recovery measure.

| Condition | Controller | Change ± SE (percentage points) | Eligible agents | Contributing worlds |
| --- | --- | --- | --- | --- |
| stationary | A0 | 26.82 ± 2.42 | 798 | 254 |
| stationary | A1 | 21.14 ± 2.44 | 790 | 253 |
| stationary | A2 | 24.87 ± 2.24 | 780 | 256 |
| stationary | B0 | 23.99 ± 2.24 | 792 | 255 |
| stationary | Biology prior | 8.65 ± 2.52 | 636 | 248 |
| stationary | Blank 0 | -15.81 ± 3.99 | 329 | 203 |
| stationary | Blank 1 | -7.31 ± 3.69 | 383 | 210 |
| stationary | Blank 2 | -6.76 ± 4.07 | 370 | 213 |
| stationary | Random walk | -10.29 ± 4.26 | 272 | 181 |
| stationary | Local rule-knowing greedy | 0.00 ± 0.00 | 1015 | 255 |
| reversal | A0 | -1.80 ± 2.66 | 722 | 254 |
| reversal | A1 | 0.00 ± 2.74 | 731 | 253 |
| reversal | A2 | -0.83 ± 2.57 | 726 | 255 |
| reversal | B0 | -1.50 ± 2.63 | 735 | 253 |
| reversal | Biology prior | -10.48 ± 2.76 | 582 | 248 |
| reversal | Blank 0 | -12.46 ± 4.18 | 313 | 192 |
| reversal | Blank 1 | -10.70 ± 3.80 | 383 | 208 |
| reversal | Blank 2 | -14.70 ± 3.72 | 347 | 207 |
| reversal | Random walk | -13.03 ± 3.93 | 284 | 185 |
| reversal | Local rule-knowing greedy | 0.00 ± 0.00 | 1020 | 255 |

## Fifty-step accuracy for final survivors

Only agents alive after step 299 contribute to **every** bin in this section, including the early bins. Accuracy pools food/poison encounters among those agents; it is not the mean of each agent’s personal ratio. Cells show **accuracy ± SE (encounters)**. The same survivor cohort is used across bins within a condition, but stationary and reversal survivor cohorts can differ. These are explicitly survival-conditioned descriptive curves.

### Stationary

| Controller | Final survivors | [0,50) | [50,100) | [100,150) | [150,200) | [200,250) | [250,300) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 861 | 0.637 ± 0.006 (5275) | 0.741 ± 0.006 (5147) | 0.784 ± 0.006 (4889) | 0.820 ± 0.006 (4854) | 0.840 ± 0.006 (4754) | 0.856 ± 0.006 (4490) |
| A1 | 867 | 0.642 ± 0.006 (5328) | 0.733 ± 0.006 (5216) | 0.779 ± 0.006 (5041) | 0.805 ± 0.006 (4949) | 0.841 ± 0.005 (4662) | 0.846 ± 0.006 (4482) |
| A2 | 871 | 0.649 ± 0.006 (5256) | 0.739 ± 0.006 (5112) | 0.787 ± 0.005 (5017) | 0.818 ± 0.006 (4613) | 0.828 ± 0.006 (4680) | 0.846 ± 0.006 (4232) |
| B0 | 850 | 0.630 ± 0.006 (5210) | 0.727 ± 0.006 (5073) | 0.772 ± 0.007 (5068) | 0.808 ± 0.006 (4925) | 0.836 ± 0.006 (4816) | 0.846 ± 0.006 (4635) |
| Biology prior | 663 | 0.620 ± 0.007 (3635) | 0.647 ± 0.008 (3523) | 0.681 ± 0.008 (3512) | 0.680 ± 0.008 (3508) | 0.689 ± 0.009 (3220) | 0.710 ± 0.010 (3104) |
| Blank 0 | 388 | 0.641 ± 0.010 (1749) | 0.628 ± 0.012 (1498) | 0.607 ± 0.011 (1420) | 0.605 ± 0.013 (1331) | 0.571 ± 0.012 (1277) | 0.554 ± 0.013 (1158) |
| Blank 1 | 401 | 0.641 ± 0.010 (1846) | 0.624 ± 0.010 (1661) | 0.627 ± 0.011 (1577) | 0.625 ± 0.011 (1597) | 0.592 ± 0.011 (1472) | 0.591 ± 0.013 (1412) |
| Blank 2 | 400 | 0.620 ± 0.009 (1905) | 0.636 ± 0.011 (1534) | 0.630 ± 0.011 (1537) | 0.607 ± 0.013 (1461) | 0.565 ± 0.012 (1414) | 0.548 ± 0.013 (1323) |
| Random walk | 378 | 0.638 ± 0.010 (1722) | 0.623 ± 0.011 (1386) | 0.617 ± 0.014 (1276) | 0.601 ± 0.013 (1231) | 0.550 ± 0.014 (1139) | 0.562 ± 0.014 (1068) |
| Local rule-knowing greedy | 1024 | 1.000 ± 0.000 (16221) | 1.000 ± 0.000 (14788) | 1.000 ± 0.000 (14139) | 1.000 ± 0.000 (13701) | 1.000 ± 0.000 (12970) | 1.000 ± 0.000 (12434) |

### Reversal

| Controller | Final survivors | [0,50) | [50,100) | [100,150) | [150,200) | [200,250) | [250,300) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 387 | 0.646 ± 0.009 (2417) | 0.754 ± 0.009 (2243) | 0.801 ± 0.009 (2107) | 0.374 ± 0.011 (1693) | 0.492 ± 0.012 (1552) | 0.549 ± 0.014 (1482) |
| A1 | 432 | 0.648 ± 0.008 (2690) | 0.741 ± 0.008 (2518) | 0.797 ± 0.008 (2415) | 0.406 ± 0.010 (2055) | 0.498 ± 0.012 (1794) | 0.562 ± 0.012 (1643) |
| A2 | 431 | 0.653 ± 0.008 (2501) | 0.741 ± 0.008 (2400) | 0.810 ± 0.008 (2406) | 0.392 ± 0.012 (2059) | 0.512 ± 0.011 (1708) | 0.548 ± 0.012 (1619) |
| B0 | 403 | 0.640 ± 0.008 (2402) | 0.740 ± 0.008 (2351) | 0.790 ± 0.008 (2293) | 0.409 ± 0.011 (2027) | 0.504 ± 0.012 (1838) | 0.562 ± 0.014 (1575) |
| Biology prior | 399 | 0.627 ± 0.009 (2111) | 0.671 ± 0.009 (2030) | 0.695 ± 0.010 (2064) | 0.502 ± 0.010 (1978) | 0.514 ± 0.011 (1821) | 0.556 ± 0.013 (1695) |
| Blank 0 | 391 | 0.632 ± 0.010 (1768) | 0.628 ± 0.011 (1563) | 0.588 ± 0.012 (1430) | 0.577 ± 0.013 (1303) | 0.577 ± 0.014 (1265) | 0.557 ± 0.014 (1180) |
| Blank 1 | 389 | 0.626 ± 0.010 (1860) | 0.635 ± 0.011 (1596) | 0.642 ± 0.011 (1508) | 0.568 ± 0.013 (1499) | 0.559 ± 0.011 (1552) | 0.537 ± 0.013 (1528) |
| Blank 2 | 376 | 0.621 ± 0.010 (1782) | 0.622 ± 0.012 (1467) | 0.633 ± 0.012 (1443) | 0.596 ± 0.013 (1367) | 0.569 ± 0.013 (1316) | 0.535 ± 0.015 (1317) |
| Random walk | 407 | 0.623 ± 0.009 (1815) | 0.621 ± 0.012 (1489) | 0.625 ± 0.014 (1351) | 0.584 ± 0.012 (1299) | 0.590 ± 0.013 (1250) | 0.533 ± 0.014 (1148) |
| Local rule-knowing greedy | 1024 | 1.000 ± 0.000 (16221) | 1.000 ± 0.000 (14788) | 1.000 ± 0.000 (14139) | 1.000 ± 0.000 (9122) | 1.000 ± 0.000 (8184) | 1.000 ± 0.000 (8227) |

## Reversal: all-agent post-switch accuracy

These rows include all encounters by agents alive at the time of eating, including agents that later die. No future-survival filter is applied. The food rule changes at step 150. The preceding [100,150) bin supplies context; each cell is **accuracy ± SE (encounters)**. The population contributing encounters changes with death and behavior, so bin differences must not be interpreted as paired within-agent recovery without further accounting.

| Controller | Before: [100,150) | After: [150,200) | After: [200,250) | After: [250,300) |
| --- | --- | --- | --- | --- |
| A0 | 0.773 ± 0.006 (5019) | 0.256 ± 0.009 (4146) | 0.355 ± 0.010 (2924) | 0.462 ± 0.014 (1854) |
| A1 | 0.767 ± 0.006 (5158) | 0.280 ± 0.009 (4358) | 0.377 ± 0.011 (2951) | 0.493 ± 0.012 (1963) |
| A2 | 0.777 ± 0.006 (5120) | 0.267 ± 0.009 (4348) | 0.381 ± 0.010 (2881) | 0.481 ± 0.013 (1906) |
| B0 | 0.759 ± 0.007 (5214) | 0.281 ± 0.009 (4507) | 0.394 ± 0.011 (2973) | 0.485 ± 0.014 (1933) |
| Biology prior | 0.630 ± 0.008 (4122) | 0.393 ± 0.009 (3525) | 0.426 ± 0.010 (2652) | 0.504 ± 0.013 (1932) |
| Blank 0 | 0.492 ± 0.010 (2693) | 0.479 ± 0.011 (2046) | 0.501 ± 0.012 (1716) | 0.511 ± 0.014 (1320) |
| Blank 1 | 0.522 ± 0.010 (2766) | 0.477 ± 0.011 (2389) | 0.497 ± 0.011 (2022) | 0.499 ± 0.013 (1679) |
| Blank 2 | 0.531 ± 0.009 (2826) | 0.480 ± 0.011 (2245) | 0.485 ± 0.012 (1850) | 0.488 ± 0.014 (1507) |
| Random walk | 0.506 ± 0.010 (2443) | 0.499 ± 0.011 (1983) | 0.520 ± 0.012 (1624) | 0.496 ± 0.013 (1267) |
| Local rule-knowing greedy | 1.000 ± 0.000 (14139) | 1.000 ± 0.000 (9122) | 1.000 ± 0.000 (8184) | 1.000 ± 0.000 (8227) |

## Provenance and interpretation limits

- All eight panels used committed source `c224f91e20d9ea12a09613d70320cfe17abea8cb` with clean Git provenance.
- Initial world keys, development keys and rule assignments match across all controllers and conditions. Actions can change subsequent food locations; “same worlds” does not mean forcing identical trajectories after different actions.
- Input and final genome files are byte-identical. No training took place. All panels use the same default configuration, 256 worlds and reversal boundary 150.
- EVAL v1 remains unchanged. Differences from historical point estimates include fresh evaluation worlds; this report does not reinterpret them as new evolution gains.
- A0, A1 and A2 provide three independently evolved genomes. B0 alone cannot support a three-seed claim about biology-start evolution. The deterministic biology reference and repeated common policies are not additional independent evolution seeds.
- Exact values, all food/poison counts, contributing-agent/world counts and complete all-agent bins are in the summaries below. Binary event arrays and world/development keys are retained in each `events.npz` for independent recalculation.

## Source artifacts

| Genome | Summary | Raw evidence | Evaluation seconds |
| --- | --- | --- | --- |
| A0 | [20261119-030209_E112-v3-A0](../runs/20261119-030209_E112-v3-A0/summary.json) | [Raw events](../runs/20261119-030209_E112-v3-A0/events.npz) | 50.390 |
| A1 | [20261119-030336_E112-v3-A1](../runs/20261119-030336_E112-v3-A1/summary.json) | [Raw events](../runs/20261119-030336_E112-v3-A1/events.npz) | 37.671 |
| A2 | [20261119-030424_E112-v3-A2](../runs/20261119-030424_E112-v3-A2/summary.json) | [Raw events](../runs/20261119-030424_E112-v3-A2/events.npz) | 24.906 |
| B0 | [20261119-030456_E112-v3-B0](../runs/20261119-030456_E112-v3-B0/summary.json) | [Raw events](../runs/20261119-030456_E112-v3-B0/events.npz) | 21.969 |
| Biology prior | [20261119-030527_E112-v3-biology](../runs/20261119-030527_E112-v3-biology/summary.json) | [Raw events](../runs/20261119-030527_E112-v3-biology/events.npz) | 30.359 |
| Blank 0 | [20261119-030608_E112-v3-blank0](../runs/20261119-030608_E112-v3-blank0/summary.json) | [Raw events](../runs/20261119-030608_E112-v3-blank0/events.npz) | 35.469 |
| Blank 1 | [20261119-030654_E112-v3-blank1](../runs/20261119-030654_E112-v3-blank1/summary.json) | [Raw events](../runs/20261119-030654_E112-v3-blank1/events.npz) | 31.297 |
| Blank 2 | [20261119-030736_E112-v3-blank2](../runs/20261119-030736_E112-v3-blank2/summary.json) | [Raw events](../runs/20261119-030736_E112-v3-blank2/events.npz) | 32.938 |

