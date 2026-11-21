# E116: smaller evolutionary steps during continuation

**Artifact validity: PASS. Registered improvement gate: FAIL.** All six runs reached generation 20. All 14 controllers and 56 conditions completed; the independent audit reconstructed 70 raw archives and checked nine copied genomes. The gate passed **3/6 return-improvement checks and 3/3 survival safeguards**.

Candidate individual minimum returns are **8.672 ± 0.304, 7.893 ± 0.306, 8.519 ± 0.304** for seeds 0, 1 and 2, respectively. Values are means ± one world-family standard error. Unrounded paired differences determine the decision.

## Fixed comparison and decision

The only scientific difference between matched arms is the ES learning rate: 0.1 for control and 0.02 for candidate. Both optimize each starting individual's lower return across the two stationary rules, then average agents and worlds. Both begin from the identical original A genome, with zero Adam moments and matched training seeds 2000, 2001 and 2002. Population 64, four base-world pairs per generation, mutation sigma 0.1, L2 0.005, original architecture, both lifetime updates and 20 generations remain fixed. Structural growth and new environmental features are off. E115 descendants are not controls.

EVAL v6 uses seed 2026112001 and 256 fresh base-world families, four starting agents per family and 300 steps per life. Each controller is replayed under both stationary rules and both initial rules with a flip before eating at step 150. Replay starts share grids, positions, identities and random/development streams; learned weights and activity reset at birth. All starting agents, including deaths and zero-encounter lives, remain in return and survival denominators.

PASS requires valid artifacts and, in every seed, candidate individual minimum return above both its matched control and unchanged start by strictly more than two paired-world SE. Candidate-minus-start joint survival must be at least minus two paired-world SE. Equality fails a return check and passes the survival boundary. Beating a degraded control, preserving an ancestor without improving it, or a better reversal score cannot substitute. This is an operational research gate, not a multiplicity-adjusted significance test or proof of equivalent survival.

| Seed | Gate component | Difference ± paired SE | Required mean difference | Decision |
| --- | --- | --- | --- | --- |
| 0 | Minimum return: candidate − control | 1.304 ± 0.419 | > 0.838729 | PASS |
| 0 | Minimum return: candidate − start | -0.033 ± 0.407 | > 0.813380 | FAIL |
| 0 | Joint survival: candidate − start | 0.008 ± 0.018 | >= -0.036598 | PASS |
| 1 | Minimum return: candidate − control | 0.879 ± 0.407 | > 0.814143 | PASS |
| 1 | Minimum return: candidate − start | -0.535 ± 0.411 | > 0.822425 | FAIL |
| 1 | Joint survival: candidate − start | -0.021 ± 0.021 | >= -0.041426 | PASS |
| 2 | Minimum return: candidate − control | 1.154 ± 0.413 | > 0.825046 | PASS |
| 2 | Minimum return: candidate − start | -0.299 ± 0.409 | > 0.818067 | FAIL |
| 2 | Joint survival: candidate − start | -0.035 ± 0.020 | >= -0.040341 | PASS |

## All fourteen controllers

Every ± value is one SE over 256 independent world families. Rules and agents are averaged within a family before estimating uncertainty. These error bars describe fixed genomes across worlds, not variability across evolution seeds. Survival is a fraction.

| Controller | Stationary fitness ± SE | Individual minimum return ± SE | Joint stationary survival ± SE | Post-flip return ± SE | Reversal final survival ± SE |
| --- | --- | --- | --- | --- | --- |
| Start 0 | 15.327 ± 0.259 | 8.705 ± 0.301 | 0.718 ± 0.014 | -2.709 ± 0.087 | 0.414 ± 0.011 |
| Control 0 | 14.072 ± 0.263 | 7.368 ± 0.304 | 0.658 ± 0.014 | -2.505 ± 0.089 | 0.406 ± 0.012 |
| Candidate 0 | 15.240 ± 0.268 | 8.672 ± 0.304 | 0.726 ± 0.014 | -2.539 ± 0.087 | 0.425 ± 0.010 |
| Start 1 | 15.440 ± 0.287 | 8.428 ± 0.340 | 0.690 ± 0.016 | -2.581 ± 0.091 | 0.419 ± 0.012 |
| Control 1 | 13.342 ± 0.242 | 7.014 ± 0.277 | 0.674 ± 0.014 | -2.539 ± 0.073 | 0.441 ± 0.010 |
| Candidate 1 | 14.799 ± 0.261 | 7.893 ± 0.306 | 0.669 ± 0.014 | -2.537 ± 0.085 | 0.400 ± 0.011 |
| Start 2 | 15.310 ± 0.243 | 8.817 ± 0.278 | 0.737 ± 0.014 | -2.745 ± 0.088 | 0.442 ± 0.012 |
| Control 2 | 13.481 ± 0.243 | 7.364 ± 0.288 | 0.699 ± 0.016 | -2.297 ± 0.092 | 0.473 ± 0.011 |
| Candidate 2 | 15.330 ± 0.264 | 8.519 ± 0.304 | 0.702 ± 0.014 | -2.573 ± 0.096 | 0.432 ± 0.012 |
| Fixed A | 38.784 ± 0.676 | -3.857 ± 0.022 | 0.010 ± 0.003 | -3.732 ± 0.045 | 0.044 ± 0.006 |
| Fixed B | 39.421 ± 0.748 | -3.889 ± 0.016 | 0.008 ± 0.003 | -3.717 ± 0.048 | 0.045 ± 0.007 |
| A/B mixture | 35.155 ± 1.224 | -3.981 ± 0.005 | 0.000 ± 0.000 | -2.532 ± 0.075 | 0.194 ± 0.010 |
| Random walk | 0.008 ± 0.057 | -2.848 ± 0.030 | 0.021 ± 0.004 | -0.005 ± 0.045 | 0.382 ± 0.008 |
| Rule-knowing greedy | 82.078 ± 0.995 | 67.938 ± 1.155 | 1.000 ± 0.000 | 24.962 ± 0.456 | 1.000 ± 0.000 |

Individual minimum return takes the lower stationary return for the same individual before averaging. Joint survival requires that individual to survive both stationary replays. Post-flip return covers steps [150,300) across both reversed lives. Zero post-flip return may reflect death before the flip. The fixed-time reversal diagnostic does not pass the later randomized-reversal rung.

## Every condition

Each row includes 1,024 starting agents. Rule 0 initially makes A food; rule 1 makes B food. Greedy-normalized fitness is 100 × mean controller fitness / mean greedy fitness in the same condition, with paired-world delta-method SE in percentage points (pp), including controller/greedy covariance. It is a ratio of means; negative values remain negative and percentages are not clipped. The rule-knowing greedy policy is a reference, not a proven upper bound, and is 100 ± 0 against itself. Food, poison, encounter and death counts are observations, not independent sample sizes.

| Controller | Condition | Fitness ± SE | Fitness (% of greedy) ± SE (pp) | Alive before 150 ± SE | Final survival ± SE | Food | Poison | Encounters | Deaths / 1,024 | No encounters / 1,024 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Start 0 | stationary_r0 | 15.478 ± 0.399 | 19.008 ± 0.562 | 0.893 ± 0.010 | 0.859 ± 0.012 | 23373 | 7524 | 30897 | 144 | 0 |
| Start 0 | stationary_r1 | 15.176 ± 0.400 | 18.344 ± 0.603 | 0.880 ± 0.010 | 0.844 ± 0.012 | 22829 | 7289 | 30118 | 160 | 0 |
| Start 0 | reversal_r0 | 3.593 ± 0.180 | 5.221 ± 0.268 | 0.893 ± 0.010 | 0.403 ± 0.016 | 14919 | 11240 | 26159 | 611 | 0 |
| Start 0 | reversal_r1 | 3.644 ± 0.187 | 5.290 ± 0.279 | 0.880 ± 0.010 | 0.425 ± 0.016 | 14744 | 11013 | 25757 | 589 | 0 |
| Control 0 | stationary_r0 | 13.927 ± 0.371 | 17.104 ± 0.539 | 0.837 ± 0.011 | 0.801 ± 0.012 | 22029 | 7768 | 29797 | 204 | 0 |
| Control 0 | stationary_r1 | 14.217 ± 0.376 | 17.184 ± 0.539 | 0.854 ± 0.011 | 0.824 ± 0.012 | 22539 | 7981 | 30520 | 180 | 0 |
| Control 0 | reversal_r0 | 2.981 ± 0.183 | 4.333 ± 0.269 | 0.837 ± 0.011 | 0.404 ± 0.015 | 14101 | 11048 | 25149 | 610 | 0 |
| Control 0 | reversal_r1 | 3.047 ± 0.175 | 4.424 ± 0.254 | 0.854 ± 0.011 | 0.408 ± 0.016 | 14408 | 11288 | 25696 | 606 | 0 |
| Candidate 0 | stationary_r0 | 14.988 ± 0.385 | 18.407 ± 0.576 | 0.880 ± 0.010 | 0.854 ± 0.012 | 22620 | 7272 | 29892 | 149 | 0 |
| Candidate 0 | stationary_r1 | 15.492 ± 0.382 | 18.726 ± 0.563 | 0.882 ± 0.010 | 0.860 ± 0.011 | 23084 | 7220 | 30304 | 143 | 0 |
| Candidate 0 | reversal_r0 | 3.681 ± 0.189 | 5.349 ± 0.288 | 0.880 ± 0.010 | 0.440 ± 0.015 | 14629 | 10860 | 25489 | 573 | 0 |
| Candidate 0 | reversal_r1 | 3.533 ± 0.177 | 5.130 ± 0.263 | 0.882 ± 0.010 | 0.409 ± 0.016 | 14597 | 10979 | 25576 | 605 | 0 |
| Start 1 | stationary_r0 | 15.318 ± 0.430 | 18.813 ± 0.609 | 0.857 ± 0.011 | 0.829 ± 0.013 | 23527 | 7841 | 31368 | 175 | 0 |
| Start 1 | stationary_r1 | 15.562 ± 0.396 | 18.811 ± 0.587 | 0.866 ± 0.011 | 0.843 ± 0.012 | 23699 | 7763 | 31462 | 161 | 0 |
| Start 1 | reversal_r0 | 3.949 ± 0.211 | 5.739 ± 0.316 | 0.857 ± 0.011 | 0.437 ± 0.015 | 15349 | 11305 | 26654 | 577 | 0 |
| Start 1 | reversal_r1 | 3.544 ± 0.187 | 5.145 ± 0.277 | 0.866 ± 0.011 | 0.401 ± 0.017 | 14921 | 11292 | 26213 | 613 | 0 |
| Control 1 | stationary_r0 | 13.584 ± 0.351 | 16.683 ± 0.515 | 0.860 ± 0.011 | 0.836 ± 0.011 | 21820 | 7910 | 29730 | 168 | 0 |
| Control 1 | stationary_r1 | 13.101 ± 0.348 | 15.835 ± 0.525 | 0.853 ± 0.011 | 0.812 ± 0.012 | 21329 | 7914 | 29243 | 192 | 0 |
| Control 1 | reversal_r0 | 3.035 ± 0.174 | 4.411 ± 0.258 | 0.860 ± 0.011 | 0.442 ± 0.015 | 14247 | 11139 | 25386 | 571 | 0 |
| Control 1 | reversal_r1 | 2.856 ± 0.159 | 4.147 ± 0.234 | 0.853 ± 0.011 | 0.440 ± 0.016 | 14126 | 11201 | 25327 | 573 | 0 |
| Candidate 1 | stationary_r0 | 15.094 ± 0.370 | 18.537 ± 0.576 | 0.853 ± 0.011 | 0.822 ± 0.011 | 23292 | 7836 | 31128 | 182 | 0 |
| Candidate 1 | stationary_r1 | 14.505 ± 0.383 | 17.533 ± 0.562 | 0.855 ± 0.010 | 0.824 ± 0.012 | 22608 | 7755 | 30363 | 180 | 0 |
| Candidate 1 | reversal_r0 | 3.702 ± 0.184 | 5.380 ± 0.274 | 0.853 ± 0.011 | 0.409 ± 0.015 | 15157 | 11366 | 26523 | 605 | 0 |
| Candidate 1 | reversal_r1 | 3.104 ± 0.187 | 4.507 ± 0.274 | 0.855 ± 0.010 | 0.391 ± 0.016 | 14461 | 11282 | 25743 | 624 | 0 |
| Start 2 | stationary_r0 | 14.896 ± 0.355 | 18.293 ± 0.516 | 0.878 ± 0.010 | 0.850 ± 0.012 | 22686 | 7433 | 30119 | 154 | 0 |
| Start 2 | stationary_r1 | 15.725 ± 0.360 | 19.007 ± 0.548 | 0.909 ± 0.009 | 0.878 ± 0.010 | 23497 | 7395 | 30892 | 125 | 0 |
| Start 2 | reversal_r0 | 3.527 ± 0.172 | 5.126 ± 0.251 | 0.878 ± 0.010 | 0.448 ± 0.015 | 14669 | 11057 | 25726 | 565 | 0 |
| Start 2 | reversal_r1 | 3.714 ± 0.170 | 5.392 ± 0.250 | 0.909 ± 0.009 | 0.436 ± 0.017 | 14921 | 11118 | 26039 | 578 | 0 |
| Control 2 | stationary_r0 | 13.901 ± 0.351 | 17.072 ± 0.514 | 0.878 ± 0.011 | 0.846 ± 0.012 | 22309 | 8074 | 30383 | 158 | 0 |
| Control 2 | stationary_r1 | 13.062 ± 0.345 | 15.788 ± 0.488 | 0.867 ± 0.012 | 0.830 ± 0.013 | 21320 | 7945 | 29265 | 174 | 0 |
| Control 2 | reversal_r0 | 3.646 ± 0.190 | 5.299 ± 0.279 | 0.878 ± 0.011 | 0.473 ± 0.017 | 14941 | 11207 | 26148 | 540 | 0 |
| Control 2 | reversal_r1 | 3.488 ± 0.187 | 5.064 ± 0.270 | 0.867 ± 0.012 | 0.473 ± 0.016 | 14572 | 11000 | 25572 | 540 | 0 |
| Candidate 2 | stationary_r0 | 15.002 ± 0.390 | 18.424 ± 0.572 | 0.865 ± 0.010 | 0.828 ± 0.012 | 22824 | 7462 | 30286 | 176 | 0 |
| Candidate 2 | stationary_r1 | 15.657 ± 0.357 | 18.926 ± 0.533 | 0.884 ± 0.010 | 0.857 ± 0.011 | 23485 | 7452 | 30937 | 146 | 0 |
| Candidate 2 | reversal_r0 | 3.591 ± 0.193 | 5.218 ± 0.285 | 0.865 ± 0.010 | 0.411 ± 0.015 | 14788 | 11111 | 25899 | 603 | 0 |
| Candidate 2 | reversal_r1 | 3.759 ± 0.187 | 5.457 ± 0.272 | 0.884 ± 0.010 | 0.452 ± 0.017 | 15142 | 11293 | 26435 | 561 | 0 |
| Fixed A | stationary_r0 | 81.426 ± 1.353 | 100.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 83380 | 0 | 83380 | 0 | 0 |
| Fixed A | stationary_r1 | -3.857 ± 0.022 | -4.663 ± 0.090 | 0.044 ± 0.007 | 0.010 ± 0.003 | 0 | 3950 | 3950 | 1014 | 10 |
| Fixed A | reversal_r0 | 35.942 ± 0.628 | 52.232 ± 0.798 | 1.000 ± 0.000 | 0.078 ± 0.013 | 44449 | 7644 | 52093 | 944 | 0 |
| Fixed A | reversal_r1 | -3.857 ± 0.022 | -5.600 ± 0.065 | 0.044 ± 0.007 | 0.010 ± 0.003 | 0 | 3950 | 3950 | 1014 | 10 |
| Fixed B | stationary_r0 | -3.889 ± 0.016 | -4.776 ± 0.081 | 0.037 ± 0.006 | 0.008 ± 0.003 | 0 | 3982 | 3982 | 1016 | 8 |
| Fixed B | stationary_r1 | 82.730 ± 1.498 | 100.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 84716 | 0 | 84716 | 0 | 0 |
| Fixed B | reversal_r0 | -3.889 ± 0.016 | -5.651 ± 0.057 | 0.037 ± 0.006 | 0.008 ± 0.003 | 0 | 3982 | 3982 | 1016 | 8 |
| Fixed B | reversal_r1 | 36.926 ± 0.664 | 53.610 ± 0.822 | 1.000 ± 0.000 | 0.082 ± 0.013 | 45424 | 7612 | 53036 | 940 | 0 |
| A/B mixture | stationary_r0 | 46.109 ± 2.304 | 56.627 ± 3.006 | 0.501 ± 0.001 | 0.500 ± 0.000 | 49257 | 2041 | 51298 | 512 | 1 |
| A/B mixture | stationary_r1 | 24.200 ± 0.850 | 29.252 ± 1.141 | 0.503 ± 0.002 | 0.500 ± 0.000 | 26817 | 2036 | 28853 | 512 | 1 |
| A/B mixture | reversal_r0 | 16.041 ± 0.722 | 23.311 ± 1.064 | 0.501 ± 0.001 | 0.173 ± 0.013 | 21230 | 4804 | 26034 | 847 | 1 |
| A/B mixture | reversal_r1 | 12.314 ± 0.356 | 17.879 ± 0.513 | 0.503 ± 0.002 | 0.215 ± 0.014 | 17089 | 4479 | 21568 | 804 | 1 |
| Random walk | stationary_r0 | 0.186 ± 0.129 | 0.228 ± 0.158 | 0.686 ± 0.015 | 0.395 ± 0.015 | 7764 | 7574 | 15338 | 620 | 0 |
| Random walk | stationary_r1 | -0.169 ± 0.124 | -0.204 ± 0.149 | 0.654 ± 0.014 | 0.362 ± 0.016 | 7271 | 7444 | 14715 | 653 | 0 |
| Random walk | reversal_r0 | 0.090 ± 0.119 | 0.131 ± 0.173 | 0.686 ± 0.015 | 0.396 ± 0.015 | 7662 | 7570 | 15232 | 618 | 0 |
| Random walk | reversal_r1 | -0.156 ± 0.116 | -0.227 ± 0.168 | 0.654 ± 0.014 | 0.368 ± 0.015 | 7316 | 7476 | 14792 | 647 | 0 |
| Rule-knowing greedy | stationary_r0 | 81.426 ± 1.353 | 100.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 83380 | 0 | 83380 | 0 | 0 |
| Rule-knowing greedy | stationary_r1 | 82.730 ± 1.498 | 100.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 84716 | 0 | 84716 | 0 | 0 |
| Rule-knowing greedy | reversal_r0 | 68.813 ± 0.656 | 100.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 70465 | 0 | 70465 | 0 | 0 |
| Rule-knowing greedy | reversal_r1 | 68.878 ± 0.685 | 100.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 70531 | 0 | 70531 | 0 | 0 |

## All 30 registered paired contrasts

Each difference is computed within a matched world family before its mean and sample-SD/√256 SE. All five metrics are reported for candidate minus control and candidate minus start in each seed. No seeds are pooled, and these correlated descriptive contrasts do not create extra independent replications.

| Seed | Comparison | Metric | Difference ± paired SE |
| --- | --- | --- | --- |
| 0 | candidate-minus-control | stationary_fitness | 1.168 ± 0.361 |
| 0 | candidate-minus-control | individual_min_return | 1.304 ± 0.419 |
| 0 | candidate-minus-control | joint_survival | 0.067 ± 0.019 |
| 0 | candidate-minus-control | post_flip_return | -0.034 ± 0.125 |
| 0 | candidate-minus-control | reversal_final_survival | 0.019 ± 0.016 |
| 0 | candidate-minus-start | stationary_fitness | -0.086 ± 0.373 |
| 0 | candidate-minus-start | individual_min_return | -0.033 ± 0.407 |
| 0 | candidate-minus-start | joint_survival | 0.008 ± 0.018 |
| 0 | candidate-minus-start | post_flip_return | 0.171 ± 0.122 |
| 0 | candidate-minus-start | reversal_final_survival | 0.011 ± 0.015 |
| 1 | candidate-minus-control | stationary_fitness | 1.457 ± 0.347 |
| 1 | candidate-minus-control | individual_min_return | 0.879 ± 0.407 |
| 1 | candidate-minus-control | joint_survival | -0.005 ± 0.020 |
| 1 | candidate-minus-control | post_flip_return | 0.001 ± 0.111 |
| 1 | candidate-minus-control | reversal_final_survival | -0.042 ± 0.015 |
| 1 | candidate-minus-start | stationary_fitness | -0.641 ± 0.350 |
| 1 | candidate-minus-start | individual_min_return | -0.535 ± 0.411 |
| 1 | candidate-minus-start | joint_survival | -0.021 ± 0.021 |
| 1 | candidate-minus-start | post_flip_return | 0.044 ± 0.117 |
| 1 | candidate-minus-start | reversal_final_survival | -0.019 ± 0.015 |
| 2 | candidate-minus-control | stationary_fitness | 1.848 ± 0.352 |
| 2 | candidate-minus-control | individual_min_return | 1.154 ± 0.413 |
| 2 | candidate-minus-control | joint_survival | 0.003 ± 0.021 |
| 2 | candidate-minus-control | post_flip_return | -0.276 ± 0.132 |
| 2 | candidate-minus-control | reversal_final_survival | -0.041 ± 0.016 |
| 2 | candidate-minus-start | stationary_fitness | 0.020 ± 0.350 |
| 2 | candidate-minus-start | individual_min_return | -0.299 ± 0.409 |
| 2 | candidate-minus-start | joint_survival | -0.035 ± 0.020 |
| 2 | candidate-minus-start | post_flip_return | 0.171 ± 0.128 |
| 2 | candidate-minus-start | reversal_final_survival | -0.010 ± 0.016 |

## Evidence and limits

The [complete run summary](../runs/20261121-072924_E116-v6/summary.json), [configuration and input manifest](../runs/20261121-072924_E116-v6/config.json) and [independent audit](diagnostics/E116-independent-audit.json) retain raw-evidence links, all meal-number and 50-step bin measurements, survivor-only bins, individual return distributions, initial-state pairing, all within-world differences and source fingerprints. The [registration and approved proposal](../experiments/E116_PROPOSAL.md) fix the comparison before evaluation.

Recorded evaluation time: **409.734 seconds**. This is the evaluator's recorded elapsed time, not the study's total supervised compute charge; guard and budget accounting belong in the lab notebook.

The independent check reconstructs saved results; it does not replay optimization or regenerate random streams. Initial zero Adam moments are recorded metadata, with complete finite final states and histories checked independently. Matching initial and final RNG positions and fixed exposure arguments support the prescribed pairing, but do not reconstruct every intermediate random draw. These measurements do not identify the cause of every E115 loss, establish causal within-life learning or a plasticity advantage.

## Five-line report

1. We tested whether taking smaller evolutionary steps for 20 generations improves an existing learner on its worse food rule.
2. Smaller steps beat larger ones by 1.30 ± 0.42, 0.88 ± 0.41 and 1.15 ± 0.41 points across three lineages, with one paired-world standard error.
3. Against the original brains the differences were −0.03 ± 0.41, −0.54 ± 0.41 and −0.30 ± 0.41. The required net improvement failed in all three, while all three survival safeguards passed.
4. The smaller update size helped within this particular training budget. The comparison leaves the role of lifetime learning and performance in other environments open.
5. Retain the original brains. The next proposed test asks whether newborns benefit from prior creatures' experience of cold and shelter on fresh worlds.
