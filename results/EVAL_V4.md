# E114: paired individual outcomes across opposite food rules

**Measurement validity: PASS.** All 11 controllers and 44 conditions completed; 55 raw archives and six copied genomes passed the independent audit. The paired measurements distinguish robust individual outcomes from a high population score, but do not by themselves prove learning. E113's failed continuation gate remains unchanged.

The three A genomes have mean individual minimum stationary returns of **8.237 ± 0.302, 8.453 ± 0.310 and 8.396 ± 0.297**, while the direct controllers are near −4. A's joint survival across opposite stationary rules is **0.710 ± 0.014, 0.705 ± 0.014 and 0.732 ± 0.014**. Its post-flip returns remain negative in all three seeds: **−2.812 ± 0.083, −2.567 ± 0.089 and −2.597 ± 0.095**. Robust outcomes across fresh stationary lives do not imply reliable reversal recovery.

## What was measured

EVAL v4 evaluates six saved controllers—A0/A1/A2 and direct0/direct1/direct2—and five reference policies. It uses 256 fresh base worlds, seed 2026111902, with four agents per world. For each controller, each base world is replayed under initial rule 0 and rule 1, both stationary and reversed before eating at step 150.

Within each base world, these four lives begin with the same grid, positions, agent identities, developmental keys and random streams. They are **fresh matched lives**: neural activity, lifetime weight changes and experiences from one replay are not inherited by the next. Later trajectories can diverge because actions and outcomes differ. Across saved controllers, initial world conditions and streams match; the genome and resulting controller differ.

No controller is evolved, selected or fitted during this diagnostic. The A genomes retain their existing lifetime plasticity; direct weights remain fixed. Structural growth is off. The five references are a fixed A-target specialist, fixed B-target specialist, two A plus two B specialists with fixed agent assignments, uniform random walk and the existing local greedy policy with privileged knowledge of the current food rule.

## Five paired metrics

Each reported ± value is one standard error over **256 base worlds**. Within a base world, average the relevant individual outcomes and paired lives before calculating that error. The agents and four replay conditions are dependent; they are not additional independent samples. These errors concern worlds for fixed saved controllers, not variability across evolution seeds.

| Metric | Calculation within each base world | Interpretation limit |
| --- | --- | --- |
| Stationary fitness | Average food-minus-poison return over both stationary rules and all four starting agents | Population averages can hide opposite specialists and early deaths. |
| Individual minimum return | For each agent, take its lower return across the two stationary rules, then average the four minima | Measures paired robustness of return, not learning itself or dependence on experience. |
| Joint stationary survival | Fraction of starting agents alive at the end of both stationary replays | Survival in matched lives does not establish a learning mechanism. |
| Post-flip return | Average food-minus-poison during steps [150,300) across both reversed lives and all starting agents | Zero may mean death before the flip; interpret with death and encounter counts. |
| Reversal final survival | Average endpoint survival across both reversed lives and all starting agents | Counts survival, not the route by which it was achieved. |

| Controller | Stationary fitness ± SE | Individual minimum return ± SE | Joint stationary survival ± SE | Post-flip return ± SE | Reversal final survival ± SE |
| --- | --- | --- | --- | --- | --- |
| A0 | 15.125 ± 0.254 | 8.237 ± 0.302 | 0.710 ± 0.014 | -2.812 ± 0.083 | 0.403 ± 0.011 |
| A1 | 14.990 ± 0.266 | 8.453 ± 0.310 | 0.705 ± 0.014 | -2.567 ± 0.089 | 0.428 ± 0.012 |
| A2 | 14.972 ± 0.261 | 8.396 ± 0.297 | 0.732 ± 0.014 | -2.597 ± 0.095 | 0.433 ± 0.011 |
| Direct 0 | 55.945 ± 0.721 | -3.992 ± 0.003 | 0.000 ± 0.000 | -3.974 ± 0.004 | 0.000 ± 0.000 |
| Direct 1 | 66.970 ± 0.707 | -3.978 ± 0.008 | 0.001 ± 0.001 | -3.941 ± 0.014 | 0.002 ± 0.001 |
| Direct 2 | 69.737 ± 0.769 | -3.962 ± 0.013 | 0.001 ± 0.001 | -3.944 ± 0.019 | 0.002 ± 0.001 |
| Fixed A | 39.080 ± 0.720 | -3.863 ± 0.019 | 0.009 ± 0.003 | -3.767 ± 0.043 | 0.036 ± 0.006 |
| Fixed B | 39.460 ± 0.734 | -3.857 ± 0.022 | 0.015 ± 0.004 | -3.732 ± 0.044 | 0.047 ± 0.007 |
| A/B mixture | 33.486 ± 1.236 | -3.970 ± 0.008 | 0.000 ± 0.000 | -2.358 ± 0.078 | 0.210 ± 0.010 |
| Random walk | 0.119 ± 0.056 | -2.897 ± 0.031 | 0.019 ± 0.004 | -0.031 ± 0.046 | 0.389 ± 0.007 |
| Rule-knowing greedy | 82.400 ± 1.023 | 68.107 ± 1.141 | 1.000 ± 0.000 | 24.300 ± 0.485 | 1.000 ± 0.000 |

Survival values are fractions from 0 to 1. Return values are food minus poison per starting agent. These five metrics are descriptive; no capability threshold was selected from their values.

## Original condition-level fitness and survival

The table retains original fitness separately for stationary rule 0, stationary rule 1, reversal from rule 0 and reversal from rule 1, with world SE, survival immediately before the flip boundary, final survival, deaths and zero-encounter counts. Each condition starts with 1,024 agents. Rule 0 initially makes A food; rule 1 makes B food.

All starting agents remain in return and survival denominators. Dead agents are not silently removed. Accuracy is conditional on an encounter and stays undefined when there are none. The raw records retain each individual's food/poison events, survival, return, death step and encounter absence; 50-step bins and survivor-conditioned summaries are retained separately.

| Controller | Condition | Fitness ± SE | Alive before step 150 ± SE | Final survival ± SE | Deaths / 1,024 | No encounters / 1,024 |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | stationary_r0 | 15.506 ± 0.410 | 0.870 ± 0.010 | 0.851 ± 0.012 | 153 | 0 |
| A0 | stationary_r1 | 14.743 ± 0.358 | 0.879 ± 0.011 | 0.849 ± 0.012 | 155 | 0 |
| A0 | reversal_r0 | 3.447 ± 0.186 | 0.870 ± 0.010 | 0.386 ± 0.016 | 629 | 0 |
| A0 | reversal_r1 | 3.489 ± 0.175 | 0.879 ± 0.011 | 0.421 ± 0.015 | 593 | 0 |
| A1 | stationary_r0 | 15.075 ± 0.386 | 0.878 ± 0.011 | 0.852 ± 0.012 | 152 | 0 |
| A1 | stationary_r1 | 14.904 ± 0.368 | 0.862 ± 0.011 | 0.839 ± 0.012 | 165 | 0 |
| A1 | reversal_r0 | 3.491 ± 0.170 | 0.878 ± 0.011 | 0.430 ± 0.016 | 584 | 0 |
| A1 | reversal_r1 | 3.548 ± 0.173 | 0.862 ± 0.011 | 0.427 ± 0.016 | 587 | 0 |
| A2 | stationary_r0 | 14.599 ± 0.367 | 0.883 ± 0.010 | 0.854 ± 0.011 | 150 | 0 |
| A2 | stationary_r1 | 15.345 ± 0.358 | 0.886 ± 0.010 | 0.859 ± 0.011 | 144 | 0 |
| A2 | reversal_r0 | 3.331 ± 0.174 | 0.883 ± 0.010 | 0.413 ± 0.016 | 601 | 0 |
| A2 | reversal_r1 | 3.998 ± 0.183 | 0.886 ± 0.010 | 0.453 ± 0.015 | 560 | 0 |
| Direct 0 | stationary_r0 | -3.992 ± 0.003 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1024 | 0 |
| Direct 0 | stationary_r1 | 115.882 ± 1.441 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0 | 0 |
| Direct 0 | reversal_r0 | -3.992 ± 0.003 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1024 | 0 |
| Direct 0 | reversal_r1 | 32.735 ± 0.633 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1024 | 0 |
| Direct 1 | stationary_r0 | 137.913 ± 1.415 | 0.997 ± 0.002 | 0.997 ± 0.002 | 3 | 0 |
| Direct 1 | stationary_r1 | -3.974 ± 0.009 | 0.006 ± 0.002 | 0.001 ± 0.001 | 1023 | 0 |
| Direct 1 | reversal_r0 | 37.695 ± 0.719 | 0.997 ± 0.002 | 0.001 ± 0.001 | 1023 | 0 |
| Direct 1 | reversal_r1 | -3.938 ± 0.025 | 0.006 ± 0.002 | 0.004 ± 0.002 | 1020 | 0 |
| Direct 2 | stationary_r0 | 143.437 ± 1.539 | 0.999 ± 0.001 | 0.999 ± 0.001 | 1 | 0 |
| Direct 2 | stationary_r1 | -3.962 ± 0.013 | 0.008 ± 0.003 | 0.001 ± 0.001 | 1023 | 0 |
| Direct 2 | reversal_r0 | 40.738 ± 0.718 | 0.999 ± 0.001 | 0.001 ± 0.001 | 1023 | 0 |
| Direct 2 | reversal_r1 | -3.908 ± 0.042 | 0.008 ± 0.003 | 0.003 ± 0.002 | 1021 | 0 |
| Fixed A | stationary_r0 | 82.023 ± 1.440 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0 | 0 |
| Fixed A | stationary_r1 | -3.863 ± 0.019 | 0.043 ± 0.006 | 0.009 ± 0.003 | 1015 | 9 |
| Fixed A | reversal_r0 | 36.688 ± 0.605 | 1.000 ± 0.000 | 0.063 ± 0.012 | 959 | 0 |
| Fixed A | reversal_r1 | -3.863 ± 0.019 | 0.043 ± 0.006 | 0.009 ± 0.003 | 1015 | 9 |
| Fixed B | stationary_r0 | -3.857 ± 0.022 | 0.044 ± 0.007 | 0.015 ± 0.004 | 1009 | 15 |
| Fixed B | stationary_r1 | 82.777 ± 1.471 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0 | 1 |
| Fixed B | reversal_r0 | -3.857 ± 0.022 | 0.044 ± 0.007 | 0.015 ± 0.004 | 1009 | 15 |
| Fixed B | reversal_r1 | 36.519 ± 0.627 | 1.000 ± 0.000 | 0.080 ± 0.013 | 942 | 1 |
| A/B mixture | stationary_r0 | 43.399 ± 2.344 | 0.504 ± 0.002 | 0.500 ± 0.000 | 512 | 0 |
| A/B mixture | stationary_r1 | 23.572 ± 0.818 | 0.504 ± 0.002 | 0.500 ± 0.000 | 512 | 2 |
| A/B mixture | reversal_r0 | 15.446 ± 0.707 | 0.504 ± 0.002 | 0.205 ± 0.014 | 814 | 0 |
| A/B mixture | reversal_r1 | 12.066 ± 0.342 | 0.504 ± 0.002 | 0.215 ± 0.014 | 804 | 2 |
| Random walk | stationary_r0 | 0.136 ± 0.140 | 0.660 ± 0.016 | 0.412 ± 0.018 | 602 | 0 |
| Random walk | stationary_r1 | 0.102 ± 0.126 | 0.655 ± 0.015 | 0.394 ± 0.016 | 621 | 0 |
| Random walk | reversal_r0 | -0.070 ± 0.118 | 0.660 ± 0.016 | 0.391 ± 0.016 | 624 | 0 |
| Random walk | reversal_r1 | 0.059 ± 0.128 | 0.655 ± 0.015 | 0.388 ± 0.015 | 627 | 0 |
| Rule-knowing greedy | stationary_r0 | 82.023 ± 1.440 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0 | 0 |
| Rule-knowing greedy | stationary_r1 | 82.777 ± 1.471 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0 | 1 |
| Rule-knowing greedy | reversal_r0 | 68.830 ± 0.648 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0 | 0 |
| Rule-knowing greedy | reversal_r1 | 67.974 ± 0.621 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0 | 0 |

For stationary conditions, “before step 150” is the same clock boundary without a rule change. Death and no-encounter entries are observed counts. An agent that dies after eating contributes its actual return and death; an agent with no encounters is retained with zero return and undefined food-choice accuracy. Complete 50-step encounter/death counts, alive-at-bin-start counts and zero-encounter counts are in the linked summary and raw records.

## Fifteen prespecified A minus direct contrasts

For each matched evolution seed 0, 1 and 2, each contrast subtracts the direct controller's base-world value from A's value for one of the five metrics. The table gives the mean and paired base-world SE. These 15 correlated descriptive contrasts were fixed before evaluation. There is no selected best contrast, multiple-testing claim, capability threshold or promotion rule based on them.

| Matched evolution seed | Metric | A − direct ± paired SE |
| --- | --- | --- |
| 0 | Stationary fitness | -40.820 ± 0.768 |
| 0 | Individual minimum return | 12.229 ± 0.301 |
| 0 | Joint stationary survival | 0.710 ± 0.014 |
| 0 | Post-flip return | 1.162 ± 0.083 |
| 0 | Reversal final survival | 0.403 ± 0.011 |
| 1 | Stationary fitness | -51.980 ± 0.736 |
| 1 | Individual minimum return | 12.431 ± 0.310 |
| 1 | Joint stationary survival | 0.704 ± 0.014 |
| 1 | Post-flip return | 1.375 ± 0.090 |
| 1 | Reversal final survival | 0.426 ± 0.012 |
| 2 | Stationary fitness | -54.766 ± 0.787 |
| 2 | Individual minimum return | 12.358 ± 0.298 |
| 2 | Joint stationary survival | 0.731 ± 0.014 |
| 2 | Post-flip return | 1.347 ± 0.098 |
| 2 | Reversal final survival | 0.431 ± 0.011 |

The direct controllers retain higher ordinary stationary fitness; A has higher paired individual-minimum returns and joint survival. Positive A − direct post-flip contrasts mean A loses less on that measure, not that A has a positive post-flip return. Survival contrasts are differences in fractions, not percentage points. No extra reference-policy significance contrasts were added.

## Reference counterexamples and what can be inferred

The fixed specialist mixture tests whether population balance conceals different agents succeeding under opposite rules. Taking a minimum after averaging each family would miss that distinction; the individual minimum is calculated first. The privileged greedy policy provides a separate counterexample: it can behave successfully without having to infer the rule from experience. Neither a high paired minimum nor survival alone proves learning.

The mixture has ordinary stationary fitness **33.486 ± 1.236**, yet mean individual minimum return **−3.970 ± 0.008** and joint survival **0.000 ± 0.000**. This directly exhibits the aggregation problem: a family can score well while the same individual fails under the opposite rule. In contrast, privileged greedy has individual minimum return **68.107 ± 1.141**, joint survival **1.000 ± 0.000**, and post-flip return **24.300 ± 0.485**, without needing to learn which rule applies. These references test what the measurements mean; they are not independent evolved seeds. No causal claim about feedback, memory, recurrence or plasticity can be inferred solely from these observational comparisons. Opposite-rule replays share initial conditions, not experiences or a learned state carried between lives.

## Validity, provenance and individual distributions

The registered gate concerns **measurement validity only**, and it passes. The independent audit reconstructed the requested statistics from every archived condition, verified common initial worlds/positions/development keys, fixed specialist assignments, matching stationary/reversal prefixes, copied input hashes and all event/death/encounter accounting. A valid measurement does not reverse E113's failed gate or establish a capability milestone.

The [run configuration](../runs/20261119-045931_E114-v4/config.json) fixes the manifest, saved-source hashes and controller formats; the [complete summary](../runs/20261119-045931_E114-v4/summary.json) contains every condition, empirical individual-return distribution and all 15 contrasts. Source commit is `5596c1a7746069548bc533931834ead9e3904b97`, with clean Git provenance. All six copied genome files retain their source bytes. Local verification passed 133 tests before this held-out evaluation. The paired archives retain individual stationary minima and joint-survival outcomes as well as the base-world vectors; per-condition files retain complete per-agent returns. Empirical individual-return distributions are descriptive summaries, not extra independent replicates.

| Controller | Stationary rule 0 | Stationary rule 1 | Reversal from rule 0 | Reversal from rule 1 | Paired individuals / vectors |
| --- | --- | --- | --- | --- | --- |
| A0 | [stationary_r0](../runs/20261119-045931_E114-v4/A0/stationary_r0.npz) | [stationary_r1](../runs/20261119-045931_E114-v4/A0/stationary_r1.npz) | [reversal_r0](../runs/20261119-045931_E114-v4/A0/reversal_r0.npz) | [reversal_r1](../runs/20261119-045931_E114-v4/A0/reversal_r1.npz) | [Paired individuals](../runs/20261119-045931_E114-v4/A0/paired.npz) |
| A1 | [stationary_r0](../runs/20261119-045931_E114-v4/A1/stationary_r0.npz) | [stationary_r1](../runs/20261119-045931_E114-v4/A1/stationary_r1.npz) | [reversal_r0](../runs/20261119-045931_E114-v4/A1/reversal_r0.npz) | [reversal_r1](../runs/20261119-045931_E114-v4/A1/reversal_r1.npz) | [Paired individuals](../runs/20261119-045931_E114-v4/A1/paired.npz) |
| A2 | [stationary_r0](../runs/20261119-045931_E114-v4/A2/stationary_r0.npz) | [stationary_r1](../runs/20261119-045931_E114-v4/A2/stationary_r1.npz) | [reversal_r0](../runs/20261119-045931_E114-v4/A2/reversal_r0.npz) | [reversal_r1](../runs/20261119-045931_E114-v4/A2/reversal_r1.npz) | [Paired individuals](../runs/20261119-045931_E114-v4/A2/paired.npz) |
| Direct 0 | [stationary_r0](../runs/20261119-045931_E114-v4/direct0/stationary_r0.npz) | [stationary_r1](../runs/20261119-045931_E114-v4/direct0/stationary_r1.npz) | [reversal_r0](../runs/20261119-045931_E114-v4/direct0/reversal_r0.npz) | [reversal_r1](../runs/20261119-045931_E114-v4/direct0/reversal_r1.npz) | [Paired individuals](../runs/20261119-045931_E114-v4/direct0/paired.npz) |
| Direct 1 | [stationary_r0](../runs/20261119-045931_E114-v4/direct1/stationary_r0.npz) | [stationary_r1](../runs/20261119-045931_E114-v4/direct1/stationary_r1.npz) | [reversal_r0](../runs/20261119-045931_E114-v4/direct1/reversal_r0.npz) | [reversal_r1](../runs/20261119-045931_E114-v4/direct1/reversal_r1.npz) | [Paired individuals](../runs/20261119-045931_E114-v4/direct1/paired.npz) |
| Direct 2 | [stationary_r0](../runs/20261119-045931_E114-v4/direct2/stationary_r0.npz) | [stationary_r1](../runs/20261119-045931_E114-v4/direct2/stationary_r1.npz) | [reversal_r0](../runs/20261119-045931_E114-v4/direct2/reversal_r0.npz) | [reversal_r1](../runs/20261119-045931_E114-v4/direct2/reversal_r1.npz) | [Paired individuals](../runs/20261119-045931_E114-v4/direct2/paired.npz) |
| Fixed A | [stationary_r0](../runs/20261119-045931_E114-v4/fixed_A/stationary_r0.npz) | [stationary_r1](../runs/20261119-045931_E114-v4/fixed_A/stationary_r1.npz) | [reversal_r0](../runs/20261119-045931_E114-v4/fixed_A/reversal_r0.npz) | [reversal_r1](../runs/20261119-045931_E114-v4/fixed_A/reversal_r1.npz) | [Paired individuals](../runs/20261119-045931_E114-v4/fixed_A/paired.npz) |
| Fixed B | [stationary_r0](../runs/20261119-045931_E114-v4/fixed_B/stationary_r0.npz) | [stationary_r1](../runs/20261119-045931_E114-v4/fixed_B/stationary_r1.npz) | [reversal_r0](../runs/20261119-045931_E114-v4/fixed_B/reversal_r0.npz) | [reversal_r1](../runs/20261119-045931_E114-v4/fixed_B/reversal_r1.npz) | [Paired individuals](../runs/20261119-045931_E114-v4/fixed_B/paired.npz) |
| A/B mixture | [stationary_r0](../runs/20261119-045931_E114-v4/mixture/stationary_r0.npz) | [stationary_r1](../runs/20261119-045931_E114-v4/mixture/stationary_r1.npz) | [reversal_r0](../runs/20261119-045931_E114-v4/mixture/reversal_r0.npz) | [reversal_r1](../runs/20261119-045931_E114-v4/mixture/reversal_r1.npz) | [Paired individuals](../runs/20261119-045931_E114-v4/mixture/paired.npz) |
| Random walk | [stationary_r0](../runs/20261119-045931_E114-v4/random_walk/stationary_r0.npz) | [stationary_r1](../runs/20261119-045931_E114-v4/random_walk/stationary_r1.npz) | [reversal_r0](../runs/20261119-045931_E114-v4/random_walk/reversal_r0.npz) | [reversal_r1](../runs/20261119-045931_E114-v4/random_walk/reversal_r1.npz) | [Paired individuals](../runs/20261119-045931_E114-v4/random_walk/paired.npz) |
| Rule-knowing greedy | [stationary_r0](../runs/20261119-045931_E114-v4/greedy/stationary_r0.npz) | [stationary_r1](../runs/20261119-045931_E114-v4/greedy/stationary_r1.npz) | [reversal_r0](../runs/20261119-045931_E114-v4/greedy/reversal_r0.npz) | [reversal_r1](../runs/20261119-045931_E114-v4/greedy/reversal_r1.npz) | [Paired individuals](../runs/20261119-045931_E114-v4/greedy/paired.npz) |

Copied inputs: [A0](../runs/20261119-045931_E114-v4/inputs/A0.npy), [A1](../runs/20261119-045931_E114-v4/inputs/A1.npy), [A2](../runs/20261119-045931_E114-v4/inputs/A2.npy), [direct0](../runs/20261119-045931_E114-v4/inputs/direct0.npy), [direct1](../runs/20261119-045931_E114-v4/inputs/direct1.npy), [direct2](../runs/20261119-045931_E114-v4/inputs/direct2.npy).

The [independent audit](diagnostics/E114-final-audit.json) and [resource guard](diagnostics/E114-guard.jsonl) preserve validation and execution evidence. Evaluation took **233.187 seconds**; supervised wall time was **248.998 seconds**. These are operational measurements, not scientific effects.

## Council decision and five-line report

The owner delegated scientific decisions to the council. With the measurement audit complete, the next separate experiment is under council review. This report does not assume that a new objective, mechanism or training run is warranted. Any selected step must be preregistered and committed before execution, preserve past results, and distinguish measurement, causal tests and training.

1. **Asked:** test saved controllers across fresh matched opposite-rule and reversal lives, with individual outcomes and specialist/mixture controls.
2. **Happened:** all 11 controllers passed measurement validity; A's individual minimum returns were **8.237 ± 0.302, 8.453 ± 0.310 and 8.396 ± 0.297**, while the specialist mixture scored **33.486 ± 1.236** on ordinary fitness but **−3.970 ± 0.008** on individual minimum return.
3. **Means:** the individual pairing exposes specialist mixtures that a population score can conceal; A's stationary outcomes are more robust across the two rules than the direct controllers' outcomes.
4. **Does not mean:** paired robustness proves learning or reversal recovery—A's post-flip return remains negative, greedy succeeds with privileged rule knowledge, and E113's failed gate remains failed.
5. **Next:** the council is reviewing a separate bounded experiment; any decision must be preregistered before new execution.
