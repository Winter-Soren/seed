# E113: directly encoded recurrent baseline

**Result: the directly encoded, fixed-weight recurrent baseline beats the plastic A agents on registered stationary fitness in all three seeds, beyond two paired standard errors. Phase 2’s continuation gate fails.** All three runs completed 200 generations and passed their validity audits. Stop for owner input; no later rung or objective repair is authorized by this result.

The raw score advantage is substantial, but the per-rule audit shows strongly asymmetric performance and almost no final survival after reversal. This is evidence against claiming a plastic advantage under the current score; it is not evidence of reliable rule adaptation.

## Registered question and decision

Can a directly encoded recurrent controller with fixed lifetime weights match or exceed the plastic A agents under the same outer evolution budget? This replaces the historical C control, whose tied sensory encoding restricted the baseline.

The primary metric is stationary-world fitness, defined as food minus poison per initial agent. For each evolution seed, compare its direct controller against the corresponding saved A genome on the same 256 EVAL v3 world-families. Compute each world's **A minus direct** fitness, then its mean and sample standard deviation divided by √256.

Continue beyond Phase 2 only if **all three** differences are strictly greater than two paired standard errors, after all three direct runs finish 200 generations and pass validity checks. A difference below minus two paired SEs favors the direct baseline. A difference within that uncertainty is unresolved, not proven equivalence. There is no majority exception or equivalence margin. Reversal is secondary and cannot change the primary decision.

## Design and comparability

Each direct genome contains 8,424 independent weights: every off-diagonal input into the 54 non-sensor neurons of the existing 157-neuron architecture. Sensor rows and the diagonal remain zero. The controller retains 103 observations, 48 hidden neurons, six motor outputs, two recurrent microticks per step, the action efference copy, signalling and the same world physics. All four family members inherit the same weights; recurrent activity can change during life, but weights cannot.

The direct weights start from independent normal values with standard deviation 1/√156. They are finite and unbounded, with no lifetime clipping, predictor updates, weight learning, growth or pruning. This matters because plastic weights are clipped to ±3. The initial and final weight statistics below disclose whether the direct controller uses values outside that range.

Each of seeds 0, 1 and 2 receives 200 generations, population 128, eight training worlds per generation, ES mutation scale 0.1, learning rate 0.1, the existing Adam update and L2 coefficient 0.005. These settings match A rather than B. The ES world-key schedule is retained. Final evaluation uses seed 2026111901, 256 worlds, 128 worlds per initial food rule and four agents per world; the reversal changes the rule before eating at step 150. Evaluation is performed only after generation 200.

## Primary paired comparison

| Evolution seed | Plastic A fitness ± SE | Direct fitness ± SE | Paired A − direct ± SE | Strict A advantage > 2 SE | Interpretation |
| --- | --- | --- | --- | --- | --- |
| 0 | 15.346 ± 0.377 | 55.520 ± 3.899 | -40.174 ± 3.922 | No | Direct advantage |
| 1 | 15.187 ± 0.385 | 66.481 ± 4.533 | -51.295 ± 4.544 | No | Direct advantage |
| 2 | 14.984 ± 0.351 | 69.869 ± 4.757 | -54.885 ± 4.737 | No | Direct advantage |

The registered continuation criterion is met in **0/3 pairs**. The direct advantage exceeds two paired SEs in **3/3**. Differences use paired world outcomes; their SE is not calculated by treating A and direct as independent samples. The [comparison data](diagnostics/E113-comparison.json) retain all 256 differences for each pair; the [saved-artifact calculator](diagnostics/E113-comparison.py) records the arithmetic and performs no simulation.

## Stationary and reversal outcomes

The tables show each direct controller and its paired A comparator separately, with raw fitness ± one world SE, fraction of the common local rule-knowing greedy score ± paired ratio SE, and final survival. Reversal is a diagnostic, not the registered promotion metric. Positive lifetime reversal fitness alone is not evidence of recovery after the switch.

| Condition | Controller | Fitness ± SE | Fraction of greedy ± SE | Final survival fraction ± SE | Survivors / 1,024 |
| --- | --- | --- | --- | --- | --- |
| stationary | A0 | 15.346 ± 0.377 | 0.1865 ± 0.0056 | 0.841 ± 0.012 | 861 |
| stationary | Direct 0 | 55.520 ± 3.899 | 0.6748 ± 0.0483 | 0.500 ± 0.031 | 512 |
| stationary | A1 | 15.187 ± 0.385 | 0.1846 ± 0.0056 | 0.847 ± 0.012 | 867 |
| stationary | Direct 1 | 66.481 ± 4.533 | 0.8080 ± 0.0572 | 0.499 ± 0.031 | 511 |
| stationary | A2 | 14.984 ± 0.351 | 0.1821 ± 0.0047 | 0.851 ± 0.012 | 871 |
| stationary | Direct 2 | 69.869 ± 4.757 | 0.8492 ± 0.0602 | 0.502 ± 0.031 | 514 |
| reversal | A0 | 3.127 ± 0.183 | 0.0453 ± 0.0027 | 0.378 ± 0.016 | 387 |
| reversal | Direct 0 | 14.016 ± 1.218 | 0.2031 ± 0.0176 | 0.000 ± 0.000 | 0 |
| reversal | A1 | 3.529 ± 0.183 | 0.0511 ± 0.0027 | 0.422 ± 0.015 | 432 |
| reversal | Direct 1 | 16.557 ± 1.388 | 0.2399 ± 0.0204 | 0.003 ± 0.002 | 3 |
| reversal | A2 | 3.576 ± 0.166 | 0.0518 ± 0.0024 | 0.421 ± 0.015 | 431 |
| reversal | Direct 2 | 19.071 ± 1.536 | 0.2763 ± 0.0226 | 0.003 ± 0.002 | 3 |

Common greedy stationary fitness: **82.278 ± 1.364**.

Common greedy reversal fitness: **69.024 ± 0.700**.

The direct controllers finish reversal with **0, 3 and 3 survivors out of 1,024**, respectively. Their positive lifetime fitness includes food obtained before death; it does not establish successful recovery after the switch.

## Initial and final weight statistics

Each row gives the direct genome's maximum absolute weight and fraction of its 8,424 weights with absolute value greater than 3, before and after evolution. These are descriptive parameter statistics, not estimates over worlds; they do not receive invented standard errors.

| Seed | Initial maximum absolute weight | Initial fraction above 3 | Final maximum absolute weight | Final fraction above 3 |
| --- | --- | --- | --- | --- |
| 0 | 0.304117 | 0.0000% | 4.661759 | 1.1277% |
| 1 | 0.304444 | 0.0000% | 5.060134 | 1.5195% |
| 2 | 0.323204 | 0.0000% | 4.846064 | 1.2821% |

## Post-result breakdown by initial food rule

This stratification was proposed after inspecting seed 0 and is explicitly **post hoc**. It explains the aggregate outcome; it is not a replacement gate, a new training objective or an evaluation-driven adjustment. Each seed and condition is split into its two 128-world initial-rule groups, with fitness ± SE and final surviving-agent counts. Initial rule 0 means A is food; initial rule 1 means B is food. Reversal changes that mapping at step 150.

The score can reward an asymmetric policy: death limits losses from repeatedly consuming poison, while food continues to score even when energy is capped. A controller that performs well in one rule group and dies in the other can therefore have high mean fitness. Per-rule outcomes and post-switch survival must be inspected before describing that score as learning or recovery. They do not by themselves identify an exact internal mechanism or rule out all use of recurrent state.

| Direct seed | Condition | Initial rule | Worlds | Fitness ± SE | Final surviving agents |
| --- | --- | --- | --- | --- | --- |
| 0 | stationary | 0 | 128 | -3.988 ± 0.005 | 0 / 512 |
| 0 | stationary | 1 | 128 | 115.027 ± 2.301 | 512 / 512 |
| 0 | reversal | 0 | 128 | -3.988 ± 0.005 | 0 / 512 |
| 0 | reversal | 1 | 128 | 32.020 ± 0.924 | 0 / 512 |
| 1 | stationary | 0 | 128 | 136.938 ± 2.088 | 511 / 512 |
| 1 | stationary | 1 | 128 | -3.975 ± 0.008 | 0 / 512 |
| 1 | reversal | 0 | 128 | 37.010 ± 1.070 | 0 / 512 |
| 1 | reversal | 1 | 128 | -3.896 ± 0.060 | 3 / 512 |
| 2 | stationary | 0 | 128 | 143.676 ± 2.258 | 512 / 512 |
| 2 | stationary | 1 | 128 | -3.938 ± 0.032 | 2 / 512 |
| 2 | reversal | 0 | 128 | 42.076 ± 1.067 | 0 / 512 |
| 2 | reversal | 1 | 128 | -3.934 ± 0.030 | 3 / 512 |

Seed 0 succeeds mainly when B is initially food; seeds 1 and 2 succeed mainly when A is initially food. In their unfavorable stationary groups, mean fitness is near −4 and nearly every agent dies. In favorable groups, mean fitness is 115.027–143.676, and nearly every agent survives. Those asymmetric outcomes explain how the baseline can score well without demonstrating robust success across both rules.

This pattern is consistent with strong item specialization benefiting from the score’s asymmetric upside and downside. It does not establish the complete internal policy or prove that recurrent state never contributes: encounter behavior can change within a life, and no causal history intervention was performed. The post hoc breakdown diagnoses an interpretation problem; the original score, three-seed result and failed continuation gate remain intact. Survival entries here are observed counts, not new probability estimates or additional gate criteria.

## What the comparison establishes—and its limits

- Every ± entry is one standard error. Fitness SE treats each world-family as one unit; paired contrasts use the same worlds. These quantify uncertainty across held-out worlds for a fixed trained genome, not variability across evolution seeds.
- A zero empirical survival SE means all sampled worlds had the same outcome; it does not establish zero uncertainty about every possible future world.
- Fraction-of-greedy uncertainty uses the existing paired world-level ratio estimate. Greedy has privileged knowledge of the current rule, including immediately after reversal, and sees only the local patch. It is not an optimal-policy upper bound; a ratio above one would not be impossible.
- Equal training budget does not isolate plasticity as the only causal difference. Encoding size, density, developmental variation, parameter meaning and weight-amplitude constraints differ. A direct-baseline loss would not establish that fixed-weight recurrence cannot learn; a win would concern the registered task and score.
- Later-meal or survivor-conditioned curves describe selected cohorts. A high accuracy among agents that reach later meals can exclude agents that died early. Matched-agent changes still condition on eligibility. The primary gate does not use those curves.
- Recurrent activity changes are distinct from changing synaptic weights, and both are distinct from forming connections. Direct weights remain fixed; structural growth is off throughout this comparison. The results cannot establish useful new-edge formation or a fly-derived wiring advantage.
- Existing EVAL v1 data remain unchanged. The paired A comparators are the completed E112 measurements, and the direct runs use the same held-out world keys, rules and reference policies. No best-seed selection, retuning or changed threshold is permitted after these outcomes.

## Decision and possible next owner authorization

The registered continuation gate **fails**, so stop for owner input. Do not change the objective, add a mechanism or begin Phase 3 under this result.

The council has prepared [E114: individual adaptation versus specialist mixtures](../experiments/E114_PROPOSAL.md), a **diagnostic-only proposal, not approved or run**. It proposes fresh paired opposite-rule and reversal evaluations of saved controllers, with fixed-specialist and specialist-mixture controls, individual paired returns and survival accounting. It authorizes no new training or reward change. Owner approval, followed by preregistration and committed verification, is required before execution.

Optimizing the minimum of the two rule-group mean scores would not by itself solve the problem: different specialist agents could succeed in opposite groups while other agents die. Group balance does not establish individual learning. Even an individual paired-success measure would not by itself prove dependence on experience; causal history/feedback interventions would require separate approval. Any later objective, numerical gate and compute budget must be fixed before new runs.

## Provenance and artifacts

All three final audits verify 200 finite contiguous generation records, checkpoint/final-genome agreement, preserved initial genome bytes, the registered controller/settings, and identical E112 world/development keys and reference-policy outputs. The direct controller’s fixed-lifetime weights, independent sensory channels, recurrence, ES key schedule and crash/resume preservation are covered by the 112-test suite; historical EVAL v1 compatibility also passed. Software tests establish implementation behavior, not scientific capability.

| Seed | Direct summary | Raw events | Weights | A comparator | Final audit |
| --- | --- | --- | --- | --- | --- |
| 0 | [20261119-033803_E113-direct-s0](../runs/20261119-033803_E113-direct-s0/summary.json) | [Events](../runs/20261119-033803_E113-direct-s0/events.npz) | [Initial](../runs/20261119-033803_E113-direct-s0/genome_start.npy) / [final](../runs/20261119-033803_E113-direct-s0/genome_final.npy) | [A0](../runs/20261119-030209_E112-v3-A0/summary.json) | [Audit](diagnostics/E113-s0-final-audit.json) |
| 1 | [20261119-040602_E113-direct-s1](../runs/20261119-040602_E113-direct-s1/summary.json) | [Events](../runs/20261119-040602_E113-direct-s1/events.npz) | [Initial](../runs/20261119-040602_E113-direct-s1/genome_start.npy) / [final](../runs/20261119-040602_E113-direct-s1/genome_final.npy) | [A1](../runs/20261119-030336_E112-v3-A1/summary.json) | [Audit](diagnostics/E113-s1-final-audit.json) |
| 2 | [20261119-041609_E113-direct-s2](../runs/20261119-041609_E113-direct-s2/summary.json) | [Events](../runs/20261119-041609_E113-direct-s2/events.npz) | [Initial](../runs/20261119-041609_E113-direct-s2/genome_start.npy) / [final](../runs/20261119-041609_E113-direct-s2/genome_final.npy) | [A2](../runs/20261119-030424_E112-v3-A2/summary.json) | [Audit](diagnostics/E113-s2-final-audit.json) |

- **Seed 0:** original source `9db7fafe9d70a6eb0974af4e137ab21e01a60be9`; continuation source `e5630033125c8409218615044f3addbd0da46d61`. Final genome hash `b03d4ccb3d931f38`.
- **Seed 1:** original source `87702c0277d63bd8f559448927662d0b296c7d83`. Final genome hash `e7b8084af7c6a4c6`.
- **Seed 2:** original source `1d7e2a3a5d8d823c1000c8b5c18d8deae49d7006`. Final genome hash `a693c973939c7fb5`.

Seed 0’s original interrupted segment and subsequent continuation are retained. The continuation source was checked against its original source and contains no engine/scripts/tests/dependency implementation difference. Seeds 1 and 2 also record clean source provenance; the run configs retain full settings and machine information.

The [seed-0 per-rule audit](diagnostics/E113-s0-per-rule-audit.json) and the seed-1/seed-2 final audits preserve the descriptive stratification. The [paired comparison JSON](diagnostics/E113-comparison.json) and [calculator](diagnostics/E113-comparison.py) preserve the gate inputs and decision. The complete A measurements and their selection limits are in [EVAL_V3.md](EVAL_V3.md).

## Five-line phase report

1. **Asked:** train a fixed-weight recurrent baseline under the same 200-generation ES budget and compare three seeds with the plastic A agents.
2. **Happened:** all three baselines completed and passed validity checks; paired A − direct fitness was **−40.174 ± 3.922, −51.295 ± 4.544 and −54.885 ± 4.737 SE**. The continuation gate failed in every pair.
3. **Means:** the direct baseline wins the registered stationary score in 3/3 seeds; per-rule outcomes reveal strongly asymmetric success, and only 0, 3 and 3 out of 1,024 agents survive reversal in the respective direct runs.
4. **Does not mean:** the baseline demonstrated reliable adaptation, that plasticity is unnecessary in general, or that either method forms useful new connections; the encodings and constraints also differ.
5. **Next:** stop for owner input. E114 is a concrete, unapproved diagnostic proposal to distinguish individual adaptation from specialist mixtures, with no new training or objective change.
