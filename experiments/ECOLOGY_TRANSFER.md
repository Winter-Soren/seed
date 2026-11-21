# Prospective ecology transfer tests

**Design only; no research run authorized.** Finish E116 unchanged and obtain the owner's next decision. Register fresh seeds, commands, source fingerprints, thresholds and compute limits before execution. Fixtures establish implementation behavior, not scientific results.

The owner requests visible environments, reuse of prior learning, and shared lessons for successors. Keep three questions separate: reuse of a genome; reuse of one individual's learned weights; and reuse of a cohort's acquired changes by different individuals. None alone establishes a curriculum benefit.

[POET compared solver transfer with transfer disabled on generated obstacle courses](https://arxiv.org/pdf/1901.01753); this motivates a transfer-disabled control here, but that paper did not test Seed's proposed sharing rule.

## Common ecology and perception

Use original A0/A1/A2, not selected descendants. All ecology controls and source/target lives share the versioned sensor layout, local shelter bitmap and weather cue; shelter is world-seeded and no hidden food rule is revealed. Even zero-cost controls retain these cues. Reusing an original genome in a new layout does not preserve its old developed network. Legacy no-ecology stays unchanged; never copy its smaller W into an expanded network.

Change one pressure per comparison: movement cost versus zero; then seasonal drain versus zero at fixed movement cost. Movement costs apply to non-stay actions while alive. Seasonal drain starts at T/2 while alive and outdoors; shelter changes exposure, not food identity. Keep energy clipping unchanged. Fix costs and shelter parameters before data.

Proposed sample: 256 base-world families per original A seed. Pair starting conditions and random streams; later trajectories may diverge. Retain per-agent arrays and compute SEs over independent families/lineages, never agents or repeated episodes.

## 1. Genome reuse: transfer without inherited lifetime weights

Evaluate fresh births under both stationary food rules in matched ecology conditions, without mid-life food reversal. Reset learned state and physics. Report pressure-minus-control return, individual minimum return, survival, food/poison, absent encounters and movement/exposure costs. Original fitness excludes energy costs; do not call it efficiency. Separate nominal debits from actual clipped energy changes. This tests genome transfer, not inherited lifetime memory.

To ask whether reusing an evolved genome helps, include a matched blank-genome start in each of the three seed groups. Both starts receive the same environment, development random streams and training budget. Comparing an evolved genome across two environments measures the effect of the environment on that genome; it does not measure an advantage over starting fresh.

## 2. Same-individual weight reuse: benefit and harmful transfer

Interface: `same_individual(g, wkeys[2], dkey, rules[2], c, *, ecologies=None, plastic=True)`. The wrapper develops and transfers internally; no arbitrary caller-supplied W.

For a fixed source-life record, compare two second-life branches in the identical target world:

| Branch | Initial second-life synapses | Everything else |
| --- | --- | --- |
| Carry | The same individual's terminal acquired W | Reset |
| Reset | Its original developed birth W | Identical reset |

Reset predictor, activity, energy, positions, grid, signals, time and counters. Match genome, development key, layout, topology, agent indices and target random streams. Both branches retain the same lifetime updates; only initial W differs.

Source deaths remain in both branches' all-initial-agent accounting with zero second-life contribution and identical ineligibility masks. Do not resurrect or silently discard them. Report source survival, target eligibility/encounters and survival through both lives. Survivor-conditioned results are secondary.

For food-rule transfer only, compare aligned `rules=(r,r)` with opposite-rule `rules=(r,1-r)`, using the **same source life and survivor cohort**. Report Carry minus Reset separately for each rule relation and ecology. A food-label flip changes recipient experience; it is not a control for the relevance of shelter knowledge. Never retune to force a negative result.

Record first-meal food/poison/no-encounter categories, early 50-step and full-life outcomes with denominators. Aligned benefit plus an opposite-rule early cost would support content-sensitive reuse. Benefits in both may instead reflect a general initialization benefit. Later gains may include new learning. Changed behavior shows use; useful transfer requires the registered return improvement and survival safeguard. Missing encounters are not correct choices.

## 3. Collective memory: different individuals and successors

Interface: `collective_lives(g, wkeys, dkeys, rules, c, *, ecologies=None, plastic=True)`. Cohorts have independent development; do not copy entire W between differently wired individuals.

Give the donor cohort a cold season **before** scoring its successors on a fresh cold world. In the software fixture, cold first appears in the last cohort, so earlier cohorts never experience the shelter's protection. Their archive can change a newborn's behavior but cannot contain an acquired lesson about that cold-weather consequence. In the scientific comparison, hold the target world and independently drawn newborn wiring fixed across recipients of relevant, reset and relevance-broken memory.

Archive acquired sensor-to-motor deltas: donor terminal-while-alive W minus its **actual initialized W**, including prior archive initialization. Average over donors possessing each edge. Present-edge zero deltas count; absent edges do not. Include dead donors' last live state, not post-death updates. Preserve donor identities, edge counts, deaths, inherited baselines and clipping. This is an engineered sharing rule, not spontaneous communication.

First use a cheap two-cohort test: **one source cohort → one new recipient cohort**, with acquired archive versus reset. Match genome, cohort size, recipient development/world keys, food rule, ecology and life budget. Count every assigned individual, including deaths and failed successors; no best-donor selection or extra lives for one arm. This transfer milestone must be decided before accumulation or curriculum-superiority studies.

For shelter specificity, keep that **recipient world identical** and add a source-relationship-mismatched or shuffled archive, matched to the acquired archive's transferred edge coverage, scale and donor accounting. Define the transformation before evaluation; preserve zero-delta contributions and absent-edge distinctions. A possible shelter-block permutation must preserve coverage and magnitude statistics and actually disrupt the relevant mapping, rather than merely alter how much information is injected. If no sound content-breaking control can be defined, leave shelter-content specificity open: archive-versus-reset can show a transfer effect, not identify a shared shelter lesson. A food-rule flip cannot substitute for this control.

Measure early shelter use and its cost consequence: first entry latency for agents starting outside, entry by a fixed early deadline, shelter occupancy during the first 50 cold steps, charged outdoor exposure, and survival of **all assigned agents**. Report agents starting inside separately: remaining there is not evidence of navigation to shelter. Death or no entry is censored/non-entry, never a successful zero latency and never silently dropped. Fix the deadlines and censoring summary before evaluation; survivor-only latency is secondary.

Only after useful transfer passes, compare cumulative sharing with **latest-cohort-only** and reset, at identical episode exposure and replacement budgets, to test whether older lessons add value. Report cumulative return, donor coverage, assigned/eligible counts and successor survival. This cannot rescue a failed primary transfer test.

Fresh-world episodes isolate memory transfer but do not fulfill continuous-world replacement on death. That owner requirement still needs a persistent-grid protocol with a global clock, fixed successor budget and matched resource resets. Keep the claim pending until tested.

## Short, falsifiable curriculum

Levels: L0, zero added costs; L1, costly movement; L2, same movement cost plus second-half outdoor cold and visible shelter. Sensors/cues stay identical. These are increasing charged costs, not assumed empirical difficulty.

Compare **L0 → L1 → L2** against **L1 → L0 → L2**, then a common fresh held-out L2 test. Match level exposure, episode count/length, final training level, memory mechanism and budget; associate world/development keys with levels across orders. Repeated L2 is an optional equal-budget control with different exposure, not a substitute for this ordering control.

Use a previously validated reuse mechanism. Fix the entire schedule, include every assigned lineage, and evaluate its final state. Register one final-world return measure and survival safeguard across all three original seeds. Failure stops the schedule; no extra stages, mechanisms, softer thresholds or compute.

## Release and reporting

Register each pressure, rule relation, archive mode, fresh seeds, contrasts and finite episode/compute budget before running. Verify compatibility on tiny fixtures, benchmark, keep existing laptop guards and one heavy job. Commit before opening held-out worlds. Timeouts/invalid artifacts remain incomplete and preserved, without automatic retry.

Report separately: genome transfer, individual memory use, useful transfer, negative transfer, collective benefit and curriculum ordering. No deliberate survival-for-return trade-off is accepted by this design. No gain in one category substitutes for a failed gate in another.
