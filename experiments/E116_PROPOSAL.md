# E116 proposal: test a smaller evolution step during continuation

**Proposed only; awaiting the owner's decision. No implementation or run is authorized by this file.** E115's registered gate remains FAIL. [PLAN.md](../PLAN.md) requires an owner decision at this failed gate now that the owner is present. Approval would cover only the comparison and limits below; exact commands and source hashes must then be preregistered in the LOG and verified code committed before training.

## Why this experiment

The [complete E115 report](../results/EVAL_V5.md) and [independent audit](../results/diagnostics/E115-independent-audit.json) establish valid measurements, with **0/6 return-improvement checks and 1/3 survival safeguards passing**. Both objectives' final individual minimum returns are below their unchanged ancestors:

| Original A seed | Unchanged start | Mean-objective final | Min-objective final |
| --- | --- | --- | --- |
| 0 | 8.690 ± 0.305 | 6.932 ± 0.296 | 7.216 ± 0.293 |
| 1 | 8.245 ± 0.314 | 6.622 ± 0.270 | 6.011 ± 0.276 |
| 2 | 8.581 ± 0.288 | 6.317 ± 0.285 | 6.753 ± 0.273 |

Values are means ± world SE for fixed genomes. These side-by-side mean-arm comparisons are descriptive; they are not additional preregistered E115 contrasts. The registered min-minus-start losses are −1.475 ± 0.406, −2.234 ± 0.427 and −1.828 ± 0.403, using paired-world SE.

Joint stationary survival also falls in both arms for every seed. Across the two stationary rules, all six trained controllers collect fewer food items than their ancestor; poison counts increase in three and decrease in three. Those counts mix behavior and survival and do not identify why performance fell. Min-arm post-flip returns become less negative but remain −2.040 ± 0.085, −2.037 ± 0.084 and −2.225 ± 0.086. This does not justify exchanging stationary survival for reversal performance.

**Hypothesis to test:** reducing the ES learning rate from 0.1 to 0.02 improves continuation of the existing learners under the paired-min objective. E115 does not establish that its learning rate caused the loss. This tests one plausible contributor, leaving mutation scale, Adam reset, population size, exposure and architecture fixed.

## Fixed comparison

| Setting | Control | Candidate |
| --- | --- | --- |
| ES learning rate | 0.1 | 0.02 |
| Training objective | Paired individual minimum | Same |
| Starting genomes | Original A0/A1/A2 | Identical bytes for each pair |
| Generations per run | 20 | 20 |
| Population / base-world pairs | 64 / 4 | Same |
| Mutation sigma / L2 | 0.1 / 0.005 | Same |
| Adam initialization | Zero moments, step 0 | Same |

Six new runs, three matched pairs. Neither arm starts from an E115 descendant. Pin the original source genomes already identified as `start0`, `start1`, `start2` in the [E115 evaluation manifest](../runs/20261120-060829_E115-v5-recovery1/config.json), and preserve their original NPY bytes.

Use fresh matched training seeds **2000, 2001, 2002** for original A seeds 0, 1, 2. These seeds and the proposed evaluation seed below were not found in existing registrations or run configurations when this proposal was written. Both arms use identical mutation/world/development/action random streams for each generation, antithetic sampling, centered ranks and the existing Adam update. Default world configuration, both lifetime learning rules and structural growth off remain unchanged. Each base world is replayed under both stationary food rules with fresh lifetime state; take each individual's lower return before averaging agents and worlds. Include every starting agent, including deaths.

Run order: control 0, candidate 0, control 1, candidate 1, control 2, candidate 2. Checkpoint every 10 generations. Evaluate generation 20 only after all six finals are valid; no best-checkpoint selection or interim held-out evaluation. The shorter budget applies to both new arms. Historical E115 finals are not the experimental control for this comparison.

## Fresh evaluation and decision

Register **EVAL v6**, preserving all older protocols. Use seed **2026112001**, 256 base worlds, and the same four stationary/reversal conditions and individual measurements as EVAL v5. Evaluate all nine genomes—three unchanged starts and six finals—plus fixed A, fixed B, mixture, random walk and rule-knowing greedy once each: **14 panels**. Software fixtures use seed 41 only.

Fix 30 descriptive contrasts before evaluation: for each original seed, candidate minus control and candidate minus unchanged start for the five existing paired metrics. Calculate differences within each matched base world and their sample-SD/sqrt(256) SE. Do not pool evolution seeds or treat agents/rule replays as independent samples. Report per-rule fitness, food/poison, encounters, survival, individual outcomes and reversal results regardless of the decision.

The numerical improvement gate passes only if all artifacts are valid and, **in each of all three seeds**:

1. Candidate individual minimum return exceeds the matched control by strictly more than two paired-world SE.
2. Candidate individual minimum return exceeds its unchanged start by strictly more than two paired-world SE.
3. Candidate-minus-start joint stationary survival is at least minus two paired-world SE.

Otherwise the completed study fails its improvement gate. Equality fails the return checks and is allowed at the survival boundary. This is an operational research gate, not a corrected significance test or proof of equivalent survival. A decline inside the survival uncertainty remains unresolved.

**No deliberate survival-for-return trade-off is accepted.** Beating a damaged control, preserving the ancestor better without improving it, or improving a reversal metric cannot substitute for this gate. Any descriptive stability benefit is reported separately. Passing would establish a bounded improvement under these settings, not explain all E115 losses, establish causal within-life learning, demonstrate a plasticity advantage, or pass the randomized-reversal rung.

## Fixed cost and stop rule

Cap the entire study at **10,800 seconds (three hours)** of supervised compute, including verification/CI, benchmark, training, evaluation and retries. Reserve **2,400 seconds** for software/CI and **1,200 seconds** for the final evaluation; training and setup share the remaining **7,200 seconds**. Stay within the owner's eight-hour daily limit as well.

The first control-0 segment is a runtime benchmark within its fixed 20 generations: 120-second training limit and 600-second full-process cap. Subsequent training invocations have 900-second training and 1,200-second full-process caps; resume only the same fixed target and preserve history. The final diagnostic has a 1,200-second full-process cap. Before releasing the rest, forecast all remaining work using 1.5 times the slowest measured warm generation, charging setup/compilation per invocation and retaining the full verification/evaluation reserves. Runtime alone determines feasibility.

Keep one low-priority heavy job at a time, temporary files on D:, startup physical-memory/commit/C:/D: floors of 8/10/10/10 GiB and runtime commit/C:/D: floors of 3/5/10 GiB. Do not stop services or alter power settings under this proposal. Count supervisor-observed elapsed time, including any standby interval, against the allowance.

Stop scheduling if resource guards or artifact validity fail, or the fixed study cannot fit the remaining allowance. Preserve incomplete attempts and leave the scientific gate undecided until a valid complete comparison exists. No automatic retry, budget extension, smaller population, extra generations, alternative learning rate or new mechanism. After any completed gate decision, report the outcome and return to the owner before another experiment.
