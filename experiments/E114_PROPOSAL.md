# E114 proposal: individual adaptation versus specialist mixtures

**Approved and completed on 19 November 2026.** All 11 panels passed the measurement-validity audit; see [the EVAL v4 report](../results/EVAL_V4.md). This document preserves the approved proposal; E114 entries in LOG.md supply preregistration and execution details. E113's failed gate remains unchanged.

## Question

Do the saved controllers perform across both food rules as individual agents, or can their population averages be explained by different specialists succeeding under different rules? E113's original fitness and gate remain unchanged. This diagnostic would inform a later objective proposal; it would not promote a capability or authorize further evolution.

## Fixed scope proposed for approval

- Add a separately versioned **EVAL v4 paired diagnostic**, leaving EVAL v1/v2/v3 unchanged.
- Use fresh evaluation seed **2026111902**, with 256 independent base worlds and the default world, observations, actions and 300-step lifetime.
- For each base world, replay the same starting grid, positions, agent development/identity and random streams under four conditions: initial rule 0 or 1, each stationary or reversed at step 150. Starting conditions are paired; subsequent trajectories may diverge.
- Evaluate the six saved controllers A0/A1/A2 and direct0/direct1/direct2. Preserve saved genome bytes. A retains its existing lifetime plasticity; direct weights remain fixed. No evolution, altered learning rule, new mechanism or changed reward.
- Evaluate five reference policies once each: fixed A specialist, fixed B specialist, a family containing two of each with color assignments fixed across paired lives, uniform random walk, and the existing local rule-knowing greedy policy.
- Specialists use the existing greedy navigation semantics with a fixed target color instead of access to the true food rule; the other color is always treated as undesirable. Their target does not change at reversal. The ordinary greedy reference still knows the true current rule, a privileged input disclosed in reporting.

## Measurements

1. Retain original mean fitness and fitness conditioned on each initial rule.
2. For each starting individual, take the minimum of its two matched stationary returns, then aggregate those individual minima. Do not take a minimum after averaging the family: that could conceal opposite specialists.
3. Report joint survival under both stationary rules, survival at reversal and final survival, using all starting agents as the denominator.
4. Retain food/poison counts, 50-step post-reversal bins, deaths and zero-encounter counts. Accuracy without encounters remains undefined, never silently dropped or set to zero.
5. Publish per-agent paired outcomes/distributions and raw events. Calculate standard errors with the base world as the independent cluster, since agents and four replay conditions are dependent. Report every saved seed separately; do not pick a best controller.

Declare the complete set of paired descriptive contrasts in the eventual preregistration. No numerical capability threshold will be selected from these outcomes.

## Gates and interpretation

The proposed gate is **measurement validity only**: all planned conditions complete; starting-state/random-stream/development pairing is verified; specialist assignments and policy semantics are fixed; saved genomes remain unchanged; event accounting and statistics reconstruct; missingness is explicit. A failed guard, incomplete output or invalid pairing is preserved and reported as incomplete/failed measurement.

The specialist mixture is a counterexample to rule-balanced population averages. Privileged greedy is a counterexample to calling successful behavior learning: it can succeed without inferring the rule from experience. Individual paired minima address the mixture failure, but are not themselves a complete learning objective. Demonstrating dependence on informative experience would require separately approved history/feedback interventions. This diagnostic would not establish a causal learning mechanism, useful structural growth or broad transfer.

## Compute and publication

The fixed scope is 11 controllers × 256 base worlds × 4 conditions = **11,264 family lifetimes**. Reserve at most **30 minutes of diagnostic execution**, plus **15 minutes of software verification**; one guarded, low-priority laptop job at a time, existing memory/disk floors and D: temporary files. Benchmark the first released panel within this budget; stop and preserve partial output if the bound or a guard is reached. Timing is an estimate, not a measured guarantee. No services are stopped.

Preregister and commit first, retain every attempt, then publish raw runs, summary, LOG, README and a five-line phase report under the owner's verified identity. No training or additional protocol is included in this approval request.
