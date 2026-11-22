# E130 · Did evolution improve autonomous relearning?

Can the three original E103 descendants, after 200 generations of selection in stationary worlds, learn a changed food rule from their own actions better than their respective founders? This is a fresh confirmation and extension of E112. Its three blank references have the same file checksums as the corresponding E103 founders. E112 already reported late post-switch contact accuracy near one half in the descendants. It did not register the paired founder–descendant recovery comparison below. E126's unfinished training and sealed worlds remain separate. [E112 measurements](../results/EVAL_V3.md)

Mice in a two-lever experiment adjusted their choices after reward probabilities changed, using outcomes of choices they had made themselves. Seed can test a narrower ability: recovering from a single unannounced deterministic change while finding items in a grid. Passing would establish that behavior here. Mammals differ widely, and a single behavioral comparison cannot establish a species-wide intelligence level. [Mouse experiment](https://pmc.ncbi.nlm.nih.gov/articles/PMC4882228/)

## Design

Use the saved E103 generation-zero and generation-200 genomes of lineages 0, 1 and 2. They encountered stationary rules during evolution. Use the standard 16 × 16 grid, four originally assigned creatures, 300 turns, ordinary plasticity and ordinary prediction. Animals choose every move. In one replay the rule stays fixed; in the other it reverses before consumption at zero-indexed turn 150. Give each family both initial food rules and pair its world and development keys across all eight replays. At birth the two genotypes can develop different brains from the same randomness. Within each genotype and initial rule, the event histories must match up to the switch. There is no imposed pause in learning.

Use four development families, seed 2026120900. Use 64 independent final families in each of three panels, seeds 2026120901, 2026120902 and 2026120903. Four creatures, two rules and the eight replays within a family are paired observations. Never count them as additional independent samples. Include every assigned creature, even if it dies or finds no item. No evaluation world is used for development.

## Outcomes and decision

For each creature in each successive 50-turn window, record its first item as +1 for food, −1 for poison and 0 for no item. A death before contact contributes 0. Call the resulting family average **Q**. Publish the first-food, first-poison and no-contact rates alongside Q and end-of-life survival. Index windows from zero. Use window 2 for the established preference, 3 for early post-switch and 5 for late post-switch. Average the four creatures and both initial rules inside each family, form each paired contrast there, then calculate its mean and sample standard error across families. Report lineage results separately.

Each lineage must meet every criterion below. “Above noise” means the mean is strictly greater than twice its own family-level standard error.

1. Generation-200 Q is above noise both immediately before the switch and in the late reversal window.
2. In late reversal, descendant Q exceeds founder Q by at least 0.05 and the difference is above noise.
3. In reversal, descendant Q rises from early to late by at least 0.05 and the rise is above noise.
4. For each genotype, take the early-to-late increase in reversal Q and subtract its early-to-late increase under the stationary rule. The descendant's value must exceed the founder's by at least 0.05 and that paired difference must be above noise. This checks whether ordinary changes over a life account for the greater recovery.
5. For each of pre-switch Q, stationary late Q and final survival in **each** schedule, the paired descendant-minus-founder mean minus two standard errors must exceed −0.05.

All three lineages must pass for a replicated evolutionary improvement claim. Report the individual components if the joint gate fails. The study compares two endpoints of evolution. It cannot establish steady improvement from generation to generation. All four creatures share a grid, so resource competition and access to feedback can influence the outcome.

## Release and stopping rules

Use the tracked runner, commit the implementation before scientific runs, and keep raw outputs immutable. First verify synthetic indexing, paired keys, first-item and death handling, then time the four-family development panel. Inspect final behavioral results only after all three panels pass independent reconstruction. Development must demonstrate sufficient headroom for the whole schedule before opening final worlds. Preserve failed and incomplete records. Do not change the sample, thresholds or dates of the switch in response to an outcome.

Allow 4,800 seconds total for E130 including software verification, development, all three panels, the audit and reporting. Allow 180 seconds for the rehearsal and at most 450 seconds per formal panel. Use the existing laptop safeguards and the daily eight-hour limit. The whole-program question remains open at either outcome. If this test fails, the council will use the failed components to identify the next distinct ability or obstacle worth testing.
