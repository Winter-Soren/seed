# E131 · Does it help to explore after a long stretch without food?

E130 found that unguided descendants recovered some of their old food preference after a hidden reversal but failed to prefer the new food late. In two lineages, at least one fifth of the creatures alive at the switch experienced a specified 25-turn gap without any item. This selected exploration under the prior [architecture decision](ARCHITECTURE_COUNCIL_E130.md). The gaps do not establish their cause. We will test whether an agent can use the energy change it actually feels to time a small amount of exploration productively.

## Three policies

Use the original E103 generation-200 genome for each lineage. Keep the native food world, four creatures, observations, predictor, Hebbian learning, ordinary stochastic policy and actual-action feedback. Do not train or choose a genome in this study. Compare three policies from identical assigned worlds and birth randomness.

- **Ordinary.** Take the native sampled action.
- **Adaptive exploration.** Each animal carries a small counter, initially zero. On every living turn after birth, a positive realized energy change from its preceding action resets the counter. Otherwise increase it by one up to 12. Before the current action, use the updated counter. It persists across blocks. Spend one exploratory move on the first eligible turn in a block when the counter is 12. If that never happens, spend the move on the last turn of the block.
- **Equal-quota exploration.** Spend one exploratory move at a uniformly sampled offset in each block. Its timing is independent of actions, outcomes, life expectancy and the food rule.

An exploratory move overrides the original sampled action with a uniformly drawn cardinal direction. The policy may happen to have sampled that same move. Record both the override and whether it changes the action. Both exploration policies use the same preassigned direction for the same animal and block and feed the actually executed move back to the brain. Use separate random streams for initial phase, block offsets and directions, separate from native world and birth randomness. Each creature's first block starts after a uniformly assigned delay of 0–24 turns, paired across policies, initial rules and schedules. This prevents every block clock from restarting when the world switches at turn 150. There are no overrides before the first block. A fully lived block lasts 25 turns and spends exactly one override. A partial or death-truncated block can spend zero or one. Record realized override counts per assigned creature and per live turn.

Only the agent's observed post-clipping energy change enters the counter. Food near the energy cap can produce zero felt gain. The controller receives neither the consumed-item label nor the reversal flag or time. Ordinary learning is active throughout. This tests an engineered change to action selection. Even a successful policy may improve exposure without improving what the brain learns from identical experience.

## Worlds and measurement

Use four development families with seed 2026121000. Evaluate the three E103 lineages separately on 64 families each with fresh seeds 2026121001, 2026121002 and 2026121003. In each family replay both initial food rules and both stationary and hidden reversal schedules. Reversal occurs before consumption on turn 150 of 300. These are 12 replays per family. Within every independent family, first average its four originally assigned animals and both initial rules. Form contrasts there and use the sample standard error across families. Report every lineage separately.

Use the six E130 50-turn windows. Q equals +1 for the first item being food, −1 for the first item being poison and 0 for no contact, including after death. The late window is [250,300). Time to first food in [150,200) is its zero-based offset 0–49 or 50 if no food is found. All originally assigned creatures count, including those already dead. Report time to first item of either kind as a separate descriptive outcome. Report absolute first-food, first-poison, no-contact and survival rates. Verify that the hidden switch occurs after identical pre-switch histories within the same mode, and verify equal realized quota for complete live blocks.

## Decision before measurement

Require every condition in each of the three lineages. Estimate changes against **both** the ordinary policy and the equal-quota policy independently on the same families.

1. Adaptive late reversal Q must be positive and exceed two family-level standard errors.
2. Time to first post-switch food must be shorter under adaptive exploration than under each control. For each difference, the mean time saved must exceed twice its paired standard error.
3. For each control, adaptive minus control late reversal Q must exceed twice its paired standard error. Also take that difference on reversal worlds and subtract the corresponding difference on stationary worlds. This second contrast must be at least 0.05 and exceed twice its paired standard error. The first condition prevents a stationary loss alone from producing an apparent reversal benefit.
4. For each control, the lower bound for adaptive minus control must exceed −0.05 for stationary late Q, stationary final survival and reversal final survival. The bound is the paired mean minus twice its standard error.

Keep a separate engineering criterion. On development families 1–3, after the first complete cold family, the median synchronized time for the adaptive policy must be at most 1.5 times the ordinary-policy median for identical workloads. Report the equal-quota time too. The nominal parameter count does not establish speed or memory cost. The science audit checks eligibility, timing, action feedback, the complete evidence and family-level statistics. It cannot infer why any individual creature moved.

If development, the cost limit or the complete-study budget fails, leave the final worlds sealed and report the exact obstacle. Adoption requires the behavioral and engineering criteria. If both exploration policies help equally, the timing rule has not earned its extra state. If the behavioral gate fails or is inconclusive, retire this setting from the improvement sequence. No new counter threshold, quota, extra seeds or rescue analysis on these worlds.

Use the standard tracked entry point, one heavy job at a time, and an inclusive budget of 5,400 seconds. Allow 240 seconds for readiness, 200 for development, 120 for its audit, 600 per final panel, 300 for joint audit, a single 2,100-second hosted job, 60 for reporting and 580 for forecasting and operating margin. The daily limit still applies. All releases require actual storage, memory and timing checks on the laptop.
