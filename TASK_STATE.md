# Current state

Updated 21 November 2026. Follow PLAN.md and AGENTS.md.

## What we know

Three evolved A genomes improve their food choices during a stationary lifetime. From the first to twentieth meal, accuracy rose by 27, 21 and 25 percentage points, each with about two points of standard error among agents that reached both meals.
They struggled after a change in the food rule. A fixed-weight recurrent baseline won the original average score in all three seeds.
Selecting for the worse of two food rules in E115 produced lower return than all three unchanged starting genomes.
E116 compared smaller and larger evolutionary steps for another twenty generations. Smaller steps beat larger ones by 1.30, 0.88 and 1.15 points, with paired standard errors of 0.42, 0.41 and 0.41.
The smaller steps missed the required improvement over the starting brains in all three seeds. All three survival safeguards passed. The E116 improvement gate failed.

## What is running

All E116 training and held-out evaluation are finished. The independent audit reconstructed every comparison from the saved events.
Three newborn cohorts ran in worlds with visible shelter, cold exposure and a cost to moving. Earlier experience changed later newborns' starting weights.
One of two newborns reached shelter in each group in the cold world. All agents survived this fixture. A separate test checks that a creature that dies contributes only its last living weights.
The optional persistence trait passed its local checks. In the cold world, the shared-memory arm pursued an out-of-sight shelter for four agent-turns. One of two creatures reached shelter with the trait enabled and one of two with it disabled. All survived.

## What is next

Keep the three original A genomes as the reference learners.
The owner approved the three-seed cold-donor test, E117. Its notebook entry fixes the comparison before evaluation.
Benchmark the full life on development worlds, then run each original genome on 256 new world families and independently check first living shelter entries and survival. The two newborn groups share their world, wiring and random stream. Only their starting weights differ. Both have the optional persistence trait off.
Keep one laptop job at a time and check both drives and memory before each launch.
