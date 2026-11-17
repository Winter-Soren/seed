# Current state

Updated 17 November 2026. Follow [PLAN.md](PLAN.md) in order under [AGENTS.md](AGENTS.md).
The thesis is a question to test, not a result to assume.

## What is known

- A met its original stationary-world criterion in all three evolution seeds.
- B0 and B1 finished: fitness 15.90234375 and 16.6015625 versus the prior's 5.12890625.
- These point estimates have no standard errors; B's gate stays open until B2 finishes.
- Adding the tested structural rule lowered observed fitness in all three saved A genomes.
- E109 is stopped by owner direction at 62 of 200 generations, with status interrupted.
- Its folder remains untouched; no final evaluation or third training segment occurred.
- Historical results lack reported per-world standard errors; Phase 1 will remeasure them.
- The external audit using seed 777 is unregistered and must not be treated as verified.
- C is historical evidence within its encoding; Phase 2 will provide the later comparison baseline.

## What is running

- B1 finished all 200 generations and its supervisor exited normally.
- Its original history, registered settings and starting genome are preserved.
- B2 is continuing from its preserved checkpoint at 161 of 200 generations, without replay.
- No later plan item has started.
- The B2 worker monitors the sole heavy job; final evaluation is still pending.
- Latest check: C: has 16.31 GiB free of 373.67 GiB; D: has 46.29 GiB free of 100 GiB.
- Fresh startup checks passed; the live guard monitors memory, disk space and wall time.

## What is next

- Phase 0 item 1 is recorded: E109 stays stopped; there is no continuation queued.
- Phase 0 item 2 is active: finish B2 exactly as registered in E104, then close gate B.
- Before any continuation, reassess today's compute and ask before exceeding the eight-hour limit.
- B passes only if at least two of all three seeds strictly exceed fitness 5.12890625.
- If B fails, report it plainly and stop for owner input.
- After B is decided, finish Phase 0 dependency pinning, dirty-check coverage and CI.
- Keep this handoff below 60 lines and benchmark one generation on a free Colab GPU.
- If the GPU is at least five times faster, propose migration and wait for the owner.
- Later phases follow PLAN.md; no new mechanism or side experiment starts early.

## Working rules

- Preregister and commit before running; never alter past LOG entries or run artifacts.
- Use three seeds for claims, one heavy job at a time, and less than three hours per run.
- Ask before stopping any laptop service, every time; earlier blanket permission no longer applies.
- Keep the repository private and follow every AGENTS.md ask-first rule.
- Create no new pull requests or routine pull-request reviews until the owner re-enables them.
- Stop for a failed gate, ambiguity, infeasibility or required owner decision.
- Review findings must be fixed or answered in the LOG; only the owner can change the plan.
- Give the five-line report from PLAN.md after each phase, with measured uncertainty.
- Detailed evidence and operational records belong in [experiments/LOG.md](experiments/LOG.md).
