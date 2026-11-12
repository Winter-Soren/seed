# Current continuation state

Updated 2026-11-12 (Asia/Calcutta). This mutable handoff records current decisions. Read [AGENTS.md](AGENTS.md), [HANDOFF.md](HANDOFF.md), and the append-only [research history](experiments/LOG.md) for detailed results, incidents, commands, and verification evidence.

## Goal and authorized scope

Active goal, verbatim as reconfirmed by `get_goal`:

> Always try to work in parallel!!
> keep your major focus on experimentation and finding out the right direction and the evolution should take place and our theory should be moving ahead. and whatever side tasks are there like git pushing, cleaning, dir arranging, documenting make sure to assign these tasks to subagents or other parallel agents!! and make sure that you implement a self learning loop inside you such that you also keep learning as what you are doing and keep optimizing stuffs

- Goal active. Continue preregistered A/B/C work and authorized pushes to private `Winter-Soren/seed`. The owner authorized overnight/continued compute beyond the earlier approximately eight-hour daily threshold. Keep one long compute job at a time; delegate independent support work.
- Record evidence → interpretation → decision → next experiment. This is an operational learning loop, not assistant model-weight changes. Preserve every run, including failed and smoke runs; never rewrite past LOG entries or manually edit run artifacts.
- Keep EVAL v1 fixed: `test_genome`, 128 worlds, evaluation seed 123. No roadmap expansion before core B/C comparisons; changes to goals or evaluation need owner authorization and protocol versioning. Preserve unrelated owner/Oracle workloads and private repository visibility.
- Runtime: Windows, Python 3.12.10 via `.\.venv\Scripts\python.exe` on D:, JAX/jaxlib 0.12.0.1, NumPy 2.5.4, CPU. No verified CUDA path; no installation is pending.

## Established result and interpretation

- E100 references, E101 smoke, and E102 benchmark are complete. E103 A completed all three 200-generation seeds and passed its preregistered gate in **3/3** (required at least 2/3). Result commit `51bdac9` and milestone `v0.1-a-evalv1` are pushed and verified; README contains the three-seed result and limits. A is not awaiting A2 and its failure-escalation queue does not apply.
- A learning fitness by seed: 16.591796875 / 15.376953125 / 14.8671875. Accuracy gains: 0.13528215885162354 / 0.14218151569366455 / 0.12878018617630005. Learning-minus-frozen fitness gaps: 16.302734375 / 15.283203125 / 14.970703125. Full evaluations and artifact checks are in [LOG](experiments/LOG.md).
- This is a replicated result in the toy domain, not general intelligence or proof every successful policy needs plasticity. The lesion jointly disables Hebbian and predictor updates; recurrent state and energy feedback remain. Pooled eating accuracy can reflect survival/encounter selection. Deterministic biology-reference repeats are reproducibility checks, not independent evidence.

## Active B0 continuation

- Run: `runs/20261112-131946_E104-B-bio-s0`. Biology start, seed 0, target 200, sigma 0.05, learning rate 0.02, population 128, eight training worlds, 128 evaluation worlds, plasticity enabled, no overrides. Start matches E100 biology seed 0, hash `72caddf0cdd93943`.
- Initial source was clean `51bdac97c7ee65cc5e6bd7249d78a6bd0152ff9f`. First segment failed on a JAX memory allocation; its evidence is preserved in `f12d3de`. See [LOG](experiments/LOG.md) for the incident; later memory counters do not establish its exact cause.
- **Segment 3 running:** started **2026-11-12 21:08:48.1749019 Asia/Calcutta**, launcher PID **28012**, supervising session **20596**, from clean **`f05751334da36f3cfeb3b7081b9ea1e4831733af`**. Use supervisor 20596 as the definitive ongoing handle; completion of bounded monitor cell g9:18 does not mean the experiment ended. Goal active; no B final evaluation or gate result.
- Current output: `.tmp/E104-B-bio-s0-segment3.stdout` and `.tmp/E104-B-bio-s0-segment3.stderr`; stderr was empty at verification. Earlier segment outputs remain. Segment 2's supervisor 57093 and monitor g9:11 are terminal after the verified memory-guard stop; see LOG for evidence.
- **Recovery verified:** resume metadata at **21:08:50.452534** records `dirty=false`, `replayed_generations=8`. Recorded rows 80–87 reproduced before writes; **89 contiguous rows 0–88** were then observed. This is a verification snapshot, not the latest generation count. The earlier three-row replay of 30–32 also passed; both recoveries preserved recorded scientific metrics.
- **Guard-stop baseline preserved in f057513:** finite checkpoint **t=80**, hash **`a2ab513f53247ece`**; 88 finite rows 0–87, retained metric seconds 4,750.455. The **17,940-byte metrics prefix**, SHA-256 **`fcf36a7062df0ff447e017b30fd9a1f581b8e1c03ed467d5c6c752645df69adc`**, matched unchanged after the eight-row replay. Preserve it through completion without duplicates.
- Original **6,723-byte metrics prefix** also matched unchanged, SHA-256 **`3d093802247823f26518fa58c1f417b87e69551d36544e1ac5bbda180d7e074c`**. No final evaluation exists. Engine, scientific settings, and EVAL v1 are unchanged; recovery is operational evidence, not a scientific result.

## Compute guards and next action

- Startup requires C: and D: free space at least **10 GiB each**, free physical memory at least **8 GiB**, and available commit at least **10 GiB**. Segment 3 met these: C:/D: 24.00/46.34 GiB, physical/available commit 12.21/15.71 GiB. At 312 seconds: C:/D: 23.99/46.34 GiB, physical/available commit 9.72/12.66 GiB. The owner closed apps and authorized an Oracle stop; the worker found Oracle already stopped with no `oracle.exe`, so took no action. Browsers were untouched. These counters do not identify the cause of earlier pressure.
- Runtime floors: **C: 5 GiB, D: 10 GiB, available commit 3 GiB**. Supervisor uses authoritative read-only memory counters with ten-second waits plus query overhead. Counter errors or breached floors invoke checked owned-process `.Kill($true)` cleanup; wait and verify tree absence. Do not reuse the failed native `taskkill` approach; detailed evidence is in LOG.
- One `BelowNormal` compute job, temporary files on D:, checkpoint every ten generations, `--max-train-seconds 7200`, full-wall limit **10,500 seconds** (under three hours). The training budget excludes setup/evaluation and may overrun one generation. Resume the same folder and unchanged target in bounded segments when needed; segment `train_minutes` is not total experiment time.
- **Next:** monitor B0 through supervisor 20596 toward its unchanged 200-generation target, preserving both metric prefixes. Commit only this LOG/TASK_STATE recovery update, excluding live artifacts and other work; then hold documentation writes until outcome. At outcome, verify and preserve artifacts before recording results. Goal remains active.
- **Queue: B0 → C0 → C1 → C2 → B1 → B2.** This reorders existing preregistered runs only; preserve every run, setting, and gate. C takes priority after A's 3/3 pass because it compares independently evolved nonplastic strategies under otherwise matched settings. B asks whether gentle ES improves the biology prior and uses different ES parameters.
- Compare paired A `final_learning` minus C **`final_frozen`** for all three seeds; C `final_learning` re-enables plasticity and is diagnostic only. C has no new numerical gate; report mixed differences as mixed. B passes only if final-learning fitness strictly exceeds **5.12890625** in at least 2/3 seeds after all three complete.
- Separate Hebbian-versus-predictor, reward-modulation, and survivor-aware diagnostics remain possible future questions. Their implementations/runs are **not authorized now**; they require separate preregistration, and evaluation changes require owner approval/versioning.

## Git identity and provenance

- Repository remains private: `Winter-Soren/seed`. Use verified repo-local identity **Winter-Soren** with **81645360+Winter-Soren@users.noreply.github.com** for author and committer through plain `git commit`; never add generic identity overrides. Verify both GitHub account attributions after pushing.
- Authorized historical metadata repair is complete: 19 repaired commits plus audit tip attributed to Winter-Soren; 21 mapped historical trees/messages/timestamps/topology were preserved, including the signed root. Use [the committed mapping](experiments/COMMIT_ATTRIBUTION_MAP_20261112.json) to resolve unchanged historical run/source hashes. Old history remains under `refs/provenance/pre-author-repair-20261112` and in the verified ignored bundle. Do not repeat the repair or rewrite history.
- This recovery record includes **LOG and TASK_STATE only**; do not stage live B0 artifacts, README, figures, source, or other workers' changes. Never manually edit run files. Detailed old runs, failed-process handling, cleanup, and timing corrections remain in [LOG](experiments/LOG.md).
