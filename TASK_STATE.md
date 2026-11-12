# Current continuation state

Updated 2026-11-12 (Asia/Calcutta). This mutable handoff records current decisions. Read [AGENTS.md](AGENTS.md), [HANDOFF.md](HANDOFF.md), and the append-only [research history](experiments/LOG.md) for detailed results, incidents, commands, and verification evidence.

## Goal and authorized scope

Active goal, verbatim as reconfirmed by `get_goal`:

> Always try to work in parallel!!
> keep your major focus on experimentation and finding out the right direction and the evolution should take place and our theory should be moving ahead. and whatever side tasks are there like git pushing, cleaning, dir arranging, documenting make sure to assign these tasks to subagents or other parallel agents!! and make sure that you implement a self learning loop inside you such that you also keep learning as what you are doing and keep optimizing stuffs

- Goal active. Continue preregistered A/B/C work and authorized pushes to private `Winter-Soren/seed`. The owner authorized overnight/continued compute beyond the earlier approximately eight-hour daily threshold. Keep one long compute job at a time; delegate independent support work.
- Record evidence → interpretation → decision → next experiment. This is an operational learning loop, not assistant model-weight changes. Preserve every run, including failed and smoke runs; never rewrite past LOG entries or manually edit run artifacts.
- Keep EVAL v1 fixed: `test_genome`, 128 worlds, evaluation seed 123. No roadmap expansion before core B/C comparisons; changes to goals or evaluation need owner authorization and protocol versioning. Preserve unrelated owner/Oracle workloads and private repository visibility.
- Runtime: Windows, Python 3.12.10 via `.\.venv\Scripts\python.exe` on D:, JAX/jaxlib 0.12.0.1, NumPy 2.5.4, CPU. No verified CUDA path; no installation is pending. Generate RESULTS with `.\.venv\Scripts\python.exe -X utf8 scripts/summarize.py`; the default cp1252 encoding cannot write its Unicode header.

## Established result and interpretation

- E100 references, E101 smoke, and E102 benchmark are complete. E103 A completed all three 200-generation seeds and passed its preregistered gate in **3/3** (required at least 2/3). Result commit `51bdac9` and milestone `v0.1-a-evalv1` are pushed and verified; README contains the three-seed result and limits. A is not awaiting A2 and its failure-escalation queue does not apply.
- A learning fitness by seed: 16.591796875 / 15.376953125 / 14.8671875. Accuracy gains: 0.13528215885162354 / 0.14218151569366455 / 0.12878018617630005. Learning-minus-frozen fitness gaps: 16.302734375 / 15.283203125 / 14.970703125. Full evaluations and artifact checks are in [LOG](experiments/LOG.md).
- This is a replicated result in the toy domain, not general intelligence or proof every successful policy needs plasticity. The lesion jointly disables Hebbian and predictor updates; recurrent state and energy feedback remain. Pooled eating accuracy can reflect survival/encounter selection. Deterministic biology-reference repeats are reproducibility checks, not independent evidence.

## Completed B0; C0 next

- Run: `runs/20261112-131946_E104-B-bio-s0`. Biology start, seed 0, target 200, sigma 0.05, learning rate 0.02, population 128, eight training worlds, 128 evaluation worlds, plasticity enabled, no overrides. Start matches E100 biology seed 0, hash `72caddf0cdd93943`.
- **Finished at generation 200**, final hash **`1a34591381de0fbb`**, EVAL v1 128 worlds. Parent verified finite checkpoint 200 == final genome, 200 contiguous finite rows 0–199, exact E100 starting genome/evaluation, and both unchanged metric prefixes. Initial clean source `51bdac97c7ee65cc5e6bd7249d78a6bd0152ff9f`; clean resumes `f12d3debc56518a029832f4231f34dfbb7d5c977` and `f05751334da36f3cfeb3b7081b9ea1e4831733af` verified 3- and 8-row replay before writes. Engine/settings/EVAL unchanged; incident details remain in LOG.
- **B0 meets the per-seed improvement criterion:** learning fitness **15.90234375** versus prior **5.12890625** (difference **10.7734375**), frozen **-0.048828125** (lesion gap **15.951171875**), accuracy **0.685681164264679 → 0.8353021144866943** (gain **0.14962095022201538**). This is one successful seed; B's gate awaits all three seeds. No new tag or roadmap advancement.
- Segment 3 exited **0**, wall **5,230.8544356 seconds**, ending **2026-11-12 22:35:59.0316370 Asia/Calcutta**. Final `train_minutes=86.56` is segment 3 only; metric seconds total **9,563.099**. Do not claim exact cumulative wall time: segment 2 only has a guard observation at elapsed 2,506 seconds, and replay compute is additional to retained timings. Supervisor **20596**, monitor **g9:35**, and follow-up polls are terminal; authoritative CIM found zero experiment processes. No job is running; goal active.
- Final metrics: **41,083 bytes**, SHA-256 **`a3feb80882288dcddf2dd41beaf0e0e036b5439c6d3b19cafaf69047cbb94118`**. Preserved **17,940-byte prefix** SHA-256 **`fcf36a7062df0ff447e017b30fd9a1f581b8e1c03ed467d5c6c752645df69adc`** and original **6,723-byte prefix** SHA-256 **`3d093802247823f26518fa58c1f417b87e69551d36544e1ac5bbda180d7e074c`** both verified through completion.
- Outputs `.tmp/E104-B-bio-s0-segment3.stdout`/`.stderr` and earlier segments remain preserved. RESULTS regenerated with 12 completed rows; README result update is ready and the coordinator authorized coherent B0 publication. No manual run edits.

## Compute guards and next action

- Startup requires C: and D: free space at least **10 GiB each**, free physical memory at least **8 GiB**, and available commit at least **10 GiB**. B0 completion snapshots: C: **23.94/373.67 GiB**, D: **46.34/100 GiB** free; physical/available commit **12.90/15.48 GiB**. Recheck before C0's clean-source launch. Earlier owner app closure and already-stopped Oracle observations are in LOG; no additional workload stop is authorized by these snapshots.
- Runtime floors: **C: 5 GiB, D: 10 GiB, available commit 3 GiB**. Supervisor uses authoritative read-only memory counters with ten-second waits plus query overhead. Counter errors or breached floors invoke checked owned-process `.Kill($true)` cleanup; wait and verify tree absence. Do not reuse the failed native `taskkill` approach; detailed evidence is in LOG.
- One `BelowNormal` compute job, temporary files on D:, checkpoint every ten generations, `--max-train-seconds 7200`, full-wall limit **10,500 seconds** (under three hours). The training budget excludes setup/evaluation and may overrun one generation. Resume the same folder and unchanged target in bounded segments when needed; segment `train_minutes` is not total experiment time.
- **Next:** publish the completed B0 result, then hold tracked writes through fresh preflight and C0 launch. Command: `.\.venv\Scripts\python.exe scripts/run_experiment.py --name E105-C-noplastic-s0 --start blank --no-plastic --generations 200 --seed 0 --max-train-seconds 7200`. Goal remains active; no new scientific settings.
- **Remaining queue: C0 → C1 → C2 → B1 → B2.** B0 is complete. This reorders existing preregistered runs only; preserve every run, setting, and gate. C takes priority after A's 3/3 pass because it compares independently evolved nonplastic strategies under otherwise matched settings. B asks whether gentle ES improves the biology prior and uses different ES parameters.
- Compare paired A `final_learning` minus C **`final_frozen`** for all three seeds; C `final_learning` re-enables plasticity and is diagnostic only. C has no new numerical gate; report mixed differences as mixed. B passes only if final-learning fitness strictly exceeds **5.12890625** in at least 2/3 seeds after all three complete.
- Separate Hebbian-versus-predictor, reward-modulation, and survivor-aware diagnostics remain possible future questions. Their implementations/runs are **not authorized now**; they require separate preregistration, and evaluation changes require owner approval/versioning.

## Git identity and provenance

- Repository remains private: `Winter-Soren/seed`. Use verified repo-local identity **Winter-Soren** with **81645360+Winter-Soren@users.noreply.github.com** for author and committer through plain `git commit`; never add generic identity overrides. Verify both GitHub account attributions after pushing.
- Authorized historical metadata repair is complete: 19 repaired commits plus audit tip attributed to Winter-Soren; 21 mapped historical trees/messages/timestamps/topology were preserved, including the signed root. Use [the committed mapping](experiments/COMMIT_ATTRIBUTION_MAP_20261112.json) to resolve unchanged historical run/source hashes. Old history remains under `refs/provenance/pre-author-repair-20261112` and in the verified ignored bundle. Do not repeat the repair or rewrite history.
- Authorized coherent result publication includes the completed **B0 folder + RESULTS + LOG + TASK_STATE + README** only; all owners are holding writes. Exclude figures/source/other changes. Never manually edit run files. Detailed old runs, failed-process handling, cleanup, and timing corrections remain in [LOG](experiments/LOG.md).
