# Seed lab notebook

This log is append-only. Each entry has: ID · date · hypothesis · command or setup · pass/fail criterion · result · interpretation · decision.
Entries E000–E007 come from the Claude design session, before this repo existed. They were run on a 2-core CPU container, mostly with **32 evaluation worlds** (not EVAL v1's 128), and some used code versions that predate the final v0. Treat them as pilot evidence, not results.

---

## E000 · 2026-11-10 · Pipeline smoke test
- **Setup:** v0 draft (12+12 blobs, energy cost 0.005/step, no efference copy). 6 gens, pop 16, 2 worlds.
- **Result:** Runs end to end, about 2 s/gen on CPU. Survival at end of life was about 0.2 to 0.3.
- **Decision:** Scale up the pilot.

## E001 · 2026-11-10 · Does learning evolve in the first draft?
- **Hypothesis:** ES discovers within-life learning from a blank start.
- **Setup:** Same draft. Pop 32 × 4 worlds. Stopped at gen 30 (of 80).
- **Result:** Fitness noisy around 0. Within-life accuracy about 0.49 → 0.52. Survival about 0.23.
- **Interpretation:** Most agents starved before they had enough eating events to learn from.
- **Decision:** A denser world and a lower energy cost (24+24 blobs, 0.003/step). These are now the v0 defaults.

## E002 · 2026-11-10 · Denser world, blank start
- **Setup:** 24+24 blobs, cost 0.003, no efference copy. 50 gens, pop 32 × 4.
- **Result (32 eval worlds):** evolved 0.23 · frozen 0.42 · random genome 0.67. Accuracy 0.50 → 0.53.
- **Interpretation:** No learning. The evolved genome was not better than random.
- **Decision:** Check whether learning is *reachable* at all with a hand-set learner.

## E003 · 2026-11-10 · Is learning reachable? (hand-set dopamine-style rule) ✔
- **Finding:** Motor neurons held tanh activity, not the action actually sampled, so reward-gated Hebbian updates strengthened the wrong synapses. **Fix:** an efference copy (motor neurons set to one-hot of the chosen action). This is now in v0.
- **Setup:** Genome with only the motor←sensor rule A = 1, modulator weight on energy change 3, log plasticity rate −1, motor←sensor wiring logit 0. This is now `biology_prior()`.
- **Result (32 worlds):** fitness **6.32**, accuracy **0.58 → 0.69**, survival 0.64 · frozen 1.08 · random −0.15.
- **Interpretation:** The architecture can learn within a life, and the learning does the work.

## E004 · 2026-11-10 · Blank start with efference copy
- **Setup:** log plasticity rate init −3, ES lr 0.03. 50 gens, pop 32 × 4.
- **Result (32 worlds):** 0.59, accuracy 0.51 → 0.53 · frozen 0.27.
- **Interpretation:** Still nothing. With an Adam step of about 0.03 per gen, the plasticity rate needs about 70 gens just to reach a useful range.
- **Decision:** Raise the initial plasticity rate and ES lr.

## E005 · 2026-11-10 · Blank start, log plasticity rate init −1.5, ES lr 0.1
- **Setup:** 50 gens, pop 32 × 4. These are the current v0 defaults.
- **Result (32 worlds):** **0.93**, accuracy 0.52 → 0.54 · frozen 0.66.
- **Interpretation:** At most a hint, within noise. **The main open question: does a blank start discover learning at larger scale?** (Experiment A.)

## E006 · 2026-11-10 · Biology start, σ 0.1, lr 0.1
- **Setup:** Mean initialised to `biology_prior`. 30 gens, pop 32 × 4.
- **Result (32 worlds):** **4.73**, accuracy 0.58 → 0.64 · frozen **0.07**. In a separate 5-gen quick run: 6.3 → 1.5.
- **Interpretation:** Plasticity carries the evolved brain entirely (the lesion collapses it), but large mutations degrade the prior.

## E007 · 2026-11-10 · Biology start, gentle ES (σ 0.05, lr 0.02)
- **Setup:** 20 gens, pop 32 × 4.
- **Result (32 worlds):** **6.27**, accuracy 0.59 → 0.70.
- **Interpretation:** Gentle settings preserve the prior. Whether evolution can *improve* it is open (Experiment B).

## E008 · 2026-11-12 · Repo created, runner refactor
- `evolve` was split into `es_init` / `es_step` for checkpoint and resume. Rerunning E007 reproduced its numbers exactly (6.265625, accuracy 0.594 → 0.702), so the refactor did not change behaviour.

---

## Next IDs start at E100 (see HANDOFF §4–5)

## First-session setup · 2026-11-12 · Before experiments
- **Environment:** Windows; Python 3.12.10. Project-local `.venv` created on D:. Dependency installation and JAX device checks are pending. CPU model, RAM, and available accelerators have not yet been verified.
- **Storage snapshot:** C: 10.49 GiB free of 373.67 GiB; D: 46.89 GiB free of 100.00 GiB. Keep the environment, installation temporary files, and experiment artifacts on D:, disable pip's download cache, and check free space before larger jobs. Do not delete historical runs to reclaim space.
- **Repository:** GitHub authentication succeeds outside the restricted sandbox. Automatic approval review rejected repository creation and push pending explicit authorization of the destination and tracked contents. Publishing remains pending; no remote result is claimed.
- **Integrity decision:** Preserve the smoke run. AGENTS.md's prohibition on deleting anything in `runs/` takes precedence over HANDOFF §4's smoke-folder deletion instruction. Smoke and benchmark runs are operational checks, not evidence for a scientific claim.
- **Working loop:** The coordinator owns experiment choices and execution; independent agents handle documentation and repository checks. After each experiment, record the observed result, what it changes about the hypothesis, and the next single change. Preserve negative results. Benchmark measurements determine run sizes; evaluation scores must not be used to tune the fixed EVAL v1 worlds.

## E101 · 2026-11-12 · Preregistered pipeline smoke test
- **Hypothesis:** The committed v0 runner executes end to end in the laptop's local Python environment.
- **Exact command (PowerShell, repository root):**
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E101-smoke --quick
  ```
- **Setup:** Runner quick settings: 3 generations, population 8, 2 training worlds, 4 evaluation worlds, checkpoint every generation. Run only after committing the code and this preregistration. No `--allow-dirty` is planned.
- **Pass:** Process completes successfully, `summary.json` reports `finished` at generation 3, and all reported evaluation metrics are finite.
- **Fail:** Crash, incomplete run, or non-finite evaluation metrics. Keep any failed run folder and diagnose before scaling.
- **Result:** Pending. This is an operational check; its four-world scores will not support an EVAL v1 claim.
- **Decision after recording:** If it passes, run E102. Regenerate the results table with `.\.venv\Scripts\python.exe scripts/summarize.py` and append the observed outcome here.

## E102 · 2026-11-12 · Preregistered full-settings benchmark
- **Hypothesis:** The default population and training-world count can run within the laptop's compute and storage limits.
- **Exact command (PowerShell, repository root):**
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E102-benchmark --generations 10 --seed 0
  ```
- **Setup:** Population 128, 8 training worlds, default world/brain configuration, 128 evaluation worlds, checkpoint every 10 generations. One low-priority compute job at a time. Use the supported Windows process-priority mechanism rather than the Unix `nice` command.
- **Pass:** Run finishes with finite metrics within 3 hours. Record total wall time, first-generation compilation cost, subsequent generation times from `metrics.jsonl`, and evaluation overhead. Use those measurements to estimate whether each proposed 200-generation run fits under 3 hours and whether the day's total stays under about 8 hours.
- **Fail:** Crash, non-finite metrics, or inability to finish within the per-run limit. Stop before exceeding the limit, retain artifacts, and use the measured constraint to choose a supported resumable schedule before longer runs.
- **Result:** Pending. A single-seed benchmark is not a claim that evolution learns.
- **Decision after recording:** Proceed to E100 reference measurements if the environment is viable. Schedule core experiments only after their wall-time and daily-compute estimates are known. Regenerate the results table and append the measured outcome.

## E100 · 2026-11-12 · Preregistered EVAL v1 reference measurements
- **Hypothesis:** The biology prior shows within-life learning and a positive plasticity lesion effect under the fixed 128-world EVAL v1 protocol; random initial genomes provide the comparison reference.
- **Exact commands (PowerShell, repository root; execute sequentially):**
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E100-blank-s0 --start blank --generations 0 --seed 0
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E100-blank-s1 --start blank --generations 0 --seed 1
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E100-blank-s2 --start blank --generations 0 --seed 2
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E100-biology-s0 --start biology --generations 0 --seed 0
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E100-biology-s1 --start biology --generations 0 --seed 1
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E100-biology-s2 --start biology --generations 0 --seed 2
  ```
- **Setup:** No evolution and no configuration overrides. All final-learning and frozen evaluations retain `test_genome`'s seed 123 and the runner's default 128 worlds. Confirm the zero-generation runner path preserves the starting genome. Do not change the evaluation protocol to manufacture independent repeats.
- **Measurement-completion pass:** All six tracked runs finish with finite metrics and the expected unchanged start/final genome. Copy learning fitness, frozen fitness, first-half accuracy, and second-half accuracy from each `summary.json`; compute accuracy gain and lesion gap transparently from those values.
- **Hypothesis pass:** The biology reference has positive second-minus-first-half accuracy and learning-minus-frozen fitness. Failure of either condition means this reproduction hypothesis fails. Report random-reference variation across seeds descriptively; do not infer an evolutionary improvement from zero-generation runs.
- **Independence limitation:** Before interpreting repetitions, inspect genome hashes and the implementation's seed use. If biology initialization is deterministic, its three zero-generation runs under fixed evaluation seed 123 are duplicate reproducibility checks, not three independent scientific trials. They cannot establish a three-seed evolutionary claim. The later A/B/C comparisons still require three evolution seeds each.
- **Result:** Pending. Pilot values from 32 worlds are not substituted for these measurements.
- **Decision after recording:** Append results, interpretation, and the next experimental decision; regenerate `results/RESULTS.md`. Consider the EVAL v1 milestone only after these measurements and their limitations are recorded. Core A/B/C experiments remain governed by HANDOFF §5 and the measured compute budget.
