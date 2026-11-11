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

## Setup update and E101 launch · 2026-11-12
- **Correction to earlier pending setup status:** Intel Core i7-1370P verified through the Windows registry. Python 3.12.10; installed JAX 0.12.0.1, jaxlib 0.12.0.1, and NumPy 2.5.4. JAX CPU device import verified. RAM capacity and accelerator availability remain unverified; no GPU execution is claimed.
- **Storage after installation:** C: 10.49 GiB free; D: 46.36 GiB free. The project environment and installation temporary files remain on D:.
- **Repository update:** The owner explicitly approved the private repository destination and contents. `Winter-Soren/seed` was created, its private visibility verified through the GitHub API, and the initial `main` branch pushed. Preregistration was committed locally as `a8ed352`; this entry does not claim that later commit was pushed.
- **E101 launch:** Started at 02:08:01 Asia/Calcutta with Windows `BelowNormal` process priority, from preregistration commit `a8ed352`. Console output is captured in `.tmp/E101-smoke.stdout` and `.tmp/E101-smoke.stderr`. Result pending.
- **Process improvement:** The owner's requested learning loop is an evidence → decision → next experiment record, with workflow changes justified by observed constraints. It does not mean modifying the assistant's model weights. Preserve the critical path: setup → smoke → benchmark → E100 → preregistered A/B/C. No roadmap stage advances without its gate or owner authorization.
- **Control interpretation for later experiment C:** When evolution uses `--no-plastic`, the control's actual disabled-plasticity performance is `final_frozen`. The runner's `final_learning` value is a diagnostic evaluation that re-enables plasticity; it must not be reported as the no-plastic control score. This clarifies reporting without changing EVAL v1.

## E101 launch correction and resource update · 2026-11-12
- **Correction to the preceding launch entry:** The detached process 31080 launched at 02:08:01 did not persist and created no tracked run folder. A retry at 02:08:29 kept its parent shell waiting; the actual tracked run is `runs/20261112-020840_E101-smoke`. Launcher 9864 and child 36412 were both verified at `BelowNormal` priority. This records a launch failure rather than a completed smoke test.
- **Progress only:** The retried run completed all 3 training generations; `metrics.jsonl` reports 32.662, 1.718, and 1.551 seconds for generations 1–3. Final evaluations are still pending. No finished-run or scientific result is claimed from these partial observations.
- **RAM snapshot:** Total physical RAM 32,870,040 KiB (about 31.35 GiB); free physical RAM 9,260,796 KiB (about 8.83 GiB) at the reported snapshot. Accelerator availability remains unverified.
- **Compute authorization update:** The oversight chat relayed the owner's explicit authorization for continued overnight compute beyond the earlier approximately 8-hour daily threshold, while remaining mindful of storage and token use. Keep individual runs bounded below 3 hours, retain checkpointing and low priority, and run only one long compute job at a time. This permits a longer daily session; it does not change the evaluation protocol or roadmap gates.
- **Execution lesson:** Verify that the tracked folder exists and the process remains alive after launch; keep the supervising shell attached when detached jobs do not persist. Continue to the benchmark only after the smoke run's final summary is checked and recorded.

## E101 result · 2026-11-12 · Smoke test passed
- **Evidence:** `runs/20261112-020840_E101-smoke/summary.json` reports `finished`, generation 3, `train_minutes` 0.6, and 4 evaluation worlds. The process exited 0 and all reported metrics were checked finite. `config.json` records clean commit `a8ed3521182663a6a9c713e1d9042cdfcaa47e59`, Python 3.12.10, JAX 0.12.0.1, and `cpu:0`.
- **Numbers copied from the summary:** Start-learning fitness 0.5625. Final-learning fitness 1.875; first-half accuracy 0.5678392052650452; second-half accuracy 0.5118110179901123; survival 0.4375; eaten 326. Final-frozen fitness 2.3125; first-half accuracy 0.5647059082984924; second-half accuracy 0.5641025900840759; survival 0.5625; eaten 287. Final genome hash `63120c42e51dc1e1`.
- **Timing:** Training generations took 32.662, 1.718, and 1.551 seconds. Elapsed time from the config-file modification timestamp to the final summary timestamp was 85.26 seconds; that is the tracked phase, not total time since the first launch attempt. Full-settings throughput remains unknown until E102.
- **Interpretation:** Operational pass: the laptop environment and tracked runner execute end to end. The four-world scores are not EVAL v1 evidence, and the run does not establish that evolution discovers learning. The smoke folder remains preserved.
- **Recording:** Ran `$env:PYTHONUTF8='1'; .\.venv\Scripts\python.exe scripts/summarize.py`; the generated `results/RESULTS.md` contains one finished smoke row with the expected commit and values.
- **Decision:** Commit the run, table, and appended record together, then execute the preregistered E102 benchmark. Measure full wall time including evaluation before scheduling longer work.
- **Scheduling limitation found during review:** `--resume` loads the original generation target, so a run created for 50 generations cannot be extended to 200 by passing a new target on resume. If E102 projects a 200-generation run above 3 hours, add and test a graceful checkpoint-boundary stop before scheduling it; do not assume the current runner already supports that split.
- **Interpretation limitation:** The frozen lesion tests reliance on synaptic plasticity. It does not eliminate every possible form of adaptation through recurrent state.

## E103–E105 · 2026-11-12 · Preregistered core comparisons
- **Scheduling condition:** Start only after E102's measured throughput projects a 200-generation run, including evaluation, below 3 hours. Otherwise implement and test graceful checkpoint-boundary stopping before scheduling the same scientific settings in bounded segments. One low-priority compute job at a time; checkpoint every 10 generations. Complete E100 references before judging these comparisons.
- **Shared protocol:** Default world/brain configuration, population 128, 8 training worlds, 200 generations, evolution seeds 0/1/2, and unchanged EVAL v1 (128 worlds, evaluation seed 123). Commit this preregistration and source before launching. Preserve every run and record full scores and gaps from summaries; no tuning on evaluation worlds.

### E103 · A: can blank-start evolution discover learning?
- **Hypothesis:** Evolution from a blank start produces within-life learning that relies on synaptic plasticity in at least 2 of 3 evolution seeds.
- **Exact commands (PowerShell, repository root):**
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E103-A-blank-s0 --start blank --generations 200 --seed 0
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E103-A-blank-s1 --start blank --generations 200 --seed 1
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E103-A-blank-s2 --start blank --generations 200 --seed 2
  ```
- **Gate:** In at least 2 of 3 seeds, final-learning second-minus-first-half accuracy is strictly greater than 0.03 **and** frozen fitness is clearly below learning fitness, preserving HANDOFF's criterion. Report each full learning-minus-frozen gap. HANDOFF supplies no numerical lesion cutoff; a tiny positive gap alone does not establish a clear effect. Qualify ambiguous effects without inventing statistical significance or advancing the roadmap from an ambiguous gate. If fewer than 2 seeds meet both requirements, the gate is not established.
- **Result:** Pending. If the gate fails, follow HANDOFF's one-change-at-a-time sequence, beginning with more training worlds; preregister any subsequent experiment separately.

### E104 · B: can evolution improve the biology prior?
- **Hypothesis:** Gentle ES improves on the E100 biology-prior learning fitness in at least 2 of 3 evolution seeds.
- **Exact commands (PowerShell, repository root):**
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E104-B-bio-s0 --start biology --sigma 0.05 --lr 0.02 --generations 200 --seed 0
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E104-B-bio-s1 --start biology --sigma 0.05 --lr 0.02 --generations 200 --seed 1
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E104-B-bio-s2 --start biology --sigma 0.05 --lr 0.02 --generations 200 --seed 2
  ```
- **Gate:** Final-learning fitness strictly exceeds the E100 biology-prior learning fitness under EVAL v1 in at least 2 of 3 seeds. Otherwise this gate fails. Record per-seed differences, within-life accuracy, and lesion gaps; do not describe a merely preserved prior as improved.
- **Result:** Pending.

### E105 · C: evolution with synaptic plasticity disabled
- **Hypothesis:** The no-plasticity evolutionary control provides a baseline below the learning performance of A and B; A versus C is the matched intervention comparison.
- **Exact commands (PowerShell, repository root):**
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E105-C-noplastic-s0 --start blank --no-plastic --generations 200 --seed 0
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E105-C-noplastic-s1 --start blank --no-plastic --generations 200 --seed 1
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E105-C-noplastic-s2 --start blank --no-plastic --generations 200 --seed 2
  ```
- **Comparison:** C's actual control performance is `final_frozen`; compare it with A/B `final_learning` by paired evolution seed and report all differences. Verify A/C starting-genome hashes match for each paired seed, along with settings other than the plasticity switch. C's `final_learning` re-enables plasticity and is diagnostic only.
- **Interpretation:** This is a baseline, not a separate roadmap gate. Positive A-minus-C differences support the proposed plasticity advantage; zero or negative differences count against it and must be reported. B versus C also changes the starting prior and ES hyperparameters, so it cannot isolate plasticity's causal effect. Do not convert this comparison into an unsupported significance claim.
- **Result:** Pending. After each completed experiment, regenerate the results table, append its results and interpretation, and commit/push the run artifacts with the record. Gate decisions require the planned three seeds and explicit treatment of uncertainty.

## E103–E105 scheduling amendment and resource checkpoint · 2026-11-12
- **Amendment to the preceding core commands:** Append `--max-train-seconds 7200` to each of the nine initial E103/E104/E105 commands. If a run pauses before generation 200, resume that same folder with `.\.venv\Scripts\python.exe scripts/run_experiment.py --resume <run-folder> --max-train-seconds 7200`. The saved target remains 200; this changes scheduling only, not scientific settings or EVAL v1.
- **Implementation verification:** The runner worker reports four passing tests, including a real tiny ES with a mocked world evaluator split into 2 + 1 + 2 generations, whose final optimizer state, RNG state, and training metrics match uninterrupted execution exactly. The coordinator reviewed the change; it is not yet committed. Commit it before using the new option. E102 continues under its original `8545333` code.
- **Limit:** The 7,200-second training budget excludes setup and evaluation and may overrun by one generation. Monitor full wall time separately to preserve the under-3-hour per-process limit; this option alone is not a hard wall-time guarantee.
- **Resource snapshot:** C: 6.94 GiB free, down from the recent 10.46 GiB reading. The pagefile allocation was 8,618 MiB, but there is no causal baseline establishing what changed it; do not attribute the free-space drop to the pagefile. The active benchmark had about 3.25 GiB private memory and 2.69 GiB resident memory. E102 had reached generation 7 of 10; final results remain pending.
- **Decision and process lesson:** Launch no additional heavy work until free-space stability and headroom are checked. Resource measurements must govern scheduling alongside runtime estimates; distinguish observed disk/RAM readings from unverified explanations. Record the completed benchmark at its next meaningful checkpoint.

## E102 result · 2026-11-12 · Full-settings benchmark passed
- **Evidence:** `runs/20261112-021242_E102-benchmark/summary.json` reports `finished`, generation 10, `train_minutes` 8.62, and 128 evaluation worlds. Process exit 0; finite metrics. Configuration records clean commit `8545333525ab26bb02740304d9f621fbfaf2dccd`, population 128, 8 training worlds, seed 0, default configuration, and CPU execution.
- **Numbers copied from the summary:** Start-learning fitness -0.00390625; first-half accuracy 0.5008518099784851; second-half accuracy 0.49779558181762695. Final-learning fitness 0.46875; first-half accuracy 0.513361930847168; second-half accuracy 0.513504147529602; survival 0.4296875; eaten 8950. Final-frozen fitness 0.001953125; first-half accuracy 0.4885801374912262; second-half accuracy 0.5247443914413452; survival 0.36328125; eaten 7699. Final genome hash `18091312a2ae9523`.
- **Measured timing:** Supervisor launch-to-exit wall time 561.9480759 seconds. Generation times sum to 517.023 seconds. First generation 71.261 seconds; generations 2–10 mean 49.5291111111 seconds, median 45.286 seconds, maximum 66.965 seconds. Full wall time minus summed generation times is about 44.925 seconds of aggregate setup, evaluation, and other overhead; it is not a separate direct evaluation measurement.
- **Projection and decision:** First-generation time plus 199 times the subsequent-generation mean projects 2.7576539 hours of training for 200 generations, plus overhead. This short benchmark cannot guarantee stable long-run throughput. Use the amended 7,200-second training segments for headroom, retaining the 200-generation target and monitoring full wall time. End-of-run storage readings were stable at C: 6.94 GiB and D: 46.36 GiB free; recheck before each heavy run.
- **Interpretation:** Operational pass: full settings completed well below 3 hours with finite metrics. This single-seed 10-generation timing run does not establish that evolution discovers learning; it is not the preregistered 200-generation A comparison.
- **Runner verification:** `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` passed four tests in 6.908 seconds, as verified by the coordinator. The bounded/resumed equivalence check uses a synthetic fitness fixture with real tiny ES, not a research world or scientific result. No engine or EVAL v1 changes were made; commit the runner and tests before using the new limit option.
- **Recording and next step:** Regenerated `results/RESULTS.md` with UTF-8 enabled and checked both finished rows against their summaries and source commits. Commit/push E102 artifacts, records, and the verified scheduling change, then execute the preregistered E100 references before judging core comparisons.
- **Additional verification:** The coordinator confirmed checkpoint generation 10 and exact equality between its mean genome and `genome_final.npy`. Final-learning accuracy gain is 0.000142216682434, which does not show a meaningful within-life learning signal in this short benchmark. The git worker subsequently committed the runner/tests separately as `67659fd`; E102's source remains `8545333`.

## E100 result · 2026-11-12 · EVAL v1 references measured
- **Evidence:** All six preregistered runs finished with exit 0, zero generations, 128 evaluation worlds, finite metrics, and clean source commit `b4f467d4f0c490e8ba82a8a7cbecdb8001d16493`. The coordinator verified checkpoint `t=0`, exact equality of start/final/checkpoint mean arrays and hashes, and `start_learning == final_learning` in every run. EVAL v1 remains fixed at seed 123; no evolution occurred.
- **Learning results:** Fitness, accuracies, survival, and eaten counts below are copied from each `summary.json`. Accuracy gain is second minus first; lesion gap is learning fitness minus frozen fitness.

| Reference | Learning fitness | Accuracy first | Accuracy second | Accuracy gain | Lesion gap | Survival | Eaten |
|---|---:|---:|---:|---:|---:|---:|---:|
| Blank seed 0 | -0.00390625 | 0.5008518099784851 | 0.49779558181762695 | -0.0030562281608581543 | -0.048828125 | 0.3828125 | 7778 |
| Blank seed 1 | 0.4140625 | 0.5018976926803589 | 0.5329878926277161 | 0.031090199947357178 | 0.369140625 | 0.38671875 | 8428 |
| Blank seed 2 | 0.23046875 | 0.5063009858131409 | 0.5091709494590759 | 0.0028699636459350586 | 0.185546875 | 0.388671875 | 8122 |
| Biology seed 0 | 5.12890625 | 0.5685573816299438 | 0.6739441156387329 | 0.10538673400878906 | 5.12890625 | 0.6015625 | 11618 |
| Biology seed 1 | 5.12890625 | 0.5685573816299438 | 0.6739441156387329 | 0.10538673400878906 | 5.12890625 | 0.6015625 | 11618 |
| Biology seed 2 | 5.12890625 | 0.5685573816299438 | 0.6739441156387329 | 0.10538673400878906 | 5.12890625 | 0.6015625 | 11618 |

| Frozen reference (identical across listed seeds) | Fitness | Accuracy first | Accuracy second | Survival | Eaten |
|---|---:|---:|---:|---:|---:|
| Blank seeds 0/1/2 | 0.044921875 | 0.49962562322616577 | 0.5053170323371887 | 0.37109375 | 7881 |
| Biology seeds 0/1/2 | 0.0 | 0.49433672428131104 | 0.5116646885871887 | 0.376953125 | 7738 |

| Run folder | Start/final genome hash | Launch-to-exit seconds |
|---|---|---:|
| `20261112-022439_E100-blank-s0` | `9a761b031011f9ee` | 34.786622 |
| `20261112-022514_E100-blank-s1` | `47c2bad05aa97822` | 34.4837528 |
| `20261112-022552_E100-blank-s2` | `03202e865b4a4255` | 50.2148786 |
| `20261112-022641_E100-biology-s0` | `72caddf0cdd93943` | 48.0871014 |
| `20261112-022728_E100-biology-s1` | `72caddf0cdd93943` | 46.5430618 |
| `20261112-022814_E100-biology-s2` | `72caddf0cdd93943` | 43.590344 |

- **Interpretation:** The reference reproduction hypothesis passes: the biology prior gains about 10.54 percentage points in within-life food-choice accuracy and loses 5.12890625 fitness when synaptic plasticity is disabled. Its three runs have the same genome and scores, so they are deterministic repeatability checks, not three independent scientific trials. Blank genomes have three distinct hashes; their identical frozen scores are expected because seed-dependent initial Hebbian-rule variation is disabled by the lesion. This measures available starting behavior, not evolution's ability to discover or improve learning.
- **Recording:** Regenerated and checked all eight rows in `results/RESULTS.md`; updated README from pilot numbers to these 128-world references. Per-run storage guards were C: at least 5 GiB and D: at least 10 GiB before launch, with one `BelowNormal` compute job at a time and temporary files on D:.
- **Decision:** Commit/push the six runs and records, and mark the EVAL v1 reference milestone. Next run the preregistered E103 A seed 0, preserving its 200-generation target with 7,200-second training segments. E103's three-seed gate remains untested; no roadmap advancement is authorized by this reference measurement alone.
- **Core resource guard:** Final reference storage was C: 6.95 GiB and D: 46.36 GiB free. Check the 5/10 GiB C:/D: guards before every invocation and monitor roughly every 45 seconds while running. If either falls below its guard, terminate only the owned run process tree to protect the laptop, preserve its checkpoint and all artifacts, and append an incident record without editing history. An emergency stop can lose progress since the last 10-generation checkpoint; use the normal graceful 7,200-second stop when resources remain healthy.

## E103 A seed 0 launch · 2026-11-12
- **Launch:** Started 02:32:43 Asia/Calcutta; tracked folder `runs/20261112-023245_E103-A-blank-s0` records clean source commit `71126a6`. Target 200 generations, training limit 7,200 seconds. Supervising shell session 92661, launcher PID 19180; output `.tmp/E103-A-blank-s0-segment1.stdout` and `.tmp/E103-A-blank-s0-segment1.stderr`. No completed A result yet.
- **Guards:** Coordinator watchdog checks about every 45 seconds; require C: at least 5 GiB and D: at least 10 GiB free. Terminate only the owned process tree if a disk guard fails or total wall time reaches 10,500 seconds, preserving artifacts.
- **Recovery caution:** Emergency termination can leave metric rows ahead of the last checkpoint, so inspect the metrics tail against checkpoint generation before resuming. Preserve all artifacts and record any discrepancy; crash recovery has not been validated by the graceful-stop equivalence test, and careless resume can duplicate metric rows.
- **Next:** After a normal graceful segment, record and commit its state before `.\.venv\Scripts\python.exe scripts/run_experiment.py --resume runs/20261112-023245_E103-A-blank-s0 --max-train-seconds 7200`. Remaining planned core runs: A seeds 1/2, B seeds 0/1/2, and C seeds 0/1/2. No gate decision is due until the required completed comparisons exist.
- **Interpretation clarification from the audit:** EVAL v1 accuracy pools eating events, so its increase can reflect survival or encounter selection as well as individual learning; dead agents contribute no later eating events. With fixed world/development randomness, the lesion supports a total fitness benefit from enabled updates (Hebbian and predictor updates together), not isolated proof that each agent infers the hidden rule. Frozen agents retain recurrent state and energy feedback. This clarifies the evidence without changing EVAL v1 or preregistered gates.

## E103 scheduling validation checkpoint · 2026-11-12
- At A seed 0 checkpoint generation 10, the coordinator's assertion check exited 0: checkpoint arrays `mu`, `m`, `v`, `t`, and `key` exactly matched E102's generation-10 checkpoint, and all ten metric dictionaries matched excluding `seconds`. E102 used source `8545333`; A seed 0 uses `71126a6`. This verifies the bounded-runner change preserved the actual full-simulation trajectory through generation 10; it is not a learning result or a completed A run. No run artifacts were modified by the check.
- Snapshot: A seed 0 at 10/200 generations, supervising session 92661 still live; C: 6.93 GiB and D: 46.36 GiB free. Revalidate current progress after handoff; next research-record update is due at the segment outcome.
