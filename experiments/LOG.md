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

## E103 A seed 0 · 2026-11-12 · First bounded segment saved at generation 167
- **Outcome:** The scheduled 7,200-second training limit stopped the segment normally, with child exit 0 and total wall time 7,221.4645452 seconds. `runs/20261112-023245_E103-A-blank-s0/summary.json` reports `interrupted`, generation 167, `train_minutes` 120.29, and genome hash `7f765cb13840c926`. This status denotes a scheduled continuation point, not a failed experiment. Source was clean `71126a6`; the target remains 200 and the saved training limit 7,200 seconds.
- **Verification:** The coordinator checked checkpoint `t=167`, finite checkpoint arrays, and a matching genome hash. There are exactly 167 contiguous metric rows, generations 0–166, totaling 7,217.155 seconds. No final-genome file or fixed-world evaluations were produced, as expected before target completion. Supervising session 92661 and monitor cell 94 are terminal; no run process remains active at this checkpoint.
- **Training observation only:** The last perturbed-population training row has mean fitness 10.440673828125 and pooled accuracy 0.6569973230361938 → 0.7717463374137878. These are not fixed-mean EVAL v1 measurements and cannot establish the A gate. Final learning and frozen fitness remain unknown.
- **Resources and hardware:** C: 6.91 GiB and D: 46.36 GiB free, stable at segment end. Read-only inventory found Intel Iris Xe Graphics, driver 32.0.101.7088; CIM found no NVIDIA adapter and `nvidia-smi` was absent from PATH and its standard location. WSL2 registrations include Ubuntu 24.04 and `podman-machine-default`, but neither was started and package usability was not tested. Keep the verified CPU path; no CUDA path or new installation is assumed.
- **Decision:** Regenerated `results/RESULTS.md` and verified nine rows, including A seed 0 as interrupted at 167/200 with evaluation cells empty. Commit/push this checkpoint and record, then resume the same experiment for its remaining 33 generations using:
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_experiment.py --resume runs/20261112-023245_E103-A-blank-s0 --max-train-seconds 7200
  ```
  Preserve the hypothesis, seed, target, configuration, and EVAL v1 protocol. Recheck disk/memory guards before resuming; no new experiment or scientific claim is introduced by the continuation.

## E103 A seed 0 resume · 2026-11-12
- Segment 2 launched at 04:35:40.686 Asia/Calcutta; stdout confirms resume from generation 167 and the recorded resume source is clean `55a459b`. Target 200 and training limit 7,200 seconds are unchanged. Launcher PID 34092, supervising session 28759; output `.tmp/E103-A-blank-s0-segment2.stdout`/`.stderr`.
- Prelaunch snapshot: C: 6.91 GiB and D: 46.36 GiB free; free physical memory 9,299,848 KiB and free virtual memory 6,621,280 KiB. Retain 45-second disk checks, C:/D: minimums 5/10 GiB, and the 10,500-second full-wall guard. No new result; next research-record update is due at final completion or an intervening stop.

## E103 A seed 0 result · 2026-11-12 · First seed meets the preregistered criterion
- **Evidence:** `runs/20261112-023245_E103-A-blank-s0/summary.json` reports `finished`, generation 200, 128 evaluation worlds, and final genome hash `ee8eaa10a7429cc5`. The coordinator verified 200 contiguous metric rows (0–199), finite checkpoint `t=200`, exact equality of checkpoint mean and `genome_final.npy`, and matching hash. Configuration is blank start, seed 0, population 128 × 8 worlds, default world/brain settings, plasticity enabled. Initial source was clean `71126a6`, resume source clean `55a459b`; engine and EVAL v1 remained unchanged.
- **Final evaluation metrics copied from the summary:**

| Evaluation | Fitness | Accuracy first | Accuracy second | Survival | Eaten |
|---|---:|---:|---:|---:|---:|
| Final learning | 16.591796875 | 0.7049121856689453 | 0.8401943445205688 | 0.880859375 | 15837 |
| Final frozen | 0.2890625 | 0.4963643252849579 | 0.5343427062034607 | 0.39453125 | 7934 |
| Start learning | -0.00390625 | 0.5008518099784851 | 0.49779558181762695 | 0.3828125 | 7778 |

- **Derived criterion values:** Final-learning accuracy gain is 0.13528215885162354 (13.528215885162354 percentage points), exceeding 0.03. Learning-minus-frozen fitness is 16.302734375, a clear lesion effect in this evaluation. Starting evaluation exactly matches E100 blank seed 0. This seed meets both preregistered per-seed requirements; the experiment's three-seed gate remains pending seeds 1 and 2.
- **Timing:** Both segment processes exited 0. Segment 1 wall time was 7,221.4645452 seconds; segment 2 was 1,455.2218498 seconds. Total process wall time was 8,676.686395 seconds (about 144.61144 minutes), excluding the pause between segments. Training metric times total 8,634.951 seconds. Final `summary.json`'s `train_minutes=23.63` covers only segment 2, not the entire experiment.
- **Interpretation and limits:** This is the first completed evolutionary seed and evidence for useful plasticity under EVAL v1, not a replicated three-seed result or proof of general intelligence. Pooled eating-event accuracy can reflect survival/encounter selection as well as individual learning. The lesion disables Hebbian and predictor updates together while preserving recurrent state and energy feedback; it does not isolate hidden-rule inference. No statistical-significance claim or roadmap advancement follows from one seed.
- **Resources and decision:** C: 6.91 GiB and D: 46.36 GiB free. Session 28759 and monitoring cell 102 are terminal; no active run remains. Regenerated and checked the nine-row results table, now showing A0 finished at 200/200. Commit/push the completed run and records, then run A seeds 1 and 2 with the original preregistered settings and bounded scheduling before deciding the A gate. Do not add a gate tag yet.
- **Independent interpretation check:** The reviewer confirmed the per-seed criterion and unchanged engine/runner between initial and resume commits. Frozen accuracy also rises by 0.0379783809185028, above 0.03 despite disabled plasticity. Accuracy gain alone therefore cannot identify individual learning or attribute all 13.53 percentage points to hidden-rule inference; the large 16.302734375 fitness lesion supports useful plasticity.

## E103 A seed 1 launch · 2026-11-12
- A0's completed result was committed and pushed as `e0befc6`; the three-seed gate remains pending. A1 launched at 05:03:17 Asia/Calcutta, PID 772, supervising session 72549, folder `runs/20261112-050319_E103-A-blank-s1`, from clean `e0befc6`. The coordinator verified its starting genome exactly matches E100 blank seed 1, hash `47c2bad05aa97822`; target 200, training limit 7,200 seconds, no stderr at launch.
- Output: `.tmp/E103-A-blank-s1-segment1.stdout`/`.stderr`. Retain the 45-second watchdog, C:/D: minimum free space 5/10 GiB, and 10,500-second total-wall guard. No A1 result yet; next record at the segment outcome.

## E103 A seed 1 incident · 2026-11-12 · Low disk headroom and unverified termination
- **Observed sequence:** The owner paused and later reactivated the goal while A1 monitoring was waiting; causation is not established. Monitoring cell 109 was subsequently unavailable. Querying original supervising session 72549 returned exit 1. Its output showed elapsed 6,399 seconds with C: 6.86 GiB free and last reported generation 149, followed by elapsed 6,444 seconds with C: 2.03 GiB and D: 46.36 GiB free, below the C: guard.
- **Guard failure:** `taskkill` reported `The user name or password is incorrect.` The supervisor then threw a message claiming the owned experiment had stopped, although that kill had failed. Treat the claimed stop as unverified. A later outside-sandbox CIM check found zero Python `run_experiment.py` processes; the coordinator's targeted-stop attempt found no matching process. The evidence establishes no matching process at the later check, not when or why it exited.
- **Preserved artifacts:** `runs/20261112-050319_E103-A-blank-s1` has a finite checkpoint at `t=150` and 151 contiguous metric rows, generations 0–150. Its summary still says `running`, generation 0, and is stale rather than evidence of a live process or completed result. Preserve every run file exactly. The metrics are one generation ahead of checkpoint state; resuming without verified replay handling risks duplicate rows. No final A1 evaluation or A gate decision exists.
- **Resource snapshot:** C: 2.18 GiB free; D: 46.36 GiB free. Free physical memory 10,749,160 KiB and free virtual memory 6,650,224 KiB. Pagefile allocation was 11,372 MiB versus the earlier 8,618 MiB observation; this change does not by itself establish the cause of the disk-space loss.
- **Decision:** The goal is active with a temporary resource/recovery block, not paused or complete. Launch no heavy work until C: is at least 5 GiB and D: at least 10 GiB, safe crash replay is verified, and the stop guard demonstrably works. A worker is preparing minimal append-only verified replay handling without engine changes; a separate worker is performing a read-only storage audit; the coordinator owns stop-guard verification. These tasks are pending, not completed fixes.
- **Audit handling:** Commit the unchanged incident artifacts with this record and TASK_STATE, excluding unfinished source/tests. Do not regenerate the results table during this incident checkpoint: the unchanged summarizer would display the stale `running` status without the recovery context. The next experimental action requires the resource and recovery checks above.
- **Stop-guard probe verified afterward:** Outside the sandbox, the coordinator spawned an owned Python `SEED_GUARD_PROBE` that slept for 60 seconds. `taskkill /PID <owned-pid> /T /F` returned 0; the launcher exited within 5 seconds, and CIM verified no probe children remained. No experiment was restarted. Future supervised runs must use the verified outside-sandbox permission path and check both the kill exit status and actual process absence before claiming a stop. C: remains 2.18 GiB free, so heavy work remains blocked. Storage review also observed a pagefile logical file size of 15.610 GiB versus a CIM allocation snapshot of 11,372 MiB; these readings are under investigation, not assumed equivalent or causal.

## E103 recovery repair verification · 2026-11-12
- **Preservation and status:** A1's incident artifacts were committed as `18202a7` and remain unchanged, including the stale summary. The actual A1 run has not been resumed or used to test replay. No new scientific result is claimed; the goal remains active while heavy compute is held for storage headroom.
- **Runner repair:** The coordinator reviewed append-only crash-tail replay. `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` passed eight tests in 4.363 seconds. Coverage includes graceful equivalence, one/two metric rows ahead of checkpoint, and interruption before the first checkpoint; final state/RNG match uninterrupted execution and existing metric bytes are preserved. Mismatched, malformed, incomplete history and insufficient replay budget are rejected before writes; verified tail-replay time counts toward the training budget. These are synthetic-fixture checks, not replay verification of the actual A1 artifacts. Runner/tests are ready to commit; engine and EVAL v1 are unchanged.
- **Termination correction and verified alternative:** The original failed supervisor used the default sandbox context. It did not capture native `taskkill`'s exit code; its observed tool exit 1 came from the later throw. A separate reproduction established `taskkill` exit 1 with the credential error and a surviving child. An independently verified default-context test terminated an owned four-process tree using the existing process handle's `.Kill($true)`. Future supervisors should use that checked owned handle, wait for exit, and verify actual tree absence before claiming termination. The earlier outside-sandbox fallback probe also passed.
- **Guard amendment:** Require at least 10 GiB free on C: before heavy startup, allowing headroom for the observed approximately 4.7 GiB free-space drop. Retain runtime C: minimum 5 GiB, D: minimum 10 GiB, the roughly 45-second checks, and full-wall limit. This changes execution safeguards, not experimental settings.
- **Next:** Commit the reviewed runner/tests with this record and TASK_STATE, excluding all run artifacts and the results table. The owner's requested npm/pip/bun/yarn cache audit is read-only and ongoing; no deletion approval has been given. Keep heavy compute held until storage meets the startup guard, then verify actual A1 recovery during its authorized continuation.
- **Actual A1 preservation baseline:** The coordinator read the 151-row metrics file without changing it: 30,922 bytes, SHA-256 `e27d93f9dd5c61d0592bd8d8cc4245e7d34c1d604e96ab14b5a1347886997ca0`. Future guarded resume must retain these exact first 30,922 bytes (verify their prefix hash), reproduce recorded generation 150, then append new rows. This records a filesystem-byte baseline independent of Git line-ending normalization; actual replay remains pending.

## Storage cleanup and resume authorization · 2026-11-12
- **Authorized cleanup:** The owner approved clearing Yarn, npm, and pip caches. The storage worker's native commands all exited 0; verification found Yarn empty, pip reduced to eight metadata files totaling 2,151 bytes, and npm at 156.179 MiB, mostly two `_npx` trees used by eight active Node processes and therefore preserved. Bun and uv were untouched. C: free space rose from 2.462 to 7.792 GiB (5.330 GiB observed difference), including concurrent system activity; do not attribute the whole change solely to cleanup.
- **Resume status:** The owner explicitly authorized resuming the experiment and is personally deleting obsolete projects' `node_modules`/`.venv` directories. Latest coordinator preflight: C: 8.74 GiB and D: 46.36 GiB free; about 9.01 GiB free physical and 3.1 GiB free virtual memory, versus the earlier job's measured 3.25 GiB private memory. No new compute started: await actual C: >=10 GiB and recheck memory headroom before resuming the unchanged A1 checkpoint.
- **Ready recovery and guards:** Reviewed recovery code is committed as `bbc9d73`. Preserve and verify the recorded 30,922-byte metric prefix while replaying generation 150. Use the checked owned-process `.Kill($true)` path and verify termination; tighten runtime polling from 45 to 10 seconds. Startup C: minimum 10 GiB, runtime C: minimum 5 GiB, D: minimum 10 GiB, training limit 7,200 seconds, and full-wall limit 10,500 seconds remain in force. This is operational recovery, not a changed hypothesis or result.

## A1 resume scheduling correction · 2026-11-12
- The owner explicitly requested immediate resume while clearing obsolete projects. The coordinator relaxes only the discretionary 10 GiB C: startup margin for this recovery; retain the C: 5 GiB and D: 10 GiB hard floors with 10-second polling, verified owned-process termination, temporary files on D:, one `BelowNormal` compute job, training limit 7,200 seconds, and full-wall limit 10,500 seconds. Latest reported free space is C: 8.88 GiB and D: 46.36 GiB; fresh launch readings are pending. Experimental settings and EVAL v1 do not change.
- A resume attempt exited before compute because the runner rejected uncommitted tracked changes. These documentation edits must be committed first; actual A1 replay remains pending, with its recorded metric-byte preservation requirement intact. Commit only LOG and TASK_STATE, then hold documentation writes until the coordinator confirms the clean-source launch.
- Fresh preflight before the retry: C: 9.47 GiB and D: 46.36 GiB free; free physical/virtual memory about 8.66/2.66 GiB. Outside-sandbox CIM found zero Python experiment processes. The failed-attempt console files remain `.tmp/E103-A-blank-s1-segment2.stdout`/`.stderr`; retry output will use distinct `segment2-retry1` names.

## A1 recovery verified and continuation running · 2026-11-12
- Retry launched at 09:40:06.7768349 Asia/Calcutta, launcher PID 9696, supervising session 92955, from clean `6639d3c72bafacd61b18cb63d07ee2f377d93cae`. Startup C: 10.20 GiB and D: 46.36 GiB free met even the original discretionary startup margin; at 62 seconds C: was 10.26 GiB. Output `.tmp/E103-A-blank-s1-segment2-retry1.stdout`/`.stderr`; the earlier failed-attempt console files remain preserved.
- **Actual recovery evidence:** The coordinator verified the full-size scientific replay of recorded generation 150 before writes. The latest resume entry at 09:40:16.298152 records `dirty=false` and `replayed_generations=1`. The metrics still contained exactly 151 contiguous rows 0–150 at the check; the first 30,922 bytes matched the original SHA-256 `e27d93f9dd5c61d0592bd8d8cc4245e7d34c1d604e96ab14b5a1347886997ca0`. No duplicate or new metric row had yet been appended at that snapshot. This verifies recovery integrity, not a completed A1 result.
- Continue the original 200-generation target with one `BelowNormal` job, D: temporary files, 10-second C:/D: floor checks at 5/10 GiB, verified owned-process stop handling, 7,200-second training budget, and 10,500-second wall limit. Goal active and running; A1 final evaluation and the A gate remain pending. Next record at outcome.

## E103 A seed 1 result · 2026-11-12 · Second completed seed meets the criterion
- **Evidence:** `runs/20261112-050319_E103-A-blank-s1/summary.json` reports `finished`, generation 200, 128 evaluation worlds, and hash `aca49c2747a994ab`. The coordinator verified all numerical metrics finite, 200 contiguous rows 0–199, finite checkpoint `t=200`, exact checkpoint-mean/final-genome equality and matching hash, and preservation of the original 30,922-byte metric prefix. Initial source was clean `e0befc6`, recovery source clean `6639d3c`, with one verified replayed generation. Engine and EVAL v1 seed 123 remained unchanged; independent review confirmed provenance and the per-seed criterion.

| Evaluation | Fitness | Accuracy first | Accuracy second | Survival | Eaten |
|---|---:|---:|---:|---:|---:|
| Final learning | 15.376953125 | 0.687602698802948 | 0.8297842144966125 | 0.84765625 | 15609 |
| Final frozen | 0.09375 | 0.5005697011947632 | 0.5078710913658142 | 0.3984375 | 7934 |
| Start learning | 0.4140625 | 0.5018976926803589 | 0.5329878926277161 | 0.38671875 | 8428 |

- **Criterion and interpretation:** Accuracy gain 0.14218151569366455 exceeds 0.03; the learning-minus-frozen fitness gap is 15.283203125, a clear lesion effect. Starting evaluation exactly matches E100 blank seed 1. Both completed A seeds meet their per-seed requirements, but the planned three-seed comparison awaits A2 before a full gate decision. Pooled accuracy can reflect survival/encounter selection, and the combined-update lesion supports useful plasticity without isolating individual hidden-rule inference. No roadmap advancement or gate tag yet.
- **Timing and recovery accounting:** Recovery process exited 0 after 2,643.3345922 seconds, ending 10:24:10.1522338 Asia/Calcutta; session 92955 and monitor 138 are terminal. Retained training metric times total 8,967.139 seconds. Final summary `train_minutes=43.29` covers the recovery segment only. Full actual wall time is unknown because the original process ended abnormally; replay consumed additional compute not represented by the retained generation-150 row's original timing. Do not report either partial timing as total experiment duration.
- **Resources:** At 09:53:49 during recovery, C: was 5.99 GiB free and memory commitment 44.61/47.92 GiB (93%). Observed pagefile allocation grew from 11,372 to 16,969 MiB; this growth contributed to disk consumption, but cannot be attributed specifically to Seed. No resource-exhaustion Event 2004 was found in the preceding two hours. The runtime guard never fired during this recovery. Completion snapshot: C: 10.67/373.67 GiB and D: 46.36/100 GiB free.
- **Decision:** Regenerated and checked the ten-row results table; A1 is finished at 200/200 and README now reflects two completed seeds. Commit/push the finished A1 folder and records, then run the unchanged third seed:
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E103-A-blank-s2 --start blank --generations 200 --seed 2 --max-train-seconds 7200
  ```
  Keep existing resource guards and EVAL v1. No new hypothesis or tuning follows from these results.

## E103 A seed 2 launch · 2026-11-12
- A1 result commit `62376af` was pushed and verified private. A2 launched at 10:27:52.2149612 Asia/Calcutta, PID 45600, supervising session 24812; folder `runs/20261112-102755_E103-A-blank-s2` records clean source `62376af0b74f7d788d8033ce8582972a47a8fa9d`. The coordinator verified its starting array exactly matches E100 blank seed 2, hash `03202e865b4a4255`, and the preregistered blank/seed-2/200-generation/population-128/8-world settings with plasticity enabled and 128 evaluation worlds.
- Training limit 7,200 seconds, checkpoint every 10 generations, one `BelowNormal` job, temporary files on D:. Startup free space C: 10.67 GiB and D: 46.36 GiB; retain 10-second guard polling, C:/D: floors 5/10 GiB, 10,500-second wall limit, and verified owned-handle `.Kill($true)` termination. Output `.tmp/E103-A-blank-s2-segment1.stdout`/`.stderr`; initial stderr empty. Goal active, A2 running, full A gate pending; next record at a meaningful checkpoint or outcome.

## E103 A seed 2 · 2026-11-12 · Planned stop at generation 167
- **Outcome:** The first bounded segment exited 0 after 7,246.8489643 seconds, ending 12:28:39.0665229 Asia/Calcutta. `runs/20261112-102755_E103-A-blank-s2/summary.json` reports `interrupted`, generation 167, `train_minutes` 120.71, and hash `07b614c4c04eb200`. This is a normal scheduled stop, not a failed experiment. Initial source was clean `62376af`; target 200 remains unchanged, with 33 generations left.
- **Verification:** The coordinator verified finite checkpoint `t=167` and matching hash, 167 contiguous finite metric rows 0–166, and starting genome equality with E100 blank seed 2. No final-genome file or evaluation exists yet, as expected. Metric seconds total 7,242.019. Preserved-file baseline: 34,156 bytes, SHA-256 `3e92da1159c2b1028cc8901ec7200f0d4c9537931ef534a0cef145d72700988c`; there is no metrics/checkpoint tail discrepancy.
- **Resources:** Session 24812 and monitor 150 are terminal; authoritative CIM found zero experiment processes. C: 10.66/373.67 GiB and D: 46.36/100 GiB free; free physical/virtual memory about 7.41/6.53 GiB. Full A gate remains pending A2 completion.
- **Decision:** Regenerated and checked eleven results rows, with A2 interrupted at 167/200 and no evaluation scores. Commit/push this run checkpoint and records, then continue unchanged settings and resource guards using:
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_experiment.py --resume runs/20261112-102755_E103-A-blank-s2 --max-train-seconds 7200
  ```

## E103 A seed 2 continuation · 2026-11-12
- Segment 2 resumed at 12:31:23.0679343 Asia/Calcutta, launcher PID 33960, supervising session 92965, from clean `1a7c4e3fb8b4896030b73737ae9dea631bf3500c`; private push verified. Config resume at 12:31:25.803986 records `dirty=false`, `replayed_generations=0`. Stdout confirms generation 167, stderr empty, and the coordinator verified the original 34,156-byte metric prefix still matches SHA-256 `3e92da1159c2b1028cc8901ec7200f0d4c9537931ef534a0cef145d72700988c`.
- Same run folder and target 200; training limit 7,200 seconds, low priority, D: temporary files, 10-second C:/D: floor checks 5/10 GiB, wall limit 10,500 seconds. Startup free space C: 10.66 GiB/D: 46.36 GiB. Console `.tmp/E103-A-blank-s2-segment2.stdout`/`.stderr`. Goal active; final A2 evaluation and full A gate pending. Next record at outcome.

## Commit attribution correction · 2026-11-12
- The owner noticed commits attributed to `@seed`. Repository review verified the authenticated GitHub account and private repository owner are `Winter-Soren`; incorrect per-command author overrides affected 19 commits. This was commit metadata, not use of the wrong authenticated account.
- Repo-local author name and privacy-preserving GitHub noreply identity are now set for verified `Winter-Soren`; effective author and committer values were checked. Future commits must use that verified local identity without the old per-command overrides. The next documentation commit will be checked through GitHub's API for both author and committer attribution to `Winter-Soren`.
- Existing history is not rewritten, and experiment source-commit references remain unchanged. No experiment artifacts or protocol were changed for this correction.

## Authorized historical attribution repair · 2026-11-12
- The owner subsequently authorized repairing historical commit authorship. The metadata-only repair is complete: local and private-remote `main` are `f73f336ae4c838d05cdbdb3383af097baaa8a1fe`. GitHub verified author and committer attribution to `Winter-Soren` for all 19 repaired commits and the audit tip (20/20).
- The committed mapping is `experiments/COMMIT_ATTRIBUTION_MAP_20261112.json`. Verification covered 21 mapped historical commits: trees, messages, timestamps, and topology unchanged; the signed root was preserved. Original history remains locally under `refs/provenance/pre-author-repair-20261112` and in a verified ignored bundle. No past LOG entry or run configuration was edited; original source hashes remain the audit references.
- For active A2, recorded initial `62376af0b74f7d788d8033ce8582972a47a8fa9d` maps to `9d56387ee7d8a7ef711474c176ceecdd9f2a7e5f`, and recorded resume `1a7c4e3fb8b4896030b73737ae9dea631bf3500c` maps to `fa013b8de9f0a2546f098f383a678344c433cfac`, with identical trees. Future commits use the verified repo-local owner identity without overrides. README now links the map and research evidence; AGENTS adds ongoing verified-result README updates and verified-identity guidance. A2 remains running pending final evaluation; no gate decision is recorded here.

## E103 A seed 2 result and full gate decision · 2026-11-12 · PASS in 3 of 3 seeds
- **Evidence:** `runs/20261112-102755_E103-A-blank-s2/summary.json` reports `finished`, generation 200, 128 evaluation worlds, and hash `17191e72299b25cc`. The coordinator verified 200 contiguous finite metric rows 0–199, finite checkpoint `t=200`, exact checkpoint-mean/final-genome equality and hash, exact starting evaluation match to E100 blank seed 2, and unchanged original 34,156-byte metric prefix. Recorded initial and resume commits were clean, with zero replayed generations; their unchanged-tree mappings are recorded above. Independent review confirmed all three A results, unchanged experimental settings, and EVAL v1 seed 123.

| A2 evaluation | Fitness | Accuracy first | Accuracy second | Survival | Eaten |
|---|---:|---:|---:|---:|---:|
| Final learning | 14.8671875 | 0.6937439441680908 | 0.8225241303443909 | 0.849609375 | 15094 |
| Final frozen | -0.103515625 | 0.4872659146785736 | 0.516306459903717 | 0.36328125 | 7885 |
| Start learning | 0.23046875 | 0.5063009858131409 | 0.5091709494590759 | 0.388671875 | 8122 |

- **Preregistered gate:** At least 2 of 3 evolution seeds must have final-learning accuracy gain strictly greater than 0.03 and clearly lower frozen fitness. All three satisfy both conditions; the observed lesion gaps are large rather than borderline. No numerical cutoff or evaluation protocol was changed after seeing results.

| Evolution seed | Accuracy gain | Learning-minus-frozen fitness | Per-seed decision |
|---|---:|---:|---|
| 0 | 0.13528215885162354 | 16.302734375 | Pass |
| 1 | 0.14218151569366455 | 15.283203125 | Pass |
| 2 | 0.12878018617630005 | 14.970703125 | Pass |

- **Interpretation:** E103 A passes in 3/3 seeds: blank-start evolution reached the preregistered learning criterion, with a substantial fitness benefit from enabled plasticity under fixed EVAL v1. This is a replicated result in the toy domain, not proof of general intelligence. Pooled eating-event accuracy can reflect survival/encounter selection, and the lesion jointly disables Hebbian and predictor updates while leaving recurrent state and energy feedback. It does not isolate individual hidden-rule inference. The matched independently evolved no-plasticity comparison C is still pending; do not claim its outcome.
- **Timing:** A2 continuation exited 0 after 2,180.0340983 seconds, ending 13:07:43.1038774 Asia/Calcutta. Combined A2 process wall time is 9,426.8830626 seconds, excluding the between-segment pause; retained training metric times total 9,361.847 seconds. Final `train_minutes=35.33` covers only the continuation segment. Session 92965 and monitor 155 are terminal; authoritative CIM found zero experiment processes.
- **Resources:** C: 10.58/373.67 GiB and D: 46.36/100 GiB free; free physical/virtual memory about 10.17/7.75 GiB. No run artifacts were manually edited for this result record. Regenerated and checked eleven finished results rows, completed README's three-seed table and limitations, and retained the reviewed AGENTS guidance additions.
- **Decision:** Commit/push the completed A2 run and coherent research records using the verified owner identity, then mark the new milestone `v0.1-a-evalv1`. Keep the original B/C comparison queue; no roadmap expansion before those core comparisons. Next run:
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E104-B-bio-s0 --start biology --sigma 0.05 --lr 0.02 --generations 200 --seed 0 --max-train-seconds 7200
  ```
  B's preregistered baseline remains E100 biology learning fitness 5.12890625; final fitness must strictly exceed it in at least 2 of 3 evolution seeds. Preserve EVAL v1, low priority, one compute job, D: temporary files, and the existing storage/wall-time safeguards.

## E104 B seed 0 launch · 2026-11-12
- A's 3/3 gate result was committed and pushed as `51bdac9`, with milestone `v0.1-a-evalv1`; GitHub author and committer attribution to `Winter-Soren` verified. B0 launched at 13:19:43.8480236 Asia/Calcutta, launcher PID 37460, supervising session 97097; folder `runs/20261112-131946_E104-B-bio-s0` records clean source `51bdac97c7ee65cc5e6bd7249d78a6bd0152ff9f`.
- The coordinator verified biology start, seed 0, target 200, sigma 0.05, learning rate 0.02, population 128 × 8 worlds, evaluation 128 worlds, plasticity enabled, and no configuration overrides. Starting genome exactly matches E100 biology seed 0, hash `72caddf0cdd93943`. B's baseline remains 5.12890625; complete all three evolution seeds before deciding its gate.
- Training limit 7,200 seconds, checkpoint every 10 generations, `BelowNormal`, D: temporary files; 10-second guard polling with C:/D: floors 5/10 GiB and 10,500-second wall limit. Startup free space C: 10.59/373.67 GiB and D: 46.36/100 GiB. Console `.tmp/E104-B-bio-s0-segment1.stdout`/`.stderr`, initial stderr empty. B0 running; B1/B2 and C remain. Next record at outcome.

## Core-comparison interpretation note · 2026-11-12
- Read-only scientific review: A's 3/3 lesion results establish a within-genome fitness benefit from enabled predictor and Hebbian updates; they do not establish that every successful policy needs synaptic plasticity.
- Preserve the planned comparisons: report paired A `final_learning` minus C `final_frozen` fitness for all three seeds. C has no new numerical gate; mixed differences remain mixed. B passes only if fitness strictly exceeds 5.12890625 in at least 2/3 seeds. A versus B cannot isolate initialization because sigma and learning rate also differ.
- After all core runs, if C favors A, examine improving plastic learning: a B pass supports improving the hand-set prior, while B failure motivates checking optimization/initialization. If C matches or exceeds A, inspect nonplastic strategies and task incentives before claiming plasticity is necessary. A passed, so do not invoke its failure-escalation queue.
- Per-agent survivor-aware accuracy and reward-modulation diagnostics remain deferred candidates requiring separate preregistration; any EVAL change requires owner approval and versioning. These are decision notes only: no new experiment, gate, or setting change.

## E104 B seed 0 memory failure and recovery plan · 2026-11-12
- **Outcome:** The first segment exited 1 at 13:59:27.2419455 Asia/Calcutta after 2,383.3909313 seconds. Preserved stderr reports `jax.errors.JaxRuntimeError` with an out-of-memory allocation of 2,596,544,168 bytes. The disk guard did not fire; the exit snapshot was C: 10.54 GiB and D: 46.36 GiB free. The cause of memory pressure at failure is not established by later resource counters. Supervising session 97097 and monitor 169 are terminal; after the owner's pause and resume, authoritative CIM found zero experiment processes. Original console files remain `.tmp/E104-B-bio-s0-segment1.stdout`/`.stderr`.
- **Preserved evidence:** The coordinator verified finite checkpoint `t=30`, hash `ad186fde58ab5fc9`, and 33 contiguous finite metric rows 0–32 in `runs/20261112-131946_E104-B-bio-s0`. Metric seconds total 2,374.328; the final recorded row took 202.234 seconds. The original metric file is 6,723 bytes, SHA-256 `3d093802247823f26518fa58c1f417b87e69551d36544e1ac5bbda180d7e074c`. Preserve this exact filesystem-byte prefix through recovery. The untouched summary still says `running`, generation 0: it is stale, and no final evaluation exists. Leave `results/RESULTS.md` unchanged pending recovery rather than promote that stale summary to a current result; no scientific result or gate decision is recorded here.
- **Recovery condition:** Use the existing verified crash-tail replay to reproduce recorded generations 30–32 before permitting writes, then continue the original 200-generation target without duplicating rows or altering settings, source, or EVAL v1. Replay remains pending. Commit the unchanged B0 folder with this appended record and `TASK_STATE.md` first, using the verified repo-local owner identity; hold tracked writes until the coordinator confirms the clean-source launch.
- **Resource preflight and guards:** A post-exit recovery-planning snapshot was C: 16.31/373.67 GiB and D: 46.34/100 GiB free, free physical memory 11.85 GiB, available commit 9.41 GiB (37.84/47.26 GiB committed), and pagefile allocation 16,290 MiB. Recheck immediately before launch: require C:/D: at least 10 GiB each and physical/available-commit headroom at least 8 GiB each. Runtime floors are C: 5 GiB, D: 10 GiB, and available commit 3 GiB, checked every 10 seconds; retain verified owned-process stop handling, one `BelowNormal` compute job, D: temporary files, training limit 7,200 seconds, and full-wall limit 10,500 seconds.
- **Next command:**
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_experiment.py --resume runs/20261112-131946_E104-B-bio-s0 --max-train-seconds 7200
  ```
  Verify the three replayed generations and unchanged metric-byte prefix before accepting continuation. B0 is stopped pending this recovery; B1/B2 and C remain queued. This is operational recovery, not a change to the preregistered hypothesis or B gate.

## E104 B seed 0 recovery verified and continuation running · 2026-11-12
- B0 resumed at 20:19:51.6502429 Asia/Calcutta, launcher PID 25240, supervising session 57093, from clean `f12d3debc56518a029832f4231f34dfbb7d5c977`. The preservation commit was pushed to private `Winter-Soren/seed` and GitHub verified both author and committer as `Winter-Soren`. Recovery output uses distinct `.tmp/E104-B-bio-s0-segment2.stdout`/`.stderr` files; original segment output remains preserved.
- **Actual recovery evidence:** The coordinator verified resume metadata at 20:19:55.103497 with `dirty=false` and `replayed_generations=3`. Recorded generations 30–32 replayed successfully before writes; the subsequent snapshot contained 35 contiguous metric rows 0–34, showing new generations appended without duplication. The original 6,723-byte prefix still matched SHA-256 `3d093802247823f26518fa58c1f417b87e69551d36544e1ac5bbda180d7e074c`, and stderr was empty. This verifies recovery integrity, not a final B result.
- **Resources and supervision:** Startup C: 18.69 GiB and D: 46.34 GiB free, physical headroom 11.62 GiB, and available commit 8.94 GiB satisfied the 10 GiB disk and 8 GiB memory startup thresholds. At 192 seconds, C:/D: remained 18.69/46.34 GiB, available commit was 5.88 GiB, and physical headroom was 9.10 GiB. The active supervisor obtains authoritative read-only memory counters; runtime C:/D:/available-commit floors remain 5/10/3 GiB, with ten-second waits plus query overhead. Counter errors invoke checked owned-process `.Kill($true)` cleanup. Preserve one `BelowNormal` compute job, D: temporary files, training limit 7,200 seconds, and full-wall limit 10,500 seconds.
- **State and next record:** Session 57093 is live; earlier monitor cells 169 and g9:6 are terminal. Goal active, original target 200 and scientific settings/EVAL v1 unchanged. Commit only this LOG and `TASK_STATE.md` update, excluding live run artifacts, RESULTS, and README; then hold documentation writes until outcome. B0 final evaluation, B1/B2, and C remain pending.

## Core queue steering toward the plasticity control · 2026-11-12
- Following the owner's question about the next plasticity-focused experiments after A passed in 3/3 seeds, the coordinator prioritizes the already-preregistered C control: finish current B0, then run C seeds 0/1/2 before remaining B1/B2. Do not interrupt B0. This changes queue order only; preserve every planned run, setting, gate, and EVAL v1.
- C is the more directly informative next comparison for plasticity: it tests independently evolved nonplastic strategies against A's plastic evolution under otherwise matched settings. Report paired A `final_learning` minus C `final_frozen` for all three seeds, with no new numerical gate. B asks whether gentle evolution improves the biology prior and uses different ES parameters; it remains planned and its gate still requires final-learning fitness strictly above 5.12890625 in at least 2/3 seeds.
- Separate Hebbian-versus-predictor and reward-modulation diagnostics remain possible future questions, not authorized implementations, runs, or evaluation changes. No source edit or new experiment follows from this queue update; record C's evidence before choosing further work.

## E104 B seed 0 memory-guard stop · 2026-11-12
- **Outcome:** At elapsed 2,506 seconds of the continuation, available commit fell to 2.68 GiB, below the 3 GiB runtime floor; C:/D: remained 18.71/46.34 GiB free. Supervisor 57093 and monitor g9:11 are terminal; supervisor exit 1 is the expected guard outcome. Launcher termination was verified, and authoritative CIM at 21:02:03.2552534 Asia/Calcutta found zero matching experiment processes. This was a guard termination, not a new JAX exception; stderr remained empty.
- **Preserved evidence:** The coordinator verified finite checkpoint `t=80`, hash `a2ab513f53247ece`, and 88 contiguous finite metric rows 0–87; retained metric seconds total 4,750.455. The complete 17,940-byte metric file has SHA-256 `fcf36a7062df0ff447e017b30fd9a1f581b8e1c03ed467d5c6c752645df69adc`. The original 6,723-byte prefix remains unchanged at SHA-256 `3d093802247823f26518fa58c1f417b87e69551d36544e1ac5bbda180d7e074c`. Preserve both baselines; the next recovery must reproduce eight recorded rows 80–87 before writes. The untouched summary says `running`, generation 33, and is stale. No final evaluation exists; README and RESULTS remain unchanged.
- **Operational lesson:** Post-exit free physical memory was 11.14 GiB and available commit 8.43 GiB. That increase does not establish which process caused pressure. The previous 8 GiB startup commit margin was insufficient for this continuation; next guarded startup requires available commit at least 10 GiB, physical headroom at least 8 GiB, and C:/D: at least 10 GiB each. Runtime C:/D:/available-commit floors remain 5/10/3 GiB; retain existing priority, temporary-file, timing, and checked-stop safeguards.
- **Decision:** Preserve the unchanged B0 folder with this appended LOG and updated STATE, commit/push under the verified owner identity, and hold writes. No restart now: goal active, B0 stopped pending headroom. The owner was asked to free about 2 GiB by closing unused apps or wait; no unrelated workload was stopped. Queue remains B0 → C0/1/2 → B1/B2. No scientific setting, EVAL v1, or gate changed, and no B scientific result is claimed.

## E104 B seed 0 second recovery verified · 2026-11-12
- **Launch:** Segment 3 started at 21:08:48.1749019 Asia/Calcutta, launcher PID 28012, supervising session 20596, from clean `f05751334da36f3cfeb3b7081b9ea1e4831733af`. Guard-stop preservation was pushed to the private repository with GitHub author and committer verified as `Winter-Soren`. Output uses `.tmp/E104-B-bio-s0-segment3.stdout`/`.stderr`; earlier segment outputs remain preserved.
- **Actual recovery:** The coordinator verified resume metadata at 21:08:50.452534 with `dirty=false`, `replayed_generations=8`. Recorded generations 80–87 reproduced before writes; 89 contiguous rows 0–88 then showed continuation. Both the original 6,723-byte prefix (SHA-256 `3d093802247823f26518fa58c1f417b87e69551d36544e1ac5bbda180d7e074c`) and guard-stop 17,940-byte prefix (`fcf36a7062df0ff447e017b30fd9a1f581b8e1c03ed467d5c6c752645df69adc`) matched unchanged. Stderr was empty. This is verified recovery, not a final B result or scientific advance.
- **Headroom:** The owner closed apps, then authorized an Oracle stop. The separate worker found Oracle already stopped and no `oracle.exe`; no stop action was taken and browsers were untouched. Startup C:/D: was 24.00/46.34 GiB, physical memory 12.21 GiB, and available commit 15.71 GiB. At elapsed 312 seconds: C:/D: 23.99/46.34 GiB, physical 9.72 GiB, available commit 12.66 GiB. The revised startup 10 GiB commit requirement and all runtime safeguards remain unchanged.
- **Current state:** Goal active, B0 running toward its unchanged 200-generation target; queue B0 → C0/1/2 → B1/B2. Session 20596 is the definitive ongoing handle; completion of bounded monitor cell g9:18 does not mean the experiment ended. Commit only LOG/TASK_STATE, excluding live run artifacts and other work, then hold writes until outcome. No settings, protocol, or gate changed.

## E104 B seed 0 result · 2026-11-12 · First seed improves the biology prior
- **Evidence:** `runs/20261112-131946_E104-B-bio-s0/summary.json` reports `finished`, generation 200, 128 evaluation worlds, and final hash `1a34591381de0fbb`. The coordinator verified finite checkpoint `t=200`, exact checkpoint-mean/final-genome equality and hash, 200 contiguous finite rows 0–199, and exact starting-genome and start-evaluation matches to E100 biology seed 0. Initial source `51bdac9` and resume sources `f12d3de`/`f057513` were clean; both recoveries verified their recorded tails (3 and 8 generations) before writes. Engine, scientific settings, and EVAL v1 remained unchanged.

| Evaluation | Fitness | Accuracy first | Accuracy second | Survival | Eaten |
|---|---:|---:|---:|---:|---:|
| Final learning | 15.90234375 | 0.685681164264679 | 0.8353021144866943 | 0.841796875 | 15990 |
| Final frozen | -0.048828125 | 0.4986666738986969 | 0.4978148639202118 | 0.37109375 | 7767 |
| Start learning | 5.12890625 | 0.5685573816299438 | 0.6739441156387329 | 0.6015625 | 11618 |

- **Per-seed criterion and interpretation:** Final-learning fitness exceeds the preregistered prior baseline by 10.7734375. Within-life accuracy gain is 0.14962095022201538 and the learning-minus-frozen fitness gap is 15.951171875. B0 meets B's per-seed improvement criterion; this is one successful seed, not the three-seed B gate, which awaits B1/B2 and requires improvement in at least 2/3. The combined predictor/Hebbian lesion supports a fitness benefit from enabled plasticity in this genome; pooled accuracy and retained recurrent-state/energy-feedback limitations remain. C is still pending, and A/B differ in both initialization and ES parameters. No gate tag or roadmap advancement follows.
- **Artifact preservation:** Final metrics are 41,083 bytes, SHA-256 `a3feb80882288dcddf2dd41beaf0e0e036b5439c6d3b19cafaf69047cbb94118`; retained metric seconds total 9,563.099. The original 6,723-byte prefix still matches `3d093802247823f26518fa58c1f417b87e69551d36544e1ac5bbda180d7e074c`, and the 17,940-byte guard-stop prefix still matches `fcf36a7062df0ff447e017b30fd9a1f581b8e1c03ed467d5c6c752645df69adc`. No run file was manually edited.
- **Timing and resources:** Segment 3 exited 0 after 5,230.8544356 seconds, ending 22:35:59.0316370 Asia/Calcutta. Final `train_minutes=86.56` covers this segment only. The first segment's wall time was 2,383.3909313 seconds; segment 2 has a guard observation at elapsed 2,506 seconds, not a verified exact total wall duration. Do not sum these into an exact cumulative duration; replay compute is also additional to retained metric timings. Supervisor 20596, monitor g9:35, and follow-up polls are terminal; authoritative CIM found zero experiment processes. Completion snapshots: physical/available commit 12.90/15.48 GiB, C: 23.94/373.67 GiB and D: 46.34/100 GiB free.
- **Record and decision:** Regenerated and checked 12 results rows with `.\.venv\Scripts\python.exe -X utf8 scripts/summarize.py`; the initial default-encoding invocation failed on Windows cp1252, and the UTF-8 invocation successfully rewrote the complete table without source changes. Coordinate the completed B0 folder, RESULTS, LOG, STATE, and README result update in one owner-attributed commit/push after the README worker is ready. Goal active; retain all guards and the queue C0 → C1 → C2 → B1 → B2. Next preregistered command after a clean-source preflight:
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_experiment.py --name E105-C-noplastic-s0 --start blank --no-plastic --generations 200 --seed 0 --max-train-seconds 7200
  ```
  Keep EVAL v1 and all settings unchanged; compare C `final_frozen` against paired A `final_learning` and report all seeds without a new numerical C gate.

## E105 C seed 0 launch · 2026-11-12
- B0's completed result was pushed as `8b42128fac8e18ac094173a2f47e771ffb71a809`; private remote and GitHub author/committer `Winter-Soren` were verified. C0 started at 22:41:31.0130171 Asia/Calcutta, launcher PID 12064 and child PID 43816, both verified `BelowNormal` through CIM. Supervising session 32581 is live. Folder `runs/20261112-224133_E105-C-noplastic-s0` records creation at 22:41:33.503011 and clean source `8b42128fac8e18ac094173a2f47e771ffb71a809`. All B0 handles were terminal and no other experiment job existed before launch.
- **Matched control verified:** Blank start, seed 0, target 200, `no_plastic=true`, population 128, eight training worlds, sigma/lr 0.1/0.1, EVAL v1 128 worlds, checkpoint every 10 generations, training budget 7,200 seconds, no overrides. The coordinator verified scientific settings/configuration match A0 except the plasticity switch; starting-genome bytes and hash exactly match A0, `9a761b031011f9ee`.
- **Resources and output:** Startup C:/D: free space 23.94/46.34 GiB, physical memory 12.54 GiB, available commit 14.89 GiB. Keep startup disk/physical/commit thresholds 10/8/10 GiB, runtime C:/D:/available-commit floors 5/10/3 GiB, ten-second waits plus query overhead, checked owned-process stopping, and full-wall limit 10,500 seconds. TEMP/TMP are on D: and PYTHONUTF8 is enabled. Unique output `.tmp/E105-C-noplastic-s0-segment1.stdout`/`.stderr` was empty at startup.
- **Current state:** Goal active, C0 running, no result claimed. Its control score is `final_frozen`, to compare with paired A0 `final_learning`; C0 `final_learning` re-enables plasticity and remains diagnostic only. Remaining queue C1 → C2 → B1 → B2. Commit only LOG/TASK_STATE for this launch, excluding live C0 and other artifacts, then hold writes until outcome. Settings, EVAL v1, and existing gates are unchanged.

## E105 C seed 0 saved segment · 2026-11-13 · Training limit at generation 127
- **Observed outcome:** Following the goal pause/resume, inspection around 23:15 Asia/Calcutta found `runs/20261112-224133_E105-C-noplastic-s0/summary.json` at `interrupted`, generation 127, `train_minutes=120.34`, hash `15cea5da1a2cb121`. Stdout explicitly reports the training time limit and checkpoint save; stderr is empty. This is a scheduled segment stop, not a final C result. Old supervisor 32581 cannot provide a current process result and monitor g9:50 belongs to a stale host; process exit code and full wall duration are unavailable. Authoritative CIM found zero experiment processes; do not infer timing or exit status from stale handles.
- **Preserved evidence:** The coordinator verified finite checkpoint `t=127`, matching hash, 127 contiguous finite metric rows 0–126, and exact A0 starting-genome equality. Metric seconds total 7,220.171. The original metric file is 25,909 bytes, SHA-256 `e88beaabbd87c1ed13de39f28a50349c45f74fb15d4bf0b778413f95643cfed7`; preserve this entire byte prefix on continuation. No checkpoint/metric tail discrepancy exists, so no tail replay is needed. There is no final-genome file or final evaluation. Clean initial source `8b42128`, target 200, `no_plastic=true`, and all preregistered settings/EVAL v1 remain unchanged; 73 generations remain.
- **Resume condition:** Current physical/available-commit headroom is 5.84/7.32 GiB, below startup floors 8/10 GiB. C:/D: free space is 22.40/46.34 GiB (24,046,596,096 / 49,757,896,704 bytes), above the 10 GiB disk floors. Oracle DB is stopped; browsers are excluded from shutdown. Goal active, C0 stopped pending headroom; do not restart now or lower safeguards. Preserve the existing runtime C:/D:/available-commit floors 5/10/3 GiB, priority, temporary-file, timing, and checked-stop controls.
- **Record and next action:** Regenerated and checked 13 results rows using `.\.venv\Scripts\python.exe -X utf8 scripts/summarize.py`; C0 is interrupted at 127/200 with no evaluation scores. Commit/push the unchanged C0 folder, RESULTS, appended LOG, and STATE under the verified owner identity, excluding README/source, then hold writes. Once a fresh preflight meets the startup guards, continue the same folder and verify the recorded metric prefix:
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_experiment.py --resume runs/20261112-224133_E105-C-noplastic-s0 --max-train-seconds 7200
  ```
  C0 `final_frozen` remains the control score against A0 `final_learning`; no scientific result or new gate is claimed. Remaining queue after C0: C1 → C2 → B1 → B2.

## E105 C seed 0 continuation verified · 2026-11-13
- Segment 2 started at 23:20:15.0854782 Asia/Calcutta, launcher PID 48296 and child PID 4180, both verified `BelowNormal`; supervising session 59158 is live. Preservation commit `1c4a60b33ce886e71a22e523b9daf1a018d1a77b` was pushed and verified private with GitHub author/committer `Winter-Soren`. Prior handles were stale, and authoritative CIM confirmed zero experiment processes before this launch.
- **Resume verification:** Same C0 folder, target 200, and no-plastic scientific settings unchanged. Resume metadata at 23:20:36.914407 records clean source `1c4a60b33ce886e71a22e523b9daf1a018d1a77b` and `replayed_generations=0`. The coordinator verified the original 25,909-byte prefix still matches SHA-256 `e88beaabbd87c1ed13de39f28a50349c45f74fb15d4bf0b778413f95643cfed7`. The initial check had 127 contiguous rows 0–126; a subsequent check verified 128 contiguous rows 0–127, including the first new generation 127 at 118.016 seconds. Independent audit confirmed the prefix and matched control. Unique `.tmp/E105-C-noplastic-s0-segment2.stdout`/`.stderr` files were empty initially; stderr remained empty after the new row. Original segment output remains preserved.
- **Headroom and safeguards:** After the owner's go-ahead and resource recovery, startup physical/available commit was 8.55/11.73 GiB, C:/D: 22.39/46.34 GiB. Original startup requirements of physical 8 GiB, available commit 10 GiB, and each disk 10 GiB were met, not lowered. At 65 seconds, physical/available commit was 6.68/8.77 GiB; available commit remained above the 3 GiB runtime floor. Runtime C:/D:/available-commit floors 5/10/3 GiB, checked stop handling, 7,200-second training limit, 10,500-second full-wall limit, and D: temporary files remain unchanged.
- **Current state:** Goal active; monitor supervisor 59158 to finish C0, then C1 → C2 → B1 → B2. C's control score remains `final_frozen`; no final result or scientific advance is claimed. Commit only LOG/TASK_STATE, excluding live artifacts, README, and RESULTS, then hold writes until outcome.
