# Seed

An experiment in AI that is never trained with labels. Tiny genomes (155 numbers) hold *rules* for growing a brain. Each brain learns on its own during its life through local, dopamine-like plasticity. Evolution keeps the genomes whose brains learn fastest in worlds they have never seen.

## Research progress

**Verified milestone, 12 Nov 2026: experiment A passes in 3 of 3 evolution seeds.** After 200 generations from a blank start, all three genomes exceed the preregistered within-life accuracy gain and lose most of their fitness when plasticity is disabled. The gate required at least 2 of 3 seeds.

Results use **EVAL v1**: 128 worlds, evaluation seed 123. Fitness is food minus poison per agent; accuracy measures food choices among eating events. Frozen means the same genome evaluated with plasticity disabled.

| A evolution seed | Learning fitness | Frozen fitness | Accuracy first → second | Gain (percentage points) |
|---|---:|---:|---|---:|
| 0 | 16.59 | 0.29 | 70.49% → 84.02% | +13.53 |
| 1 | 15.38 | 0.09 | 68.76% → 82.98% | +14.22 |
| 2 | 14.87 | -0.10 | 69.37% → 82.25% | +12.88 |

The fitness lesion gaps are **16.30**, **15.28**, and **14.97**. All three seeds exceed the required accuracy gain of 3 percentage points and show a large fitness benefit from enabled plasticity. Evolution from a blank start has reached this project's first replicated learning criterion.

The unevolved biology reference scores **5.13**, frozen **0.00**, with accuracy **56.86% → 67.39%**. Its three evaluations repeat the same genome and worlds, so they are repeatability checks, not independent trials. The random starting genomes score **-0.004**, **0.414**, and **0.230**.

**Limits:** this is a small grid-world task. Pooled accuracy can rise through survival or encounter selection as well as individual learning. The lesion disables Hebbian and predictor updates together while preserving recurrent state and energy feedback. It supports useful plasticity here, not isolated proof of hidden-rule inference or general intelligence. Superiority over an independently evolved no-plasticity control remains untested.

**Next comparisons:** run the preregistered biology-start experiment B and matched no-plasticity evolutionary control C. B tests whether gentle evolution improves the biology reference; C tests the matched evolutionary comparison with plasticity disabled. Both remain pending, with settings and evaluation worlds fixed. Complete these core comparisons before expanding the roadmap.

Values here are rounded. Exact values are in the [seed 0 summary](runs/20261112-023245_E103-A-blank-s0/summary.json), [seed 1 summary](runs/20261112-050319_E103-A-blank-s1/summary.json), [seed 2 summary](runs/20261112-102755_E103-A-blank-s2/summary.json), and [lab notebook](experiments/LOG.md); the generated [results table](results/RESULTS.md) indexes all runs. Update this section after verified results and gate decisions.

**Provenance:** the [commit mapping](experiments/COMMIT_ATTRIBUTION_MAP_20261112.json) connects recorded source hashes to commits with corrected authorship. Historical run records retain their original hashes; the authorship repair does not change experiment contents.

## Run it
From a checkout on D:, use PowerShell to keep the environment and installation temporary files off C:.

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TEMP = $env:TMP = (Resolve-Path .tmp).Path
$env:PYTHONUTF8 = '1'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
.\.venv\Scripts\python.exe scripts/run_experiment.py --name smoke --quick
.\.venv\Scripts\python.exe scripts/summarize.py
```

The `--quick` smoke run is an integration check only, not research evidence.

Preregister research runs in `experiments/LOG.md` and commit before launching. Run one long job at low priority, check free space on both drives, and preserve all run folders. The laptop benchmark averaged about 49.5 seconds per generation after the first; 200 generations need bounded sessions for headroom:

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --name A-blank --start blank --generations 200 --seed 0 --max-train-seconds 7200
.\.venv\Scripts\python.exe scripts/run_experiment.py --resume 'runs/<run-folder>' --max-train-seconds 7200
```

Resume only an unfinished run; its original generation target is retained. The training limit excludes setup/evaluation and can overrun by one generation, so monitor full wall time to stay below 3 hours. Scientific comparisons require evolution seeds 0, 1, and 2. The session's launch guards require at least 5 GiB free on C: and 10 GiB on D:.

In Colab, open `notebooks/Seed_v0.ipynb`, set the runtime to T4 GPU, then Run all.

## Where things are
- `HANDOFF.md`: full context and plan for agents continuing the work
- `AGENTS.md`: research and integrity rules
- `seed/seed.py`: the whole engine (JAX)
