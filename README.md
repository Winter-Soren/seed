# Seed

An experiment in AI that is never trained with labels. Tiny genomes (155 numbers) hold *rules* for growing a brain. Each brain learns on its own during its life through local, dopamine-like plasticity. Evolution keeps the genomes whose brains learn fastest in worlds they have never seen.

**Status (v0, 12 Nov 2026):** The first blank-start evolution seed meets the preregistered learning criterion after 200 generations under EVAL v1 (128 worlds, evaluation seed 123). Food-choice accuracy rises from **70.49% to 84.02%** within life; fitness is **16.591796875**, falling to **0.2890625** with plasticity disabled. **The three-seed gate is still pending:** seeds 1 and 2 must finish before a replicated conclusion. See [results](results/RESULTS.md) and the [lab notebook](experiments/LOG.md).

The zero-generation biology reference scores 5.12890625 fitness, frozen 0, with accuracy 56.86% → 67.39%. Its three runs are deterministic duplicates, not independent trials. Evolution from the biology prior and the matched no-plasticity evolutionary control remain untested.

Aggregate accuracy does not separate individual learning from survival or encounter selection; the lesion supports useful plasticity without isolating those mechanisms.

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
