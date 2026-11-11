# Seed

An experiment in AI that is never trained with labels. Tiny genomes (155 numbers) hold *rules* for growing a brain. Each brain learns on its own during its life through local, dopamine-like plasticity. Evolution keeps the genomes whose brains learn fastest in worlds they have never seen.

**Status (v0, Nov 2026):** The architecture can learn within a lifetime. A biology-seeded brain improves its food choices from 58% to 69% over one life, and freezing its synapses removes most of its fitness. Evolution from a blank start has not yet discovered learning at small scale. That's the open question. See `results/RESULTS.md` and `experiments/LOG.md`.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiment.py --name smoke --quick            # 1-minute check
python scripts/run_experiment.py --name B-bio --start biology --sigma 0.05 --lr 0.02
python scripts/summarize.py                                      # update results/RESULTS.md
```
In Colab, open `notebooks/Seed_v0.ipynb`, set the runtime to T4 GPU, then Run all.

## Where things are
- `HANDOFF.md`: full context and plan for agents continuing the work
- `AGENTS.md`: research and integrity rules
- `seed/seed.py`: the whole engine (JAX)
