# HANDOFF: Seed, an architecture that grows, learns and evolves on its own

**From:** Claude (Anthropic), the design and v0 build session, 10–12 Nov 2026
**To:** the Codex agent on the owner's laptop, continuing the project autonomously
**Your job:** set up the private GitHub repo, run the experiments, keep the research evolving, and keep every result audited and traceable.

Read this whole file before you run anything. Then read `AGENTS.md` (the standing rules) and `experiments/LOG.md` (what has been tried).

---

## 1. The project in one paragraph

Seed bets that general intelligence needs two things: a good *starting* brain, and a lifelong *self-learning loop*. It does not need one giant training run with labels. Three loops nest inside each other:

1. **Evolution** (generations). A population of tiny genomes improves.
2. **Development** (birth). Each genome *grows* a brain from rules, not weights, so siblings differ.
3. **Learning** (lifetime). There are no labels and no backprop. Synapses change by a local three-factor Hebbian rule, gated by a neuromodulator ("that mattered"). A local delta rule predicts the next sensation.

The owner's long-term idea: brain *wiring* (as in the fruit-fly connectome, FlyWire / MaleCNS 2026) carries much of the intelligence. Biology should provide the base layer, and evolution plus lifetime learning build on top.

## 2. What exists (v0)

| Path | What it is |
|---|---|
| `seed/seed.py` | The whole engine in JAX: world, development, brain, lifetime rollout, evolution strategies (`es_init`/`es_step`), evaluation (`test_genome`), `biology_prior` |
| `scripts/run_experiment.py` | Runs **one tracked, resumable experiment**. Use it for every run. |
| `scripts/summarize.py` | Rebuilds `results/RESULTS.md` from all run folders |
| `experiments/LOG.md` | Lab notebook: hypotheses, decisions and interpretations, append-only |
| `results/RESULTS.md` | Generated results table (never hand-edit) |
| `notebooks/Seed_v0.ipynb` | Colab version for the owner. Rebuild with `python tools/build_notebook.py` after changing `seed/seed.py`. |
| `runs/` | One folder per run (small files; **commit them all**) |

### The world and agent (v0 defaults in `Cfg`)

- A 16×16 grid that wraps around, with 24 "A" blobs and 24 "B" blobs. **A hidden rule per world decides which blob is food (+0.25 energy) and which is poison (−0.25).** Half the worlds use each rule, so a fixed instinct scores chance. Only within-life learning wins.
- Each world holds a family of 4 agents sharing one genome (relatives), each with different random wiring. Agents see a 5×5 patch with 4 channels (A, B, other agents, their signals), plus energy, energy change and a bias. They move in 4 directions or stay, and emit a signal from −1 to 1. Energy costs 0.003 per step; at 0 the agent dies. A life is 300 steps.
- Brain: 157 rate neurons in 5 types (103 sensors; 3 hidden types of 16 each; 6 motors), recurrent with 2 updates per step. **Motor neurons receive an efference copy** of the action actually taken. Without it, nothing could learn (see LOG E002–E003).
- Genome: 155 numbers. Per type pair: wiring probability, initial strength, and a 4-term plasticity rule (A·pre·post + B·pre + C·post + D). Plus a plasticity rate, 3 modulator weights (energy change, surprise, bias) and a prediction learning rate.
- Fitness: food minus poison eaten, averaged over the family.
- Evolution: OpenAI-style ES (antithetic pairs, centered ranks, Adam, L2 0.005). Everyone in a generation faces the same worlds.

### Evaluation protocol (EVAL v1; keep it fixed so runs stay comparable)

`test_genome(mu, c, n_worlds=128, seed=123)` evaluates the genome in unseen worlds, half with each rule. There are two key measures:
- **Within-life learning:** food-choice accuracy in the 2nd half of life minus the 1st half. A value above 0 means it learned this world's rule.
- **Lesion test:** fitness with plasticity (`plastic=True`) versus frozen (`plastic=False`). A large drop means the brain relies on learning.

If you ever must change the protocol, call it EVAL v2, record why in the LOG, and re-evaluate the reference genomes (random, biology prior, best so far) under it.

## 3. What we know so far (CPU tests, small scale; details in `experiments/LOG.md`)

| Genome | Fitness | Accuracy 1st → 2nd half | Frozen fitness |
|---|---|---|---|
| Random | about −0.6 to −0.1 | 0.48 → 0.48 | n/a |
| Evolved from blank (50 gens, pop 32 × 4 worlds) | 0.9 | 0.52 → 0.54 | 0.7 |
| Biology prior (dopamine-gated Hebbian, hand-set) | 6.3 | 0.58 → 0.69 | 1.1 |
| Biology prior + 20 gens of gentle ES (σ 0.05, lr 0.02) | 6.3 | 0.59 → 0.70 | not run |

What these show:
1. **The architecture can learn within a life.** The biology prior proves it.
2. **Evolution from a blank start has not discovered learning yet**, at least at small scale. That's the main open question.
3. **Large mutations wreck the biology prior** (σ 0.1 / lr 0.1 gave 6.3 → 1.5 in 5 gens). Use σ 0.05 / lr 0.02 when starting from it.
4. These numbers come from **32 evaluation worlds**, not 128. Re-measure the reference genomes under EVAL v1 as your first real task.

## 4. Your first session: step by step

1. **Set up the environment.** Python ≥ 3.10, then `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. If the laptop has an NVIDIA GPU on Linux, install `jax[cuda12]` instead of the CPU build. On a Mac, use the CPU build (do not use `jax-metal`; it's experimental). Record the hardware in the LOG.
2. **Create the private GitHub repo.** This folder is already a git repo with one commit.
   - `gh auth status`. If it isn't logged in, **stop and ask the owner** to run `gh auth login`. Never handle their tokens yourself.
   - `gh repo create seed --private --source=. --remote=origin --push`. If the name is taken, use `seed-agi`. **The repo must stay private.** Never change its visibility.
3. **Smoke test.** Run `python scripts/run_experiment.py --name smoke --quick`, then delete that run folder (smoke runs are not results).
4. **Benchmark.** Do a 10-generation run at full settings and read the `s/gen` column. Size everything else from this. Estimate: on a 2-core CPU, pop 128 × 8 worlds takes about 30 s/gen.
5. **Re-measure the references under EVAL v1.** Evaluate the random genome and the biology prior at 128 worlds. Log the numbers as E100.
6. **Run the core experiments** (§5), commit, push, and write the interpretation in the LOG.

## 5. Experiment queue (in priority order)

Run every comparison with **3 seeds** (`--seed 0/1/2`) before you believe it. One run is an anecdote.

| ID | Question | Command (add `--seed N`) | Passes if |
|---|---|---|---|
| A | Can evolution discover learning from a blank start? | `--name A-blank --start blank --generations 200` | Final accuracy rises within life (> +0.03) **and** frozen fitness is clearly below learning fitness, in ≥ 2 of 3 seeds |
| B | Can evolution improve on biology? | `--name B-bio --start biology --sigma 0.05 --lr 0.02 --generations 200` | Final fitness > the prior's fitness under EVAL v1, in ≥ 2 of 3 seeds |
| C | Control: is plasticity what helps? | `--name C-noplastic --start blank --no-plastic --generations 200` | It's a baseline. A and B should beat it. |

If A fails at 200 generations, try in this order, one change at a time:
1. More worlds per genome (`--worlds 16`), which reduces fitness noise.
2. Longer evolution (500 gens; resumable).
3. Bigger initial plasticity (edit the `init_genome` defaults in a commit).
4. A curriculum: easier early worlds (`--cfg n_items=32`).

### Then the roadmap (each version has a gate; don't start it until the previous gate passes or the owner says so)

- **v1: Seed from the fly.** Initialise the genome's wiring logits (`conn`, a 5×5 type-pair matrix) from connectome statistics. Aggregate published FlyWire or MaleCNS synapse counts between broad classes (sensory → intrinsic → motor/descending) into connection probabilities. Also add **eligibility traces**: a genome-controlled fading memory of recent activity, so rewards that arrive late still reach the right synapses. *Gate:* fly-seeded starts learn faster than random starts (3 seeds). Before downloading data, check its licence and cite the source in the LOG and README.
- **v2: Sleep and memory.** Periodic offline replay through the prediction loop, and worlds whose food rule flips mid-life. *Gate:* sleepers forget less after a rule flip.
- **v3: Culture.** Overlapping generations, strangers instead of only family, harder worlds and tools. *Gate:* agents' signals predict food nearby (measure mutual information).

## 6. Known quirks (not urgent; fix with a test if they start to matter)

- A regrowing blob can overwrite an existing one, so blob counts drift down slightly over a life.
- Two agents on the same cell both eat the same blob.
- The Hebbian update uses the *previous* step's activity with the *current* energy change. That's deliberate: it credits the activity that caused the last action. There's no longer trace yet (v1).
- `test_genome` always uses seed 123. That's deliberate, so evaluations stay comparable.
- The population-mean fitness during ES is that of *perturbed* genomes, so it sits below the mean genome's own fitness. Judge progress by `summary.json`, not training curves alone.

## 7. How to report to the owner

The owner is learning computer science and AI. They're enthusiastic and ambitious, but not an ML engineer yet. When you report:
- Lead with the plain-English answer ("evolution found learning in 2 of 3 seeds"), then the numbers.
- Be honest about negative results. They're progress, and the owner explicitly wants to build something real.
- Keep `README.md` current as the human-facing front page: what Seed is, the latest headline result, and how to run it.
