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
