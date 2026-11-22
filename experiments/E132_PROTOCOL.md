# E132 protocol · Fast and slow learning systems

Registered 22 November 2026, before any E132 code, benchmark or training exists.
This file fixes the question, the controllers, the score and the gate. A dated
pre-final amendment in `LOG.md` will add the exact commands and the laptop cost
admission. That amendment may not change anything in this file except to correct
a plain error, and any correction must be dated and explained.

## Question

Does adding a fast, decaying memory to Seed's existing plastic brain help a
creature recover after an unannounced food-rule change? The human-brain idea
behind it is complementary learning systems. In that account the hippocampus
stores single experiences quickly in sparse, well-separated codes. The neocortex
learns slowly from many experiences. Dopamine signals which outcomes mattered.
[McClelland, McNaughton and O'Reilly (1995)](https://doi.org/10.1037/0033-295X.102.3.419)

The comparison is with Seed's own controllers, not with animals. Seed has one
creature, a 5×5 view, two item colors and a deterministic food rule. It has no
hippocampal anatomy, no replay and no consolidation. The name describes the
design principle only.

## The new controller: `fast-slow-v1`

**Slow learner.** This is the unchanged `rules-v0` brain: the same 155-number
genome, development, recurrent dynamics, prediction loop, modulator and
three-factor Hebbian rule.

**Fast learner.** A separate pathway from the visual input to the five movement
outputs.

- **Separation layer.** 128 units. Each unit receives input from 6 visual sensors
  chosen uniformly without replacement from the 100 visual sensors (four channels
  × 25 cells), with weight +1. Each birth draws this wiring from the development
  key, so siblings differ. A unit is active (value 1) if it is among the 6
  highest-input units and its input is above zero; otherwise it is 0. Ties are
  broken by index, as `jax.lax.top_k` does. About 5% of units are active at once.
- **Fast weights.** `F` is a 5 × 128 matrix that starts at zero at every birth.
- **Readout.** At each neural microstep, `gain · F @ z` is added to the
  pre-activation of the five movement neurons. Here `z` is the current
  separation-layer activity.
- **Update.** The update happens once per world step, before the neural ticks,
  with the same timing as the slow rule:
  `F ← clip((1 − decay) · F + eta_f · m · outer(efference_prev, z_prev), −3, 3)`.
  Here `m` is the brain's existing modulator, `efference_prev` is the one-hot
  copy of the action actually taken on the previous step, and `z_prev` is the
  separation activity that led to it.

**Genome.** The genome adds three numbers after the 155 `rules-v0` values, for
158 in total. All three use the same ES and L2 as the other numbers.

| Number | Meaning | Blank value | Why this value |
| --- | --- | --- | --- |
| `log_eta_f` | Fast learning rate (log) | −1.5 | Same blank value as the slow rate, so no speed advantage is hand-set |
| `logit_decay_f` | Fast decay per step (logit) | −4.6 | About 1% per step, so a stored association halves in about 69 steps |
| `log_gain_f` | Readout strength (log) | 0.0 | Gain 1, so fast and slow outputs start on the same scale |

The fast rule's form is fixed at modulator × pre × post with coefficient 1. The
slow rule's coefficients start small and random. So even at matched blank rates
the fast pathway changes faster at birth. That is part of the design, not a
tuned advantage. Evolution can shrink it.

**Compatibility.** The fast pathway is off by default. With it off, every old
genome, EVAL v1 and every earlier protocol must be byte-identical on the same
machine. Tests must prove this before any benchmark.

## Contestants

All contestants train fresh from their established blank starts on the E126
training world. No saved genome is used.

1. **`fast-slow-v1`**: the candidate.
2. **`rules-v0`**: the same brain without the fast pathway. This is the primary,
   one-change comparison.
3. **`direct-rnn-v1`**: the fixed-weight recurrent controller from E113. It can
   adapt only through its recurrent activity. It differs from the candidate in
   many ways (encoding, parameter count, development), so it is a second
   required comparison, not a clean single-change test.

## Training

- **World.** The `reversal-evolution-v1` world from E126: one creature, 300
  turns, and one unannounced swap of food and poison. Each generation draws four
  base worlds and replays each under both starting rules. The flip times 100,
  125, 150 and 175 are each used once per generation, in a permuted order. That
  makes eight lives per candidate.
- **Objective.** The E126 training score: for each base world, the smaller of the
  two replays' net returns (food minus poison) in turns `[flip+50, flip+100)`,
  averaged over the four bases. Taking the smaller return stops a one-color
  specialist from scoring well.
- **Evolution.** Antithetic ES, population 128, σ 0.1, Adam learning rate 0.1,
  L2 0.005 and average ranks for ties. These are the E126 settings, identical
  for all three contestants. The world and random-key schedules are matched
  across contestants within a seed.
- **Budget.** **50 generations** for every contestant. The claim is a head
  start, meaning better learning early in evolution, so a short matched budget
  is the honest test. Generation counts are matched. Wall time is measured and
  reported, not matched.
- **Seeds.** Development benchmark seed 2026121100. Training seeds 2026121101,
  2026121102 and 2026121103, shared across the three contestants. None of these
  seeds has been used before. Each run checkpoints every ten generations.

## Held-out evaluation

The evaluation stays sealed until all nine training endpoints exist.

- For training seed *i*, evaluate all three endpoints on 128 new base worlds
  (32 per flip time), each replayed under both starting rules. Evaluation seeds
  are 2026121201, 2026121202 and 2026121203. They are new, and they differ from
  E126's sealed 2026120501–03.
- Within a seed, layouts, births and action randomness are matched across all
  conditions.
- **Conditions:**
  - each contestant intact in the reversal world and in a stationary copy;
  - **fast-slow lesion:** freeze `F` from neural tick `flip+1` onward in the
    reversal world, and at the same shadow time in the stationary copy. The slow
    Hebbian rule, the prediction loop and the recurrent dynamics stay active.
    Behavior before the switch and the first changed outcome stay identical to
    the intact run.

## Score and gate

**Primary score J.** For each held-out base world, take the smaller net return
of its two starting-rule replays in turns `[flip+50, flip+100)`, then average
over the 128 bases. Standard errors run over base worlds. Every contrast below
is paired by base world, so its SE is the SE of the per-base difference. Death
or no contact contributes zero.

A training seed **passes** if all four of these hold:

1. **Beats the one-speed brain:** J(fast-slow) − J(rules-v0) > 2 paired SE.
2. **Beats the fixed-weight recurrent brain:** J(fast-slow) − J(direct) > 2 paired SE.
3. **The fast memory is the reason:** J(fast-slow intact) − J(fast-slow lesion)
   > 2 paired SE, in the reversal world.
4. **Survival safeguard:** survival is the fraction of bases where both replays
   are alive at turn 300. The lower two-SE bound of survival(fast-slow) −
   survival(rules-v0) must be above −0.10.

**E132 passes if the same two or more of the three training seeds pass.**
Anything else is a fail. No other condition, metric or subset can turn a fail
into a pass.

**Report regardless of outcome:** J for every contestant and condition, all
paired differences with SE, stationary-world J, survival, whole-life fitness,
the evolved values of the three fast numbers, training curves, and wall time and
storage per run. Report each seed's result separately.

## Architecture roundtable (required by the 22 November direction)

- **Proponent: complementary learning systems.** Seed's single learner has one
  speed. When it learns fast it overwrites old knowledge; when it learns slowly
  it cannot follow a change. That fits E125's persistence of an obsolete
  lesson, E128's loss of a new rule across a gap and E130's weak autonomous
  recovery. A separate fast store with its own decay should let new evidence act
  quickly without rewriting the slow network.
- **Challenger: a simpler design.** One plastic network with an evolvable
  weight decay, or a second learning rate, might do just as well. That was
  PLAN's original rung-1 mechanism. Another possibility is that any extra
  input-to-action pathway helps, whatever its sparseness or decay. A third
  challenger is the fixed-weight recurrent controller, which can adapt through
  activity alone (meta-RL, as in [Wang et al. 2018](https://doi.org/10.1038/s41593-018-0147-8)). It is included as contestant 3.
- **How E132 separates them.** The lesion shows whether the fast weights'
  *learning* after the switch matters, not just the extra pathway. The direct
  controller tests the activity-only challenger. **E132 cannot separate the
  full design from decay alone.**
- **Evidence that would change the decision.** If E132 passes, the next step
  (E133) compares `fast-slow-v1` with a decay-only variant of `rules-v0`. If the
  decay-only variant does as well, adopt the simpler design and drop the
  complementary-systems label. If E132 fails, do not tune the fast pathway on
  these worlds. Record the failure and return to the owner before moving to the
  next principle.

## Compute and admission

All costs are measured on the owner's laptop. Timings from other machines do not
admit a run.

- **Benchmark first.** Run three full-size generations of each contestant on the
  development seed.
- **Resource criterion for adoption.** A warm `fast-slow-v1` generation must
  take at most 2× a warm `rules-v0` generation, measured in the same benchmark.
  Missing this does not stop the science, but `fast-slow-v1` cannot be adopted.
- **Rough expectation.** This is from the E126 laptop timings, not a forecast.
  Plastic warm generations took about 37–59 s and direct about 7–13 s. The nine
  50-generation runs plus evaluation would be roughly 5–7 hours, so they span
  at least two laptop days.
- **Study ceiling.** 36,000 seconds, including benchmarks, training, evaluation
  and verification.
- **Forecast rule.** For each run: cold start + 49 × 1.5 × the slowest warm
  generation + overhead. Sum the runs and add 1,800 s each for evaluation and
  verification. If the forecast exceeds the ceiling, stop and ask the owner.
  Never shorten training or change the gate to fit.
- **Limits.** Under 3 hours per process. At most 28,800 s of Seed compute per
  local day. One heavy job at a time at low priority. Checkpoint every 10
  generations.

## What this does not test

E132 does not test human-level intelligence, delayed reward, new tasks, sleep or
real connectome wiring. A pass would show only that this fast pathway helps
these creatures recover from one rule change after 50 generations of evolution.
## Pre-implementation correction · 22 November 2026

This correction was made before any E132 code, benchmark or training existed.
Where it disagrees with the sections above, this section governs.

**1. The lesion stops learning, not decay.** The lesion sets `eta_f` to 0 from
neural tick `flip+1`. Decay and readout continue unchanged. This replaces "freeze
`F`" in *Held-out evaluation* and in gate condition 3. Freezing `F` entirely would
also stop decay, so an intact win could then come from decay erasing old-rule
associations rather than from new learning. The corrected lesion matches the
roundtable claim and mirrors E126's Hebbian-only lesion.

**2. An indeterminate outcome is added.** In a training seed, call a contestant
**learned** if its held-out J is more than 2 SE above zero. Call the seed
**uninformative** if no contestant learned. The overall result is then:

- **PASS**, if the same two or more seeds pass, unchanged from above;
- **INDETERMINATE**, otherwise, if two or more seeds are uninformative;
- **FAIL**, in every other case.

An indeterminate result is reported plainly and returned to the owner. Any longer
budget would need a new registration. This does not make a pass easier.

*Reason, from training scores only:* E126 trained `rules-v0` and `direct-rnn-v1`
from blank on this same world and objective. After 50 generations the population
means of the training objective were −0.66, −0.04 and −0.17 for the plastic
lineages, and about −0.62 for the first direct lineage. A creature that avoids
every item scores 0, so a zero score can mean nothing was learned. These are
training curves already in `runs/`. They are not held-out evidence about
recovery, and E126's sealed evaluation stays sealed.

**3. Births share the slow brain's draws.** The separation-layer wiring is drawn
from `jax.random.fold_in(agent_development_key, 0xE132)`. The slow brain's
development draws must stay byte-identical to `rules-v0` for the same key, so
the one-change comparison changes only the fast pathway.

**4. Caveat added to "What this does not test."** The fast pathway's fixed rule
(modulator × pre × post, coefficient 1, from sensation to action) is close to the
hand-built biology-prior rule. That rule is now running on a separate, decaying
path. E104 showed that the biology prior gives an early lead that blank
evolution matched by generation 200. A pass at 50 generations could therefore
partly repeat that early lead rather than show something specific to
complementary learning systems. If E132 passes, E133 must include two controls:
a decay-only variant of `rules-v0`, and `rules-v0` started from the biology
prior. Only a fast-slow advantage over both would support the
complementary-systems design specifically.
