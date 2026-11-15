# Owner research plan for Seed

Owner direction, 15 November 2026. This replaces the previous next action in
`TASK_STATE.md`. `AGENTS.md` still applies in full: preregister, commit first, use
three seeds, never edit `runs/` or past LOG entries, and report failures plainly.
Where this plan changes project goals or adds evaluation protocols, the owner has
approved exactly what is written here and nothing beyond it.

## Thesis to test, not assume

> Structured evolutionary pressure plus a good wiring prior produces increasingly
> general within-life learners.

Each phase has a gate that can fail. A failed gate is a valid result: record it,
say so plainly in README, and stop for owner input. Never soften a gate after
seeing results. No AGI, general-intelligence, novelty or state-of-the-art claims
anywhere.

## Findings from an external audit

These are **unregistered probes on 256 fresh worlds, seed 777**. Remeasure them
properly in Phase 1 before relying on them.

- Per-agent learning is real: accuracy by meal number rises from about 0.49 on the
  first meal to about 0.80 on the twentieth.
- All three A genomes converged on the hand-built biology-prior rule
  (sensor-to-motor reward-gated Hebbian). Removing all hidden-to-motor wiring
  changes nothing.
- Removing the surprise term from the modulator collapses fitness from about 15
  to 2–7.
- A greedy policy that knows the rule scores about 92; evolved agents score about 15.
- When the food rule flips at step 150, accuracy falls to about 0.26 and recovers
  only to about 0.49.
- Control C cannot win by construction: all sensors share one type, so a frozen
  genome cannot distinguish blob A from blob B.
- The same genome scores 16.49 on Linux versus 16.59 on the Windows laptop:
  exact reproduction is machine-specific. Per-world standard error is about 0.35.

## Phase 0 — housekeeping, no new science

1. Stop E109. Leave its folder as-is with status interrupted; append a LOG entry
   recording the decision and the reason: the rule lowered fitness in 3/3 genomes,
   and an eight-hour single-seed cost is not justified.
2. Run B1 and B2 exactly as preregistered in E104 and close gate B either way.
3. Pin exact versions in `requirements.txt`. Make the runner's dirty check include
   untracked files under `seed/` and `scripts/`. Add a CI workflow that runs the
   test suite.
4. Rewrite `TASK_STATE.md` to under 60 lines of plain English: what is known, what
   is running, and what is next. No hashes, process IDs or byte counts; those
   belong in the LOG.
5. Benchmark one generation on a free Colab GPU and record seconds per generation
   next to the laptop figure. If it is at least five times faster, propose moving
   training there and wait for the owner's answer before doing it.

## Phase 1 — better measurement

Use saved genomes only, for minutes of compute. Register this as **EVAL v3
diagnostics**; EVAL v1 stays untouched. For A0/A1/A2, B0, the biology prior and
blank references, on fresh held-out worlds with a new seed and 256 worlds, report:

- Fitness with per-world standard error.
- Accuracy by meal number: first, second, third, fifth, tenth and twentieth item
  eaten, per agent.
- Accuracy in 50-step bins restricted to agents alive at the end.
- Two reference policies in the same worlds: random walk and a greedy
  rule-knowing policy. Report every fitness as a fraction of the greedy score
  as well as raw.
- A rule-reversal test, with the rule flipping at step 150: accuracy in each
  50-step bin after it.

From now on every summary reports standard errors, and README claims use them.

## Phase 2 — fair non-plastic baseline

Implement a directly encoded recurrent controller with the same observations,
actions and hidden size. Its weights are the genome; it has no lifetime weight
changes. Train it with the same ES, population, worlds and generation budget,
using three seeds. Preregister the comparison.

This replaces C as the control for all later claims. If it matches or beats the
plastic agents, report that as the headline and stop for owner input.

## Phase 3 — the task ladder

Proceed one rung at a time, each as a versioned world variant:

1. Rule reversal mid-life, with flip time randomized per world.
2. Delayed reward: energy change arrives k steps after eating; start with k = 5.
3. Context-dependent rule: which blob is food depends on a visible cue.
4. Many item types, eight or more, with new types appearing mid-life; measure
   forgetting.

For each rung, in this order:

a. Evaluate existing genomes on it to establish the failure, without training.
b. Evolve from blank, using three seeds and the unchanged architecture.
c. Only if (b) fails its gate, add **one** mechanism, default off, with
   compatibility tests: rung 1 gets evolvable weight decay or modulator sign
   handling; rung 2 gets a genome-controlled eligibility trace; rung 3 gets the
   finer genome described below.
d. Run a lesion test showing which machinery the solution depends on: hidden
   layer, trace or modulator terms. A rung counts as passed only if plastic
   agents beat the Phase 2 baseline by **more than two combined standard errors
   in at least two of three seeds**, on held-out worlds, **and** the lesion test
   identifies the mechanism.

Set each rung's numerical gate in the LOG before running. Do not start a rung
until the previous one is decided.

## Phase 4 — finer genome and the connectome test

This tests the owner's core idea.

1. Let the genome distinguish sensor channels and add more hidden types, for
   example one sensor type per visual channel and six to eight hidden types.
   Version the genome format. Old 155-number genomes and EVAL v1 must keep
   working and stay bit-identical on the same machine.
2. Check the licence of the published fly connectome data first; cite it in LOG
   and README. Ask the owner before any download over 1 GB. Aggregate synapse
   counts between broad cell classes into connection probabilities for the
   genome's wiring table.
3. Compare three starting points on the hardest rung passed so far, with three
   seeds each and the same budget: (i) fly-derived wiring; (ii) shuffled fly
   wiring with the same connection counts but class structure destroyed;
   (iii) random wiring at matched density.

Gate: fly-derived wiring reaches a preregistered fitness level in fewer
generations than **both** controls in at least two of three seeds. Beating (iii)
but not (ii) means density, not structure, and must be reported that way.

## Phase 5 — public demo

Start only after Phase 1; it needs no new science. Build a static web page with
no server: a grid-world replay where a visitor can pick a genome (random,
biology prior, evolved, and later fly-wired), watch four agents live a life,
see the accuracy-by-meal curve fill in, and press a button to flip the food rule.

The simplest route is to export recorded trajectories and weights as JSON from
the engine and replay them in JavaScript. State clearly on the page what this
is—a small research simulation—and what it is not. The repository stays private.
Do not publish the page or change repository visibility without the owner's
explicit go-ahead.

## Optional work

Only on owner request after Phase 3 rung 2: replace ES with gradient-based
meta-learning of the plasticity rules through the lifetime (differentiable
plasticity). Benchmark it against ES on a passed rung before using it for
anything new.

## Do not

- Tune anything on evaluation worlds or reuse an evaluation seed for development.
- Change more than one thing between a run and its comparison.
- Delete, overwrite or edit `runs/` or past LOG entries, or rewrite Git history.
- Add mechanisms speculatively; add one only after a rung demonstrably fails
  without it.
- Run more than one heavy job at a time, exceed three hours per run, or stop
  services on the owner's laptop without asking each time.
- Describe an operational fix, a single seed, or a difference inside the error
  bars as a result.

## Reporting

After each phase, give the owner five lines in plain English:

1. What was asked.
2. What happened, with numbers and error bars.
3. What it means.
4. What it does not mean.
5. What is next.

Keep README current with the same content.

## Execution and review

The owner's active goal is to execute this plan in order under `AGENTS.md`, one
phase and one rung at a time. Test whether the thesis is true; do not try to make
it look true. A failed gate reported plainly with numbers and error bars counts
as success.

- Work on the earliest unfinished item. Do not start a later phase, a new
  mechanism or a side experiment until the current gate is decided and logged.
- Preregister the gate in `experiments/LOG.md` before running and never change it
  afterward.
- Use subagents only for support work such as tests, documentation and figures,
  never for extra experiments. Keep one heavy compute job at a time.
- Keep `TASK_STATE.md` under 60 lines of plain English and README current.
- After each phase, provide the five-line report above, then continue.
- Stop and ask the owner when a gate fails, this plan is ambiguous, something in
  it turns out to be wrong or infeasible, or an action is on `AGENTS.md`'s
  ask-first list. Do not work around the plan silently.
- A reviewer session audits commits against this plan and `AGENTS.md`. Treat its
  findings as review comments: fix them or answer them in the LOG before moving
  on. The reviewer cannot change the plan; only the owner can.
- Never claim AGI, general intelligence, novelty or state of the art.
