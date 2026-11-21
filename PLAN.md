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

## Working directive

**Owner delegation, 19 November 2026:** after approving E114, the owner said they
were going to sleep and asked the research council to make the scientific decisions
that would otherwise wait for them. During this absence, record council review and
its decision at failed gates or proposed protocol changes instead of repeatedly
requesting owner input. Preserve every original gate result; preregister each next
experiment before running. This delegates research decisions, not permission to
change repository visibility, stop services, weaken resource safeguards or exceed
the existing compute limits. E114 retains its approved diagnostic-only scope and budget.

Learn from experience, adapt when conditions change, and retain what remains useful.
This frames the owner's 19 November direction within the existing research plan;
it is an aim, not a demonstrated capability or a new reward function. Measure
learning in Phase 1, adaptation in the task ladder, and forgetting in rung 4.
Keep evolution bounded by the registered budgets and gates below.

### Human-inspired research direction · 21 November 2026

The owner asked to use human evolution to guide the research. Human ancestry branches, and different human species once lived at the same time. Seed will use a sequence of measurable behaviors as its engineering curriculum. The sequence concerns what the simulated agents can do. It assigns them no species or historical era. [Smithsonian Human Origins Program](https://humanorigins.si.edu/education/frequently-asked-questions)

1. **Learn from individual experience.** Keep the original within-life food-learning results and the shelter task as reference measurements. Genetic changes, learned connection strengths and the programmed persistence trait are separate mechanisms.
2. **Learn by observing another agent.** The next proposed test gives a newborn a demonstration of a useful choice. A matched control sees the same amount of activity with the link between the demonstration and the useful choice removed. Both newborns start with identical conditions and an empty weight archive. Remove the demonstrator before testing the observer alone on a new layout governed by the same hidden rule. Counterbalance the rule. Register success and survival thresholds, three lineages, and fresh held-out worlds before training. This tests whether useful information survives after the demonstrator leaves.
3. **Test whether successive cohorts can build on a skill.** If observational transfer passes, let each cohort learn from its immediate predecessor and become the next demonstrator. Compare independent chains with matched individual practice and a repeated, unchanged demonstration. Use unseen layouts to test whether a later cohort preserves a useful innovation and improves it. Direct copying of synaptic weights is disabled in this behavioral transmission study.

A three-stage puzzle-box study by Dean and colleagues found that children's higher-stage performance was associated with behaviors including teaching and imitation. It motivates the observable-demonstration control. E119's engineered transfer of acquired weight changes answers a different question. The observational and multi-cohort tests above are proposed, and each requires its own preregistration. [Dean et al., Science (2012)](https://lalandlab.wp.st-andrews.ac.uk/files/2015/08/dean_kendal_schapiro_thierry_laland_2012.pdf)

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
5. **Owner amendment, 19 November 2026:** use the laptop only. The unrun Colab
   GPU benchmark and migration proposal are withdrawn. Preserve the completed
   CPU measurement and all benchmark history; no GPU speed comparison was made.
   Phase 0 closes with this item waived by the owner, not passed experimentally.

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
- Owner amendment, 17 November 2026: subagents may run the registered experiments
  under the main coordinator, as well as support work. The owner explicitly
  confirmed: keep one heavy compute job at a time.
- Keep `TASK_STATE.md` under 60 lines of plain English and README current.
- After each phase, provide the five-line report above, then continue.
- Stop and ask the owner when a gate fails, this plan is ambiguous, something in
  it turns out to be wrong or infeasible, or an action is on `AGENTS.md`'s
  ask-first list. Do not work around the plan silently.
- A reviewer session audits commits against this plan and `AGENTS.md`. Treat its
  findings as review comments: fix them or answer them in the LOG before moving
  on. The reviewer cannot change the plan; only the owner can.
- Never claim AGI, general intelligence, novelty or state of the art.

## Owner direction, 21 November 2026

Develop a world where a creature can see shelter and choose whether to spend
energy reaching it when the weather changes. Retain the food task as a benchmark.
The owner asked to reuse both evolved genomes and acquired experience, including
experience left by creatures that die. Test three questions separately.

1. Does starting from an evolved genome help in a different environment?
2. Does the same individual benefit from keeping its learned connections for a
   second life?
3. Can independently wired newborns benefit from what earlier creatures learned,
   and does keeping lessons from several cohorts help more than keeping one?

E116 finishes its registered comparison on the original world. Build the new
environment and transfer controls separately. In the new world, register each
scientific comparison before running it and measure survival and useful actions
on held-out lineages. A sequence of harder worlds can test transfer or forgetting.
Showing that the sequence itself helps requires a comparison with equal exposure.
- Owner workflow update, 15 November 2026: create no new pull requests until the
  owner re-enables them, and do not request routine pull-request reviews. This
  changes publication workflow only, not gates or approval requirements.
