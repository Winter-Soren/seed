# E132 implementation instructions for the coding agent

Written 22 November 2026 for the agent working on the owner's laptop. These are
step-by-step instructions. Where this file and
[`E132_PROTOCOL.md`](E132_PROTOCOL.md) disagree, the protocol wins, including its
dated correction at the end. If anything here is ambiguous, stop and ask the
owner. Do not guess.

## 0. Before you start

1. Read, in this order:
   - `AGENTS.md`;
   - `PLAN.md`, especially "Owner direction, 22 November 2026 (evening)";
   - `experiments/E132_PROTOCOL.md`, all of it including the correction;
   - the E126 closure and E132 entries at the end of `experiments/LOG.md`.
2. Pull the branch `claude/zen-gauss-dfd2oe`. It contains the registration commits.
3. Confirm the repo-local git identity is the owner's GitHub noreply address,
   the same as earlier commits. Never use a personal email address.
4. Follow these rules throughout:
   - Change one thing at a time.
   - Never edit `runs/` or past LOG entries. Append dated entries instead.
   - Never claim AGI or general intelligence.
   - One heavy job at a time, at low priority.
   - No process over 3 hours; no more than 28,800 s of Seed compute per local day.
   - Do not touch E126's sealed held-out seeds 2026120501–03, or its partial genomes.
   - Do not change `scripts/reversal_evolution.py` or `scripts/run_reversal_evolution.py`.
     E126's behavior must stay exactly as it is. Import their helpers instead.

## Owner checkpoints (stop and wait at each)

| When | What to bring to the owner |
| --- | --- |
| **A.** After the benchmarks and forecast (step 6) | Measured times, the 2× resource check, the forecast against 36,000 s, and the budget question in step 7 |
| **B.** If any compatibility fixture differs, a forecast exceeds a limit, or a run crashes | What happened, with files and numbers. Do not work around it |
| **C.** After the gate decision (step 10) | The five-line report |

Do not start final training until the owner has answered checkpoint A in writing
and you have logged that answer.

## 1. Engine change in `seed/seed.py`: the fast pathway, off by default

Add the pathway so that **all existing code paths stay byte-identical** when it
is off.

**New rollout arguments.** Add two keyword-only arguments to `rollout`:

- `fast_memory=None`. When on, it is a tuple of three floats
  `(eta_f, decay_f, gain_f)`, already transformed:
  `eta_f = exp(log_eta_f)`, `decay_f = sigmoid(logit_decay_f)`,
  `gain_f = exp(log_gain_f)`.
- `_fast_off_step=None`. This is the lesion tick.

Add `_fast_off_step` to the `static_argnames` of the `jax.jit` decorator.
`fast_memory` holds traced values, so do not make it static. Reject every
combination that is not E132's, as the existing argument checks do:

- the direct controller;
- structural plasticity;
- ecology or persistence;
- social belief;
- exploration;
- any lineage hook;
- `plastic=False`;
- `_fast_off_step` without `fast_memory`;
- `_fast_off_step != flip_step + 1` when a flip is given.

**Separation wiring, drawn once per agent at birth.**

```python
SEP_UNITS, SEP_FANIN, SEP_ACTIVE, SEP_TAG = 128, 6, 6, 0xE132
VISUAL = 4 * V * V            # 100 for R = 2; obs[:, :VISUAL] are the A, B, others, signals patches
agent_keys = jax.random.split(dkey, c.M)          # the same split develop() already uses
sep_keys = jax.vmap(lambda k: jax.random.fold_in(k, SEP_TAG))(agent_keys)
sep_index = jax.vmap(lambda k: jnp.argsort(jax.random.uniform(k, (SEP_UNITS, VISUAL)), axis=1)[:, :SEP_FANIN])(sep_keys)
# sep_index: (M, 128, 6) int32, six distinct visual sensors per unit
```

The slow brain must still be grown with `develop(g, k, c)` from the same
`agent_keys`, unchanged. Folding the tag leaves those draws untouched. With one
creature (`M = 1`), the "others" and "signals" patches are always zero, so about
half the drawn inputs never fire. That is expected under the registered design.
Do not change it.

**Separation activity.** Compute it for every observation:

```python
def separate(obs_visual, idx):            # obs_visual (100,), idx (128, 6)
    u = obs_visual[idx].sum(1)            # weight +1 on each input
    top = jax.lax.top_k(u, SEP_ACTIVE)[1] # ties go to the lower index
    z = jnp.zeros(SEP_UNITS).at[top].set(1.0)
    return z * (u > 0)                    # silent if its input is zero
```

**Carry state.** Add `F` with shape `(M, 5, 128)`, starting at zeros, and
`z_prev` with shape `(M, 128)`, starting at zeros. Treat them exactly as the
existing code treats `W`. Do not mask them by `alive` unless the existing code
masks `W`.

**`brain_step` change.** Add an optional `fast=None` argument. When it is `None`,
the function must return its current 4-tuple through the current code, byte for
byte. When it is given as `(F, z_prev, idx, eta_f_eff, decay_f, gain_f)`, do this
in order:

1. Prediction-loop update, `surprise`, modulator `m` and slow Hebbian update `W`,
   all **unchanged**.
2. Fast update. `eff_prev = x[n - O:n - 1]` is the one-hot efference copy of the
   previous action, which is already in `x` at this point. Then
   `F = clip((1 - decay_f) * F + eta_f_eff * m * outer(eff_prev, z_prev), -c.wmax, c.wmax)`.
3. `z = separate(obs[:VISUAL], idx)`.
4. Think:

   ```python
   x = x.at[:D].set(obs)
   for _ in range(c.micro):
       pre = W @ x
       pre = pre.at[n - O:n - 1].add(gain_f * (F @ z))
       x = jnp.tanh(pre).at[:D].set(obs)
   ```

5. Return `W, x, Wp, surprise, F, z`. In `rollout`, store `F` and `z` in the
   carry; `z` becomes the next step's `z_prev`.

**Lesion.** `eta_f_eff = eta_f` while `t < _fast_off_step`, and `0` from then on.
Use `jnp.where` on the tick, as `_hebbian_off_step` already does. Decay and
readout continue. The slow Hebbian rule and the predictor are **not** lesioned.

**Diagnostics.** When `diagnostics` or `record` is set with the new world
protocol, record:

- a per-tick `fast_enabled` flag (`eta_f_eff != 0`);
- SHA-256 hashes of `F` at birth, at the probe tick `flip+1` and at the end,
  next to the existing `W`, `x` and `Wp` hashes;
- a SHA-256 hash of `sep_index`.

Then rebuild the notebook with `python tools/build_notebook.py`, because
`seed/seed.py` changed.

## 2. Genome format: new file `seed/fast_slow.py`

Use `seed/direct.py` as the template.

- `FORMAT = "fast-slow-v1"`.
- `genome_size(c) = seed.genome_size(c) + 3`, which is 158.
- **Layout.** Entries `[0:155]` are the rules-v0 genome, flattened in exactly the
  order `scripts/shelter_evolution._unravel(c)` expects. Entries `[155:158]` are
  `[log_eta_f, logit_decay_f, log_gain_f]`.
- **Blank start.** The first 155 values must equal what
  `scripts/reversal_evolution.initialize_state("rules-v0", seed)` produces for
  the same training seed. Build the 155-value mean the same way and append
  `[-1.5, -4.6, 0.0]` as float32. A test must check this equality.
- **`decode(flat, c)`.** Returns `(rules_genome_dict, (eta_f, decay_f, gain_f))`.
  It rejects wrong shapes, wrong dtypes and non-finite values, as `direct.decode`
  does.
- **ES noise.** Draw it for the full 158-vector the standard way. Perturbations of
  the slow parameters therefore are not matched to rules-v0's candidates. Only
  the world, flip-time and birth keys are matched. Write this down in the LOG
  amendment.

## 3. Protocol module: new file `scripts/fast_slow_evolution.py`

Set `PROTOCOL = "fast-slow-reversal-v1"`. Copy the structure of
`scripts/reversal_evolution.py` by importing its helpers, never by editing it:

- `training_assignment`;
- `sampling`;
- `es_step`;
- `average_centered_ranks`;
- the window logic;
- `evaluation_schedule`;
- `pack_evidence` and `unpack_events`.

- **Controllers.** `CONTROLLERS = ("rules-v0", "direct-rnn-v1", "fast-slow-v1")`.
  Rules-v0 and direct must run through exactly the same rollout calls E126 used.
  A test must show that, for identical inputs, this module's scorer gives
  byte-identical results to `reversal_evolution._group_scorer` for those two
  controllers.
- **Training objective.** E126's objective, unchanged: the per base world minimum
  of the two rule replays' late `[flip+50, flip+100)` net return, averaged over
  four bases.
- **Evaluation arms.** For one held-out base world, evaluate all three endpoints
  of one seed in eight arms:
  - `rules_stationary`, `rules_reversal`;
  - `fastslow_stationary`, `fastslow_reversal`;
  - `fastslow_lesion_stationary`, `fastslow_lesion_reversal`;
  - `direct_stationary`, `direct_reversal`.

  Every arm uses the same world key and birth key. The lesion arms pass
  `_fast_off_step = flip + 1`. In the stationary copy, the lesion is applied at
  the same shadow tick.
- **Validation.** Mirror `validate_world`, and add these checks:
  - Before the switch, the stationary and reversal histories are identical for
    each controller.
  - Through tick `flip+1`, intact and lesion runs have identical prefix hashes
    and identical probe hashes for `W`, `x`, `Wp` and `F`.
  - In the lesion arms, `fast_enabled` is false from `flip+1` on.
  - For rules-v0 and fast-slow, the slow brain's birth `W` hash is identical
    across all arms of one world.
- **Gate.** Implement `summarize` and `aggregate` as written in the protocol.
  1. J per base world is the minimum over the two rules of late food minus poison.
  2. Every contrast uses the **paired per-base difference** and its SE. Do not
     use E126's `hypot` combined SE; its test deliberately differs.
  3. Seed pass requires all four conditions. Condition 4 is about joint
     end-of-life survival against rules-v0: the lower 2-SE bound of the
     difference must be above −0.10.
  4. A contestant is **learned** if its reversal-world J has a lower 2-SE bound
     above 0. A seed is **uninformative** if no contestant learned.
  5. Overall: PASS if the same ≥2 seeds pass; INDETERMINATE if not a pass and
     ≥2 seeds are uninformative; FAIL otherwise.
  6. Also return every descriptive quantity the protocol's "Report regardless of
     outcome" lists.

## 4. Runner: new file `scripts/run_fast_slow_evolution.py`, plus dispatch

Mirror `scripts/run_reversal_evolution.py` with these registered values:

| Setting | Value |
| --- | --- |
| Protocol | `fast-slow-reversal-v1` |
| Architectures | `fastslow` → `fast-slow-v1`, `plastic` → `rules-v0`, `direct` → `direct-rnn-v1` |
| Benchmark names and seed | `E132-benchmark-{fastslow,plastic,direct}`, seed 2026121100, 3 attempts |
| Training names | `E132-{fastslow,plastic,direct}-s{0,1,2}-part{NNN}` |
| Training seeds | 2026121101 + seed index |
| Total attempts per target | 50, unless the owner changes it at checkpoint A |
| Evaluation names and seeds | `E132-eval-s{0,1,2}`, seeds 2026121201 + index, 128 held-out bases |
| ES | pop 128, 4 base pairs (8 lives), σ 0.1, lr 0.1, L2 0.005, checkpoint every 10 |
| Time limits | development: soft 780 s, hard 900 s; final: soft 9,000 s, hard 10,800 s per process |
| Development tags | `0xE13200`, `0xE13201`, `0xE13202` |

The runner must:

- refuse uncommitted code;
- refuse any other name, seed, budget or flag;
- refuse to evaluate until all nine endpoints are complete;
- preserve interrupted parts, with continuation only through `--resume` into a
  new numbered part.

In `scripts/run_experiment.py`, add the protocol constant and a dispatch block
next to the reversal-evolution one. Allow `--controller fast-slow-v1` **only**
with this protocol. All other protocols must keep rejecting it.

## 5. Tests, then an independent audit tool

Write `tests/test_fast_slow.py` and `tests/test_fast_slow_runner.py`. Each test
below is required.

**Compatibility.** These prove nothing old changed.

1. The entire existing suite passes unchanged.
2. The EVAL v1 fixture `tests/fixtures/eval_v1_compatibility.*` matches exactly.
3. For rules-v0 and direct, `rollout` outputs are byte-identical to the
   historical engine at commit `720a834` for several keys, flip and no-flip,
   with and without diagnostics. Load it from Git as the existing compatibility
   tests do.
4. All E126 engine and runner tests pass unchanged.

**One-change test.** A fast-slow genome with `log_gain_f = -inf` (gain 0) and
`log_eta_f = -inf` (rate 0) produces exactly the same actions, events and slow
weights as rules-v0 built from the same first 155 numbers and the same keys. If
`-inf` cannot pass validation, test through the internal `fast_memory`
argument with `(0.0, decay, 0.0)`.

**Births.** For the same development key, the slow `W` and mask are identical
between rules-v0 and fast-slow.

**Separation layer.**
- Each unit has 6 distinct indices, all below 100.
- At most 6 units are active.
- A visual input that is all zero gives `z` all zero.
- Different agent keys give different wiring.
- The same key gives the same wiring.

**Update arithmetic.** One hand-computed step of the `F` update and readout
matches the code.

**Lesion.**
- `F` is identical to the intact run through tick `flip+1`.
- After that, every step applies exactly `F ← clip((1 - decay_f) · F)`.
- The slow `W` keeps learning.

**Genome.**
- Size is 158.
- Blank values are `[-1.5, -4.6, 0.0]`, and the first 155 equal rules-v0's
  blank start for the same seed.
- `decode` rejects bad input.

**Gate logic.** Synthetic per-base vectors give:
- a PASS;
- a FAIL;
- an INDETERMINATE;
- the "same seeds" requirement, where two seeds pass different conditions and
  the result is not a pass;
- paired SE, not `hypot`.

**Runner identity.** Wrong seed, name, budget or controller, dirty code, early
evaluation and a second attempt under the same name are each refused.

Then write `tools/audit_fast_slow_evolution.py`, mirroring
`tools/audit_reversal_evolution.py`. From the saved files it independently
reconstructs:
- the ES updates;
- the per-candidate objectives;
- the evaluation windows and the gate.

Give it a `--self-check` mode.

**Commit the code before any benchmark**, following the pattern
`E132: implement fast-slow-v1 controller and protocol (no runs)`. Push. Run the
full local suite and record its result and wall time in the LOG. Use hosted CI
in the project's usual way: CI runs on `main` pushes, pull requests or manual
dispatch. The owner has paused new pull requests.

## 6. Laptop benchmark and forecast (development only)

Run the three benchmarks one at a time, at low priority:

```powershell
$python = '.\.venv\Scripts\python.exe'
$entry = 'scripts/run_experiment.py'
$common = @('--eval-protocol', 'fast-slow-reversal-v1', '--pop', 128, '--worlds', 4,
    '--sigma', 0.1, '--lr', 0.1, '--eval-worlds', 128, '--checkpoint-every', 10, '--start', 'blank')
$architectures = [ordered]@{ fastslow = 'fast-slow-v1'; plastic = 'rules-v0'; direct = 'direct-rnn-v1' }
foreach ($label in $architectures.Keys) {
    & $python $entry @common --controller $architectures[$label] --condition-split development `
        --name "E132-benchmark-$label" --seed 2026121100 --generations 3 --max-train-seconds 780
}
```

From the benchmark files, compute:

- **Warm-time ratio:** warm `fast-slow` ÷ warm `rules-v0`, using the median of
  generations 2–3. The resource criterion requires this to be ≤ 2.0. Record
  pass or fail; it affects adoption only.
- **Per-run forecast:** `cold + (attempts − 1) × 1.5 × slowest warm + overhead`.
  Sum over all nine runs. Add 1,800 s for evaluation, 1,800 s for verification
  and the benchmark charges already spent.
- **Admission:** compare the total with the 36,000 s study ceiling and with the
  remaining daily allowance. Plan the nine runs across local days, never more
  than 28,800 s per day.
- **Audit:** run the audit tool's development mode on the benchmark folders.

Append a dated LOG entry with these numbers. Then stop at **checkpoint A**.

## 7. Owner decision at checkpoint A: the training budget

Explain this to the owner in plain words:

- **50 generations (registered).** Cheap, at about 5–7 hours over at least two
  days. E126's training scores on this world were still at or below zero after
  50 generations, so an INDETERMINATE result is a real possibility.
- **200 generations.** Much more likely to give an informative result. It costs
  roughly four times as much training, which is likely beyond the 36,000 s
  ceiling. The owner would have to raise the ceiling explicitly.

Record the owner's answer as a dated LOG entry. If the owner picks a different
budget or ceiling, that entry is the only change to the registration. The gate,
seeds and scoring stay fixed. Then append the **pre-final amendment** with the
exact final commands and the admitted forecast. It may add only commands and
costs.

## 8. Final training (after the amendment)

Run one target at a time, at low priority, in this order for each seed index
0, 1, 2: `direct`, `plastic`, `fastslow`.

```powershell
# For seed index $i, label $label, part $part (1 unless resuming):
$partName = 'E132-{0}-s{1}-part{2:D3}' -f $label,$i,$part
$continuation = if ($part -gt 1) { @('--resume', $previous) } else { @() }
& $python $entry @common --controller $architectures[$label] --condition-split final `
    --name $partName --seed (2026121101+$i) --generations 50 --max-train-seconds 9000 @continuation
```

Replace `50` with the owner's budget if checkpoint A changed it. Preserve
interrupted parts and resume only into a new numbered part. Do not look at
held-out results; there are none yet. After all nine endpoints are complete, run
the audit's training mode.

## 9. Held-out evaluation (only after all nine endpoints exist)

```powershell
foreach ($i in 0..2) {
    & $python $entry --eval-protocol fast-slow-reversal-v1 --condition-split final `
        --name "E132-eval-s$i" --seed (2026121201+$i) --generations 0 --eval-worlds 128 --start blank
}
```

Run the audit's final mode, then `python scripts/summarize.py`.

## 10. Record, report, tag

1. Append the result to the LOG:
   - for every seed, J for every arm and every paired difference with its SE,
     with numbers copied from the summary files;
   - the PASS, FAIL or INDETERMINATE decision, stated plainly;
   - which conditions failed;
   - the evolved values of the three fast numbers;
   - wall time and storage per run.
2. Update `README.md`: change the E132 milestone row and add a short section with
   the result, what it means and what it does not mean. Update `TASK_STATE.md`
   and keep it under 60 lines. Make no AGI claims anywhere.
3. Commit the run folders, `results/RESULTS.md`, the README, `TASK_STATE.md` and
   the LOG together, with the message `E132: <one-line result>`. Push. Tag
   `v0.21-fast-slow-v1`.
4. Give the owner the five-line report: what was asked, what happened with
   numbers and SE, what it means, what it does not mean, and what comes next.
   Then stop at **checkpoint C**.

**What comes next depends on the outcome.** Each outcome needs its own
registration and the owner's approval:

- **PASS:** E133 compares fast-slow with a decay-only rules-v0 and with rules-v0
  started from the biology prior.
- **FAIL:** report the failure and ask the owner before moving to the next
  principle. Do not tune the fast pathway on these worlds.
- **INDETERMINATE:** ask the owner about a longer budget, which needs a new
  registration.

## Do not

- Tune any number after seeing a training or evaluation result.
- Change the protocol, the gate or the seeds. The only change allowed is the
  owner's budget decision at checkpoint A.
- Modify E126 code, folders or sealed seeds.
- Add any other mechanism to the brain.
- Run two heavy jobs at once, or exceed the time limits.
- Describe a difference inside the error bars as a result.
