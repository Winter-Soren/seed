# Ecology v1 development API

This is an opt-in implementation and a future-study starting point, not evidence
of adaptation. Historical `Cfg`, 155-number rule genomes and scientific
evaluation protocols are unchanged. Do not run an unregistered training loop
from an example or reuse held-out worlds as training data.

## Two environmental pressures

`EcologyV1(movement_energy_cost=0.0, seasonal_energy_cost=0.0)` accepts only finite,
nonnegative numbers. These are additional energy costs, not new food rewards.

- Movement cost is charged for each non-stay action by a living agent, even if
  toroidal wrapping returns it to the same cell.
- Cold starts at step `T // 2`. Seasonal cost is charged each cold step only when
  a living agent ends its action outside shelter. Movement into shelter can avoid
  the seasonal cost while still paying movement cost. Costs precede energy clipping.

Shelter is a toroidal square of width `max(1, G // 4)`. Its anchor is sampled from
`fold_in(wkey, 0x45434F31)`, a separate stream that leaves food placement, agent
spawns and action randomness unchanged. Worlds with the same configuration have
the same shelter area, at seeded locations. Shelter neither supplies food nor
changes food/poison identities.

## Perception and compatibility

Every ecology stage, including zero-cost controls, appends the visible local
shelter bitmap and current cold-season bit after the original sensor block.
Agents receive local observations, not the global map or an optimal action.
`dims(c, ecology=pressure)` describes this layout. With default `Cfg`, sensors
increase from 103 to 129 and total neurons from 157 to 183. Sensor type labels
remain shared, so the genome still has 155 numbers.

`ecology=None` retains the legacy sensor layout and arithmetic. An explicit
zero-cost `EcologyV1()` is a neutral **ecology-layout** control, not a claim of
trajectory identity with the smaller legacy brain. All transfer/control arms
must be born with the same ecology layout. Do not copy legacy learned weight
arrays into the larger network.

Direct controllers have the same optional ecology keyword in `genome_size`,
`init_genome`, `decode` and `make_evaluator`. A legacy direct genome/matrix fails
shape validation under ecology. No silent embedding or historical direct-v1
comparison is made.

## Prepare a known evolved genome

Run this preparation from the repository root. It validates the saved original
A0 genome, creates a fresh optimizer and connects the existing ES evaluator. It
does not train or evaluate a world.

```python
import hashlib
from pathlib import Path
import numpy as np
from seed import seed
from seed.ecology import EcologyV1

path = Path("runs/20261112-023245_E103-A-blank-s0/genome_final.npy")
assert hashlib.sha256(path.read_bytes()).hexdigest() == (
    "477d9d03c713dcab8072349ff9155813e5a384d3709eff67595a342b55111f53"
)
flat = np.load(path, allow_pickle=False)
c = seed.Cfg()
assert flat.shape == (seed.genome_size(c),) and np.isfinite(flat).all()
pressure = EcologyV1(movement_energy_cost=.01, seasonal_energy_cost=.02)
state = seed.es_init(c, mean=flat, seed=41)  # development preparation only
evaluate, decode = seed.make_evaluator(c, E=2, ecology=pressure)
genome = decode(state["mu"])
print(pressure.version, flat.shape, seed.dims(c, ecology=pressure))
```

The unchanged `es_step(state, evaluate, pop, E, sigma, lr)` consumes this evaluator.
`evolve(..., mean=flat, ecology=pressure)` also connects the same code path, but
scientific runs must be preregistered and dispatched through the tracked runner.
The example pressures are engineering values, not selected research settings.

## Measurements and controls

With `record=True` or `diagnostics=True`, ecology rollouts return a shelter map,
`initial_sheltered` for initially alive agents, per-step `sheltered` occupancy
while alive before the action, and `energy_lost`
from the extra costs after clipping. Existing per-agent encounter/alive events
retain the information needed to report arrival timing, survival and censoring.
The global shelter map is diagnostic output; the creature only sees its patch.

The reference rollout utility accepts the same ecology object and charges the
same costs in the same worlds. Random walk, fixed-color and greedy policies keep
their original action rules: none deliberately navigates toward shelter, and the
greedy reference is not an optimal ecological ceiling. A purposeful shelter
policy would need a separate preregistered comparison.

The registered `test_genome` and `evaluate_conditions` APIs do not accept ecology.
New ecological studies cannot be labeled EVAL v1/v2/v3/v4/v5 or treated as directly
comparable to those fixed protocols. See `ECOLOGY_TRANSFER.md` for the proposed
controls and gates; no new scientific gate is passed by this implementation.

## Learned weights across lives

The controlled lineage API owns transfer and provenance. Private rollout hooks
export birth/initial weights, the birth mask, per-agent last-living weights and
final eligibility; they are not an arbitrary weight-import API. Only weights
from the same layout and allowed connections may enter a new life. All other
state resets. Donor export stops when an agent dies: because the historical
learning rule credits rewards one step later, the final fatal transition is not
credited after death. Historical no-lineage trajectories are left unchanged.

Same-individual survivor controls retain dead original slots as zeros. Collective
inheritance is a distinct question, with newborn cohorts and explicit donor and
archive accounting. Neither intervention establishes biological inheritance or
general intelligence. Never pass a held-out evaluation archive into training.

## Run the three-cohort software check

From a clean checkout, run the following command with the project's Python
interpreter. Use a new run name each time. The example loads the first evolved
genome and follows two independently wired newborns in each of three worlds.

```powershell
python -X utf8 scripts/run_experiment.py --name ecology-local-example --eval-protocol ecology-lineage-v1 --quick --condition-split development --seed 41 --generations 0 --memory-condition shared --start file:runs/20261112-023245_E103-A-blank-s0/genome_final.npy
```

The first cohort experiences neither extra cost. The next pays to move. The
last also loses energy outside shelter during the cold half of its life. Each
cohort has a matched group with fresh weights. Both groups learn during life.
The shared group also starts with connections adjusted using earlier groups'
experience. The genome stays fixed in this check.

Open `summary.json` in the new run folder for shelter use, survival and both
groups' outcomes. Each cohort has a JSON index and a matching NumPy archive
containing its events and the shared table before and after its life.
The fixed sequence exercises the mechanics. A test of whether the memory helps
needs independent lineages and held-out worlds.

To recover the group's memory in a later local session, use
`load_development_archive(run_dir)` from `scripts.ecology_lineage`. Pass its
return value as `archive_state` to `seed.lineage.collective_lives` with the
same genome and sensor layout and new world and development keys. It continues
the cohort numbering and checks which keys have already been used. The paired
fresh-weight group starts from birth weights on every life.

## Optional persistence rule

`from seed.persistence import PersistenceV1` supplies the optional
`persistence=PersistenceV1(memory_steps=4, stall_steps=2)` keyword to ecology
`rollout`, `make_evaluator`, `evolve`, `same_individual` and `collective_lives`.
Omitting it preserves the previous policy and output schema exactly. It requires
the ecology observation layout; registered historical protocols reject it.

This is an explicit agent rule, not an evolved trait or a claim about motivation.
When alive, cold and outside shelter, a creature that sees shelter adds one to
one movement logit pointing toward its nearest visible shelter cue. It can retain
that relative offset for at most four missing-cue decisions by subtracting its
own previous movement. No global map, absolute position or hidden food rule is
available to the rule. Warmth, shelter, death or stale memory clears pursuit.
The existing stochastic policy still selects the action and pays the same costs.

After two consecutive visible observations without a decrease in shelter
distance, the rule biases an alternative movement and subtracts one from the
previous action's logit. Repeatedly staying can therefore count as a stall only
while actively seeking cold-weather shelter. There are at most two such tactic
attempts per uninterrupted pursuit episode. The local distance heuristic can
switch between visible shelter pixels and does not infer progress from a missing
cue. Memory is configurable from 1 to 16 decisions, with the stall threshold
between 1 and that bound. No extra energy or additional lifetime is provided.

Recorded/diagnostic rollouts include per-agent `persistence` arrays: `active`,
`remembered`, `progress_observed` (actual visible distance decrease), `stalled`,
`tactic_attempted`, `previous_action`, `selected_action`, and `action_switched`.
The last is true only when a tactic was attempted and the sampled action differs
from the preceding action. Action 0 is stay and is the initial previous-action
sentinel. As in the original engine, actions are still sampled for dead slots;
all six boolean behavior events are false there. Reweighting is not evidence of
a successful switch or of better shelter use.

The same setting applies to every parent, successor and matched control arm.
Only learned weights transfer; pursuit memory resets at birth. Archive metadata
binds the setting and rejects a different setting on continuation. An archive
without this metadata means the trait was off. Shared versus fresh weights with
the trait enabled is not an enabled-versus-disabled persistence comparison.
