# Seed

Seed is a small simulation about learning from experience. We evolve simple brains for creatures that have to find food and avoid poison. Then we watch what each creature learns during its life and how it reacts when the rules change.

## What the creatures do

Four creatures move around a 16 × 16 grid. Each sees a small patch around itself and can move one square or stay put. There are two kinds of items, A and B. In one world, A is food and B is poison. In another world, B is food and A is poison. The rule is hidden from the creatures. Food adds energy, poison drains it, and staying alive uses a little energy each turn. A life lasts up to 300 turns and ends early if energy reaches zero. In some tests, the food rule switches halfway through.

## How a creature gets its brain

The **genome** is a set of 155 numbers. It tells Seed how likely different kinds of brain cells are to connect, how strong those connections start out, and how experience can change them. Each of the four creatures in a world starts from the same genome. Seed draws the wiring separately for each one, so they can start with different brains from the same instructions.

## How a creature learns during its life

1. **Look and act.** The brain takes in nearby items, other creatures and their signals, and its own energy. Brain activity carries over from one turn to the next. The brain uses that activity to choose a move and keeps track of the move it actually made.
2. **Get feedback.** If the creature lands on an item, it eats it. On the next turn, the brain gets the change in energy and compares what it sees with what it expected to see.
3. **Adjust the brain.** One update helps the brain make a better prediction next time. The other changes the strength of existing connections using the activity from the last move. The genome controls how the energy change and the size of the prediction error influence that update.

For example, one starting genome is designed to strengthen the link between seeing an item on the right and moving right when that move gains energy. If the move loses energy, the same rule can weaken that link. That gives the next decision a chance to use the last experience. This example comes from Seed's hand-written starting rule. Evolution can discover different ways to use the feedback. [Brain update code](seed/seed.py#L135) · [Example starting genome](seed/seed.py#L71)

## What a generation means

A **turn** is one action in the grid. A **life** is one creature's time in that world. A **generation** is one round of training that tests many slightly different genomes, each across several fresh lives.

1. **Make candidates.** Start with the genome that training has reached so far. Make copies with small random changes to its numbers. The candidates come in pairs, with one copy adding a change and the other subtracting it.
2. **Let them live.** Try each candidate on the same set of training worlds for that round. Build new brains from the candidate, then let the creatures act and learn through a full life.
3. **Score and update.** In the original experiment, a creature's score was food eaten minus poison eaten. Seed averages scores across creatures and worlds, ranks the candidates, and uses those ranks to update the genome. The next generation tries new changes around that updated genome.

In the original A runs, one generation tested **128 candidate genomes in 8 worlds each**, with 4 creatures per world. Each run repeated that process **200 times**. A new creature inherits the instructions for building and updating a brain. It starts life with fresh wiring and no memory of the lives that came before it. [One run's settings](runs/20261112-023245_E103-A-blank-s0/config.json) · [Evolution code](seed/seed.py#L332)

A **random seed** is a number that sets the sequence of random choices in a training run. We repeat experiments with different seeds to see whether the same pattern shows up across runs.

## How we check whether learning helps

We test saved genomes on worlds set aside from training. To check learning during a life, we look at food choices over time and compare a genome with its brain updates on and off. To check flexibility, we can switch the food rule halfway through a life or replay the same starting creature in two separate lives, one under each rule. We track survival alongside food choices because some creatures die before they have many chances to eat. [Learning and rule-switch tests](results/EVAL_V3.md) · [Paired-rule tests](results/EVAL_V4.md)

## What we found

- **The creatures got better at choosing food when the rule stayed fixed.** Among creatures that ate at least twenty times, the twentieth meal was more likely to be food than the first. The gains in three separate runs were 27 ± 2, 21 ± 2 and 25 ± 2 percentage points. When the rule switched during a life, food choices still hovered around half correct near the end. [Learning and rule-switch results](results/EVAL_V3.md)
- **A high average score can hide a one-rule specialist.** We trained a second type of brain whose connections stay the same strength throughout its life. It outscored the learning brains in all three runs on the original measure of food eaten minus poison eaten. Each of these brains did well under one food rule and poorly under the other. That led us to replay the same starting creature under both rules and score it on its worse result. [Fixed-weight comparison](results/E113_BASELINE.md) · [Paired-rule test](results/EVAL_V4.md)

## What happened when we trained for both rules?

We took the three brains from the first learning experiments and trained two versions of each for another 50 generations. One was rewarded for its average score across the two rules. The other was rewarded for each creature's worse result. Both versions scored below their starting brain on the harder-rule test, across 256 worlds set aside for testing. [Full results and raw data](results/EVAL_V5.md)

The numbers below show how much the harder-rule score changed when we trained specifically to improve it. The score counts food minus poison per creature. A negative change means the new brain scored lower than its own starting point. The ± values show one standard error across matched test worlds.

| Starting brain | Change after training |
| --- | ---: |
| A0 | −1.47 ± 0.41 |
| A1 | −2.23 ± 0.43 |
| A2 | −1.83 ± 0.40 |

This experiment, **E115**, passed none of its six required score comparisons. Two of the three brains also missed the target for keeping creatures alive under both rules. The original three brains are still our benchmark for this test. [See the chart for both training versions and all three original brains](results/figures/e115-objective-comparison.svg)

## What comes next

**E116 is awaiting approval.** It would start from the original brains and test whether taking a smaller step when updating the genome between generations helps. A new brain would have to beat both its starting brain and a version trained with the original step size while meeting the survival target. [Read the proposal](experiments/E116_PROPOSAL.md)

## Try it locally

From the root of a clean checkout, use Python 3.12.10. The commands below create a local Python environment, install the pinned dependencies and run a small check on Windows PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
.\.venv\Scripts\python.exe scripts/run_experiment.py --name smoke --quick
```

On macOS or Linux, use `python3` to create the environment and `.venv/bin/python` for the next two commands. The small check confirms the simulator runs and saves a `summary.json` in a new folder under `runs/`. For an example you can inspect without running anything, open the [first original brain's summary](runs/20261112-023245_E103-A-blank-s0/summary.json).

<details>
<summary>More setup and testing notes</summary>

On the owner's laptop, work from the checkout on D: and set temporary files to use that drive before setup.

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TEMP = $env:TMP = (Resolve-Path .tmp).Path
```

For software tests, create `.tmp/` in the checkout and run `python -X utf8 -m unittest discover -s tests -v` with the environment's Python. Some tests need full Git history. The verified setup uses JAX on the Windows CPU. [CI workflow](.github/workflows/tests.yml)

</details>

## Where to look next

| If you want to… | Start here |
| --- | --- |
| See every recorded run | [Results index](results/RESULTS.md) |
| Read the hypotheses, commands and decisions | [Lab notebook](experiments/LOG.md) |
| Find the next experiment and the rules for running it | [Research plan](PLAN.md) · [Current status](TASK_STATE.md) · [Research rules](AGENTS.md) |
| See how the world and brains work in code | [Simulation](seed/seed.py) · [Experiment runner](scripts/run_experiment.py) |

The [earlier README](https://github.com/Winter-Soren/seed/blob/v0.5-evalv5/README.md) has the original tables and charts for the older experiments, including the hand-set starting brain and the connection growth tests.
