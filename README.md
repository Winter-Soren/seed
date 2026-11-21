# Seed

Seed is a small simulation about learning from experience. We evolve simple brains for creatures that have to find food and avoid poison. Then we watch what each creature learns during its life and how it reacts when the rules change.

## What the creatures do

Four creatures move around a grid. They can only see what is nearby. Each world has two kinds of items. In one world, the first kind is food and the second is poison. In another world, their roles are reversed. The creatures get feedback through changes in their energy after eating. In some tests, the rule switches halfway through their lives.

Each creature starts with a brain built from a **genome**, a set of 155 numbers that describes how to wire the brain and how it should learn. During a life, experience changes the strength of connections. Between generations, evolution tries small changes to the genome and uses the creatures' scores to decide which changes to keep. Each new life starts with a freshly built brain.

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

**E116 is awaiting approval.** It would start from the original brains and test whether making smaller changes to the genome each generation helps. A new brain would have to beat both its starting brain and a version trained with the original step size while meeting the survival target. [Read the proposal](experiments/E116_PROPOSAL.md)

## A world with shelter and memory across lives

Creatures in the new world can see nearby shelter. Moving costs energy. Halfway through a life the weather turns cold, and a creature that ends its turn outside loses extra energy. In a controlled check, staying outdoors left a creature with 0.750 energy and moving into shelter left it with 0.875.

When a cohort's life ends, Seed saves what its members learned on connections from senses to actions. A newborn develops its own wiring. It receives earlier changes only on connections it actually has. An agent that dies early contributes its last living weights. The saved table can be loaded in a later local session.

One software check reused the original A0 genome across **three cohorts of two newborns**, with matched fresh-weight controls. Both newborns in each later cohort received changed starting weights. In the cold world, one of two creatures reached shelter in each group, and all survived. This verifies the transfer mechanism. Measuring a survival or navigation benefit requires independent lineages on held-out worlds. [Run and inspect the example](experiments/ECOLOGY_API.md) · [Saved observations](runs/20261121-062635_ecology-transfer-smoke-a0/summary.json) · [Proposed comparisons](experiments/ECOLOGY_TRANSFER.md)

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
