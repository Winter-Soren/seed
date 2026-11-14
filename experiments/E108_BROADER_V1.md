# E108: BROADER v1 proposed final panel

Completion update, 2026-11-14: all seven final panels completed with finite metrics and
unchanged genome bytes. All three A seeds retain positive advantages in every shift.
See the completed E108 record in LOG.md and E108_VERIFICATION.json. The submitted
conditions, commands, metrics and falsifiers below were not changed after evaluation.

Status: implementation and development fixtures authorized; **final scientific launch
requires owner approval of this exact protocol and a coordinated compute slot**.
No final-world result is available. EVAL v1/v2 remain unchanged.

Result update, 2026-11-14: the first final A0 panel is complete, with the result and
measured cost appended to LOG.md. This supersedes the initial submission status above;
the remaining six panels await coordinator cost review. Conditions/criteria are unchanged.

Approval update, 2026-11-14: the owner answered “Approve BROADER v1” to the exact panel.
The first A0 final panel has a coordinated bounded slot; later panels await its cost
review. See the appended E108 approval/development records in LOG.md.

## Question, scope and fixed conditions

Does the plasticity fitness advantage in the three saved E103 A genomes persist when
one environmental parameter changes? All genomes are fixed, with fresh lifetime
learning allowed and a matched frozen control. No evolution, optimization, selection,
hyperparameter search or tuning on either final split or EVAL v1 worlds is permitted.

| Label | Only change from default Cfg | Meaning |
|---|---|---|
| control | none | Original task family, new held-out randomness |
| sparse | n_items=12 (was 24) | Half the initial items of each kind |
| dense | n_items=36 (was 24) | 1.5 times the initial items of each kind |
| low-cost | decay=0.0015 (was 0.003) | Half the metabolic cost |
| high-cost | decay=0.006 (was 0.003) | Twice the metabolic cost |

The same observation/action interface, G=16, M=4, T=300, R=2, H=48 and three hidden
types apply to every final condition. These shifts remain the same food/poison task
family. Success here would show parameter robustness, not transfer to a new task,
open-ended general intelligence, or state-of-the-art performance.

## Splits and pairing

`--eval-protocol broader-v1` is separate runner dispatch; the default remains EVAL v1.
Development uses PRNG seed **8108**. Final uses reserved seed **8109**, **128 worlds**,
64 with each rule, default Cfg only. This extends the evaluation inventory without
relabeling or replacing any historic EVAL v1 score. Tests of final dispatch use a
synthetic evaluator and do not generate or inspect final worlds.

For each genome, condition and learning/frozen pair, `test_genome` receives identical
world/development/action random keys, n_worlds and seed. A changed density gives a
different item layout from the same random stream; it is not guaranteed to preserve
the control's subset of item cells. Development and action streams are held matched
at initialization but trajectories can diverge in response to conditions/learning.

Development smoke is one rule-0 world per condition, with full default brain/lifetime,
for cost and wiring checks only. Do not choose scientific thresholds or changes from
those scores. If debugging requires changing a condition, append an amendment and
version the proposed protocol before accessing any final worlds. Final scores are
reported once, including failures; they cannot select later variants of BROADER v1.

## Frozen input panel and hashes

All sources are existing `.npy` artifacts. Record complete file SHA-256, short genome
array hash, source commit and machine. The runner loads and hashes the same bytes,
rejects pickle/non-finite/mismatched genomes, and copies identical bytes into new run
folders. Check the following hashes before and after the final panel.

| Input | Source (`runs/` prefix) | SHA-256 of whole file |
|---|---|---|
| A0 | 20261112-023245_E103-A-blank-s0/genome_final.npy | 477d9d03c713dcab8072349ff9155813e5a384d3709eff67595a342b55111f53 |
| A1 | 20261112-050319_E103-A-blank-s1/genome_final.npy | 35196cd30c2de5c1ed5bb9e23ce445ff6ad13bd1a33bdbf7a77f54d53edce899 |
| A2 | 20261112-102755_E103-A-blank-s2/genome_final.npy | f299069753636a303f6ba7f91be78099e8491b20a7511bdcef47fd8b4c5bbaa6 |
| blank0 | 20261112-022439_E100-blank-s0/genome_final.npy | 621e5a735e408e487653f7ffc15561be773fbd1053af847ef1bb633fd5243e07 |
| blank1 | 20261112-022514_E100-blank-s1/genome_final.npy | 3ec510cab0205915614a7df2d3b217eca201972897842592fcef419b215e5d99 |
| blank2 | 20261112-022552_E100-blank-s2/genome_final.npy | c62051a478c75f3211645968e5a15bb64bcfcdb8f61c366357020f4cff8294b9 |
| biology | 20261112-022641_E100-biology-s0/genome_final.npy | f39d8960f182b832e2700d6274e5d838d7a0f410add2b7753b6f259aba732108 |

A0/A1/A2 are the primary three independently evolved seeds. Blank0/1/2 and one unique
biology genome are reference anchors under the same changed conditions; repeated
biology seeds are identical genomes and must not count as independent replications.

## Metrics, hypotheses and falsifiers

Record all five existing metrics for learning and frozen in every condition:
food-minus-poison fitness per agent, pooled first/second-half food-choice accuracy,
survival fraction, and total food/poison encounters (`eaten`). Also record:

- learning fitness minus frozen fitness (the primary within-genome plasticity advantage);
- condition minus control fitness, separately for learning and frozen;
- condition plasticity advantage minus control plasticity advantage;
- second-minus-first accuracy for both learning and frozen, without treating it alone
  as evidence of a synaptic learning mechanism.

The directional hypothesis is positive plasticity advantage in each of the four
shifted conditions in at least two of three A seeds. For a given condition, <=0 in
two or more seeds falsifies this condition's retention hypothesis. If any shifted
condition fails, the hypothesis of retained advantage across the whole suite fails;
report which one and every per-seed signed effect. A control failure also weakens
the original task-family result on this new split. Do not invent a new roadmap gate,
significance claim or threshold after seeing scores. Report references descriptively.

Survival or encounter selection can change pooled accuracy. Frozen agents retain
recurrent state, energy feedback and fixed developed wiring. Learning enables both
Hebbian and predictor updates as in EVAL v1; E106 separately tests their contributions.
Absolute fitness is resource-dependent, so a fall under sparse conditions alone is
not a failure of adaptation. Compact aggregate metrics do not estimate per-world
uncertainty or individual recovery; no confidence interval is claimed.

## Exact final commands (pending approval and slot)

Run sequentially from this lab or an integrated clean checkout. All scientific runs
must use `scripts/run_experiment.py` from committed code, with no `--allow-dirty`.
CLI seeds record provenance; the protocol's final evaluation seed stays 8109.

```powershell
D:\cool-projects\general-intelligence\.venv\Scripts\python.exe -X utf8 scripts/run_experiment.py --name E108-final-A0 --eval-protocol broader-v1 --condition-split final --start file:runs/20261112-023245_E103-A-blank-s0/genome_final.npy --generations 0 --eval-worlds 128 --seed 0
D:\cool-projects\general-intelligence\.venv\Scripts\python.exe -X utf8 scripts/run_experiment.py --name E108-final-A1 --eval-protocol broader-v1 --condition-split final --start file:runs/20261112-050319_E103-A-blank-s1/genome_final.npy --generations 0 --eval-worlds 128 --seed 1
D:\cool-projects\general-intelligence\.venv\Scripts\python.exe -X utf8 scripts/run_experiment.py --name E108-final-A2 --eval-protocol broader-v1 --condition-split final --start file:runs/20261112-102755_E103-A-blank-s2/genome_final.npy --generations 0 --eval-worlds 128 --seed 2
D:\cool-projects\general-intelligence\.venv\Scripts\python.exe -X utf8 scripts/run_experiment.py --name E108-final-blank0 --eval-protocol broader-v1 --condition-split final --start file:runs/20261112-022439_E100-blank-s0/genome_final.npy --generations 0 --eval-worlds 128 --seed 0
D:\cool-projects\general-intelligence\.venv\Scripts\python.exe -X utf8 scripts/run_experiment.py --name E108-final-blank1 --eval-protocol broader-v1 --condition-split final --start file:runs/20261112-022514_E100-blank-s1/genome_final.npy --generations 0 --eval-worlds 128 --seed 1
D:\cool-projects\general-intelligence\.venv\Scripts\python.exe -X utf8 scripts/run_experiment.py --name E108-final-blank2 --eval-protocol broader-v1 --condition-split final --start file:runs/20261112-022552_E100-blank-s2/genome_final.npy --generations 0 --eval-worlds 128 --seed 2
D:\cool-projects\general-intelligence\.venv\Scripts\python.exe -X utf8 scripts/run_experiment.py --name E108-final-biology --eval-protocol broader-v1 --condition-split final --start file:runs/20261112-022641_E100-biology-s0/genome_final.npy --generations 0 --eval-worlds 128 --seed 0
```

Before launch, approve the protocol and reserve the compute slot. Benchmark final A0
first; measure actual full-panel memory/time before extrapolating the remaining six.
One BelowNormal process, D: temporary files, authoritative memory/disk guards and the
under-three-hour process cap remain required. Development one-world timing is not a
reliable estimate of 128-world memory. A full final panel is not safe concurrent work
merely because the development fixture was small. Preserve interrupted folders and
start a new named panel after a failure; this zero-generation harness refuses resume.

After completion run `D:\cool-projects\general-intelligence\.venv\Scripts\python.exe -X utf8 scripts/summarize.py`,
record numbers from summaries, update README with results and limitations, and commit
the new run folders/RESULTS/LOG/README together. Main chat coordinates integration and
publication; this lab's push URL stays disabled.

## Separate future condition: food-rule reversal

Not implemented or implicitly authorized by this parameter suite. A reversal changes
the task dynamics: the food/poison mapping flips inside a lifetime while learned state
is retained. It needs a separately versioned protocol, retained-state versus newborn
controls, matched no-flip runs, pre/post-flip encounter counts, survival at the flip,
early/late post-flip accuracy, and a preregistered recovery definition that handles
agents that die or never encounter food. The current two-half aggregate cannot measure
recovery reliably. Choose flip timing/windowing and owner authorization before writing
that world change; no reversal or sleep benefit is claimed here.
