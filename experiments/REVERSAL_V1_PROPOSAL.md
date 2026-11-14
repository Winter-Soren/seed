# REVERSAL v1: changing-world panel ready for review

Software prototype and tiny direct fixtures are authorized. **Final scientific panels
are not authorized yet.** No final seed8211 world has been evaluated. E108 is complete
and pinned separately at9766538; its conditions and artifacts remain unchanged.

## Fixed intervention and comparisons

Use the existing seven saved genomes (A0/A1/A2, blank0/1/2, one unique biology prior),
default Cfg (T300, G16, M4, R2, H48, n_items24, decay0.003), 128 worlds balanced by
initial rule, and reserved final seed8211. No evolution or tuning.

Four conditions share identical world/development/action keys: noflip_learning,
noflip_frozen, flip_learning, flip_frozen. Learning enables both Hebbian and predictor
updates; Frozen disables both. The world comparison changes one factor: at zero-based
step150 the food/poison mapping reverses. Nothing resets at that boundary: grid, energy,
recurrent state, predictor, weights, positions and random streams continue. No flip
announcement, new observation channel or changed action interface is provided.

The no-flip controls use the same measurement boundary. Development seed8210 is
separate; CLI development mode enforces tiny configurations. Legacy/default rollout
behavior and E106's predictor_plastic API remain intact. E107 is not in this prototype's
base18818de and needs its ordinary merge/compatibility checks before a combined panel;
do not replace its controller code. The proposed first reversal panel is weight-only.

## Measurements fixed before final-world access

Retain compact per-agent pre/post food/poison counts, nullable accuracies, survival at
the boundary and life end, and recovery-window-hit latency/censor reason. Retain pooled
counts/accuracy/survival for every non-overlapping 25-step post-boundary window.

A **window hit** is the first such window with >=5 encounters and >=70% food choices,
while the agent is still alive at the window end. Latency is that end minus150.
This is a declared behavioral recovery proxy, not proof of rule inference or sustained
learning. Report late performance too. An early hit followed by death remains visible.

Censor reasons distinguish dead_at_boundary, died_before_window_hit,
no_post_encounters, insufficient_window_encounters, and criterion_not_met. No encounters
means accuracy=null. Report hit fractions both over all512 agents and over the actual
agents alive at the boundary, with their counts; the latter denominator can differ
across learning/frozen, so it is not an isolated causal estimate. Report raw censor
counts, survival and encounters alongside every conditional result.

Report signed post-boundary fitness differences: learning-minus-frozen separately in
no-flip/flip worlds, and flip-minus-no-flip separately for learning/frozen. All negative,
zero and mixed results remain. Never infer recovery from whole-life fitness alone.

## Prospective hypothesis and falsifiers

For each A genome, ask whether enabled updates improve post-flip fitness over frozen,
and whether last-window food accuracy exceeds first-window food accuracy. The proposed
directional hypothesis requires both in >=2/3 independently evolved A seeds; a window
with <5 pooled encounters makes its accuracy criterion unestablished. Nonpositive
fitness differences or nonincreasing first-to-last accuracy in >=2 seeds count against
the hypothesis; insufficient observations do not become a pass. Report every seed,
per-agent window-hit/censor result and reference regardless of the criterion.

This is a temporal rule-change test within the food/poison task family. Pooled accuracy
still reflects encounter/survival selection; frozen brains retain recurrent state and
energy feedback. No significance, new-task transfer, SOTA, sleep benefit, structural
benefit or AGI claim is proposed. This measurement definition itself needs owner review.

## Exact final commands (pending protocol approval and compute slot)

Run sequentially through the committed runner, starting with A0 as a resource benchmark.
The input genomes are the immutable E100/E103 artifacts; record their full file SHA-256
from the exact loaded bytes and preserve identical copies. CLI seeds record provenance,
not the protocol's fixed final randomness. Never tune on seed123,8109 or8211.

```powershell
D:\cool-projects\general-intelligence\.venv\Scripts\python.exe -X utf8 scripts/run_experiment.py --name reversal-v1-final-A0 --eval-protocol reversal-v1 --condition-split final --start file:runs/20261112-023245_E103-A-blank-s0/genome_final.npy --generations 0 --eval-worlds 128 --seed 0
D:\cool-projects\general-intelligence\.venv\Scripts\python.exe -X utf8 scripts/run_experiment.py --name reversal-v1-final-A1 --eval-protocol reversal-v1 --condition-split final --start file:runs/20261112-050319_E103-A-blank-s1/genome_final.npy --generations 0 --eval-worlds 128 --seed 1
D:\cool-projects\general-intelligence\.venv\Scripts\python.exe -X utf8 scripts/run_experiment.py --name reversal-v1-final-A2 --eval-protocol reversal-v1 --condition-split final --start file:runs/20261112-102755_E103-A-blank-s2/genome_final.npy --generations 0 --eval-worlds 128 --seed 2
D:\cool-projects\general-intelligence\.venv\Scripts\python.exe -X utf8 scripts/run_experiment.py --name reversal-v1-final-blank0 --eval-protocol reversal-v1 --condition-split final --start file:runs/20261112-022439_E100-blank-s0/genome_final.npy --generations 0 --eval-worlds 128 --seed 0
D:\cool-projects\general-intelligence\.venv\Scripts\python.exe -X utf8 scripts/run_experiment.py --name reversal-v1-final-blank1 --eval-protocol reversal-v1 --condition-split final --start file:runs/20261112-022514_E100-blank-s1/genome_final.npy --generations 0 --eval-worlds 128 --seed 1
D:\cool-projects\general-intelligence\.venv\Scripts\python.exe -X utf8 scripts/run_experiment.py --name reversal-v1-final-blank2 --eval-protocol reversal-v1 --condition-split final --start file:runs/20261112-022552_E100-blank-s2/genome_final.npy --generations 0 --eval-worlds 128 --seed 2
D:\cool-projects\general-intelligence\.venv\Scripts\python.exe -X utf8 scripts/run_experiment.py --name reversal-v1-final-biology --eval-protocol reversal-v1 --condition-split final --start file:runs/20261112-022641_E100-biology-s0/genome_final.npy --generations 0 --eval-worlds 128 --seed 0
```

Before scientific launch, append the approved version and these commands in LOG, commit
clean code/registration, reserve compute, and recheck the established laptop guards.
Preserve failures and all immutable run artifacts. Three seeds are required for a
retention/adaptation conclusion. Rebuild RESULTS and record observed numbers afterward.
