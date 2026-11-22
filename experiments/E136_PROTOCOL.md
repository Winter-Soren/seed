# E136 · Select for first choices across ordinary backgrounds

## Registered question and scope

E135's independently audited stationary attenuation gate passed in all three lineages, selecting this background branch. Its source/result closes at `fa07d896b12e1acd1630e863c1fa9e859d58cc94` with local tag `v0.24-scene-transfer-v1` and audit SHA-256 `879186a4140a7da3074069b58bca612524c53055a02632f859703a8461d5096d`. A1's separate ordinary stationary deficit remains unresolved. E136 changes only which of two matched first-choice scores enters genetic selection. It compares newly trained ordinary-background and isolated-view native 155-number genomes with each other and their own E103 parents. No mechanism, lifetime update, pretraining, exposure, objective mixture, sample count or endpoint is selected using final results.

Hypothesis: selecting ordinary-background first-choice S instead of isolated S repairs transfer to fresh ordinary solo grids after matched experience. Stationary repair and reversal advancement are separate, preregistered decisions. A higher training score alone passes neither. Autonomous acquisition and reversal from birth remain outside scope.

## Subjects, paired training and one-factor intervention

Start both arms from identical E103 generation-200 parents and zero optimizer moments:

| Lineage | Parent genome |
|---|---|
| A0 | `runs/20261112-023245_E103-A-blank-s0/genome_final.npy` |
| A1 | `runs/20261112-050319_E103-A-blank-s1/genome_final.npy` |
| A2 | `runs/20261112-102755_E103-A-blank-s2/genome_final.npy` |

The runner pins and verifies source metadata and raw genome SHA-256. Use 10 ES attempts, population 16 (eight antithetic pairs), four fresh families per attempt, sigma .05, learning rate .02, L2 .005 and native updates. Pair the optimizer noise, scripts, birth keys, scene keys and action keys across objective arms at each attempt. The same birth key serves both signs of an antithetic pair. Each genome develops its own brain and samples its own free action. Lineage masters are independent; pairing is within a lineage and attempt. Always freeze attempt-ten means, including all-tied attempts. No best-checkpoint selection.

Each training attempt contains the four old-contact/gap cells `(8,0),(8,8),(16,0),(16,8)` once in a rule-blind shuffled order. Within each family include both initial food mappings and stationary and reversed post blocks equally. Replay eight or sixteen old-rule contacts then sixteen post-block contacts, each followed by its empty-hub return. Geometry and executed contact paths are identical across arms, genotypes and mappings. The final return delivers the last contact's realized energy change before any gap. The gap has native learning, empty visual input and imposed stay. Guided replay reward is never the fitness.

From every acquired state, both arms calculate both discarded, frozen first-choice probes over all 12 ordered adjacent cardinal A/B placements. The full physical grid is identical in both views: center (8,8), adjacent A/B, the other two cardinal cells and center empty, plus 46 nonfocal items (23 A,23 B) sampled without replacement outside the central cross. Isolated view replaces only the two item observation planes by their matching isolated focal planes. Background view retains actual ordinary-density item planes. Preserve all other sensory features, neural state, neutral body (energy 1, change 0), five action alternatives, readout timing and disabled first-probe Hebbian/predictor writes. Scores and discarded probes never feed back into exposure. Use the same background and action keys across repeats and candidates. Keep zero-visible-distractor families and verify view identity there.

For each placement, `S = P(current food move) - P(current poison move)` using all five native probabilities. Each arm's scalar is its assigned view's S averaged equally over four families, two initial rules, two histories and twelve placements. Both arms execute the same primitives, copies, exposure lengths and scoring opportunities; only this preassigned scalar enters centered tied ranks and ES. Archive both view readouts for reconstruction, but never use the unused objective to tune, stop, choose endpoints or change evaluation. Record distinct fitness values, separated antithetic pairs and all-tied attempts descriptively. Preserve E122 tied-rank/all-tied behavior; sparse scores receive no added shaping bonus. Compare actual per-arm wall costs against equal caps, rather than claiming bit-identical elapsed times.

## Physics and sealed assignments

An adjacent contact pair consumes A and B under one mapping; every switch is at a pair boundary. Net unmatched payoff is at most one .25 gain or loss. Starting energy is 1, decay .003, maximum held-out exposure is `2*(16+16)+32 = 96` ticks. Thus all unclipped energies lie within `[1-.25-96*.003, 1+.25] = [.462,1.25]`, strictly inside `[0,2]`, independently of genome and action randomness. Training has shorter gaps. Keep explicit finite-value, exact contact-count, unclipped-energy and all-alive checks; a violation is operational invalidity, not a scientific failure. At readout, standardize the physical body and use matched states, so energy, orientation and last position cannot disclose the history being scored.

Protocol is `background-selection-v1`, namespace `0xE136`. ES starts with the registered training master through the existing `seed.es_init` and splits its key into mutation, world and birth streams as in E134. Within each training family split the world key into ordered roles `script,chamber,background,probe`. The training schedule key folds `0xE136` into the first world key. Final family keys fold the panel master first with the namespace then the zero-based family index and split into seven ordered roles `script,birth,chamber,background,probe,solo_layout,solo_action`. Probe keys split into twelve placements. The independent audit checks root/role identity and disjointness without generating final assignments during development.

Development: two cold training processes, two attempts each, use the same A0 parent and master `2026121600`; names `E136-benchmark-isolated` and `E136-benchmark-background`. A separate cold four-family, three-genotype evaluation uses master `2026121700` and name `E136-benchmark-eval`. Final paired training masters are `2026121601,2026121602,2026121603`; separate common evaluation masters are `2026121701,2026121702,2026121703`. Final training names are `E136-Ai-isolated` and `E136-Ai-background`; final evaluation names are `E136-Ai`. Arm follows the registered run name, not a new CLI flag. No previous study world is reused.

All four evaluation cells cycle in fixed order `(8,16),(16,32),(8,32),(16,16)`. Each final panel has 128 genuinely fresh paired families, 32 per cell, with both initial rules and both histories. Freeze and hash all six final attempt-ten genomes, training summaries and source receipts in `.tmp/E136-endpoints.json` before deriving any final panel key. A frozen endpoint is not replaced after reading another lineage or view.

## Held-out readouts and decision rules

Evaluate each parent, isolated endpoint and background endpoint on both frozen local views and on a separate unprobed fresh ordinary grid. Preserve acquired brain state, standardize the body, and use shared rule-blind solo layouts, action and regrowth keys within family. All 50 ordinary-grid ticks use native perception, native updates, regrowth and `forced=-1`; no focal item is inserted and no action being scored is imposed. The first actual food contact scores Y=+1, poison -1, and no contact or death before contact 0. Include a lethal first contact. Run the fixed length even when the endpoint is already known, preserving absorbing death. Never condition the primary denominator on contact or survival.

For each lineage, average repeated rules, placements and cells within each independent family before calculating means and standard errors. Contrasts are paired within those same 128 families. Clones, histories and rules are not independent replicates. Report separate history-specific Y, first-food, first-poison, no-contact and survival, with assigned/contact denominators. Conditional-on-contact food accuracy is descriptive and null at zero contact; it cannot replace Y. Report the descriptive decomposition `Y = contact_rate * (2*conditional_food_fraction - 1)`.

Define `gain(x)=mean(x)>=.05 and mean(x)>2*paired_SE(x)`, `positive(x)=mean(x)>2*SE(x)`, and `safe(x)=mean(x)-2*paired_SE(x)>-.05`. Equality at a strict bound fails. Use each lineage independently and require all applicable criteria in all three lineages; no pooling away a failure.

**Stationary repair** requires, in every lineage:

1. Positive absolute background-endpoint stationary ordinary solo Y.
2. Stationary Y gain against both its retrained isolated endpoint and own parent.
3. Stationary background-view local S gain over the retrained isolated endpoint.
4. Reversal Y safeguards against both comparators.
5. Survival safeguards against both comparators separately in both histories.

**Reversal advancement**, assessed separately, requires in every lineage:

1. Positive absolute background-endpoint reversal ordinary solo Y.
2. Reversal Y gain against both its retrained isolated endpoint and own parent.
3. Stationary Y safeguards against both comparators.
4. Survival safeguards against both comparators separately in both histories.

Repair neither substitutes for nor is a prerequisite to reversal advancement. Report absolute and incremental skill separately. The signed local logit ceiling cannot relax a gate. These criteria concern behavior after matched replay and after gaps, not autonomous learning from birth, causal retention, feedback-caused solo discrimination or an architectural superiority claim.

Report both local views, their interaction and focal approach `F=p_food+p_poison` and conditional focal preference `q=p_food/F` (undefined at zero), so `S=F*(2q-1)`. Where decomposing paired scene differences, calculate the symmetric approach and relative-color terms per placement before averaging families, as in E135. Nonfocal items may provide later reward; changed first-move probabilities do not by themselves establish an attention deficit.

Prospective precision: with paired family SD .50, n=128 gives SE .04419 and two-SE .08839. A nominal .05 gain can remain unresolved. Sample size is fixed before outcomes. Report mean±one SE and two-SE intervals. Failure to pass is not proof of equivalence: a comparison whose upper bound is below .05 excludes that minimum only for the tested objective, schedule and budget; a wide interval leaves the gain unresolved. A lineage with a resolved opposite effect cannot be repaired by pooling or treating replication as merely imprecise. Retire this exact setting after a complete failure; any later study needs fresh registration and keys.

## Exact commands and release sequence

Commands run only through the guarded entry point below after registration and code commits, with verified repo-local owner identity. No GitHub, publication, remote CI, visibility change or service is authorized. The canonical experiment entry is `.venv\Scripts\python.exe -B -X utf8 scripts/run_experiment.py`; common arguments are `--eval-protocol background-selection-v1 --pop 16 --worlds 4 --sigma .05 --lr .02`.

```powershell
$python = '.\.venv\Scripts\python.exe'
$entry = 'scripts/run_experiment.py'
$common = @('-B','-X','utf8',$entry,'--eval-protocol','background-selection-v1','--pop',16,'--worlds',4,'--sigma','.05','--lr','.02')
$parents = @('runs/20261112-023245_E103-A-blank-s0/genome_final.npy','runs/20261112-050319_E103-A-blank-s1/genome_final.npy','runs/20261112-102755_E103-A-blank-s2/genome_final.npy')
& $python @common --condition-split development --name E136-benchmark-isolated --seed 2026121600 --generations 2 --eval-worlds 0 --start "file:$($parents[0])"
& $python @common --condition-split development --name E136-benchmark-background --seed 2026121600 --generations 2 --eval-worlds 0 --start "file:$($parents[0])"
& $python @common --condition-split development --name E136-benchmark-eval --seed 2026121700 --generations 0 --eval-worlds 4 --start "file:$($parents[0])"
foreach ($i in 0..2) {
  foreach ($view in @('isolated','background')) {
    & $python @common --condition-split final --name "E136-A$i-$view" --seed (2026121601+$i) --generations 10 --eval-worlds 0 --start "file:$($parents[$i])"
  }
}
# Only after all six frozen endpoints are pinned and released:
foreach ($i in 0..2) {
  & $python @common --condition-split final --name "E136-A$i" --seed (2026121701+$i) --generations 0 --eval-worlds 128 --start "file:$($parents[$i])"
}
```

Readiness: focused `test_background_selection*.py` plus `tools/audit_background_selection.py --self-check` in a fresh process. Independent development: `tools/audit_background_selection.py --source-commit <scientific SHA> --development`. Joint reconstruction: same without `--development`. Each command uses `-B -X utf8`. Reconstruct assignment roles, actual view fitness and ranks, every ES attempt, complete raw checksums, shared opportunities and noise, energy and movement, five-action orientation, ordinary first contacts, all-assigned family contrasts and both decisions. Independently reconstruct frozen readout logits from archived acquired states; document that development and recurrent plasticity are not wholly rerun if that remains the audit's limitation.

## Laptop admission, receipts and bounded recovery

One low-priority guarded process tree at a time under `Local\Seed-E126-compute-supervisor`, only after E135's explicit slot release. Pin registration, scientific source, actual owner attribution, runtime and resource helper hashes. Keep read-only final keys sealed until complete development audit, cost admission and all six endpoint pins. Never use `--allow-dirty` for scientific runs. Preserve every receipt and every failed/partial run; no resume or restart after scientific measurements without a prospective source-compatible operational recovery that leaves assignments, science and gates fixed.

Hard slots in seconds are 130 for readiness, 105 for each of two development training processes, 100 for development evaluation, 100 for development audit, 240 for each of six final training processes, 300 for each of three final evaluation processes, 300 for final audit, 60 for reporting and 360 for operations. Each guarded stage retains 30 seconds for cleanup. The total ceiling is 3,600 seconds, subject to the actual day balance. The day had 3,774.273 seconds at handoff. Set aside 4.91 seconds for the predecessor chart separately from the measured ledger. Re-read the canonical ledger and actual capacity before every stage. Do not reset an existing date or infer an unused new day.

Admission uses only the complete registered development schedule's timings and archive sizes, never final outcomes. For each training arm forecast `guarded_nonattempt + cold_attempt1 + 9*1.5*warm_attempt2`; price both cold processes separately. For each three-genotype final panel forecast `guarded_nonfamily + sum(family0,1) +126*1.5*max(family2,3)`. Confirm source covers both chamber/gap lengths and both view and ordinary-grid kernels before calling the last two rows warm. The independent auditor additionally compiles total exposure action tapes64,96,80 in its first three families; price those cold once and scale the warm fourth80-tick family by96/80 and1.5 for the remaining385. Audit all64 ES attempts,388 families and12 run-level overheads, including development. Preserve cold/warm attempt timing by view and account for distinct shape costs actually reported. A1/A2 do not inherit A0 process caches.

Require both the measured complete forecast with all cleanup/report/operating reserves and the sum of remaining hard stage caps to fit study and actual day balances. If measured caps can fit only by prospectively reducing future administrative slots, freeze that bounded allocation before final release and retain every scientific count; do not silently borrow from closeout or exceed a registered stage. If the whole schedule cannot fit, stop with keys sealed and preserved receipts. Reserve both archive and local Git copies,1.5 expansion, current development evidence and largest atomic temporary artifact; verify D remains at least10GiB afterward and check C separately. Startup available commit/C/D floors10/10/10GiB; live floors3/5/10GiB. Record physical RAM; retain the established physical-startup waiver only. Run below normal priority with hidden windows and terminate the owned tree on a floor or wall stop.

Every attempt saves an immutable stdout/stderr, telemetry JSONL, source hash inventory and terminal receipt with source/HEAD, mode, attempt, processStarted, childExitCode, status, wall time, calendar segments, canonical-ledger hash, prior cumulative stage charge, remaining reserve and retry eligibility. Only a charged no-process operational stop can retry automatically, within the original cumulative cap and unchanged source. A committed, prospectively recorded readiness/checker-only amendment can explicitly cover a no-process source mismatch; verify both receipts and diff allowlist at both guard layers. Any started scientific failure requires diagnosis and a new bounded prospective authorization record, not an automatic retry. No claim may precede the independent joint reconstruction.

After valid audit, run `scripts/summarize.py`, preserve all previous diagnostic bytes and LOG entries, append result and operational receipts, update local README and <60-line TASK_STATE, append the ignored milestone note, commit exact local results and tag the decision. No external publication. Report audited lineage effects/SEs, both separate scientific statuses, actual inclusive cost/day remainder and explicit slot release.
