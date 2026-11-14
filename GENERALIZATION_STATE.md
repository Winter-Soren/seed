# Generalization lab state

Work only in this isolated checkout; origin push URL is DISABLED. Base: a0cb4a7.

Objective: implement a separate saved-genome broader-condition harness, directly test it,
and prepare E108 without changing EVAL v1 or claiming smoke scores as science.

Plan / acceptance:
1. Inspect runner/rollout/tests and run existing test baseline under a short guarded process.
2. Add explicit `broader-v1` dispatch, fixed one-variable conditions, input-file provenance,
   matched learning/frozen evaluations, and separate development/final randomness.
3. Test dispatch, validation, metrics, hashes and EVAL v1 compatibility; benchmark a
   development one-world saved-genome panel from committed code through the runner.
4. Preserve compact run evidence, regenerate RESULTS, and finish a concrete E108 final
   preregistration whose launch remains pending owner protocol approval and compute slot.

Scope: runner, new condition module, focused tests, summarizer, E108 protocol/log;
README gets only a pointer/status, never a scientific result from development fixtures.
No engine/notebook changes, training, reversal-world implementation, final-world access,
remote push, or edits outside this lab. C2/E106 and structural work belong to other chats.

Current: harness/proposal implemented. Eight-test baseline passed; final implementation
check passed 18 tests in 2.872 s (5.568 s guard wall), sampled max private/resident memory
791,289,856/216,047,616 bytes. Further JAX launches are on the coordinator's temporary
shared-resource hold; this running test finished before honoring the hold.
No graph index exists for Seed, so focused source reads used. The upstream chat separately
owns EVAL v2; reconcile shared runner/summarizer hunks during integration, preserving both.
Authoritative memory checks use approved read-only CIM; initial unprivileged CIM failed.
Implementation committed as fcc1f6b. Human approval “Approve BROADER v1” verified in
oversight; hold lifted for bounded fixtures and full A0 only. Startup C/D10 GiB,
physical8/availablecommit10 GiB; runtime C5/D10/commit5 GiB, BelowNormal, max1200 s.
Development smoke `runs/20261114-231414_E108-D-A0-oneworld` passed from clean21676c5;
63.8466413 s supervisor wall, sampled private/resident max975536128/399548416 bytes,
input/copies byte-identical, all five conditions finite. Evidence in
experiments/E108_D_A0_SUPERVISION.json; unit checks18 passed before smoke.
Full A0 passed: runs/20261114-231716_E108-final-A0, clean source a45f51a,
172.5020589 s guard wall, 163.594 s evaluation; sampled private/resident max
1306451968/729059328 bytes; min sampled available commit17480540160 bytes. Inputs/copies
byte-identical, all metrics finite. Advantages control14.44921875, sparse5.0546875,
dense17.93359375, low-cost14.560546875, high-cost12.703125. One genome only.
Next: commit A0 artifacts/evidence/records; coordinator reviews actual cost before
remaining six full panels. No process from this lab remains active. No protocol/tuning
changes; EVAL v1 and all original historical artifacts remain unchanged. Push disabled.
