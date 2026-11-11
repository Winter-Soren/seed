# First-session state

Updated 2026-11-12 (Asia/Calcutta). This file is a mutable continuation note; `experiments/LOG.md` is the append-only research record.

- Objective: establish the local environment and private repository, smoke-test and benchmark v0, measure EVAL v1 references, then choose and run the next preregistered core experiment within the measured budget.
- Read: `HANDOFF.md`, `AGENTS.md`, and pilot history E000–E008. Keep EVAL v1 fixed at 128 worlds and evaluation seed 123.
- Environment: Python 3.12.10 on Windows; project-local `.venv` created on D:. Dependencies, hardware details, and JAX device verification pending.
- Storage at initial check: C: 10.49/373.67 GiB free/total; D: 46.89/100.00 GiB. Put environment/download temporary files on D:, disable pip cache, and recheck before larger runs. Preserve all run artifacts.
- Repository: authentication verified outside sandbox. Creation/push was blocked by automatic approval review; explicit user authorization of destination and payload is pending. No repository creation or push is recorded as successful.
- Preregistered: E101 smoke; E102 10-generation full-settings benchmark; E100 zero-generation random and biology references for seeds 0/1/2. Exact commands and pass/fail criteria are in the LOG. No experiment result recorded yet.
- Compute constraints: one low-priority long job at a time, checkpoint every 10 generations, under 3 hours per run and about 8 hours per day without owner approval. Parallelize independent support work, not simultaneous long compute jobs.
- Research decisions: preserve smoke runs despite the conflicting handoff deletion step; treat deterministic biology reference repeats as reproducibility checks rather than independent evidence.
- Next: finish dependency/device checks, review and commit preregistration, run E101 through the tracked runner, then use E102 timings to plan the remaining work. Append observed results and the resulting decision after each experiment; do not replace prior LOG text.
