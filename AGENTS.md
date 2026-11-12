# AGENTS.md: standing rules for any coding agent working on Seed

Start by reading `HANDOFF.md` (context and next steps) and `experiments/LOG.md` (history).

## The research loop (every experiment, every time)

1. **Pre-register.** Before running, add a LOG entry with an ID (E101, E102…), the hypothesis, the exact command, and what result would count as pass and as fail.
2. **Commit code first.** `scripts/run_experiment.py` refuses to run from uncommitted code. Every run records its git commit, so any result can be traced to the exact code. Use `--allow-dirty` only for throwaway debugging, and say so in the LOG.
3. **Run** only through `scripts/run_experiment.py`, with 3 seeds for any claim.
4. **Record.** Run `python scripts/summarize.py`, then complete the LOG entry: the result (numbers copied from `summary.json`), the interpretation, and the decision taken next. After verified scientific results or gate decisions, update `README.md` with the results, interpretation, limitations, and next experiments. Operational fixes alone are not breakthroughs.
5. **Commit and push** the run folders, `results/RESULTS.md`, `README.md` and the LOG together. Commit message format: `E1xx: <one-line result>`.

## Integrity rules

- **Never delete, overwrite or edit** anything in `runs/` or past LOG entries. Failed, crashed and interrupted runs stay, with their status. To correct an earlier entry, append a dated correction below it.
- Change **one thing at a time** between a run and its comparison.
- Never tune on the evaluation worlds (EVAL v1: `test_genome`, seed 123, 128 worlds). Don't change the protocol without versioning it (see HANDOFF §2).
- Report what happened, not what was hoped. If a gate fails, say so plainly.
- Tag milestones: `git tag v0.1-evalv1` and so on, after each gate decision.
- **Use verified commit identity.** Use the owner's verified GitHub account and its privacy-preserving noreply identity from repo-local Git config. Never invent generic author overrides. After identity changes, push a scoped commit and verify GitHub's author and committer attribution match that account.

## Compute and safety limits (it's the owner's laptop)

- Benchmark first. Keep any single run under **3 hours** of wall time; split longer ones with `--resume`.
- Run long jobs at low priority (`nice -n 10`), one at a time, and checkpoint every 10 generations (the default).
- **Ask the owner before you:** make the repo public or add collaborators; rewrite git history or force-push; delete any branch or tag; spend more than about 8 hours of compute in a day; install system-level software or drivers; download datasets over 1 GB; or change the project's goals or the evaluation protocol.
- Never commit secrets, tokens, or personal data (including the owner's email address).

## Repo layout

`seed/` engine · `scripts/` runner and summary · `experiments/LOG.md` lab notebook · `results/RESULTS.md` generated table · `runs/` one folder per run · `notebooks/` the Colab version (rebuild with `tools/build_notebook.py`).
