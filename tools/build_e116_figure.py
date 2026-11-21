"""Draw E116's registered paired contrasts; no engine or training-curve imports.

Real data require a passed final independent audit and its unchanged v6 summary.
--synthetic is a plainly marked layout check and reads no experiment artifacts.
All outputs and Matplotlib cache files stay under this checkout's .tmp on D:.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / ".tmp"
os.environ["MPLCONFIGDIR"] = str(SCRATCH / "mplconfig")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.build_readme_figures import INK, MUTED, TEAL, ORANGE, GRID, style_axes

SOURCE = "861405d7651421abd216fec28683be54b0c4edf5"
COMPARISONS = ("candidate-minus-control", "candidate-minus-start")
PANELS = tuple(f"{arm}{i}" for i in range(3) for arm in ("start", "control", "candidate")) + (
    "fixed_A", "fixed_B", "mixture", "random_walk", "greedy")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_data(path):
    path = path.resolve()
    require(path.is_relative_to(ROOT), "audit must be inside the repository")
    audit_bytes = path.read_bytes()
    audit = json.loads(audit_bytes.decode("utf-8-sig"))
    require(audit.get("status") == "passed" and audit.get("final_evaluation_audited") is True
            and audit.get("mode") == "full" and audit.get("protocol") == "v6"
            and audit.get("registered_source") == SOURCE, "require a passed complete E116 audit against its registered source")
    require(len(audit.get("training", [])) == 6 and all(row.get("generation") == 20 and row.get("history_rows") == 20
            for row in audit["training"]) and len(audit.get("matched_training", [])) == 3, "six matched final generation-20 states required")
    result = audit["evaluation"]
    run = (ROOT / result["run"]).resolve()
    require(run.is_relative_to(ROOT / "runs") and run.name.endswith("_E116-v6"), "registered E116-v6 run required")
    relative = (run / "summary.json").relative_to(ROOT).as_posix()
    expected_sha = audit.get("checked_file_sha256", {}).get(relative)
    raw = (run / "summary.json").read_bytes()
    require(expected_sha is not None and hashlib.sha256(raw).hexdigest() == expected_sha, "evaluation summary changed after the audit")
    summary = json.loads(raw)
    require(summary.get("status") == "finished" and summary.get("artifact_validity") == "passed"
            and summary.get("eval_protocol") == "v6" and summary.get("condition_split") == "final"
            and summary.get("eval_seed") == 2026112001 and summary.get("eval_worlds") == 256
            and summary.get("completed_panels") == 14 and tuple(summary.get("panels", {})) == PANELS
            and len(summary.get("contrasts", [])) == 30 and summary["gate"]["passed"] == result["gate"]["passed"],
            "complete matching 14-panel EVAL v6 summary required")
    contrasts = {(row["seed"], row["comparison"], row["metric"]): row for row in summary["contrasts"]}
    require(len(contrasts) == 30, "duplicate registered contrast")
    values, survival = [], []
    for seed in range(3):
        for comparison, metric in [(item, "individual_min_return") for item in COMPARISONS] + [
                ("candidate-minus-start", "joint_survival")]:
            row = contrasts[(seed, comparison, metric)]
            delta = row["per_world_difference"]
            require(row["n_worlds"] == len(delta) == 256 and all(type(x) in (int, float) and math.isfinite(x) for x in delta),
                    "contrast must contain 256 finite paired differences")
            mean, se = statistics.fmean(delta), statistics.stdev(delta)/16
            require(math.isclose(row["mean"], mean, rel_tol=1e-12, abs_tol=1e-12)
                    and math.isclose(row["se"], se, rel_tol=1e-12, abs_tol=1e-12), "contrast estimate differs from its world vector")
            key = comparison + ("_minimum_return" if metric == "individual_min_return" else "_joint_survival")
            component = summary["gate"]["per_seed"][seed][key]
            passed = row["mean"] > 2*row["se"] if metric == "individual_min_return" else row["mean"] >= -2*row["se"]
            require(component["passed"] == passed, "gate disagrees with plotted contrast")
            record = dict(seed=seed, training_seed=2000+seed, comparison=comparison, metric=metric,
                          mean=row["mean"], se=row["se"], two_se=2*row["se"], passed=passed)
            (values if metric == "individual_min_return" else survival).append(record)
    require(summary["gate"]["passed"] == all(row["passed"] for row in values + survival), "overall plotted gate mismatch")
    return dict(synthetic=False, protocol="v6", n_world_families=256, candidate_lr=.02, control_lr=.1,
        generations=20, uncertainty="two paired-world-family standard errors", values=values, survival=survival,
        gate_passed=summary["gate"]["passed"], audit=path.relative_to(ROOT).as_posix(),
        audit_sha256=hashlib.sha256(audit_bytes).hexdigest(), summary=relative, summary_sha256=expected_sha)


def synthetic_data():
    values = []
    examples = (((2., .35), (.3, .25)), ((.05, .2), (-.6, .28)), ((-.4, .15), (.8, .3)))
    for seed, rows in enumerate(examples):
        for comparison, (mean, se) in zip(COMPARISONS, rows):
            values.append(dict(seed=seed, training_seed=2000+seed, comparison=comparison, metric="individual_min_return",
                               mean=mean, se=se, two_se=2*se, passed=mean > 2*se))
    survival = [dict(seed=seed, comparison="candidate-minus-start", metric="joint_survival",
        mean=mean, se=se, two_se=2*se, passed=mean >= -2*se)
        for seed, (mean, se) in enumerate(((.01, .008), (-.03, .01), (0., .006)))]
    return dict(synthetic=True, protocol="v6-layout-only", n_world_families=256,
        uncertainty="two paired-world-family standard errors; illustrative values only",
        values=values, survival=survival, gate_passed=None)


def plot(data, stem):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
        "axes.titlecolor": INK, "axes.labelsize": 11, "figure.facecolor": "white", "axes.facecolor": "white",
        "svg.fonttype": "path", "svg.hashsalt": "seed-e116-paired-gate-v1"})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 8.), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.96, top=.72, bottom=.42, wspace=.16)
    synthetic = data["synthetic"]
    decision = "SYNTHETIC RENDER CHECK — NOT RESULTS" if synthetic else (
        "REGISTERED GATE: " + ("PASS" if data["gate_passed"] else "FAIL"))
    fig.text(.07, .95, "E116  /  LEARNING RATE COMPARISON  /  " + decision,
             color=ORANGE if synthetic else TEAL, size=10.5, weight="bold")
    fig.text(.07, .89, "Smaller-step continuation: the registered contrasts", size=22, weight="bold")
    fig.text(.07, .84, "Candidate lr 0.02  ·  control lr 0.1  ·  20 generations  ·  256 matched world families",
             color=MUTED, size=11.2)
    values = data["values"]
    low = min(0., min(row["mean"]-row["two_se"] for row in values))
    high = max(0., max(row["mean"]+row["two_se"] for row in values))
    pad = max((high-low)*.18, .2)
    titles = ("Candidate − matched control", "Candidate − unchanged original")
    for ax, comparison, title, color, marker in zip(axes, COMPARISONS, titles, (ORANGE, TEAL), ("s", "D")):
        for row in (row for row in values if row["comparison"] == comparison):
            y = 2-row["seed"]
            ax.errorbar(row["mean"], y, xerr=row["two_se"], fmt=marker, color=color,
                        markersize=7, capsize=5, elinewidth=1.6, capthick=1.5)
            ax.text(row["mean"], y+.18, f"{row['mean']:+.3f} ± {row['two_se']:.3f}",
                    color=color, ha="center", va="bottom", size=10)
        style_axes(ax)
        ax.grid(visible=False, axis="y")
        ax.grid(axis="x", color=GRID, linewidth=.8)
        ax.axvline(0, color=INK, linewidth=1.1)
        ax.set(xlim=(low-pad, high+pad), ylim=(-.48, 2.55), yticks=[2, 1, 0],
               yticklabels=["Seed 0", "Seed 1", "Seed 2"])
        ax.set_title(title, loc="left", pad=16, size=13, weight="bold")
        ax.set_xlabel("Difference in individual worse-rule return\n(food − poison per starting agent)", labelpad=10)
    axes[1].tick_params(labelleft=True)
    fig.text(.07, .265, "Joint-survival safeguard: candidate − original, percentage points ± 2 paired SE",
             size=10.5, weight="bold")
    for x, row in zip((.07, .38, .69), data["survival"]):
        verdict = "PASS" if row["passed"] else "FAIL"
        fig.text(x, .222, f"Seed {row['seed']}: {100*row['mean']:+.2f} ± {100*row['two_se']:.2f} pp  ·  {verdict}",
                 size=10.5, color=MUTED)
    caption = [
        "Each point is a within-world paired difference; bars and numeric labels show ±2 standard errors across 256 world families.",
        "Return gate: all six intervals must lie strictly right of zero. Survival safeguard: each upper end must reach zero or above.",
        "All starting agents count. SE describes fixed genomes across worlds, not variability across the three evolution seeds.",
        "SYNTHETIC DATA FOR LAYOUT ONLY. No experiment outputs were opened." if synthetic else
        "This operational gate is not a corrected significance test and does not establish causal learning or reversal competence."]
    for y, line in zip((.16, .122, .084, .046), caption):
        fig.text(.07, y, line, color=ORANGE if synthetic and y == .046 else MUTED, size=9.1)
    description = " ".join(caption)
    title = "Synthetic E116 figure layout — not results" if synthetic else "E116 paired gate evidence"
    for suffix in ("svg", "png"):
        metadata = dict(Title=title, Description=description)
        metadata.update(dict(Date=None, Creator="Seed / build_e116_figure.py") if suffix == "svg"
                        else dict(Software="Seed / build_e116_figure.py"))
        path = stem.with_suffix("." + suffix)
        fig.savefig(path, dpi=170, facecolor="white", metadata=metadata)
        if suffix == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    stem.with_suffix(".json").write_text(json.dumps(dict(data, caption=description), indent=2, allow_nan=False)+"\n",
                                       encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audit", type=Path)
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output-stem", type=Path)
    args = parser.parse_args()
    stem = (args.output_stem or SCRATCH / "E116-figure-review" / (
        "e116-synthetic-render" if args.synthetic else "e116-paired-gate")).resolve()
    require(stem.is_relative_to(SCRATCH.resolve()), "review outputs must remain under .tmp")
    stem.parent.mkdir(parents=True, exist_ok=True)
    data = synthetic_data() if args.synthetic else load_data(args.audit)
    plot(data, stem)
    print(json.dumps(dict(output_stem=str(stem), synthetic=data["synthetic"],
        plotted_return_contrasts=len(data["values"]), survival_components=len(data["survival"]))))


if __name__ == "__main__":
    main()
