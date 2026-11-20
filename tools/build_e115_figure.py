"""Plot one completed, audited E115 comparison; never imports or runs the engine.

The input must pass the independent audit. Outputs go to results/figures/.
Example: .venv/Scripts/python.exe tools/build_e115_figure.py \
    --audit results/diagnostics/E115-independent-audit.json
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp" / "mplconfig"))
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.build_readme_figures import INK, MUTED, TEAL, ORANGE, style_axes

ARMS = ("start", "mean", "min")
PANELS = tuple(f"{arm}{i}" for i in range(3) for arm in ARMS) + (
    "fixed_A", "fixed_B", "mixture", "random_walk", "greedy")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_audited_data(audit_path):
    audit_path = audit_path.resolve()
    require(audit_path.is_relative_to(ROOT), "audit must be inside the repository")
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    require(audit.get("status") == "passed" and audit.get("final_evaluation_audited") is True,
            "a passed, complete independent audit is required")
    run = (ROOT / audit["evaluation"]["run"]).resolve()
    require(run.is_relative_to(ROOT / "runs"), "audited evaluation must be inside runs/")
    summary_path = run / "summary.json"
    relative = summary_path.relative_to(ROOT).as_posix()
    expected = audit.get("checked_file_sha256", {}).get(relative)
    require(expected is not None, "audit must fingerprint the evaluation summary")
    raw = summary_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "evaluation summary changed after audit")
    summary = json.loads(raw)
    require(summary.get("status") == "finished" and summary.get("artifact_validity") == "passed"
            and summary.get("eval_protocol") == "v5" and summary.get("condition_split") == "final"
            and summary.get("eval_worlds") == 256 and summary.get("completed_panels") == 14
            and tuple(summary.get("panels", {})) == PANELS and len(summary.get("contrasts", [])) == 30
            and summary.get("gate") == audit["evaluation"]["gate"],
            "complete, matching fourteen-panel EVAL v5 summary required")
    recovery = audit.get("recovery")
    if run.name.endswith("_E115-v5-recovery1"):
        require(recovery is not None and recovery.get("recovery_run") == run.relative_to(ROOT).as_posix()
                and recovery.get("original_files_unchanged") == 54
                and recovery.get("overlapping_condition_archives") == 35
                and recovery.get("overlapping_paired_archives") == 8
                and recovery.get("scientific_summaries_identical") is True,
                "complete interruption-preservation and exact recovery evidence required")
    rows = []
    for seed in range(3):
        for arm in ARMS:
            panel = summary["panels"][f"{arm}{seed}"]
            metric = panel["paired"]["individual_min_return"]
            require(panel.get("status") == "finished"
                    and all(type(metric[key]) in (int, float) and math.isfinite(metric[key]) for key in ("mean", "se"))
                    and metric["se"] >= 0, "invalid or incomplete individual-minimum estimate")
            rows.append(dict(seed=seed, training_seed=1000+seed, arm=arm,
                             mean=metric["mean"], se=metric["se"]))
    return dict(metric="individual_min_return", uncertainty="one base-world-family standard error",
                n_base_worlds=256, agents_per_world=4, gate=summary["gate"], values=rows,
                summary=relative, summary_sha256=expected,
                audit=audit_path.relative_to(ROOT).as_posix(),
                audit_sha256=hashlib.sha256(audit_bytes).hexdigest())


def plot(data, stem):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.titlecolor": INK, "axes.labelsize": 11,
        "svg.fonttype": "path", "svg.hashsalt": "seed-e115-objective-comparison-v1",
        "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 7.6), sharey=True)
    fig.subplots_adjust(left=.09, right=.97, top=.72, bottom=.36, wspace=.18)
    decision = "PASS" if data["gate"]["passed"] else "FAIL"
    fig.text(.07, .95, f"E115  /  OBJECTIVE COMPARISON  /  REGISTERED GATE: {decision}",
             color=TEAL, size=10.5, weight="bold")
    fig.text(.07, .89, "Worse-rule return before and after selection", size=23, weight="bold")
    fig.text(.07, .84, "3 saved A genomes  ·  50-generation continuations  ·  256 matched base worlds",
             color=MUTED, size=11.5)
    values = data["values"]
    low = min(0, min(row["mean"]-row["se"] for row in values))
    high = max(0, max(row["mean"]+row["se"] for row in values))
    pad = max((high-low)*.10, .5)
    for seed, ax in enumerate(axes):
        rows = [row for row in values if row["seed"] == seed]
        for x, row, color, marker in zip(range(3), rows, (MUTED, ORANGE, TEAL), ("o", "s", "D")):
            ax.errorbar(x, row["mean"], yerr=row["se"], fmt=marker, color=color,
                        markersize=7, capsize=5, elinewidth=1.5, capthick=1.5)
            ax.text(x, row["mean"]+row["se"]+pad*.4,
                    f"{row['mean']:.2f} ± {row['se']:.2f}", ha="center", color=color, size=10)
        ax.axhline(0, color=MUTED, linewidth=.7, alpha=.6)
        ax.set(xlim=(-.45, 2.45), ylim=(low-pad*.6, high+pad*1.5),
               xticks=[0, 1, 2], xticklabels=["Before", "After:\nmean\nselection", "After:\nminimum\nselection"])
        ax.set_title(f"A{seed}  ·  training seed {1000+seed}", loc="left", pad=18, size=13, weight="bold")
        style_axes(ax)
    axes[0].set_ylabel("Worse-rule return per starting agent\n(food − poison)", labelpad=12)
    caption = (
        "Minimum is taken across the two rule returns for each agent, then averaged within its four-agent family.",
        "Points show means ± 1 SE across 256 base worlds. All starting agents, including deaths and zero encounters, remain included.",
        "The gate uses paired within-world contrasts: minimum selection must beat both controls by > 2 paired SE in all 3 seeds.",
        "It also requires each minimum-minus-start joint-survival change to be ≥ −2 paired SE; that safeguard is not plotted.",
        "These marginal error bars are not paired-difference bars. Stationary performance alone does not establish causal learning or reversal."
    )
    for y, line in zip((.235, .198, .161, .124, .087), caption):
        fig.text(.07, y, line, color=MUTED, size=9.5)
    description = " ".join(caption)
    for extension in ("png", "svg"):
        metadata = dict(Title="E115 objective comparison", Description=description)
        metadata.update(dict(Date=None, Creator="Seed / tools/build_e115_figure.py")
                        if extension == "svg" else dict(Software="Seed / tools/build_e115_figure.py"))
        fig.savefig(stem.with_suffix("."+extension), dpi=170, facecolor="white", metadata=metadata)
    plt.close(fig)
    data["caption"] = description
    stem.with_suffix(".json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    stem = ROOT / 'results' / 'figures' / 'e115-objective-comparison'
    data = load_audited_data(args.audit)
    plot(data, stem)
    print(json.dumps(dict(output_stem=str(stem), plotted_estimates=len(data["values"]),
                          audit_sha256=data["audit_sha256"], summary_sha256=data["summary_sha256"]), indent=2))


if __name__ == "__main__":
    main()
