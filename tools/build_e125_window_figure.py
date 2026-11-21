"""Plot E125's primary pupil from a passed independent final audit only.

After release: python tools/build_e125_window_figure.py --audit
results/diagnostics/E125-independent-audit.json
Use --self-check to validate synthetic inputs without opening experiment data.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "HANDOFF.md").is_file())
PROTOCOL = "reversal-random-v1"
WINDOWS = ("pre", "early", "late")
CONDITIONS = ("stationary", "reversal")
FAMILIES = 128


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probability(value, label):
    mean, se = value["mean"], value["se"]
    require(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1
                for x in (mean, se)), label + ": expected a probability and nonnegative SE")
    return dict(mean=mean, se=se)


def decode_audit(report):
    require(report.get("status") == "passed" and report.get("mode") == "final"
            and report.get("final_evaluation_audited") is True
            and report.get("protocol") == PROTOCOL,
            "E125 requires a passed independent final artifact audit")
    runs = report.get("runs", [])
    require(len(runs) == 3, "E125 requires all three canonical final lineages")
    rows = {}
    for run in runs:
        identity = str(run.get("run", "")).replace("\\", "/")
        match = re.fullmatch(r"runs/\d{8}-\d{6}_E125-(A[012])", identity)
        require(match is not None, "unexpected E125 canonical run identity")
        lineage = match.group(1)
        require(lineage not in rows, "duplicate E125 lineage")
        measurements = run["measurements"]
        require(measurements["families"] == FAMILIES and measurements["primary"] == "observed",
                lineage + ": expected 128 assigned families and the primary observed pupil")
        conditions = measurements["arms"]["observed"]["conditions"]
        values = {}
        for condition in CONDITIONS:
            panel = conditions[condition]
            require(panel["families"] == FAMILIES, lineage + ": condition drops assigned families")
            values[condition] = []
            for window in WINDOWS:
                cell = panel["windows"][window]
                require(cell["assigned_families"] == FAMILIES,
                        lineage + ": window drops assigned families")
                outcomes = {key: probability(cell[key], lineage + "/" + condition + "/" + window)
                            for key in ("first_food", "first_poison", "no_meal")}
                require(math.isclose(sum(v["mean"] for v in outcomes.values()), 1., abs_tol=1e-12),
                        lineage + ": first outcomes do not cover every assigned family")
                values[condition].append(dict(window=window, **outcomes["first_food"]))
        require(values["stationary"][0] == values["reversal"][0],
                lineage + ": paired pre-change estimates disagree")
        rows[lineage] = dict(lineage=lineage, run=identity, families=FAMILIES, conditions=values)
    require(set(rows) == {"A0", "A1", "A2"}, "missing E125 lineage")
    return dict(protocol=PROTOCOL, primary="observed", rows=[rows[f"A{i}"] for i in range(3)])


def load_audit(path):
    raw = path.read_bytes()
    data = decode_audit(json.loads(raw.decode("utf-8-sig")))
    data["audit_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def self_check():
    """Exercise publication guards using synthetic dictionaries, with no file I/O."""
    runs = []
    for i in range(3):
        conditions = {}
        for condition in CONDITIONS:
            windows = {}
            for j, window in enumerate(WINDOWS):
                p = .75 if condition == "stationary" or j == 0 else (.25 if j == 1 else .5)
                windows[window] = dict(assigned_families=FAMILIES,
                    first_food=dict(mean=p, se=.03), first_poison=dict(mean=.875-p, se=.03),
                    no_meal=dict(mean=.125, se=.02))
            conditions[condition] = dict(families=FAMILIES, windows=windows)
        runs.append(dict(run=f"runs/20990101-000000_E125-A{i}", measurements=dict(
            families=FAMILIES, primary="observed", arms=dict(observed=dict(conditions=conditions)))))
    report = dict(status="passed", mode="final", final_evaluation_audited=True,
                  protocol=PROTOCOL, scientific_gate_passed=False, runs=runs)
    data = decode_audit(report)
    assert data["rows"][2]["conditions"]["reversal"][1] == dict(window="early", mean=.25, se=.03)
    broken = [dict(report, status="incomplete"), dict(report, mode="development"),
              dict(report, final_evaluation_audited=False), dict(report, runs=runs[:2]),
              dict(report, runs=[runs[0], runs[0], runs[2]])]
    for key, value in (("assigned_families", 127), ("first_food", dict(mean=float("nan"), se=.03)),
                       ("first_food", dict(mean=.9, se=.03))):
        changed = copy.deepcopy(report)
        changed["runs"][0]["measurements"]["arms"]["observed"]["conditions"]["reversal"]["windows"]["late"][key] = value
        broken.append(changed)
    for bad in broken:
        try:
            decode_audit(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid or incomplete publication input was accepted")


def plot(data, root, output_dir):
    cache = root / ".tmp" / "mplconfig"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from tools.build_readme_figures import INK, MUTED, TEAL, ORANGE, style_axes

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": INK, "axes.titlecolor": INK,
                         "svg.hashsalt": "seed-e125-window-outcomes"})
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 6.1), sharey=True)
    fig.subplots_adjust(left=.11, right=.985, top=.73, bottom=.31, wspace=.16)
    title = "When the food rule changes"
    fig.suptitle(title, x=.075, y=.97, ha="left", fontsize=20, fontweight="bold")
    fig.text(.075, .895, "E125 · Original pupils with calibrated social memory", fontsize=12, color=MUTED)
    colors, markers = (TEAL, ORANGE), ("o", "s")
    labels = ("Stationary rule", "Hidden reversal")
    handles = [Line2D([], [], color=c, marker=m, linestyle="none", markersize=7, label=label)
               for c, m, label in zip(colors, markers, labels)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .855), ncol=2, frameon=False)

    for ax, row in zip(axes, data["rows"]):
        style_axes(ax)
        ax.set_title(f"{row['lineage']} · 128 families", loc="left", pad=16, fontsize=12, fontweight="bold")
        ax.axvline(.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(.5, 1.01, "Rule flips", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color=MUTED)
        for offset, condition, color, marker in zip((-.07, .07), CONDITIONS, colors, markers):
            values = row["conditions"][condition]
            ax.errorbar([i + offset for i in range(3)], [v["mean"] for v in values],
                        yerr=[v["se"] for v in values], fmt=marker, linestyle="none", color=color,
                        markersize=7, capsize=4, elinewidth=1.5, capthick=1.3)
        ax.set(xlim=(-.4, 2.4), ylim=(-.025, 1.045), yticks=(0, .25, .5, .75, 1),
               xticks=(0, 1, 2), xticklabels=("Before\n−50 to −1", "Early\n0 to 49", "Late\n50 to 99"))
        ax.tick_params(axis="x", labelsize=10)
    axes[0].set_ylabel("First-food probability\n(all assigned pupils)", labelpad=12)
    fig.text(.53, .20, "50-turn windows relative to each family's scheduled change", ha="center", color=MUTED)
    caption = (
        "Points summarize whole windows. Whiskers show ±1 standard error across 128 independent families per lineage.",
        "First food = 1. Poison, no item contact or death before contact = 0. Every assigned pupil counts.",
        "The change is hidden from the pupil. Its paired stationary world keeps the original food rule.",
    )
    for y, line in zip((.12, .075, .03), caption):
        fig.text(.075, y, line, fontsize=9.5, color=MUTED)
    description = " ".join(caption) + " Independent audit SHA256: " + data["audit_sha256"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("svg", "png"):
        path = output_dir / f"e125-changing-world.{extension}"
        metadata = dict(Title=title, Description=description)
        metadata.update(dict(Date=None, Creator="Seed / build_e125_window_figure.py")
                        if extension == "svg" else dict(Software="Seed / build_e125_window_figure.py"))
        fig.savefig(path, dpi=180, facecolor="white", metadata=metadata)
        paths.append(str(path))
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--audit", type=Path, help="released independent final E125 audit JSON")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        require(args.audit is None, "self-check accepts no experiment audit path")
        self_check()
        print("Synthetic publication checks passed. No experiment files opened or figures rendered.")
        return
    if args.audit is None:
        parser.error("--audit is required after release of the independent final audit")
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "results" / "figures").resolve()
    require(not output_dir.is_relative_to(root / "runs"), "figure output must stay outside immutable runs")
    data = load_audit(args.audit.resolve())
    print(json.dumps(dict(outputs=plot(data, root, output_dir), audit_sha256=data["audit_sha256"])))


if __name__ == "__main__":
    main()
