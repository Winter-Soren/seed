"""Compare one registered movie contrast across completed E120, E122 and E123 audits.

Inputs are the independent audit JSON files only. No simulator or raw run is read.
Use --synthetic for a visibly labeled layout check that opens no experiment data.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


ROOT = next(path for path in Path(__file__).resolve().parents if (path / "HANDOFF.md").is_file())
STUDIES = (
    ("E120", "Original observers", "social-learning-v1"),
    ("E122", "Observers after selection", "social-observer-train-v1"),
    ("E123", "Engineered calibrated readout", "social-belief-readout-v1"),
)
METRICS = (("first_food_difference", "First solo meal is food"),
           ("survival_difference", "Alive at the end of life"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def estimate(value, label):
    require(isinstance(value, dict), label + " must contain a mean and SE")
    mean, se = value.get("mean"), value.get("se")
    require(type(mean) in (int, float) and type(se) in (int, float)
            and math.isfinite(mean) and math.isfinite(se) and -1 <= mean <= 1 and 0 <= se <= 1,
            label + " must contain finite probability differences and nonnegative SE")
    return dict(mean=100 * mean, se=100 * se, two_se=200 * se)


def decode_audit(report, study):
    identifier, title, protocol = study
    require(report.get("status") == "passed" and report.get("mode") == "final"
            and report.get("final_evaluation_audited") is True and report.get("protocol") == protocol,
            identifier + " requires a passed independent final artifact audit")
    runs = report.get("runs", [])
    require(len(runs) == 3, identifier + " requires all three completed lineages")
    rows = []
    for index, run in enumerate(runs):
        label = f"{identifier}/A{index}"
        require(str(run.get("run", "")).endswith(f"_{identifier}-A{index}"), label + " canonical run identity")
        measurements = run["measurements"]
        require(measurements["families"] == measurements["assigned_newborns"] == 128,
                label + " must retain all 128 assigned newborns")
        contrasts = measurements["contrasts"]["yoked"]
        values = {}
        for key, arm_key in (("first_food_difference", "first_food"), ("survival_difference", "survival")):
            values[key] = estimate(contrasts[key], label + " " + key)
            means = [measurements["arms"][arm][arm_key]["mean"] for arm in ("observed", "yoked")]
            require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                        for value in means), label + " arm probability bounds")
            require(math.isclose(contrasts[key]["mean"], means[0] - means[1], abs_tol=1e-12, rel_tol=1e-12),
                    label + " paired contrast differs from its arm means")
        rows.append(dict(lineage=f"A{index}", run=run["run"], **values))
    return dict(study=identifier, title=title, protocol=protocol, families_per_lineage=128, rows=rows)


def load_data(root, paths):
    studies = []
    for study, path in zip(STUDIES, paths):
        raw = path.read_bytes()
        data = decode_audit(json.loads(raw.decode("utf-8-sig")), study)
        reference = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
        data.update(audit=reference, audit_sha256=hashlib.sha256(raw).hexdigest())
        studies.append(data)
    return dict(synthetic=False, studies=studies)


def synthetic_report(study):
    rows = []
    for index in range(3):
        first, survival = (-.06, .02, .17)[index], (.04, -.08, .10)[index]
        rows.append(dict(run=f"runs/SYNTHETIC_{study[0]}-A{index}", measurements=dict(
            families=128, assigned_newborns=128,
            arms=dict(observed=dict(first_food=dict(mean=.5 + first), survival=dict(mean=.5 + survival)),
                      yoked=dict(first_food=dict(mean=.5), survival=dict(mean=.5))),
            contrasts=dict(yoked=dict(first_food_difference=dict(mean=first, se=.025),
                                      survival_difference=dict(mean=survival, se=.035))))))
    return dict(status="passed", mode="final", final_evaluation_audited=True,
                protocol=study[2], runs=rows)


def self_check():
    report = synthetic_report(STUDIES[0])
    decoded = decode_audit(report, STUDIES[0])
    assert decoded["rows"][0]["first_food_difference"] == dict(mean=-6., se=2.5, two_se=5.)
    survival = decoded["rows"][1]["survival_difference"]
    assert survival["mean"] == -8. and math.isclose(survival["se"], 3.5) and math.isclose(survival["two_se"], 7.)
    for broken in (dict(report, status="incomplete"), dict(report, mode="development"),
                   dict(report, final_evaluation_audited=False), dict(report, runs=report["runs"][:2])):
        try:
            decode_audit(broken, STUDIES[0])
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete or development evidence was accepted")
    report["runs"][0]["measurements"]["contrasts"]["yoked"]["first_food_difference"]["mean"] = .5
    try:
        decode_audit(report, STUDIES[0])
    except ValueError:
        pass
    else:
        raise AssertionError("an inconsistent paired contrast was accepted")


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
    from matplotlib.ticker import MaxNLocator
    from tools.build_readme_figures import INK, MUTED, GRID, TEAL, BLUE, ORANGE, style_axes

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": INK, "axes.titlecolor": INK,
                         "svg.hashsalt": "seed-social-learning-comparison"})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 8.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.97, top=.79, bottom=.27, wspace=.24)
    title = "Using a demonstration across three studies"
    if data["synthetic"]:
        title = "SYNTHETIC LAYOUT CHECK · social-learning comparison"
    fig.suptitle(title, x=.075, y=.97, ha="left", fontsize=19, fontweight="bold")
    fig.text(.075, .921, "Intact demonstration minus its matched color-yoked movie",
             fontsize=12, color=MUTED)
    colors, markers = (TEAL, BLUE, ORANGE), ("o", "s", "^")
    handles = [Line2D([], [], color=color, marker=marker, linestyle="none", markersize=7, label=f"A{index}")
               for index, (color, marker) in enumerate(zip(colors, markers))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .912), ncol=3,
               frameon=False, title="Original genome lineage")

    estimates = [row[key] for study in data["studies"] for row in study["rows"] for key, _ in METRICS]
    low = min(0., min(value["mean"] - value["two_se"] for value in estimates))
    high = max(0., max(value["mean"] + value["two_se"] for value in estimates))
    pad = max((high - low) * .09, 3.)
    positions = [8. - group * 3.2 - lineage * .75 for group in range(3) for lineage in range(3)]
    for ax, (key, panel_title) in zip(axes, METRICS):
        style_axes(ax)
        ax.grid(visible=False, axis="y")
        ax.grid(axis="x", color=GRID, linewidth=.8)
        ax.axvline(0, color=INK, linewidth=1.1)
        ax.set(xlim=(low - pad, high + pad), ylim=(-.5, 9.1), yticks=positions,
               yticklabels=[f"A{index}" for _ in STUDIES for index in range(3)])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_title(panel_title, loc="left", pad=17, fontsize=13, fontweight="bold")
        ax.set_xlabel("Difference in percentage points", labelpad=11)
        ax.tick_params(labelleft=True)
        for group, study in enumerate(data["studies"]):
            top = 8. - group * 3.2
            ax.text(low - pad, top + .64, f"{study['study']} · {study['title']}", fontsize=10.8,
                    fontweight="bold", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=2))
            for index, row in enumerate(study["rows"]):
                y = top - index * .75
                value = row[key]
                ax.errorbar(value["mean"], y, xerr=value["two_se"], fmt=markers[index],
                            linestyle="none", color=colors[index], markersize=6.5,
                            capsize=4, elinewidth=1.6, capthick=1.4)
                ax.text(value["mean"], y + .18, f"{value['mean']:+.1f} ± {value['two_se']:.1f}",
                        ha="center", va="bottom", fontsize=8.8, color=colors[index],
                        bbox=dict(facecolor="white", edgecolor="none", pad=.6))
    caption = [
        "Points show paired differences. Bars and labels show ±2 standard errors across 128 world families per lineage.",
        "Both panels use the same scale. Positive values favor the intact movie. Every assigned newborn counts.",
        "Studies use independent worlds. E123 adds an engineered calibrated readout to the original observer genomes.",
        "SYNTHETIC VALUES FOR LAYOUT ONLY." if data["synthetic"] else
        "Cross-study differences are descriptive. Full experiment gates include additional controls reported in the README."]
    for y, line in zip((.145, .106, .067, .028), caption):
        fig.text(.075, y, line, fontsize=9.2, color=ORANGE if data["synthetic"] and y == .028 else MUTED)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "social-learning-synthetic" if data["synthetic"] else "social-learning-comparison"
    outputs = []
    for extension in ("svg", "png"):
        path = output_dir / f"{stem}.{extension}"
        metadata = dict(Title=title, Description=" ".join(caption))
        metadata.update(dict(Date=None, Creator="Seed / build_social_learning_figure.py")
                        if extension == "svg" else dict(Software="Seed / build_social_learning_figure.py"))
        fig.savefig(path, dpi=170, facecolor="white", metadata=metadata)
        outputs.append(str(path))
    plt.close(fig)
    manifest = output_dir / f"{stem}.json"
    manifest.write_text(json.dumps(dict(data, uncertainty="two paired family standard errors",
                                        units="percentage points", caption=caption), indent=2, allow_nan=False) + "\n",
                        encoding="utf-8", newline="\n")
    return outputs + [str(manifest)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--e120-audit", type=Path)
    parser.add_argument("--e122-audit", type=Path)
    parser.add_argument("--e123-audit", type=Path)
    check = parser.add_mutually_exclusive_group()
    check.add_argument("--synthetic", action="store_true")
    check.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("Synthetic validation checks passed; no experiment files opened.")
        return
    paths = [args.e120_audit, args.e122_audit, args.e123_audit]
    if args.synthetic and any(paths):
        parser.error("synthetic mode accepts no experiment audit paths")
    root = args.root.resolve()
    if args.synthetic:
        data = dict(synthetic=True, studies=[decode_audit(synthetic_report(study), study) for study in STUDIES])
        default_output = root / ".tmp" / "social-figure-review"
    else:
        paths = [(path or root / "results" / "diagnostics" / f"{study[0]}-independent-audit.json").resolve()
                 for path, study in zip(paths, STUDIES)]
        data = load_data(root, paths)
        default_output = root / "results" / "figures"
    output_dir = (args.output_dir or default_output).resolve()
    require(not output_dir.is_relative_to(root / "runs"), "figure output must be outside immutable runs")
    print(json.dumps(dict(outputs=plot(data, root, output_dir), synthetic=data["synthetic"])))


if __name__ == "__main__":
    main()
