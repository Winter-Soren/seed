"""Rebuild the README's E103 A figures from the three completed run records."""

from pathlib import Path
import hashlib
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
RUNS = (
    "20261112-023245_E103-A-blank-s0",
    "20261112-050319_E103-A-blank-s1",
    "20261112-102755_E103-A-blank-s2",
)
INK, MUTED, GRID = "#172B3A", "#566976", "#E4EAEE"
TEAL, BLUE, ORANGE = "#087F78", "#4668AC", "#BD6329"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_data():
    runs = []
    for seed, folder in enumerate(RUNS):
        path = ROOT / "runs" / folder
        raw = {name: (path / name).read_bytes()
               for name in ("config.json", "summary.json", "metrics.jsonl")}
        config = json.loads(raw["config.json"])
        summary = json.loads(raw["summary.json"])
        rows = [json.loads(line) for line in raw["metrics.jsonl"].splitlines()]
        require(summary["status"] == "finished" and summary["generation"] == 200
                and summary["eval_worlds"] == 128, f"{folder}: incomplete EVAL v1 result")
        expected = dict(seed=seed, start="blank", generations=200, pop=128, worlds=8,
                        eval_worlds=128, sigma=0.1, lr=0.1, no_plastic=False, quick=False)
        require(all(config["args"][key] == value for key, value in expected.items()),
                f"{folder}: unexpected experiment configuration")
        require(config["cfg_overrides"] == {} and not config["git"]["dirty"]
                and all(not resume["git"]["dirty"] for resume in config.get("resumes", [])),
                f"{folder}: unverified source or configuration override")
        require(len(rows) == 200 and [row["gen"] for row in rows] == list(range(200)),
                f"{folder}: expected 200 contiguous generation records")
        require(all(isinstance(value, (int, float)) and math.isfinite(value)
                    for row in rows for value in row.values()),
                f"{folder}: nonnumeric or nonfinite training data")
        for label in ("final_learning", "final_frozen"):
            require(all(math.isfinite(value) for value in summary[label].values()),
                    f"{folder}: nonfinite evaluation data")
            require(all(0 <= summary[label][key] <= 1
                        for key in ("acc_first", "acc_second", "survived")),
                    f"{folder}: evaluation fraction outside [0, 1]")
        enabled, frozen = summary["final_learning"], summary["final_frozen"]
        runs.append({
            "evolution_seed": seed,
            "run": path.relative_to(ROOT).as_posix(),
            "source_sha256": {name: hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
                              for name, data in raw.items()},
            "fitness_enabled": enabled["fitness"],
            "fitness_frozen": frozen["fitness"],
            "accuracy_first_half": enabled["acc_first"],
            "accuracy_second_half": enabled["acc_second"],
            "accuracy_gain_percentage_points": 100 * (enabled["acc_second"] - enabled["acc_first"]),
            "training_population_mean_fitness": [row["fit_mean"] for row in rows],
        })
    return {
        "experiment": "E103 A: blank-start evolution",
        "source_sha256_normalization": "Replace CRLF with LF in source text bytes; all other bytes are unchanged.",
        "evaluation": {"protocol": "EVAL v1", "worlds": 128, "seed": 123},
        "training": {"population": 128, "worlds_per_generation": 8,
                     "generations": list(range(1, 201)),
                     "note": "Generation labels are one-based; source gen indices are 0-199."},
        "accuracy_gain_criterion_percentage_points": 3.0,
        "runs": runs,
    }


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", which="both", length=0, pad=9, colors=MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)


def save_figure(fig, stem, description):
    for extension in ("svg", "png"):
        metadata = {"Title": stem, "Description": description}
        if extension == "svg":
            metadata.update(Date=None, Creator="Seed / tools/build_readme_figures.py")
        else:
            metadata["Software"] = "Seed / tools/build_readme_figures.py"
        fig.savefig(OUT / f"{stem}.{extension}", dpi=150, facecolor="white",
                    metadata=metadata)
    plt.close(fig)


def plot_evaluation(data):
    runs = data["runs"]
    enabled = np.array([run["fitness_enabled"] for run in runs])
    frozen = np.array([run["fitness_frozen"] for run in runs])
    gains = np.array([run["accuracy_gain_percentage_points"] for run in runs])
    criterion = data["accuracy_gain_criterion_percentage_points"]
    require(np.all(enabled > frozen) and np.all(gains > criterion),
            "The E103 A milestone headline no longer matches its source results")

    fig, (fit_ax, gain_ax) = plt.subplots(1, 2, figsize=(12.4, 7.0))
    fig.subplots_adjust(left=0.085, right=0.965, top=0.70, bottom=0.23, wspace=0.30)
    fig.text(0.07, 0.94, "EXPERIMENT A  /  FINAL EVALUATION", color=TEAL,
             size=10.5, weight="bold")
    fig.text(0.07, 0.875, "A repeatable benefit from plasticity", size=24, weight="bold")
    fig.text(0.07, 0.823,
             "200 generations  ·  3 independent evolution seeds  ·  EVAL v1: 128 worlds, seed 123",
             color=MUTED, size=12)
    x = np.arange(3)
    width = 0.29
    fit_ax.bar(x - width / 2 - 0.02, enabled, width, color=TEAL, label="Updates enabled")
    fit_ax.bar(x + width / 2 + 0.02, frozen, width, color=ORANGE, label="Frozen")
    fit_ax.axhline(0, color=MUTED, linewidth=0.8)
    fit_ax.set(ylim=(-1.9, 19), yticks=[0, 5, 10, 15],
               ylabel="Fitness (food − poison per agent)")
    fit_ax.set_title("Fitness depends on plasticity", loc="left", pad=47, size=15, weight="bold")
    fit_ax.legend(loc="lower left", bbox_to_anchor=(-0.015, 1.015), ncol=2,
                  frameon=False, handlelength=1.1, handletextpad=0.5,
                  columnspacing=1.5, fontsize=11.5)
    for i, (on, off) in enumerate(zip(enabled, frozen)):
        fit_ax.text(i - width / 2 - 0.02, on + 0.48, f"{on:.2f}", ha="center",
                    va="bottom", size=13, weight="bold", color=TEAL)
        fit_ax.text(i + width / 2 + 0.02, off + (0.45 if off >= 0 else -0.48),
                    f"{off:.2f}", ha="center", va="bottom" if off >= 0 else "top",
                    size=12.5, color=ORANGE)

    gain_ax.bar(x, gains, width=0.48, color=BLUE)
    gain_ax.axhline(criterion, color=ORANGE, linewidth=1.4, linestyle=(0, (5, 3)))
    gain_ax.text(0.02, criterion + 0.45, "Preregistered criterion: > 3 points",
                 transform=gain_ax.get_yaxis_transform(), size=10.5, color=ORANGE,
                 bbox=dict(facecolor="white", edgecolor="none", pad=2))
    gain_ax.set(ylim=(0, 17.2), yticks=[0, 5, 10, 15],
                ylabel="Accuracy gain (percentage points)")
    gain_ax.set_title("Pooled accuracy rises during life", loc="left", pad=47,
                      size=15, weight="bold")
    gain_ax.text(0, 1.035, "Second half − first half  ·  updates enabled",
                 transform=gain_ax.transAxes, color=MUTED, size=11.5)
    for i, gain in enumerate(gains):
        gain_ax.text(i, gain + 0.45, f"+{gain:.2f}", ha="center", va="bottom",
                     size=13, weight="bold", color=BLUE)
    for ax in (fit_ax, gain_ax):
        ax.set_xticks(x, [f"Seed {run['evolution_seed']}" for run in runs])
        style_axes(ax)
    fig.text(0.07, 0.125,
             "Frozen = the same genome with Hebbian and predictor updates disabled.",
             color=MUTED, size=11)
    fig.text(0.07, 0.079,
             "Pooled accuracy can reflect survival and encounter selection. Independently evolved nonplastic controls are pending.",
             color=MUTED, size=10.5)
    save_figure(fig, "a-final-evaluation",
                "Final E103 A evaluations for all three evolution seeds. Paired enabled/frozen fitness "
                "and pooled within-life accuracy gains; fixed EVAL v1, 128 worlds, evaluation seed 123.")


def plot_training(data):
    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.72, bottom=0.24)
    fig.text(0.07, 0.94, "EXPERIMENT A  /  TRAINING RECORD", color=TEAL,
             size=10.5, weight="bold")
    fig.text(0.07, 0.867, "Three paths through 200 generations", size=24, weight="bold")
    fig.text(0.07, 0.806,
             "3 independent evolution seeds  ·  128 genomes × 8 training worlds per generation",
             color=MUTED, size=12)
    for run, color, style in zip(data["runs"], (TEAL, BLUE, ORANGE), ("-", "--", "-.")):
        ax.plot(data["training"]["generations"], run["training_population_mean_fitness"],
                color=color, linestyle=style, linewidth=1.65,
                label=f"Seed {run['evolution_seed']}")
    ax.set(xlim=(1, 200), xticks=[1, 50, 100, 150, 200],
           xlabel="Evolution generation", ylabel="Population-mean fitness\n(food − poison per agent)")
    ax.set_ylim(bottom=min(-0.5, ax.get_ylim()[0]))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.legend(loc="upper left", ncol=3, frameon=False, handlelength=2.6,
              columnspacing=1.7, fontsize=12)
    style_axes(ax)
    fig.text(0.07, 0.118,
             "All 200 raw population means per seed are shown. These are fitness values of perturbed training genomes.",
             color=MUTED, size=11)
    fig.text(0.07, 0.067,
             "Training worlds vary by generation; final genomes are evaluated separately on the fixed 128-world EVAL v1 set.",
             color=MUTED, size=11)
    save_figure(fig, "a-training-progress",
                "All 600 raw E103 A training-population fitness means, with no smoothing or run selection. "
                "These values describe perturbed training genomes, not final EVAL v1 scores.")


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
        "axes.labelsize": 11.5, "axes.labelpad": 12, "axes.titlecolor": INK,
        "xtick.labelsize": 12, "ytick.labelsize": 11.5,
        "svg.fonttype": "path", "svg.hashsalt": "seed-e103-a-readme-v1",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    data = load_data()
    OUT.mkdir(parents=True, exist_ok=True)
    plot_evaluation(data)
    plot_training(data)
    (OUT / "a-figure-data.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print("Validated 3 completed E103 A runs and all 600 training records.")
    for path in sorted(OUT.glob("a-*")):
        print(f"{path.relative_to(ROOT).as_posix()}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
