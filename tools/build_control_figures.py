"""Build the matched A/C figure only after all six registered runs finish.

No source record is modified. All validation precedes output creation.
"""

from pathlib import Path
import hashlib
import io
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
PAIRS = (
    ("20261112-023245_E103-A-blank-s0", "20261112-224133_E105-C-noplastic-s0"),
    ("20261112-050319_E103-A-blank-s1", "20261114-003141_E105-C-noplastic-s1"),
    ("20261112-102755_E103-A-blank-s2", "20261114-031608_E105-C-noplastic-s2"),
)
INK, MUTED, GRID = "#172B3A", "#566976", "#E4EAEE"
TEAL, BLUE, ORANGE = "#087F78", "#4668AC", "#BD6329"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_source(path, name):
    source = path / name
    require(source.is_file(), f"{path.name}: missing required source {name}")
    return source.read_bytes()


def load_run(folder, seed, condition):
    path = ROOT / "runs" / folder
    raw = {"summary.json": read_source(path, "summary.json")}
    summary = json.loads(raw["summary.json"])
    require(summary.get("status") == "finished",
            f"{folder}: unfinished run (status={summary.get('status')!r}); no figure written")
    require(summary.get("generation") == 200 and summary.get("eval_worlds") == 128,
            f"{folder}: expected completed generation 200 and 128 evaluation worlds")
    raw.update({name: read_source(path, name) for name in ("config.json", "metrics.jsonl")})
    config = json.loads(raw["config.json"])
    expected = dict(seed=seed, start="blank", generations=200, pop=128, worlds=8,
                    eval_worlds=128, sigma=0.1, lr=0.1, no_plastic=(condition == "C"),
                    quick=False, allow_dirty=False, cfg=None)
    require(all(config.get("args", {}).get(key) == value for key, value in expected.items()),
            f"{folder}: unexpected experiment arguments")
    require(config.get("cfg_overrides") == {} and isinstance(config.get("cfg"), dict)
            and bool(config["cfg"]) and all(finite_number(v) for v in config["cfg"].values()),
            f"{folder}: configuration override or invalid developmental configuration")
    source_history = [config.get("git", {})] + [r.get("git", {}) for r in config.get("resumes", [])]
    require(all(g.get("dirty") is False and isinstance(g.get("commit"), str)
                and len(g["commit"]) == 40 for g in source_history),
            f"{folder}: missing clean source provenance")

    rows = [json.loads(line) for line in raw["metrics.jsonl"].splitlines()]
    metric_keys = {"gen", "fit_mean", "fit_best", "fit_std", "acc_first", "acc_second", "survived", "seconds"}
    require(len(rows) == 200 and [row.get("gen") for row in rows] == list(range(200)),
            f"{folder}: expected 200 contiguous metric rows (indices 0-199)")
    require(all(metric_keys <= row.keys() and all(finite_number(v) for v in row.values())
                and type(row["gen"]) is int and row["seconds"] >= 0 and row["fit_std"] >= 0
                and all(0 <= row[key] <= 1 for key in ("acc_first", "acc_second", "survived"))
                for row in rows), f"{folder}: invalid or nonfinite training metrics")
    for label in ("final_learning", "final_frozen", "start_learning"):
        evaluation = summary.get(label, {})
        require({"fitness", "acc_first", "acc_second", "survived", "eaten"} <= evaluation.keys()
                and all(finite_number(v) for v in evaluation.values())
                and all(0 <= evaluation[key] <= 1 for key in ("acc_first", "acc_second", "survived"))
                and evaluation["eaten"] >= 0, f"{folder}: invalid {label} evaluation")

    binary = {name: read_source(path, name)
              for name in ("genome_start.npy", "genome_final.npy", "checkpoint.npz")}
    start = np.load(io.BytesIO(binary["genome_start.npy"]), allow_pickle=False)
    final = np.load(io.BytesIO(binary["genome_final.npy"]), allow_pickle=False)
    require(config.get("genome_size") == 155
            and all(v.shape == (155,) and np.issubdtype(v.dtype, np.floating)
                    and np.isfinite(v).all() for v in (start, final)),
            f"{folder}: invalid genome arrays")
    start_hash = hashlib.sha256(start.tobytes()).hexdigest()[:16]
    final_hash = hashlib.sha256(final.tobytes()).hexdigest()[:16]
    require(start_hash == config.get("start_genome_hash")
            and final_hash == summary.get("final_genome_hash"),
            f"{folder}: genome hash does not match the recorded result")
    with np.load(io.BytesIO(binary["checkpoint.npz"]), allow_pickle=False) as checkpoint:
        require({"mu", "m", "v", "t", "key"} <= set(checkpoint.files)
                and all(np.isfinite(checkpoint[k]).all() for k in checkpoint.files),
                f"{folder}: invalid or nonfinite checkpoint")
        require(checkpoint["t"].shape == () and checkpoint["t"].item() == 200
                and np.array_equal(checkpoint["mu"], final)
                and checkpoint["m"].shape == final.shape and checkpoint["v"].shape == final.shape
                and np.all(checkpoint["v"] >= 0) and checkpoint["key"].shape == (2,),
                f"{folder}: checkpoint does not match the final genome at generation 200")
        checkpoint_key = checkpoint["key"].tolist()
    field = "final_learning" if condition == "A" else "final_frozen"
    record = {
        "run": path.relative_to(ROOT).as_posix(), "condition": condition, "evolution_seed": seed,
        "evaluation_field": field, "evaluation": summary[field],
        "start_genome_hash": start_hash, "final_genome_hash": final_hash,
        "source_commits": [g["commit"] for g in source_history],
        "source_sha256": {name: hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
                          for name, content in raw.items()},
        "binary_source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in binary.items()},
        "retained_training_metric_seconds": math.fsum(row["seconds"] for row in rows),
    }
    return record, config, start, checkpoint_key


def load_data():
    pairs, reference_cfg = [], None
    for seed, (a_folder, c_folder) in enumerate(PAIRS):
        a, a_config, a_start, a_key = load_run(a_folder, seed, "A")
        c, c_config, c_start, c_key = load_run(c_folder, seed, "C")
        if reference_cfg is None:
            reference_cfg = a_config["cfg"]
        require(a_config["cfg"] == c_config["cfg"] == reference_cfg,
                f"Seed {seed}: developmental configurations do not match")
        require(a_start.dtype == c_start.dtype and np.array_equal(a_start, c_start),
                f"Seed {seed}: initial genomes do not match exactly")
        require(a_key == c_key, f"Seed {seed}: final optimizer random states do not match")
        pairs.append({"evolution_seed": seed, "A": a, "C": c,
                      "fitness_difference_A_minus_C": a["evaluation"]["fitness"] - c["evaluation"]["fitness"]})
    return {
        "comparison": "E103 A versus E105 C; three matched, independently evolved pairs",
        "source_sha256_normalization": "Replace CRLF with LF in source text bytes; all other bytes are unchanged.",
        "binary_source_sha256_normalization": "None; SHA-256 of the exact binary file bytes.",
        "evaluation": {"protocol": "EVAL v1", "worlds": 128, "seed": 123,
                       "fitness_units": "Food minus poison eaten, averaged per agent",
                       "A": "final_learning: plasticity enabled during evolution and evaluation",
                       "C": "final_frozen: plasticity disabled during evolution and evaluation",
                       "excluded_diagnostic": "C.final_learning re-enables plasticity; it is not the control score."},
        "training": {"generations": 200, "population": 128, "worlds_per_generation": 8,
                     "sigma": 0.1, "learning_rate": 0.1, "start": "blank", "genome_size": 155,
                     "cfg": reference_cfg,
                     "timing_note": "Retained metric seconds sum training rows only; excludes evaluation, process overhead, pauses and discarded/replayed work. Not total wall time."},
        "scope": "Same developmental encoding and optimizer budget; C retains recurrent state and energy feedback. An independently evolved control, distinct from A's within-genome frozen lesion. Three evolution seeds in one fixed toy-world protocol; no confidence intervals or universal-necessity claim.",
        "pairs": pairs,
    }


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", which="both", length=0, pad=9, colors=MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(range(3), ["Seed 0", "Seed 1", "Seed 2"])
    ax.set_xlim(-0.6, 2.6)


def plot_comparison(data):
    a = np.array([p["A"]["evaluation"]["fitness"] for p in data["pairs"]])
    c = np.array([p["C"]["evaluation"]["fitness"] for p in data["pairs"]])
    gaps = np.array([p["fitness_difference_A_minus_C"] for p in data["pairs"]])
    fig, (fit_ax, gap_ax) = plt.subplots(1, 2, figsize=(12.4, 7.5))
    fig.subplots_adjust(left=0.085, right=0.965, top=0.66, bottom=0.25, wspace=0.32)
    fig.text(0.07, 0.945, "EXPERIMENTS A + C  /  MATCHED FINAL EVALUATION", color=TEAL, size=10.5, weight="bold")
    fig.text(0.07, 0.885, "Evolution with and without plasticity", size=24, weight="bold")
    fig.text(0.07, 0.835, "3 matched evolution seeds  ·  EVAL v1: 128 worlds, seed 123", color=MUTED, size=12)
    x, width = np.arange(3), 0.29
    fit_ax.bar(x - 0.165, a, width, color=TEAL, label="A · plastic evolution")
    fit_ax.bar(x + 0.165, c, width, color=ORANGE, label="C · nonplastic evolution")
    low, high = min(0.0, a.min(), c.min()), max(0.0, a.max(), c.max())
    span = max(high - low, 1.0)
    fit_ax.set_ylim(low - 0.16 * span, high + 0.20 * span)
    fit_ax.set_ylabel("Fitness (food − poison per agent)")
    fit_ax.set_title("Final fitness by seed", loc="left", pad=61, size=15, weight="bold")
    fit_ax.legend(loc="lower left", bbox_to_anchor=(-0.015, 1.015), frameon=False,
                  handlelength=1.1, handletextpad=0.5, fontsize=11.5)
    for offset, values, color in ((-0.165, a, TEAL), (0.165, c, ORANGE)):
        for i, value in enumerate(values):
            sign = 1 if value >= 0 else -1
            fit_ax.text(i + offset, value + sign * 0.045 * span, f"{value:.2f}",
                        ha="center", va="bottom" if sign > 0 else "top", size=12, color=color)
    gap_ax.bar(x, gaps, width=0.48, color=BLUE)
    extent = max(float(np.max(np.abs(gaps))), 1.0)
    gap_ax.set_ylim(-1.32 * extent, 1.32 * extent)
    gap_ax.set_ylabel("Fitness difference (food − poison per agent)")
    gap_ax.set_title("Paired difference (A − C)", loc="left", pad=61, size=15, weight="bold")
    gap_ax.text(0, 1.055, "Positive favors A; negative favors C", transform=gap_ax.transAxes,
                size=11.5, color=MUTED)
    for i, value in enumerate(gaps):
        sign = 1 if value >= 0 else -1
        gap_ax.text(i, value + sign * 0.075 * extent, f"{value:+.3f}", ha="center",
                    va="bottom" if sign > 0 else "top", size=12.5, color=BLUE, weight="bold")
    for ax in (fit_ax, gap_ax):
        style_axes(ax)
    fig.text(0.07, 0.16, "Matched budget: 200 generations × 128 candidates × 8 worlds; σ = 0.1, lr = 0.1.", size=11, color=INK)
    fig.text(0.07, 0.12, "Same developmental encoding. C retains recurrent state and energy feedback.", size=11, color=INK)
    fig.text(0.07, 0.08, "C is independently evolved without plasticity; A's within-genome frozen lesion is a separate comparison.", size=10.5, color=MUTED)
    fig.text(0.07, 0.045, "Scope: three evolution seeds in one fixed toy-world protocol. All seed pairs shown; no confidence intervals.", size=10.5, color=MUTED)
    for extension in ("svg", "png"):
        metadata = {"Title": "Matched A versus C final evaluation", "Description": data["scope"]}
        if extension == "svg":
            metadata.update(Date=None, Creator="Seed / tools/build_control_figures.py")
        else:
            metadata["Software"] = "Seed / tools/build_control_figures.py"
        fig.savefig(OUT / f"ac-matched-comparison.{extension}", dpi=150,
                    facecolor="white", metadata=metadata)
        if extension == "svg":
            path = OUT / "ac-matched-comparison.svg"
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                            encoding="utf-8", newline="\n")
    plt.close(fig)


def main():
    data = load_data()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
                         "axes.labelsize": 11.5, "axes.labelpad": 12, "axes.titlecolor": INK,
                         "xtick.labelsize": 12, "ytick.labelsize": 11.5, "svg.fonttype": "path",
                         "svg.hashsalt": "seed-e103-e105-ac-v1", "figure.facecolor": "white",
                         "axes.facecolor": "white"})
    OUT.mkdir(parents=True, exist_ok=True)
    plot_comparison(data)
    (OUT / "ac-figure-data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n",
                                            encoding="utf-8", newline="\n")
    print("Validated six finished runs; wrote ac-matched-comparison.svg/.png and ac-figure-data.json")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"Cannot build A/C comparison: {error}") from error
