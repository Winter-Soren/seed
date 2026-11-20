"""Plot only the completed E114 data; no engine import or simulation.

From any working directory: python -X utf8 path/to/build_e114_figure.py
Outputs are written to results/figures/.
"""
from pathlib import Path
import hashlib
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
OUT = ROOT / "results/figures"
RUN = ROOT / "runs/20261119-045931_E114-v4"
SUMMARY_SHA256 = "1a4801efcf638848793c3eb20e0acd35ccd9323a0b27484e98679840e4c1258d"
PANELS = ("A0", "A1", "A2", "direct0", "direct1", "direct2", "mixture")
LABELS = ("A · seed 0", "A · seed 1", "A · seed 2", "Direct · seed 0",
          "Direct · seed 1", "Direct · seed 2", "A/B specialist mixture")
METRICS = ("stationary_fitness", "individual_min_return")
BLUE, TEAL, INK, MUTED, GRID = "#4668AC", "#087F78", "#172B3A", "#566976", "#E4EAEE"


def load_data():
    raw = (RUN / "summary.json").read_bytes()
    digest = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    assert digest == SUMMARY_SHA256, "The registered complete E114 summary changed"
    summary = json.loads(raw)
    assert summary["status"] == "finished" and summary["generation"] == 0
    assert summary["eval_protocol"] == "v4" and summary["condition_split"] == "final"
    assert summary["eval_seed"] == 2026111902 and summary["eval_worlds"] == 256
    assert summary["expected_panels"] == summary["completed_panels"] == len(summary["panels"]) == 11
    assert all(p["status"] == "finished" for p in summary["panels"].values())
    config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    assert config["git"]["dirty"] is False and config["cfg"]["M"] == 4
    rows = []
    for key, label in zip(PANELS, LABELS):
        panel = summary["panels"][key]
        files = {name: RUN / panel["raw_files"][name]
                 for name in ("paired", "stationary_r0", "stationary_r1")}
        with np.load(files["stationary_r0"], allow_pickle=False) as r0, np.load(files["stationary_r1"], allow_pickle=False) as r1:
            for name in ("wkeys", "dkeys", "initial_grid", "initial_pos", "agent_ids"):
                np.testing.assert_array_equal(r0[name], r1[name])
            a, b = r0["stationary_r0_agent_return"], r1["stationary_r1_agent_return"]
            assert a.shape == b.shape == (256, 4)
            expected = ((a.astype(float) + b).mean(axis=1) / 2,
                        np.minimum(a, b).mean(axis=1))
        row = {"panel": key, "label": label, "metrics": {},
               "raw_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in files.items()}}
        with np.load(files["paired"], allow_pickle=False) as paired:
            for metric, values in zip(METRICS, expected):
                np.testing.assert_array_equal(values, paired[f"paired_{metric}"])
                assert values.shape == (256,) and np.isfinite(values).all()
                score = panel["paired"][metric]
                mean, se = float(values.mean()), float(values.std(ddof=1) / np.sqrt(256))
                assert math.isclose(mean, score["mean"], rel_tol=0, abs_tol=1e-12)
                assert math.isclose(se, score["se"], rel_tol=0, abs_tol=1e-12)
                row["metrics"][metric] = {"mean": score["mean"], "se": score["se"]}
        rows.append(row)
    assert (RUN / "summary.json").read_bytes() == raw, "Source changed during figure generation"
    return {"experiment": "E114", "run": RUN.relative_to(ROOT).as_posix(),
            "source_commit": config["git"]["commit"], "summary_sha256_lf": digest,
            "eval_seed": summary["eval_seed"], "world_families": 256, "agents_per_family": 4,
            "error_bars": "One SE across 256 world-family means, not evolution seeds",
            "definition": "Worse rule: minimum return across two fresh matched stationary lives for each starting individual, then average four individuals within each world family",
            "validation": "All 14 means and SEs checked against the complete summary and independently recomputed from paired raw individual returns",
            "rows": rows}


def main():
    data = load_data()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK,
                         "svg.hashsalt": "E114-stationary-paired-v1", "svg.fonttype": "none"})
    fig = plt.figure(figsize=(13.8, 7.1), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(3.2, 2), left=.16, right=.98,
                           bottom=.23, top=.76, wspace=.055)
    ax, table = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    y = np.arange(len(data["rows"]))
    for axis in (ax, table):
        axis.set_ylim(6.6, -.7)
        for boundary in (2.5, 5.5):
            axis.axhline(boundary, color=GRID, lw=1)
        for spine in axis.spines.values():
            spine.set_visible(False)
    ax.set_yticks(y, LABELS)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_xlim(-9, 78)
    ax.set_xticks([-5, 0, 10, 20, 30, 40, 50, 60, 70])
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.axvline(0, color=INK, lw=1.5, zorder=2)
    ax.set_xlabel("Food minus poison per starting agent", labelpad=12)
    table.set_xlim(0, 1)
    table.axis("off")
    table.text(.02, 1.025, "Mean ± SE", transform=table.transAxes, color=BLUE, weight="bold")
    table.text(.53, 1.025, "Worse rule ± SE", transform=table.transAxes, color=TEAL, weight="bold")
    for i, row in enumerate(data["rows"]):
        mean, worse = (row["metrics"][metric] for metric in METRICS)
        ax.plot([worse["mean"], mean["mean"]], [i, i], color="#BFCBD3", lw=1.5, zorder=2)
        for score, color, marker, xtext in ((mean, BLUE, "s", .02), (worse, TEAL, "o", .53)):
            ax.errorbar(score["mean"], i, xerr=score["se"], fmt=marker, color=color,
                        markersize=6, elinewidth=1.8, capsize=5, capthick=1.4, zorder=4)
            table.text(xtext, i, f'{score["mean"]:.3f} ± {score["se"]:.3f}',
                       color=color, va="center", fontsize=11.5)
    handles = [plt.Line2D([], [], marker="s", color=BLUE, linestyle="", markersize=7,
                          label="Mean across both stationary rules"),
               plt.Line2D([], [], marker="o", color=TEAL, linestyle="", markersize=7,
                          label="Same individual's worse rule")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.145, .855),
               ncol=2, frameon=False, fontsize=11, columnspacing=2)
    fig.text(.035, .95, "E114 · Mean fitness and individual robustness", fontsize=19, weight="bold")
    fig.text(.035, .904, "Opposite food rules · three evolution seeds per controller class · 256 world families",
             fontsize=12, color=MUTED)
    fig.text(.035, .127, "Worse rule: take each starting agent's lower return across the two rules, then average within each world.",
             fontsize=10.8)
    fig.text(.035, .089, "Fresh matched lives; four agents per family. Points and labels show mean ± 1 SE over 256 world families.",
             fontsize=10.8, color=MUTED)
    fig.text(.035, .051, "Stationary outcomes only. This diagnostic does not by itself establish learning or recovery after a rule reversal.",
             fontsize=10.8, color=MUTED)
    for suffix in ("png", "svg", "pdf"):
        metadata = {"Creator": "Seed E114 figure", "CreationDate": None, "ModDate": None} if suffix == "pdf" else {"Date": None} if suffix == "svg" else {"Software": "Seed E114 figure"}
        fig.savefig(OUT / f"e114-stationary-vs-worse-rule.{suffix}", dpi=180, metadata=metadata)
    plt.close(fig)
    (OUT / "e114-figure-data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print("Verified 7 controllers, 14 means and 14 SEs; wrote PNG/SVG/PDF and exact-data JSON.")


if __name__ == "__main__":
    main()
