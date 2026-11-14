"""Rebuild the fixed-genome E107/E108 figures from their finished run artifacts.

Uses NumPy/Matplotlib only: no evaluator import, training, or source-record writes.
Run without Python's -O option so the provenance assertions remain enabled.
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
A_RUNS = (
    "20261112-023245_E103-A-blank-s0",
    "20261112-050319_E103-A-blank-s1",
    "20261112-102755_E103-A-blank-s2",
)
INPUT_HASHES = (
    "477d9d03c713dcab8072349ff9155813e5a384d3709eff67595a342b55111f53",
    "35196cd30c2de5c1ed5bb9e23ce445ff6ad13bd1a33bdbf7a77f54d53edce899",
    "f299069753636a303f6ba7f91be78099e8491b20a7511bdcef47fd8b4c5bbaa6",
)
E107_RUNS = (
    "20261114-232719_E107-structural-A-s0",
    "20261114-232924_E107-structural-A-s1",
    "20261114-233501_E107-structural-A-s2",
)
E108_RUNS = (
    "20261114-231716_E108-final-A0",
    "20261114-232930_E108-final-A1",
    "20261114-233434_E108-final-A2",
)
DEFAULT_CFG = dict(G=16, M=4, T=300, R=2, n_items=24, H=48, n_htypes=3,
                   e0=1.0, decay=0.003, gain=0.25, e_max=2.0, micro=2, wmax=3.0)
RULE = dict(name="activity-grow-prune-v1", interval=10, prune_fraction=0.01,
            extra_fanin=1, stream_tag=107, timing="after brain tick; previous activity")
SHIFTS = {"control": {}, "sparse": {"n_items": 12}, "dense": {"n_items": 36},
          "low-cost": {"decay": 0.0015}, "high-cost": {"decay": 0.006}}
METRICS = {"fitness", "acc_first", "acc_second", "survived", "eaten"}
INK, MUTED, GRID = "#172B3A", "#566976", "#E4EAEE"
TEAL, BLUE, ORANGE = "#087F78", "#4668AC", "#BD6329"


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def check_score(score, structural=False):
    assert set(score) == METRICS | ({"structure"} if structural else set()), "Unexpected score fields"
    assert all(type(score[k]) in (int, float) and math.isfinite(score[k]) for k in METRICS)
    assert all(0 <= score[k] <= 1 for k in ("acc_first", "acc_second", "survived"))
    assert type(score["eaten"]) is int and score["eaten"] >= 0
    if structural:
        counters = score["structure"]
        assert all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in counters.values())
        assert all(counters[k] == 0 for k in ("cap_violations", "illegal_edges", "invalid_proposals", "invalid_weights"))
        assert counters["max_abs_weight"] <= DEFAULT_CFG["wmax"]
        assert counters["min_edges"] <= counters["final_edges"] <= counters["max_edges"] <= counters["edge_cap"]


def load_run(folder, seed, protocol):
    path, a_path = ROOT / "runs" / folder, ROOT / "runs" / A_RUNS[seed]
    source = (a_path / "genome_final.npy").relative_to(ROOT).as_posix()
    raw = {name: (path / name).read_bytes() for name in ("config.json", "summary.json")}
    config, summary = (json.loads(raw[name]) for name in ("config.json", "summary.json"))
    assert summary["status"] == "finished" and summary["generation"] == 0, f"{folder}: unfinished diagnostic"
    assert summary["eval_worlds"] == 128 and summary["eval_protocol"] == protocol
    expected = dict(start=f"file:{source}", generations=0, seed=seed, eval_protocol=protocol,
                    eval_worlds=128, pop=128, worlds=8, sigma=0.1, lr=0.1, no_plastic=False,
                    quick=False, allow_dirty=False, cfg=None, resume=None)
    assert all(k in config["args"] and config["args"][k] == v for k, v in expected.items()), f"{folder}: arguments changed"
    assert config["cfg"] == DEFAULT_CFG and config["cfg_overrides"] == {} and config["genome_size"] == 155
    assert config["git"]["dirty"] is False and not config.get("resumes"), f"{folder}: unclean source"
    commit = config["git"]["commit"]
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    a_raw = {name: (a_path / name).read_bytes() for name in ("config.json", "summary.json")}
    a_config, a_summary = (json.loads(a_raw[name]) for name in ("config.json", "summary.json"))
    assert a_summary["status"] == "finished" and a_summary["generation"] == 200 and a_summary["eval_worlds"] == 128
    assert a_config["cfg"] == DEFAULT_CFG and a_config["args"]["seed"] == seed
    assert a_config["args"]["no_plastic"] is False and not a_config["cfg_overrides"]
    assert a_config["git"]["dirty"] is False and all(r["git"]["dirty"] is False for r in a_config.get("resumes", []))
    input_bytes = (ROOT / source).read_bytes()
    input_hash = sha256(input_bytes)
    assert input_hash == INPUT_HASHES[seed] == config["input_genome"]["sha256"], f"{folder}: input mismatch"
    assert config["input_genome"]["path"].replace("\\", "/").endswith(source)
    genome = np.load(io.BytesIO(input_bytes), allow_pickle=False)
    assert genome.shape == (155,) and np.issubdtype(genome.dtype, np.floating) and np.isfinite(genome).all()
    assert sha256(genome.tobytes())[:16] == config["start_genome_hash"] == summary["final_genome_hash"] == a_summary["final_genome_hash"]
    binary = {name: (path / name).read_bytes() for name in ("genome_start.npy", "genome_final.npy")}
    assert all(content == input_bytes for content in binary.values()), f"{folder}: genome changed during evaluation"
    if protocol == "structural-v1":
        raw["metrics.jsonl"] = (path / "metrics.jsonl").read_bytes()
        assert not raw["metrics.jsonl"].strip(), f"{folder}: unexpected training records"
        binary["checkpoint.npz"] = (path / "checkpoint.npz").read_bytes()
        with np.load(io.BytesIO(binary["checkpoint.npz"]), allow_pickle=False) as checkpoint:
            assert checkpoint["t"].item() == 0 and np.array_equal(checkpoint["mu"], genome)
            assert all(np.isfinite(checkpoint[k]).all() for k in checkpoint.files)
            assert np.all(checkpoint["m"] == 0) and np.all(checkpoint["v"] == 0)
    provenance = {
        "evolution_seed": seed, "run": path.relative_to(ROOT).as_posix(), "source_commit": commit,
        "input_genome": {"path": source, "sha256": input_hash},
        "source_sha256": {name: sha256(content.replace(b"\r\n", b"\n")) for name, content in raw.items()},
        "binary_source_sha256": {name: sha256(content) for name, content in binary.items()},
        "input_run_source_sha256": {name: sha256(content.replace(b"\r\n", b"\n")) for name, content in a_raw.items()},
    }
    return config, summary, a_summary, provenance


def load_data():
    common = {
        "source_sha256_normalization": "Replace CRLF with LF in source text bytes; all other bytes unchanged. Binary hashes use exact file bytes.",
        "fitness_units": "Food minus poison eaten, averaged per agent", "default_cfg": DEFAULT_CFG,
        "input_scope": "All three saved E103 A genomes; 200 generations of evolution with weight learning and no structural updates. No evolution or tuning in these evaluations.",
    }
    structural = dict(common, experiment="E107", protocol="structural-v1", worlds=128, eval_seed=123,
                      eval_seed_basis="Fixed by Structural v1 preregistration; not stored separately in these summaries.",
                      structural_rule=RULE, runs=[])
    broader = dict(common, experiment="E108", protocol="broader-v1", split="final", worlds=128,
                   eval_seed=8109, conditions=SHIFTS, runs=[])
    for seed, folder in enumerate(E107_RUNS):
        config, summary, reference, record = load_run(folder, seed, "structural-v1")
        assert config["structural_rule"] == RULE and config["args"]["structural_plasticity"] == RULE["name"]
        scores = summary["conditions"]
        assert set(scores) == {"Weight-only", "Weight+structure", "Frozen"}
        for name, score in scores.items():
            check_score(score, structural=(name != "Weight-only"))
        assert scores["Weight-only"] == reference["final_learning"], f"{folder}: weight-only reference mismatch"
        assert {k: scores["Frozen"][k] for k in METRICS} == reference["final_frozen"]
        assert scores["Frozen"]["structure"]["added"] == scores["Frozen"]["structure"]["removed"] == 0
        record["fitness"] = {name: score["fitness"] for name, score in scores.items()}
        record["structure_minus_weight_only"] = scores["Weight+structure"]["fitness"] - scores["Weight-only"]["fitness"]
        structural["runs"].append(record)
    for seed, folder in enumerate(E108_RUNS):
        config, summary, _, record = load_run(folder, seed, "broader-v1")
        assert config["args"]["condition_split"] == summary["condition_split"] == "final" and summary["eval_seed"] == 8109
        assert config["args"].get("structural_plasticity", "off") == "off"
        assert config["eval_protocol"] == dict(name="broader-v1", split="final", seed=8109, n_worlds=128,
                                               rule_counts={"0": 64, "1": 64}, conditions=SHIFTS,
                                               plasticity={"learning": True, "frozen": False})
        assert summary["input_genome"] == config["input_genome"]
        scores = summary["condition_evaluations"]
        assert set(scores) == set(SHIFTS), f"{folder}: unexpected environment conditions"
        record["conditions"] = {}
        for name, shift in SHIFTS.items():
            score = scores[name]
            assert score["cfg_shift"] == shift
            for mode in ("learning", "frozen"):
                check_score(score[mode])
            delta = score["learning"]["fitness"] - score["frozen"]["fitness"]
            assert delta == score["learning_minus_frozen_fitness"] and math.isfinite(delta)
            assert all(math.isfinite(v) for v in score.values() if type(v) in (int, float))
            assert score["learning_fitness_shift"] == score["learning"]["fitness"] - scores["control"]["learning"]["fitness"]
            assert score["frozen_fitness_shift"] == score["frozen"]["fitness"] - scores["control"]["frozen"]["fitness"]
            assert score["learning_advantage_shift"] == delta - scores["control"]["learning_minus_frozen_fitness"]
            record["conditions"][name] = dict(learning_fitness=score["learning"]["fitness"],
                                               frozen_fitness=score["frozen"]["fitness"],
                                               learning_minus_frozen_fitness=delta)
        broader["runs"].append(record)
    return structural, broader


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", which="both", length=0, pad=9, colors=MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.axhline(0, color=MUTED, linewidth=0.8)


def save(fig, stem, data, description):
    for extension in ("svg", "png"):
        metadata = {"Title": stem, "Description": description}
        metadata.update({"Date": None, "Creator": "Seed / tools/build_extension_figures.py"} if extension == "svg"
                        else {"Software": "Seed / tools/build_extension_figures.py"})
        path = OUT / f"{stem}.{extension}"
        fig.savefig(path, dpi=150, facecolor="white", metadata=metadata)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                            encoding="utf-8", newline="\n")
    plt.close(fig)
    (OUT / f"{data['experiment'].lower()}-figure-data.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def plot_structural(data):
    weight = np.array([r["fitness"]["Weight-only"] for r in data["runs"]])
    structure = np.array([r["fitness"]["Weight+structure"] for r in data["runs"]])
    delta = np.array([r["structure_minus_weight_only"] for r in data["runs"]])
    assert np.array_equal(delta, structure - weight)
    fig, (fit_ax, delta_ax) = plt.subplots(1, 2, figsize=(12.4, 7.1))
    fig.subplots_adjust(left=0.085, right=0.965, top=0.67, bottom=0.25, wspace=0.32)
    fig.text(0.07, 0.945, "E107  /  FIXED-GENOME STRUCTURAL DIAGNOSTIC", size=10.5, weight="bold", color=TEAL)
    fig.text(0.07, 0.88, "Adding structural updates to evolved genomes", size=23, weight="bold")
    fig.text(0.07, 0.825, "3 saved A genomes  ·  Structural v1: 128 matched worlds, seed 123", size=12, color=MUTED)
    x = np.arange(3)
    span = max(float(np.ptp(np.r_[0, weight, structure])), 1.0)
    for offset, values, color, label in ((-0.165, weight, TEAL, "Weight-only"), (0.165, structure, ORANGE, "Weight + structure")):
        fit_ax.bar(x + offset, values, width=0.29, color=color, label=label)
        label_dx = -4 if offset < 0 else 4
        for i, value in enumerate(values):
            fit_ax.annotate(f"{value:.2f}", (i + offset, value), xytext=(label_dx, 7 if value >= 0 else -7),
                            textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top", color=color, size=12)
    fit_ax.set_ylim(min(0, weight.min(), structure.min()) - 0.1 * span, max(0, weight.max(), structure.max()) + 0.2 * span)
    fit_ax.set_ylabel("Fitness (food − poison per agent)")
    fit_ax.set_title("Paired evaluation", loc="left", pad=47, size=15, weight="bold")
    fit_ax.legend(loc="lower left", bbox_to_anchor=(-0.015, 1.025), ncol=2, frameon=False,
                  handlelength=1.1, handletextpad=0.5, columnspacing=1.1, fontsize=11.5)
    delta_ax.bar(x, delta, width=0.48, color=ORANGE)
    extent = max(float(np.max(np.abs(delta))), 1.0)
    delta_ax.set_ylim(min(0, delta.min()) - 0.3 * extent, max(0, delta.max()) + 0.25 * extent)
    delta_ax.set_ylabel("Fitness difference (food − poison per agent)")
    delta_ax.set_title("Change from weight-only", loc="left", pad=47, size=15, weight="bold")
    delta_ax.text(0, 1.055, "Weight + structure − weight-only", transform=delta_ax.transAxes, size=11.5, color=MUTED)
    for i, value in enumerate(delta):
        delta_ax.annotate(f"{value:+.3f}", (i, value), xytext=(0, 8 if value >= 0 else -8), textcoords="offset points",
                          ha="center", va="bottom" if value >= 0 else "top", color=ORANGE, size=12.5, weight="bold")
    for ax in (fit_ax, delta_ax):
        style_axes(ax)
        ax.set_xticks(x, ["A0", "A1", "A2"])
        ax.set_xlim(-0.6, 2.6)
    fig.text(0.07, 0.16, "Fixed activity-grow-prune-v1 rule; no evolution or tuning in this evaluation.", size=11)
    fig.text(0.07, 0.115, "Inputs evolved with weight learning only. This is not a result from evolution with structural plasticity.", size=10.5, color=MUTED)
    fig.text(0.07, 0.07, "Descriptive paired differences, n = 3 evolution seeds. No confidence intervals or significance claim.", size=10.5, color=MUTED)
    save(fig, "e107-structural-diagnostic", data,
         "E107: fixed-rule structural intervention on three previously evolved A genomes, 128 worlds/seed 123. Signed paired differences; no structural training or significance claim.")


def plot_broader(data):
    fig, ax = plt.subplots(figsize=(12.4, 7.1))
    fig.subplots_adjust(left=0.105, right=0.965, top=0.71, bottom=0.29)
    fig.text(0.07, 0.945, "E108  /  FRESH-WORLD PARAMETER SHIFTS", size=10.5, weight="bold", color=TEAL)
    fig.text(0.07, 0.88, "Plasticity advantage under parameter shifts", size=23, weight="bold")
    fig.text(0.07, 0.825, "3 fixed A genomes  ·  BROADER v1 final: 128 fresh worlds, seed 8109", size=12, color=MUTED)
    x = np.arange(len(SHIFTS))
    all_values = []
    for offset, color, record in zip((-0.24, 0.0, 0.24), (TEAL, BLUE, ORANGE), data["runs"]):
        values = [record["conditions"][name]["learning_minus_frozen_fitness"] for name in SHIFTS]
        all_values.extend(values)
        ax.bar(x + offset, values, width=0.22, color=color, label=f"A{record['evolution_seed']}")
        for i, value in enumerate(values):
            ax.annotate(f"{value:.2f}", (i + offset, value), xytext=(0, 6 if value >= 0 else -6),
                        textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top", size=10.5, color=color)
    style_axes(ax)
    ax.set_ylabel("Learning − frozen fitness\n(food − poison per agent)")
    ax.set_xticks(x, ["Control\nDefault", "Sparse\n12 items / kind", "Dense\n36 items / kind",
                     "Lower cost\nDecay 0.0015", "Higher cost\nDecay 0.006"])
    span = max(max(all_values) - min(0, min(all_values)), 1.0)
    ax.set_ylim(min(0, min(all_values)) - 0.06 * span, max(0, max(all_values)) + 0.17 * span)
    ax.set_xlim(-0.6, 4.6)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.015), ncol=3, frameon=False, handlelength=1.1)
    fig.text(0.07, 0.16, "Learning enables both Hebbian and predictor updates; frozen retains recurrent state and energy feedback.", size=10.5)
    fig.text(0.07, 0.115, "One parameter changes per condition. Default: 24 items per kind; energy decay 0.003 per step.", size=10.5, color=MUTED)
    fig.text(0.07, 0.07, "Same food/poison task family, n = 3 evolution seeds. No uncertainty estimate, significance or open-world claim.", size=10.5, color=MUTED)
    save(fig, "e108-parameter-shifts", data,
         "E108: learning-minus-frozen fitness for three fixed A genomes under five registered conditions, 128 fresh worlds/seed 8109. Same task family; no confidence intervals, significance or open-world claim.")


def main():
    if not __debug__:
        raise SystemExit("Run without -O: provenance assertions must remain enabled")
    structural, broader = load_data()  # Validate both complete panels before any output.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
                         "axes.labelsize": 11.5, "axes.labelpad": 12, "axes.titlecolor": INK,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "svg.fonttype": "path",
                         "svg.hashsalt": "seed-e107-e108-figures-v1", "figure.facecolor": "white", "axes.facecolor": "white"})
    OUT.mkdir(parents=True, exist_ok=True)
    plot_structural(structural)
    plot_broader(broader)
    print("E107 structural-minus-weight-only:", [r["structure_minus_weight_only"] for r in structural["runs"]])
    for name in SHIFTS:
        print("E108", name, [r["conditions"][name]["learning_minus_frozen_fitness"] for r in broader["runs"]])


if __name__ == "__main__":
    main()
