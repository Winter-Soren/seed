"""Plot the 11 saved genome states from each completed canonical E118 lineage.

The figure describes genetic change. Held-out shelter and survival measurements
determine whether the descendants perform better. No simulator is imported.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "shelter-evolution-v1"
STEM = "e118-genome-evolution"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def genome_metrics(trajectory):
    require(trajectory.shape == (11, 155) and trajectory.dtype == np.float32
            and np.isfinite(trajectory).all(), "expected 11 finite float32 genomes of length 155")
    values = trajectory.astype(np.float64)
    rms = np.sqrt(np.mean(np.square(values - values[0]), axis=1))
    # rules-v0 flattening starts with conn[post_type, pre_type], a 5 by 5 table.
    logits = values[:, 4 * 5]
    probability = 100 * np.exp(-np.logaddexp(0., -logits))
    return rms, probability


def load_series(root):
    folders = []
    for index in range(3):
        matches = sorted((root / "runs").glob(f"*_E118-A{index}"))
        require(len(matches) == 1, f"expected one canonical E118-A{index} run")
        folder = matches[0].resolve()
        require(folder.is_relative_to(root / "runs"), "run must be inside the selected repository")
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        require(status.get("status") == "finished" and status.get("generation") == 10
                and status.get("completed_families") == status.get("expected_families") == 128,
                f"{folder.name} must finish training and evaluation before plotting")
        folders.append(folder)

    series = []
    for index, folder in enumerate(folders):
        label = f"A{index}"
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        require(config.get("protocol") == summary.get("protocol") == PROTOCOL
                and config.get("condition_split") == summary.get("condition_split") == "final"
                and config.get("controller") == "rules-v0" and config.get("genome_size") == 155
                and config.get("seed") == 2026112201 + index
                and config.get("expected_generations") == 10
                and config.get("source", {}).get("label") == label,
                f"{label} configuration differs from the registered protocol")
        require(summary.get("status") == "finished" and summary.get("artifact_validity") == "passed"
                and summary.get("scientific_result") is True and summary.get("generation") == 10
                and summary.get("completed_families") == summary.get("expected_families") == 128
                and summary.get("source_label") == label,
                f"{label} requires a complete valid final summary")

        source = (root / config["source"]["path"]).resolve()
        require(source.is_relative_to(root / "runs"), "original genome must be a recorded run artifact")
        source_bytes = source.read_bytes()
        require(hashlib.sha256(source_bytes).hexdigest() == config["source"]["sha256"],
                f"{label} original genome hash differs from its recorded source")
        require((folder / "genome_start.npy").read_bytes() == source_bytes,
                f"{label} starting genome differs from the original")
        original = np.load(io.BytesIO(source_bytes), allow_pickle=False)
        trajectory = np.load(folder / "genome_trajectory.npy", allow_pickle=False)
        rms, probability = genome_metrics(trajectory)
        require(np.array_equal(trajectory[0], original), f"{label} generation zero differs from its parent")
        final = np.load(folder / "genome_final.npy", allow_pickle=False)
        require(np.array_equal(trajectory[-1], final), f"{label} final genome differs from the trajectory")

        manifests = summary.get("generations", [])
        require(len(manifests) == 11, f"{label} requires all 11 generation manifests")
        for generation, manifest in enumerate(manifests):
            relative = f"generations/generation_{generation:04d}.npz"
            require(manifest.get("path") == relative, f"{label} generation order differs from its manifest")
            content = (folder / relative).read_bytes()
            require(hashlib.sha256(content).hexdigest() == manifest.get("sha256"),
                    f"{label} generation {generation} hash mismatch")
            with np.load(io.BytesIO(content), allow_pickle=False) as checkpoint:
                require(checkpoint["t"].shape == () and int(checkpoint["t"]) == generation
                        and np.array_equal(checkpoint["mu"], trajectory[generation]),
                        f"{label} generation {generation} differs from its immutable checkpoint")
        series.append(dict(label=label, training_seed=config["seed"],
                           run=folder.relative_to(root).as_posix(),
                           rms=rms, probability=probability))
    return series


def plot(series, root, output_dir):
    cache = root / ".tmp" / "mplconfig"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.build_readme_figures import INK, MUTED, TEAL, BLUE, ORANGE, style_axes

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "text.color": INK, "axes.titlecolor": INK,
                         "svg.hashsalt": STEM})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(left=.09, right=.97, bottom=.24, top=.76, wspace=.32)
    fig.suptitle("E118 genomes across ten generations", x=.09, ha="left", y=.98,
                 fontsize=18, fontweight="bold")
    generations = np.arange(11)
    for row, color, marker in zip(series, (TEAL, BLUE, ORANGE), ("o", "s", "^")):
        label = f"{row['label']} · seed {row['training_seed']}"
        for ax, metric in zip(axes, ("rms", "probability")):
            ax.plot(generations, row[metric], color=color, marker=marker, markersize=4.5,
                    linewidth=1.8, label=label)
    for ax in axes:
        style_axes(ax)
        ax.set(xlim=(-.2, 10.2), xticks=generations, xlabel="Evolution generation")
    axes[0].set_title("Distance from each original genome", loc="left", pad=13)
    axes[0].set_ylabel("RMS parameter change\n(raw genome units)", labelpad=10)
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("Sensor-to-output wiring probability", loc="left", pad=13)
    axes[1].set_ylabel("Connection probability (%)", labelpad=10)
    values = np.concatenate([row["probability"] for row in series])
    pad = max(float(np.ptp(values)) * .12, .03)
    axes[1].set_ylim(max(0., values.min() - pad), min(100., values.max() + pad))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(.53, .91), frameon=False)
    caption = [
        "Each marker is one saved population-mean genome. Generation 0 is that lineage's original E103 parent.",
        "RMS uses all 155 raw parameters without rescaling. The wiring curve is sigmoid(conn[4, 0]).",
        "A larger displacement means greater genetic change. Held-out shelter and survival tests measure performance."]
    for y, line in zip((.13, .085, .04), caption):
        fig.text(.09, y, line, fontsize=9.2, color=MUTED)
    description = " ".join(caption) + " Sources: " + ", ".join(row["run"] for row in series)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("svg", "png"):
        path = output_dir / f"{STEM}.{extension}"
        metadata = dict(Title="E118 genome evolution", Description=description)
        metadata.update(dict(Date=None, Creator="Seed / build_e118_genome_figure.py")
                        if extension == "svg" else dict(Software="Seed / build_e118_genome_figure.py"))
        fig.savefig(path, dpi=170, facecolor="white", metadata=metadata)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def self_check():
    trajectory = np.zeros((11, 155), np.float32)
    trajectory[1:] = 2
    rms, probability = genome_metrics(trajectory)
    assert rms[0] == 0 and np.array_equal(rms[1:], np.full(10, 2.))
    assert probability[0] == 50 and np.allclose(probability[1:], 100 / (1 + np.exp(-2.)))
    trajectory[1:] = 0
    trajectory[1:, 20] = -2
    rms, probability = genome_metrics(trajectory)
    assert np.allclose(rms[1:], 2 / np.sqrt(155)) and np.all(probability[1:] < 50)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository containing the completed E118 runs")
    parser.add_argument("--output-dir", type=Path, help="defaults to <root>/results/figures")
    parser.add_argument("--self-check", action="store_true", help="check arithmetic with synthetic arrays only")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("Synthetic arithmetic checks passed; no experiment files read.")
        return
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "results" / "figures").resolve()
    require(not output_dir.is_relative_to(root / "runs"), "figure outputs must be outside immutable runs")
    series = load_series(root)
    outputs = plot(series, root, output_dir)
    print(json.dumps(dict(figures=outputs, lineages=[row["label"] for row in series],
                          states_per_lineage=11, data="saved genome parameters")))


if __name__ == "__main__":
    main()
