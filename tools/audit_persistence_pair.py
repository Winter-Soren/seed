"""Read-only NumPy audit of completed, registered persistence development checks."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ecology_evidence", ROOT / "tools/audit_ecology_fixture.py")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
require, same, subset = base.require, base.same, base.subset
TRAIT = dict(version="persistence-v1", memory_steps=4, stall_steps=2)
FLAGS = ("active", "remembered", "progress_observed", "stalled", "tactic_attempted", "action_switched")


def guard(path, source_commit):
    data = path.read_bytes()
    records = [json.loads(line) for line in data.decode("utf-8-sig").splitlines()]
    terminal = records[-1]
    require(terminal.get("event") == "finished" and terminal.get("exitCode") == 0
            and 0 <= terminal.get("wallSeconds", 121) <= 120, "successful bounded terminal guard required")
    require(not any(row.get("event") == "stopped" for row in records), "guard recorded a stop")
    preflight = [row for row in records if row.get("event") == "preflight"]
    require(len(preflight) == 1 and preflight[0]["commit"] == source_commit, "guard source commit")
    return dict(sha256=base.sha(data), terminal=terminal, source_commit=source_commit,
                arguments=preflight[0]["arguments"])


def load(auditor, folder, trait):
    summary = auditor.json(folder + "/summary.json")
    status = auditor.json(folder + "/status.json")
    require(summary.get("status") == status.get("status") == "finished", "incomplete fixture")
    subset(summary, dict(protocol=base.PROTOCOL, condition_split="development", scientific_result=False,
           completed_cohorts=3, expected_cohorts=3, memory_condition="shared"), "fixture summary")
    config = auditor.json(folder + "/config.json")
    require(config["git"]["dirty"] is False, "fixture source must be committed")
    same(config["cfg"], base.CFG, "registered development configuration")
    require(config.get("persistence") == summary.get("persistence") == trait, "trait binding")
    start = auditor.read(folder + "/genome_start.npy")
    require(base.sha(start) == config["source"]["sha256"] == base.ORIGINAL_SHA, "original A0 bytes")
    history = [json.loads(line) for line in auditor.read(folder + "/metrics.jsonl").splitlines()]
    same(summary["observations"], history, "summary/history equality")
    require(len(history) == 3, "three completed records required")
    rows = [auditor.bundle(folder, record["raw_files"], f"cohort_{i:03d}") for i, record in enumerate(history)]
    final = auditor.bundle(folder, summary["final_memory"], "lineage_final")
    require(final["completed_lives"] == 3, "three final lives")
    flat = np.load(io.BytesIO(start), allow_pickle=False)
    same(np.concatenate([final["genome"][key].ravel() for key in sorted(final["genome"])]),
         flat, "unchanged original A0 runtime genome")
    digest = hashlib.sha256()
    for key in sorted(final["genome"]):
        value = final["genome"][key]
        digest.update(key.encode() + value.dtype.str.encode() + str(value.shape).encode() + value.tobytes())
    require(final["archive_state"]["genome_sha256"] == digest.hexdigest(), "archive genome binding")
    return dict(config=config, history=history, rows=rows, final=final, summary=summary)


def activity(raw, label):
    events = raw["persistence"]
    require(set(events) == set(FLAGS) | {"previous_action", "selected_action"}, label + " fields")
    values = {name: np.asarray(events[name]) for name in FLAGS}
    require(all(x.shape == (12, 2) and x.dtype == bool for x in values.values()), label + " binary shape")
    alive = raw["events"]["alive_before"].astype(bool)
    sheltered = np.concatenate((raw["ecology"]["initial_sheltered"][None], raw["ecology"]["sheltered"][:-1]))
    eligible = alive & ~sheltered & (np.arange(12)[:, None] >= 6)
    require(all(not np.any(value & ~eligible) for value in values.values()), label + " inactive eligibility")
    active = values["active"]
    require(all(not np.any(value & ~active) for value in values.values()), label + " activity subsets")
    require(not np.any(values["remembered"] & (values["progress_observed"] | values["stalled"]))
            and not np.any(values["progress_observed"] & values["stalled"]), label + " cue/progress consistency")
    require(not np.any(values["tactic_attempted"] & ~values["stalled"]), label + " attempt requires stall")
    previous, selected = (np.asarray(events[name]) for name in ("previous_action", "selected_action"))
    require(all(x.shape == (12, 2) and np.issubdtype(x.dtype, np.integer) and np.isin(x, range(5)).all()
                for x in (previous, selected)), label + " action ranges")
    same(previous[0], np.zeros(2, np.int32), label + " initial action")
    same(previous[1:], selected[:-1], label + " previous sampled actions")
    same(values["action_switched"], values["tactic_attempted"] & (selected != previous), label + " realized switches")
    for agent in range(2):
        attempts = 0
        for step in range(12):
            attempts = attempts + int(values["tactic_attempted"][step, agent]) if active[step, agent] else 0
            require(attempts <= 2, label + " two attempts per active episode")
    return {"persistence_" + name + "_agent_steps": int(value.sum()) for name, value in values.items()}


def check_lineage(data, trait):
    previous = np.zeros((6, 49), np.float32)
    world_keys, birth_keys, output = [], [], []
    for i, (row, record) in enumerate(zip(data["rows"], data["history"])):
        label = f"cohort {i}"
        require(row.get("persistence") == trait and row["life"] == i and row["rule"] == 0, label + " settings")
        same(row["agent_ids"], np.column_stack((np.full(2, i), np.arange(2))), label + " identities")
        world_keys.append(row["world_key"])
        birth_keys.append(row["development_key"])
        shared, empty = row["inherited"], row["fresh"]
        same(shared["weights"]["birth_W_sha256"], empty["weights"]["birth_W_sha256"], label + " birth signatures")
        same(shared["ecology"]["shelter"], empty["ecology"]["shelter"], label + " shelter")
        same(shared["ecology"]["initial_sheltered"], empty["ecology"]["initial_sheltered"], label + " initial occupancy")
        delta, mask = row["donor_delta"], row["donor_mask"]
        require(delta.shape == mask.shape == (2, 6, 49) and np.isin(mask, (0, 1)).all()
                and np.all(delta[mask == 0] == 0), label + " donor edges")
        same(row["donor_counts"], mask.sum(0), label + " donor counts")
        same(row["archive_before"], previous, label + " archive continuity")
        previous = np.clip(previous + delta.sum(0) / np.maximum(mask.sum(0), 1), -3., 3.)
        same(row["archive_after"], previous, label + " archive update")
        checkpoint = row["archive_state"]
        same(checkpoint["archive"], previous, label + " checkpoint archive")
        same(checkpoint["used_world_keys"], np.asarray(world_keys), label + " world history")
        same(checkpoint["used_development_keys"], np.asarray(birth_keys), label + " birth history")
        require(checkpoint["next_life"] == i + 1 and checkpoint.get("persistence") == trait, label + " checkpoint binding")
        pair = {}
        for name, arm in base.ARMS.items():
            raw = row[arm]
            obs, details = base.observations(raw, label + "/" + name, *base.COSTS[i])
            if trait:
                obs.update(activity(raw, label + "/" + name))
            else:
                require("persistence" not in raw, label + " off diagnostics absent")
            same(record["paired"][name], obs, label + "/" + name + " computed metrics")
            pair[name] = dict(obs, deaths=details["deaths"], per_agent_return=details["per_agent_return"])
        same(record["selected"], record["paired"]["shared"], label + " selected record")
        output.append(dict(cohort=i, phase=base.PHASES[i], paired=pair,
                           archive_nonzero=int(np.count_nonzero(previous))))
    for values in (world_keys, birth_keys):
        require(all(x.shape == (2,) and x.dtype == np.uint32 for x in values)
                and len(np.unique(values, axis=0)) == 3, "fresh cohort keys")
    same(data["final"]["final_archive"], previous, "final archive")
    same(data["final"]["archive_state"], data["rows"][-1]["archive_state"], "final checkpoint")
    require(data["final"].get("persistence") == trait, "final trait binding")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--off", required=True)
    parser.add_argument("--on", required=True)
    parser.add_argument("--off-guard", type=Path, required=True)
    parser.add_argument("--on-guard", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    require(len(args.source_commit) == 40 and all(ch in "0123456789abcdef" for ch in args.source_commit), "full source commit")
    guards = {"off": guard(args.off_guard, args.source_commit), "on": guard(args.on_guard, args.source_commit)}
    auditor = base.Auditor(args.root)
    off, on = load(auditor, args.off, None), load(auditor, args.on, TRAIT)
    for mode, data in (("off", off), ("on", on)):
        require(data["config"]["git"]["commit"] == args.source_commit, "registered source")
        require(("--persistence" in guards[mode]["arguments"]) == (mode == "on"), "guard trait argument")
        require(data["config"]["args"].get("persistence", False) == (mode == "on"), "CLI trait argument")
    require(auditor.git("cat-file", "-t", args.source_commit) == "commit", "source commit exists")
    old = load(auditor, "runs/20261121-062635_ecology-transfer-smoke-a0", None)
    off_config, on_config = (dict(value["config"]) for value in (off, on))
    for cfg in (off_config, on_config):
        cfg.pop("persistence", None)
        cfg["args"] = dict(cfg["args"])
        cfg["args"].pop("name")
        cfg["args"].pop("persistence", None)
    same(on_config, off_config, "only trait and run name differ in configuration")
    output = {"off": check_lineage(off, None), "on": check_lineage(on, TRAIT)}
    for i in range(3):
        a, b, original = off["rows"][i], on["rows"][i], old["rows"][i]
        for field in ("agent_ids", "world_key", "development_key", "rule", "ecology", "donor_mask"):
            same(a[field], b[field], f"cohort {i} paired {field}")
            same(a[field], original[field], f"cohort {i} historical {field}")
        for arm in base.ARMS.values():
            for field in ("events", "ecology", "weights"):
                same(a[arm][field], original[arm][field], f"cohort {i}/{arm} historical {field}")
            same(a[arm]["weights"]["birth_W_sha256"], b[arm]["weights"]["birth_W_sha256"], "on/off raw birth")
            same(a[arm]["ecology"]["initial_sheltered"], b[arm]["ecology"]["initial_sheltered"], "on/off initial shelter")
            same(a[arm]["ecology"]["shelter"], b[arm]["ecology"]["shelter"], "on/off shelter map")
    same(off["final"]["genome"], on["final"]["genome"], "same final rule genome")
    for relative, digest in auditor.hashes.copy().items():
        require(base.sha(auditor.read(relative)) == digest, "artifact changed during audit")
    print(json.dumps(dict(status="passed", scientific_result=False, helper_sha256=base.sha(Path(__file__).read_bytes()),
        registered_source=off["config"]["git"]["commit"],
        guards=guards, cohorts=output, checked_artifact_sha256=auditor.hashes,
        limits=["One development seed, no performance inference or standard error.",
                "Initial positions, full topology and local cue patches were not saved; keys, birth signatures and occupancy are checked.",
                "Remembered activity is a recorded flag; cue disappearance cannot be independently reconstructed from these artifacts."]), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
