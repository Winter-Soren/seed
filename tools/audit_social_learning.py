"""Read-only E120 audit of demonstration inputs and paired solo outcomes.

Only NumPy and existing evidence helpers are imported. Final mode waits for all
three complete panels before opening their scientific outcomes.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_cold_donor as base


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "social-learning-v1"
CFG = dict(base.CFG, M=1)
ARMS = ("observed", "yoked", "no_demo", "scene_only")
CONTROLS = ARMS[1:]
KEY_ORDER = ["teacher_world", "teacher_birth", "solo_world", "pupil_birth"]
require, same, subset, typed = base.require, base.same, base.subset, base.typed


def fingerprint(value):
    return np.frombuffer(hashlib.sha256(np.asarray(value).tobytes()).digest(), np.uint8)


def events(raw, prefix, steps):
    result = {name: typed(raw, prefix + name, (steps, 1), "biuf", binary=True).astype(bool)
              for name in ("food", "poison", "alive_before", "alive_after")}
    food, poison, before, after = (result[name] for name in ("food", "poison", "alive_before", "alive_after"))
    require(before[0].all() and np.array_equal(before[1:], after[:-1])
            and not np.any(after & ~before) and not np.any(food & poison)
            and not np.any((food | poison) & ~before), prefix + " event/death timing")
    meals = (food | poison)[:, 0]
    first = int(np.flatnonzero(meals)[0]) if meals.any() else steps
    result.update(first_food=bool(food[first, 0]) if first < steps else False,
                  first_poison=bool(poison[first, 0]) if first < steps else False,
                  no_meal=first == steps, first_meal_step=first + 1,
                  fitness=int(food.sum()) - int(poison.sum()), survival=bool(after[-1, 0]))
    return result


def reconstruct_movie(raw, cfg=CFG):
    """Recover the public camera view using only the saved world and visible agent."""
    steps, size, radius = cfg["T"], cfg["G"], cfg["R"]
    watch_start, width = steps // 2, 2 * radius + 1
    initial = typed(raw, "teacher_initial_grid", (size, size), "iu")
    grid = typed(raw, "teacher_grid", (steps, size, size), "iu")
    initial_position = typed(raw, "teacher_initial_position", (1, 2), "iu")
    position = typed(raw, "teacher_position", (steps, 1, 2), "iu")
    signal = typed(raw, "teacher_signal", (steps, 1), "f")
    energy = typed(raw, "teacher_energy", (steps, 1), "f")
    require(np.isin(initial, (0, 1, 2)).all() and np.isin(grid, (0, 1, 2)).all(), "teacher item labels")
    require(np.sum(initial == 1) == cfg["n_items"] and np.sum(initial == 2) == cfg["n_items"],
            "teacher initial item counts")
    require(np.all((initial_position >= 0) & (initial_position < size))
            and np.all((position >= 0) & (position < size)), "teacher position bounds")
    require(np.all(np.abs(signal) <= 1) and np.all((energy >= 0) & (energy <= cfg["e_max"])),
            "teacher signal/energy bounds")
    outcome = events(raw, "teacher_", steps)
    before, after = outcome["alive_before"], outcome["alive_after"]
    pre_grid = np.concatenate((initial[None], grid[:-1]), axis=0)
    pre_pos = np.concatenate((initial_position[None], position[:-1]), axis=0)
    pre_sig = np.concatenate((np.zeros((1, 1), signal.dtype), signal[:-1]), axis=0)
    pre_energy = np.concatenate((np.full((1, 1), cfg["e0"], energy.dtype), energy[:-1]), axis=0)
    item = pre_grid[np.arange(steps), position[:, 0, 0], position[:, 0, 1]]
    rule = int(raw["rule"])
    require(np.array_equal(outcome["food"][:, 0], before[:, 0] & (item == 1 + rule))
            and np.array_equal(outcome["poison"][:, 0], before[:, 0] & (item == 2 - rule)),
            "teacher meal classification")
    expected_energy = np.clip(pre_energy - np.float32(cfg["decay"]) * before
                              + np.float32(cfg["gain"]) * outcome["food"]
                              - np.float32(cfg["gain"]) * outcome["poison"], 0, cfg["e_max"])
    require(np.allclose(energy, expected_energy, atol=3e-7, rtol=2e-6)
            and np.array_equal(after, before & (energy > 0)), "teacher energy and survival")
    displacement = (position - pre_pos) % size
    permitted = np.array([[0, 0], [size - 1, 0], [1, 0], [0, size - 1], [0, 1]])
    require(np.all(np.any(np.all(displacement[:, 0, None, :] == permitted, axis=2), axis=1))
            and not np.any(displacement[~before] != 0), "teacher movement")
    require(not np.any(signal[~before] != 0), "teacher signals after death")
    frames, cameras, visible = [], [], []
    offset = np.arange(-radius, radius + 1)
    for turn in range(watch_start, steps):
        camera = (pre_pos[max(turn - 1, 0), 0] + np.array([1, 0])) % size
        rows, cols = (camera[0] + offset) % size, (camera[1] + offset) % size
        patch = pre_grid[turn][np.ix_(rows, cols)]
        occupancy = ((rows[:, None] == pre_pos[turn, 0, 0])
                     & (cols[None, :] == pre_pos[turn, 0, 1])) & before[turn, 0]
        frame = np.concatenate(((patch == 1).ravel(), (patch == 2).ravel(), occupancy.ravel(),
                                (occupancy * pre_sig[turn, 0]).ravel(),
                                np.array([cfg["e0"] - 1., 0., 1.])))
        frames.append(frame.astype(np.float32))
        cameras.append(camera)
        visible.append(bool(occupancy.any()))
    frames = np.stack(frames)
    require(frames.shape == (steps - watch_start, 4 * width * width + 3), "movie dimensions")
    return frames, np.stack(cameras), np.asarray(visible), outcome


def control_frames(movie, swap, radius):
    pixels = (2 * radius + 1) ** 2
    yoked = movie.copy()
    if swap:
        yoked[:, :pixels] = movie[:, pixels:2 * pixels]
        yoked[:, pixels:2 * pixels] = movie[:, :pixels]
    blank = movie.copy()
    blank[:, :4 * pixels] = 0
    scene = movie.copy()
    scene[:, 2 * pixels:4 * pixels] = 0
    return dict(observed=movie, yoked=yoked, no_demo=blank, scene_only=scene)


def family_events(raw, index, cfg=CFG):
    steps, radius, size, hidden = cfg["T"], cfg["R"], cfg["G"], cfg["H"]
    start, pixels = steps // 2, (2 * radius + 1) ** 2
    sensory, neurons = 4 * pixels + 3, 4 * pixels + 3 + hidden + 6
    require(all(np.isfinite(value).all() for value in raw.values()), "nonfinite raw family")
    for field, expected in (("family_index", index), ("steps", steps), ("watch_start", start), ("agents", 1),
                            ("radius", radius), ("grid_size", size), ("rule", index % 2),
                            ("hidden_units", hidden), ("yoked_swap", index // 2 % 2)):
        require(int(typed(raw, field, (), "biu")) == expected, field + " differs from registration")
    require(float(typed(raw, "initial_energy", (), "f")) == cfg["e0"], "newborn energy")
    keys = typed(raw, "keys", (4, 2), "u")
    require(keys.dtype == np.uint32 and len({key.tobytes() for key in keys}) == 4, "distinct family role keys")
    solo_grid = typed(raw, "solo_initial_grid", (size, size), "iu")
    solo_position = typed(raw, "solo_initial_position", (1, 2), "iu")
    require(np.isin(solo_grid, (0, 1, 2)).all()
            and np.sum(solo_grid == 1) == cfg["n_items"] and np.sum(solo_grid == 2) == cfg["n_items"]
            and np.all((solo_position >= 0) & (solo_position < size)), "solo initial world")
    movie, cameras, visible, teacher = reconstruct_movie(raw, cfg)
    require(np.array_equal(typed(raw, "camera_positions", (steps - start, 2), "iu"), cameras), "camera movement")
    expected_inputs = control_frames(movie, bool(index // 2 % 2), radius)
    result = dict(teacher=teacher, visible_frames=int(visible.sum()),
                  informative_changed_frames=int(np.any(expected_inputs["observed"] != expected_inputs["yoked"], axis=1).sum()))
    birth = None
    initial_states = {}
    zeros = dict(x=fingerprint(np.zeros(neurons, np.float32)),
                 Wp=fingerprint(np.zeros((sensory, hidden), np.float32)))
    for arm in ARMS:
        prefix = arm + "_"
        frame = typed(raw, prefix + "frames", movie.shape, "f")
        require(frame.dtype == np.float32 and np.array_equal(frame, expected_inputs[arm]), arm + " observer inputs")
        motor = typed(raw, prefix + "watch_motor", (steps - start, 5), "bifu")
        require(np.array_equal(motor, np.tile([1, 0, 0, 0, 0], (steps - start, 1))), arm + " passive motor clamp")
        require(not typed(raw, prefix + "watch_denergy", (steps - start,), "f").any(), arm + " observer own reward")
        current_birth = typed(raw, prefix + "birth_sha256", (32,), "u")
        require(current_birth.dtype == np.uint8, arm + " birth hash dtype")
        birth = current_birth if birth is None else birth
        require(np.array_equal(current_birth, birth), "observer birth mismatch")
        for state in ("W", "x", "Wp"):
            initial = typed(raw, prefix + f"watch_initial_{state}_sha256", (32,), "u")
            final = typed(raw, prefix + f"watch_final_{state}_sha256", (32,), "u")
            solo = typed(raw, prefix + f"solo_initial_{state}_sha256", (32,), "u")
            require(initial.dtype == final.dtype == solo.dtype == np.uint8, arm + " state fingerprint dtype")
            initial_states.setdefault(state, initial)
            require(np.array_equal(initial, initial_states[state]) and np.array_equal(final, solo),
                    arm + " observer state continuity")
            if state in zeros:
                require(np.array_equal(initial, zeros[state]), arm + " newborn neural reset")
        require(np.array_equal(raw[prefix + "watch_initial_W_sha256"], current_birth), arm + " begins with own birth weights")
        for field, expected in (("energy", cfg["e0"]), ("alive", 1), ("denergy", 0)):
            require(np.array_equal(typed(raw, prefix + "solo_initial_" + field, (1,), "bifu"), [expected]),
                    arm + " solo body reset " + field)
        outcome = events(raw, prefix, steps)
        for field in ("first_food", "first_poison", "no_meal"):
            require(bool(typed(raw, prefix + field, (), "biu", binary=True)) == outcome[field], prefix + field)
        require(int(typed(raw, prefix + "first_meal_step", (), "iu")) == outcome["first_meal_step"]
                and float(typed(raw, prefix + "fitness", (), "fiu")) == outcome["fitness"], arm + " meal summary")
        outcome["weight_change_count"] = int(typed(raw, prefix + "weight_change_count", (), "iu"))
        require(0 <= outcome["weight_change_count"] <= neurons * neurons, arm + " changed weights count")
        result[arm] = outcome
    for field in ("teacher_birth_sha256", "teacher_terminal_sha256"):
        require(typed(raw, field, (32,), "u").dtype == np.uint8, field + " dtype")
    return result


def scientific_gate(first_food, survival):
    decided = first_food["se"] is not None and survival["se"] is not None
    first_pass = bool(first_food["mean"] > 2 * first_food["se"]) if decided else None
    survival_pass = bool(survival["mean"] >= -2 * survival["se"]) if decided else None
    return dict(passed=first_pass and survival_pass if decided else None,
                first_food_passed=first_pass, survival_passed=survival_pass)


def measurements(rows):
    """World families are the independent units, with every assigned pupil retained."""
    count = len(rows)
    require(count > 0, "empty sample")
    by_arm = {}
    for arm in ARMS:
        outcomes = [row[arm] for row in rows]
        by_arm[arm] = dict(first_food=base.world_stats([row["first_food"] for row in outcomes]),
            first_food_count=sum(row["first_food"] for row in outcomes),
            first_poison_count=sum(row["first_poison"] for row in outcomes),
            no_meal_count=sum(row["no_meal"] for row in outcomes),
            survivors=sum(row["survival"] for row in outcomes),
            survival=base.world_stats([row["survival"] for row in outcomes]),
            fitness=base.world_stats([row["fitness"] for row in outcomes]),
            changed_weights=base.world_stats([row["weight_change_count"] for row in outcomes]))
    contrasts = {}
    for control in CONTROLS:
        first = base.world_stats([int(row["observed"]["first_food"]) - int(row[control]["first_food"]) for row in rows])
        survival = base.world_stats([int(row["observed"]["survival"]) - int(row[control]["survival"]) for row in rows])
        contrasts[control] = dict(first_food_difference=first, survival_difference=survival,
                                  gate=scientific_gate(first, survival))
    steps = len(rows[0]["teacher"]["food"])
    teacher = {}
    for phase, bounds in (("practice", (0, steps // 2)), ("demonstration", (steps // 2, steps))):
        food = np.array([row["teacher"]["food"][slice(*bounds)].sum() for row in rows])
        poison = np.array([row["teacher"]["poison"][slice(*bounds)].sum() for row in rows])
        teacher[phase] = dict(food=base.world_stats(food), poison=base.world_stats(poison),
            accuracy=base.ratio_stats(food, food + poison), encounters=int((food + poison).sum()))
    teacher.update(survival=base.world_stats([row["teacher"]["survival"] for row in rows]),
                   visible_frames=base.world_stats([row["visible_frames"] for row in rows]))
    return dict(families=count, assigned_newborns=count, arms=by_arm, contrasts=contrasts, teacher=teacher)


class Auditor(base.Auditor):
    def register_keys(self, keys, label):
        keys = np.asarray(keys)
        require(keys.ndim == 2 and keys.shape[1] == 2 and keys.dtype == np.uint32, label + " key shape")
        tokens = [key.tobytes() for key in keys]
        require(len(set(tokens)) == len(tokens) and not self.used_keys.intersection(tokens), label + " reused keys")
        self.used_keys.update(tokens)

    def completed_bundle(self, development=False):
        names = ["E120-benchmark"] if development else [f"E120-A{i}" for i in range(3)]
        count = 4 if development else 128
        folders, pending = [], []
        for name in names:
            candidates = sorted(self.path("runs").glob("*_" + name))
            require(len(candidates) <= 1, name + " duplicate canonical attempts")
            if not candidates or not (candidates[0] / "status.json").is_file():
                pending.append(name)
                continue
            folder = candidates[0].relative_to(self.root).as_posix()
            status = self.json(folder + "/status.json")
            files = {path.name for path in self.path(folder + "/families").glob("*.npz")}
            if (status.get("status") != "finished" or status.get("completed_families") != count
                    or status.get("expected_families") != count
                    or files != {f"family_{i:04d}.npz" for i in range(count)}
                    or not self.path(folder + "/summary.json").is_file()):
                pending.append(name)
                continue
            folders.append(folder)
        if pending:
            raise base.Incomplete(", ".join(pending) + " incomplete; no outcomes opened")
        return folders

    def config(self, folder, index, development):
        config = self.json(folder + "/config.json")
        count, master = (4, 47) if development else (128, 2026112601 + index)
        split, name = ("development", "E120-benchmark") if development else ("final", f"E120-A{index}")
        source_path, digest = base.ORIGINALS[index]
        subset(config, dict(protocol=PROTOCOL, cfg=CFG, cfg_overrides={}, condition_split=split,
            seed=master, eval_seed=master, eval_worlds=count, expected_families=count,
            practice_steps=150, watch_steps=150, solo_steps=300, key_order=KEY_ORDER, arms=list(ARMS),
            controls=list(CONTROLS), ecology=None, persistence="off", plastic=True,
            controller="rules-v0", genome_size=155, generation=0, raw_schema=PROTOCOL,
            max_seconds=900 if development else 3600), name + " config")
        subset(config["args"], dict(name=name, eval_protocol=PROTOCOL, condition_split=split,
            seed=master, generations=0, eval_worlds=count, start="file:" + source_path,
            no_plastic=False, allow_dirty=False, quick=False, resume=None, cfg=None,
            objective=None, memory_condition=None, max_train_seconds=None), name + " arguments")
        require(config["args"].get("persistence", False) is False
                and config["args"].get("controller") in (None, "rules-v0")
                and config["args"].get("structural_plasticity") in (None, "off"), name + " mechanism flags")
        subset(config["source"], dict(label=f"A{index}", path=source_path, sha256=digest), name + " source")
        provenance = config["git"]
        require(provenance.get("dirty") is False and len(provenance.get("commit", "")) == 40
                and self.code_trees(provenance["commit"]) == self.expected_trees, name + " clean source")
        original = self.read(source_path)
        require(hashlib.sha256(original).hexdigest() == digest
                and self.read(folder + "/genome_start.npy") == original, name + " original genome bytes")
        genome = np.load(io.BytesIO(original), allow_pickle=False)
        require(genome.shape == (155,) and genome.dtype == np.float32 and np.isfinite(genome).all(), name + " genome")
        return config

    def run(self, folder, index, development=False):
        config = self.config(folder, index, development)
        summary = self.json(folder + "/summary.json")
        count = config["expected_families"]
        subset(summary, dict(status="finished", protocol=PROTOCOL, condition_split=config["condition_split"],
            generation=0, completed_families=count, expected_families=count, source_label=f"A{index}",
            eval_seed=config["seed"], eval_worlds=count, artifact_validity="passed",
            scientific_result=not development), folder + " summary")
        require(0 < summary["elapsed_seconds"] <= config["max_seconds"], "wall limit")
        lines = self.read(folder + "/metrics.jsonl").decode("utf-8").splitlines()
        require(len(lines) == count and len(summary["raw_files"]) == count, "family accounting")
        rows, elapsed = [], 0.
        for number, line in enumerate(lines):
            path = f"families/family_{number:04d}.npz"
            raw = self.arrays(folder + "/" + path)
            artifact = dict(index=number, path=path, sha256=self.hashes[folder + "/" + path])
            same(summary["raw_files"][number], artifact, "raw manifest")
            row = family_events(raw, number)
            self.register_keys(raw["keys"], "social learning family")
            record = json.loads(line)
            subset(record, dict(family=number, rule=number % 2, yoked_swap=number // 2 % 2,
                keys=raw["keys"].tolist(), raw_file=artifact), "family record")
            require(elapsed <= record["elapsed_seconds"] <= summary["elapsed_seconds"], "family elapsed order")
            elapsed = record["elapsed_seconds"]
            rows.append(row)
        estimates = measurements(rows)
        same(summary["measurements"], estimates, "independent measurement reconstruction")
        report = dict(run=folder, source_commit=config["git"]["commit"], measurements=estimates,
                      demonstration=dict(visible_frames=sum(row["visible_frames"] for row in rows),
                          yoked_changed_frames=sum(row["informative_changed_frames"] for row in rows),
                          teacher_alive_at_demo_start=sum(bool(row["teacher"]["alive_before"][150, 0]) for row in rows)))
        if development:
            require(summary["gate_status"] == "not_applicable" and "gate" not in summary, "benchmark gate")
        else:
            gates = [entry["gate"] for entry in estimates["contrasts"].values()]
            gate = dict(passed=all(x["passed"] for x in gates),
                        first_food_passed=all(x["first_food_passed"] for x in gates),
                        survival_passed=all(x["survival_passed"] for x in gates))
            same(summary["gate"], gate, "three-control primary gate")
            same(summary["gate_status"], "passed" if gate["passed"] else "failed", "scientific gate status")
            report["gate"] = gate
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    auditor, report = None, dict(protocol=PROTOCOL, mode="development" if args.development else "final")
    try:
        auditor = Auditor(args.root, args.source_commit)
        folders = auditor.completed_bundle(args.development)
        if not args.development:
            report["development_audit"] = auditor.run(auditor.completed_bundle(True)[0], 0, True)
        report["runs"] = [auditor.run(folder, index, args.development) for index, folder in enumerate(folders)]
        report.update(status="passed", registered_source=args.source_commit,
                      registered_code_trees=auditor.expected_trees, final_evaluation_audited=not args.development,
                      limitations=["World physics, neural updates and random-key generation are not replayed.",
                          "The public movie is reconstructed. Neural transfer is checked through recorded state hashes.",
                          "Teachers receive practice; dead or uninformative demonstrations remain assigned.",
                          "A failed gate limits these saved genomes and this observation route."])
        if not args.development:
            report["scientific_gate_passed"] = all(run["gate"]["passed"] for run in report["runs"])
        code = 0
    except base.Incomplete as error:
        report.update(status="incomplete", error=str(error), final_evaluation_audited=False)
        code = 2
    except (base.AuditError, OSError, ValueError, KeyError, TypeError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", final_evaluation_audited=False)
        code = 1
    if auditor is not None:
        report["checked_file_sha256"] = auditor.hashes
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
