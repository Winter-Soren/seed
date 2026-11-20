"""Read-only E115 artifact audit; imports no Seed/JAX/evaluator code.

Run --inventory before all results exist. The default audits all six completed
training runs and the final EVAL v5 bundle. JSON goes only to stdout. Exit 0 is a
completed audit (or inventory), 2 means missing/incomplete evidence, 1 a mismatch.
No simulation, inference, optimizer replay, file write, or artifact modification.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

SOURCE = "27dfb88f25185c891e6bac57d2fcd86079ea65a0"
INTERRUPTED_RUN = "runs/20261119-081846_E115-v5"
RECOVERY_NAME = "E115-v5-recovery1"
ORIGINALS = (
    ("runs/20261112-023245_E103-A-blank-s0", "477d9d03c713dcab8072349ff9155813e5a384d3709eff67595a342b55111f53"),
    ("runs/20261112-050319_E103-A-blank-s1", "35196cd30c2de5c1ed5bb9e23ce445ff6ad13bd1a33bdbf7a77f54d53edce899"),
    ("runs/20261112-102755_E103-A-blank-s2", "f299069753636a303f6ba7f91be78099e8491b20a7511bdcef47fd8b4c5bbaa6"),
)
CFG = dict(G=16, M=4, T=300, R=2, n_items=24, H=48, n_htypes=3,
           e0=1., decay=.003, gain=.25, e_max=2., micro=2, wmax=3.)
REFERENCES = ("fixed_A", "fixed_B", "mixture", "random_walk", "greedy")
GENOMES = tuple(f"{arm}{i}" for i in range(3) for arm in ("start", "mean", "min"))
PANELS = GENOMES + REFERENCES
CONDITIONS = ("stationary_r0", "stationary_r1", "reversal_r0", "reversal_r1")
EVENTS = ("food", "poison", "alive_before", "alive_after")
PAIRING = ("wkeys", "dkeys", "initial_grid", "initial_pos", "agent_ids")
METRICS = ("stationary_fitness", "individual_min_return", "joint_survival",
           "post_flip_return", "reversal_final_survival")


class AuditError(Exception):
    pass


class Incomplete(AuditError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def compare(actual, expected, label):
    """Exact structure/count checks and tight float tolerance for independent sums."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{label}: dictionary fields")
        for key, value in expected.items():
            compare(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{label}: list length")
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, f"{label}[{i}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10),
                f"{label}: {actual!r} != {expected!r}")
    else:
        require(actual == expected, f"{label}: {actual!r} != {expected!r}")


def subset(actual, expected, label):
    require(all(key in actual for key in expected), f"{label}: missing fields")
    compare({key: actual[key] for key in expected}, expected, label)


def stats(vector):
    x = np.asarray(vector, np.float64)
    require(x.ndim == 1 and x.size >= 2 and np.isfinite(x).all(), "invalid independent world vector")
    average = float(x.sum() / x.size)
    error = math.sqrt(float(np.square(x - average).sum()) / (x.size * (x.size - 1)))
    return dict(mean=average, se=error)


def ratio(numerator, denominator):
    n, d = np.asarray(numerator, float), np.asarray(denominator, float)
    if d.sum() == 0:
        return dict(mean=None, se=None)
    value = float(n.sum() / d.sum())
    return dict(mean=value, se=stats((n - value * d) / d.mean())["se"])


def distribution(values):
    unique, count = np.unique(values, return_counts=True)
    return dict(values=unique.tolist(), counts=count.tolist())


def accuracy(food, poison):
    f = food.reshape(256, -1).sum(1, dtype=np.int64)
    p = poison.reshape(256, -1).sum(1, dtype=np.int64)
    result = ratio(f, f+p)
    return dict(accuracy=result["mean"], accuracy_se=result["se"], food=int(f.sum()),
                poison=int(p.sum()), encounters=int((f+p).sum()), contributing_worlds=int(np.count_nonzero(f+p)))


def select_evaluation(audit, requested, interrupted_manifest):
    """Never select a recovery implicitly or accept an unregistered retry name."""
    name = "E115-v5"
    if requested:
        basename = audit.path(requested).name
        if basename.endswith("_"+RECOVERY_NAME):
            name = RECOVERY_NAME
            require(bool(interrupted_manifest), "recovery requires the preserved interruption manifest")
        else:
            require(basename.endswith("_E115-v5"), "unregistered evaluation name; only E115-v5 or explicit E115-v5-recovery1 is allowed")
    require(name == RECOVERY_NAME or not interrupted_manifest, "interruption manifest is only valid for explicit recovery1")
    folders = list(audit.path("runs").glob("*_"+name))
    require(len(folders) <= 1, "duplicate registered "+name+" attempts")
    if not folders:
        raise Incomplete("registered evaluation folder does not exist: "+name)
    if requested:
        require(audit.path(requested) == folders[0], "explicit evaluation is not the unique registered "+name+" folder")
    return folders[0].relative_to(audit.root).as_posix(), name


class Auditor:
    def __init__(self, root, source):
        self.root = root.resolve()
        self.hashes = {}
        self.trees = {}
        self.source = self.git("rev-parse", source)
        self.expected_trees = self.code_trees(self.source)

    def git(self, *args):
        result = subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True, text=True, timeout=15)
        return result.stdout.strip()

    def code_trees(self, commit):
        if commit not in self.trees:
            self.trees[commit] = {name: self.git("rev-parse", f"{commit}:{name}") for name in ("seed", "scripts")}
        return self.trees[commit]

    def path(self, relative):
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root), f"path leaves repository: {relative}")
        return path

    def read(self, relative):
        path = self.path(relative)
        if not path.is_file():
            raise Incomplete(f"missing file: {relative}")
        data = path.read_bytes()
        name = path.relative_to(self.root).as_posix()
        digest = hashlib.sha256(data).hexdigest()
        require(name not in self.hashes or self.hashes[name] == digest, f"file changed during read-only audit: {name}")
        self.hashes[name] = digest
        return data

    def json(self, relative):
        return json.loads(self.read(relative), parse_constant=lambda x: (_ for _ in ()).throw(AuditError(f"nonfinite JSON: {relative}")))

    def arrays(self, relative):
        with np.load(io.BytesIO(self.read(relative)), allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), f"duplicate array names: {relative}")
            return {name: archive[name] for name in archive.files}

    def genome(self, relative):
        data = self.read(relative)
        flat = np.load(io.BytesIO(data), allow_pickle=False)
        require(isinstance(flat, np.ndarray) and flat.shape == (155,) and np.issubdtype(flat.dtype, np.floating)
                and np.isfinite(flat).all(), f"invalid 155-value genome: {relative}")
        return data, flat, hashlib.sha256(np.asarray(flat, np.float32).tobytes()).hexdigest()[:16]

    def verify_interrupted_manifest(self, manifest_path):
        manifest = self.json(manifest_path)
        require(isinstance(manifest, dict) and manifest
                and all(path.startswith(INTERRUPTED_RUN+"/") and isinstance(digest, str) and len(digest) == 64
                        for path, digest in manifest.items()), "invalid original interruption manifest")
        actual = {path.relative_to(self.root).as_posix() for path in self.path(INTERRUPTED_RUN).rglob("*") if path.is_file()}
        require(actual == set(manifest), "interrupted attempt file set changed")
        for path, expected in manifest.items():
            require(hashlib.sha256(self.read(path)).hexdigest() == expected, "interrupted attempt file changed: "+path)
        return manifest

    def provenance(self, config, *, pinned):
        records = [config.get("git", {})] + [x.get("git", {}) for x in config.get("resumes", [])]
        for record in records:
            commit = record.get("commit", "")
            require(record.get("dirty") is False and len(commit) == 40
                    and all(ch in "0123456789abcdef" for ch in commit), "unclean or malformed source commit")
            if pinned:
                require(self.code_trees(commit) == self.expected_trees, f"scientific source differs from registered {self.source}: {commit}")
        return records

    def original(self, i):
        folder, expected_sha = ORIGINALS[i]
        data, flat, genome_hash = self.genome(folder + "/genome_final.npy")
        require(hashlib.sha256(data).hexdigest() == expected_sha, f"original A{i} byte hash")
        config, summary = self.json(folder+"/config.json"), self.json(folder+"/summary.json")
        require(config["cfg"] == CFG and config["genome_size"] == 155, f"original A{i} config")
        subset(config["args"], dict(seed=i, start="blank", generations=200, pop=128, worlds=8,
                                   sigma=.1, lr=.1, no_plastic=False, quick=False, allow_dirty=False), f"original A{i} arguments")
        subset(summary, dict(status="finished", generation=200, final_genome_hash=genome_hash), f"original A{i} final")
        provenance = self.provenance(config, pinned=False)
        metadata = dict(label=f"start{i}", controller="rules-v0", seed=i, run=folder, sha256=expected_sha,
                        genome_hash=genome_hash, genome_size=155, source_commits=provenance,
                        config_sha256=self.hashes[folder+"/config.json"], summary_sha256=self.hashes[folder+"/summary.json"])
        input_metadata = dict(path=folder+"/genome_final.npy", sha256=expected_sha, genome_hash=genome_hash,
                              controller="rules-v0", source_run=folder, source_commits=provenance,
                              source_config_sha256=metadata["config_sha256"], source_summary_sha256=metadata["summary_sha256"])
        return dict(data=data, flat=flat, metadata=metadata, input_metadata=input_metadata)

    def training(self, i, arm, original, *, inventory):
        name = f"E115-{arm}-s{i}"
        folders = sorted(self.path("runs").glob("*_"+name))
        require(len(folders) <= 1, f"duplicate canonical training attempts: {name}")
        if not folders:
            return dict(name=name, status="missing")
        folder = folders[0].relative_to(self.root).as_posix()
        config = self.json(folder+"/config.json")
        objective = "paired-"+arm
        expected_args = dict(name=name, eval_protocol="paired-train-v1", objective=objective,
            condition_split="final", controller="rules-v0", seed=1000+i, generations=50, pop=64,
            worlds=4, sigma=.1, lr=.1, checkpoint_every=10, eval_worlds=256, no_plastic=False,
            quick=False, allow_dirty=False, structural_plasticity="off",
            start="file:"+ORIGINALS[i][0]+"/genome_final.npy")
        subset(config["args"], expected_args, name+" arguments")
        subset(config, dict(cfg=CFG, cfg_overrides={}, controller="rules-v0", genome_size=155,
            eval_protocol="paired-train-v1", objective=objective, evaluation_status="deferred",
            start_genome_hash=original["metadata"]["genome_hash"], input_genome=original["input_metadata"]), name+" config")
        provenance = self.provenance(config, pinned=True)
        require(self.read(folder+"/genome_start.npy") == original["data"], name+" starting bytes")
        optimizer = config["initial_optimizer"]
        subset(optimizer, dict(method="es_init", generation=0, adam_m_zero=True, adam_v_zero=True, l2=.005), name+" optimizer")
        require(len(optimizer["key"]) == 2 and all(type(k) is int and 0 <= k < 2**32 for k in optimizer["key"]), name+" initial RNG key")
        summary_path = self.path(folder+"/summary.json")
        summary = self.json(folder+"/summary.json") if summary_path.exists() else {}
        result = dict(name=name, run=folder, status=summary.get("status", "no-summary"),
                      generation=summary.get("generation"), initial_optimizer=optimizer)
        if inventory or summary.get("status") != "finished":
            return result
        data, flat, genome_hash = self.genome(folder+"/genome_final.npy")
        digest = hashlib.sha256(data).hexdigest()
        subset(summary, dict(status="finished", generation=50, training_complete=True, evaluation_status="deferred",
            controller="rules-v0", condition_split="final", eval_protocol="paired-train-v1", objective=objective,
            eval_worlds=256, final_genome_hash=genome_hash, final_genome_sha256=digest), name+" summary")
        require(not any(field in summary for field in ("evaluations", "final_learning", "final_frozen", "start_learning"))
                and not self.path(folder+"/events.npz").exists(), name+" unexpected held-out evaluation")
        checkpoint = self.arrays(folder+"/checkpoint.npz")
        require(set(checkpoint) == {"mu", "m", "v", "t", "key"}, name+" checkpoint fields")
        for field in ("mu", "m", "v"):
            value = checkpoint[field]
            require(value.shape == (155,) and np.issubdtype(value.dtype, np.floating) and np.isfinite(value).all(), name+" checkpoint "+field)
        require(np.all(checkpoint["v"] >= 0) and np.array_equal(checkpoint["mu"], flat), name+" final state")
        require(checkpoint["t"].shape == () and np.issubdtype(checkpoint["t"].dtype, np.integer)
                and int(checkpoint["t"]) == 50, name+" checkpoint generation")
        require(checkpoint["key"].shape == (2,) and checkpoint["key"].dtype == np.uint32, name+" checkpoint key")
        text = self.read(folder+"/metrics.jsonl").decode("utf-8")
        require(text.endswith("\n"), name+" incomplete history line")
        history = [json.loads(line) for line in text.splitlines()]
        require(len(history) == 50, name+" history length")
        for generation, row in enumerate(history):
            require(set(row) == {"gen", "fit_mean", "fit_best", "fit_std", "acc_first", "acc_second", "survived", "seconds"}
                    and row["gen"] == generation and all(isinstance(v, (int, float)) and math.isfinite(v) for v in row.values()), name+" history row")
            require(row["fit_std"] >= 0 and row["seconds"] >= 0 and row["fit_best"] >= row["fit_mean"]
                    and all(0 <= row[key] <= 1 for key in ("acc_first", "acc_second", "survived")), name+" history range")
        metadata = dict(label=arm+str(i), controller="rules-v0", seed=i, training_seed=1000+i, objective=objective,
            run=folder, sha256=digest, genome_hash=genome_hash, genome_size=155, input_genome=original["input_metadata"],
            config_sha256=self.hashes[folder+"/config.json"], summary_sha256=self.hashes[folder+"/summary.json"], source_commits=provenance)
        result.update(metadata=metadata, final_key=checkpoint["key"].tolist(), history_rows=50,
                      first_training_row=history[0], final_training_row=history[-1], data=data)
        return result

    def condition(self, archive, label, controller, recorded, common):
        for key in PAIRING:
            require(key in archive and np.array_equal(archive[key], common[key]), f"{controller}/{label} pairing {key}")
        prefix = label+"_"
        raw = {key: archive[prefix+key] for key in EVENTS}
        require(all(value.shape == (256, 300, 4) and value.dtype == np.bool_ for value in raw.values()), f"{controller}/{label} event shape/type")
        f, p, before, after = (raw[key] for key in EVENTS)
        require(before[:, 0].all() and not (f & p).any() and not ((f | p) & ~before).any()
                and not (after & ~before).any() and np.array_equal(before[:, 1:], after[:, :-1]), f"{controller}/{label} event accounting")
        eat, death = f | p, before & ~after
        returns = f.sum(1, dtype=np.int64)-p.sum(1, dtype=np.int64)
        expected = dict(agent_return=returns, survival_at_flip=before[:, 150], final_alive=after[:, -1],
                        death_step=np.where(death.any(1), death.argmax(1), -1).astype(np.int32), no_encounters=~eat.any(1),
                        fitness=returns.mean(1), rule=np.full(256, int(label[-1]), np.int32))
        for key, value in expected.items():
            require(np.array_equal(archive[prefix+key], value), f"{controller}/{label} saved {key}")
        expected_keys = set(PAIRING) | {prefix+key for key in (*EVENTS, *expected)}
        if controller in ("fixed_A", "fixed_B", "mixture"):
            expected_keys.add("specialist_targets")
            target = [0]*4 if controller == "fixed_A" else [1]*4 if controller == "fixed_B" else [0, 0, 1, 1]
            require(np.array_equal(archive["specialist_targets"], target), f"{controller} specialist assignments")
        require(set(archive) == expected_keys, f"{controller}/{label} archive fields")
        fit, survived, flip = stats(returns.mean(1)), stats(after[:, -1].mean(1)), stats(before[:, 150].mean(1))
        meals, bins, survivor_bins = [], [], []
        ordinal = np.cumsum(eat, axis=1)
        for number in (1, 2, 3, 5, 10, 20):
            selected = eat & (ordinal == number)
            meals.append(dict(meal=number, eligible_agents=int(selected.sum()), total_agents=1024,
                              **accuracy(f & selected, p & selected)))
        for start in range(0, 300, 50):
            for output, keep in ((bins, np.ones((256, 1, 4), bool)), (survivor_bins, after[:, -1, None, :])):
                food, poison = f[:, start:start+50] & keep, p[:, start:start+50] & keep
                row = dict(start=start, stop=start+50, complete=True, contributing_agents=int((food | poison).any(1).sum()),
                           **accuracy(food, poison))
                if output is bins:
                    missing = ~eat[:, start:start+50].any(1)
                    row.update(deaths=int(death[:, start:start+50].sum()), zero_encounter_agents=int(missing.sum()),
                               agents_alive_at_start=int(before[:, start].sum()),
                               alive_at_start_without_encounters=int((missing & before[:, start]).sum()))
                output.append(row)
        eligible = eat.sum(1) >= 20
        first, twentieth = (f & (ordinal == 1)).any(1), (f & (ordinal == 20)).any(1)
        gain = ratio(((twentieth.astype(int)-first)*eligible).sum(1), eligible.sum(1))
        metrics = dict(fitness=fit["mean"], fitness_se=fit["se"], n_worlds=256, total_agents=1024,
            survived=survived["mean"], survived_se=survived["se"], food=int(f.sum()), poison=int(p.sum()), eaten=int(eat.sum()),
            meal_accuracy=meals, bins=bins, survivor_bins=survivor_bins,
            meal20_minus_meal1_matched=dict(difference=gain["mean"], se=gain["se"], eligible_agents=int(eligible.sum()),
                                          contributing_worlds=int(np.count_nonzero(eligible.sum(1)))),
            survival_at_flip=flip["mean"], survival_at_flip_se=flip["se"], deaths=int(death.sum()),
            zero_encounter_agents=int((~eat.any(1)).sum()), agent_return_distribution=distribution(returns),
            initial_rule=int(label[-1]), flip_step=150 if label.startswith("reversal") else None, controller=controller)
        compare(recorded, metrics, f"{controller}/{label} summary")
        return dict(returns=returns, final_alive=after[:, -1], food=f, poison=p, events=raw), metrics

    def evaluate(self, folder, sources, expected_name="E115-v5"):
        config, summary = self.json(folder+"/config.json"), self.json(folder+"/summary.json")
        if summary.get("status") != "finished":
            raise Incomplete(f"evaluation is {summary.get('status')!r}: {folder}")
        self.provenance(config, pinned=True)
        subset(config, dict(cfg=CFG, eval_protocol="v5", eval_seed=2026111903, eval_worlds=256,
                            reference_policies=list(REFERENCES), internal_seconds_limit=1140), "v5 config")
        subset(config["args"], dict(name=expected_name, eval_protocol="v5", condition_split="final", generations=0,
            eval_worlds=256, allow_dirty=False, quick=False, resume=None, no_plastic=False), "v5 arguments")
        subset(summary, dict(status="finished", generation=0, eval_protocol="v5", condition_split="final", eval_seed=2026111903,
            eval_worlds=256, expected_panels=14, completed_panels=14, artifact_validity="passed"), "v5 summary")
        require(tuple(summary["panels"]) == PANELS, "v5 panel order")
        require([record["label"] for record in config["input_genomes"]] == list(GENOMES), "v5 input manifest order")
        for record in config["input_genomes"]:
            label = record["label"]
            expected = dict(sources[label]["metadata"], copied_genome=f"inputs/{label}.npy")
            compare(record, expected, f"v5 {label} manifest")
            require(self.read(folder+"/"+record["copied_genome"]) == sources[label]["data"], f"v5 {label} copied genome bytes")
        first = self.arrays(folder+"/start0/stationary_r0.npz")
        common = {key: first[key] for key in PAIRING}
        for key in ("wkeys", "dkeys"):
            require(common[key].shape == (256, 2) and common[key].dtype == np.uint32
                    and len(np.unique(common[key], axis=0)) == 256, f"v5 {key} shape/type/unique worlds")
        require(common["initial_grid"].shape == (256, 16, 16) and np.isin(common["initial_grid"], (0, 1, 2)).all()
                and all(np.all((common["initial_grid"] == kind).sum((1, 2)) == 24) for kind in (1, 2)), "v5 initial grids")
        require(common["initial_pos"].shape == (256, 4, 2) and np.issubdtype(common["initial_pos"].dtype, np.integer)
                and np.all((common["initial_pos"] >= 0) & (common["initial_pos"] < 16)), "v5 initial positions")
        require(np.array_equal(common["agent_ids"], np.broadcast_to(np.arange(4), (256, 4))), "v5 individual identities")
        vectors, panels = {}, {}
        for label in PANELS:
            entry = summary["panels"][label]
            controller = "rules-v0" if label in GENOMES else label
            subset(entry, dict(status="finished", controller=controller), label+" status/controller")
            require(tuple(entry["conditions"]) == CONDITIONS, label+" conditions")
            compare(entry["raw_files"], {key: f"{label}/{key}.npz" for key in (*CONDITIONS, "paired")}, label+" raw paths")
            data, condition_report = {}, {}
            for condition in CONDITIONS:
                arrays = self.arrays(folder+"/"+entry["raw_files"][condition])
                data[condition], condition_report[condition] = self.condition(arrays, condition, controller, entry["conditions"][condition], common)
            for rule in (0, 1):
                for event in EVENTS:
                    require(np.array_equal(data[f"stationary_r{rule}"]["events"][event][:, :150],
                                           data[f"reversal_r{rule}"]["events"][event][:, :150]), label+" preflip identity")
            r0, r1 = data["stationary_r0"], data["stationary_r1"]
            q0, q1 = data["reversal_r0"], data["reversal_r1"]
            minimum = np.minimum(r0["returns"], r1["returns"])
            joint = r0["final_alive"] & r1["final_alive"]
            post = [(q["food"][:, 150:].sum(1, dtype=np.int64)-q["poison"][:, 150:].sum(1, dtype=np.int64)) for q in (q0, q1)]
            values = dict(stationary_fitness=((r0["returns"]+r1["returns"])/2).mean(1), individual_min_return=minimum.mean(1),
                joint_survival=joint.mean(1), post_flip_return=((post[0]+post[1])/2).mean(1),
                reversal_final_survival=((q0["final_alive"].astype(float)+q1["final_alive"])/2).mean(1),
                reversal_fitness=((q0["returns"]+q1["returns"])/2).mean(1))
            expected = {"paired_"+key: value for key, value in values.items()}
            expected.update(paired_individual_minimum=minimum.astype(float), paired_individual_joint_survival=joint)
            saved = self.arrays(folder+"/"+entry["raw_files"]["paired"])
            require(set(saved) == set(expected), label+" paired fields")
            for key, value in expected.items():
                require(np.array_equal(saved[key], value), label+" paired array "+key)
            estimates = {key: stats(value) for key, value in values.items()}
            estimates["individual_min_return"]["distribution"] = distribution(minimum.astype(float))
            compare(entry["paired"], estimates, label+" paired summary")
            vectors[label] = values
            panels[label] = dict(paired=estimates, conditions=condition_report)
        require(len(list(self.path(folder).glob("*/*.npz"))) == 70, "v5 archive count")
        contrasts, per_seed = [], []
        for i in range(3):
            parts = {}
            for control in ("mean", "start"):
                comparison = "min-minus-"+control
                for metric in METRICS:
                    delta = vectors[f"min{i}"][metric]-vectors[f"{control}{i}"][metric]
                    estimate = stats(delta)
                    contrasts.append(dict(seed=i, comparison=comparison, metric=metric, minuend=f"min{i}", subtrahend=f"{control}{i}",
                        **estimate, n_worlds=256, per_world_difference=delta.tolist()))
                    if metric == "individual_min_return":
                        limit = 2*estimate["se"]
                        parts[comparison+"_minimum_return"] = dict(**estimate, threshold=limit, operator=">", passed=estimate["mean"] > limit)
                    if metric == "joint_survival" and control == "start":
                        limit = -2*estimate["se"]
                        parts["min-minus-start_joint_survival"] = dict(**estimate, threshold=limit, operator=">=", passed=estimate["mean"] >= limit)
            per_seed.append(dict(seed=i, **parts, passed=all(x["passed"] for x in parts.values())))
        improvement = all(row[key]["passed"] for row in per_seed for key in ("min-minus-mean_minimum_return", "min-minus-start_minimum_return"))
        survival = all(row["min-minus-start_joint_survival"]["passed"] for row in per_seed)
        gate = dict(passed=improvement and survival, all_minimum_improvements=improvement,
                    no_joint_survival_regression=survival, per_seed=per_seed)
        compare(summary["contrasts"], contrasts, "v5 all30 contrasts")
        compare(summary["gate"], gate, "v5 numerical gate")
        return dict(run=folder, archive_count=70, panels=panels, contrasts=contrasts, gate=gate,
                    pairing_sha256={key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in common.items()})

    def recovery_evidence(self, folder, manifest_path):
        manifest = self.verify_interrupted_manifest(manifest_path)
        old_config, old_summary = self.json(INTERRUPTED_RUN+"/config.json"), self.json(INTERRUPTED_RUN+"/summary.json")
        config, summary = self.json(folder+"/config.json"), self.json(folder+"/summary.json")
        self.provenance(old_config, pinned=True)
        subset(old_summary, dict(status="interrupted", generation=0, eval_protocol="v5", condition_split="final",
               eval_seed=2026111903, eval_worlds=256, expected_panels=14, completed_panels=8), "original interrupted summary")
        require(not any(key in old_summary for key in ("gate", "contrasts", "artifact_validity")), "original incomplete attempt must have no gate")
        require(old_config["args"]["name"] == "E115-v5" and config["args"]["name"] == RECOVERY_NAME, "recovery attempt names")
        old_args, new_args = (dict(c["args"]) for c in (old_config, config))
        old_args.pop("name"); new_args.pop("name")
        require(old_args == new_args, "recovery changed arguments beyond its registered name")
        for key in ("cfg", "eval_protocol", "eval_seed", "eval_worlds", "reference_policies", "internal_seconds_limit", "input_genomes"):
            require(old_config[key] == config[key], "recovery changed scientific configuration: "+key)
        require(tuple(old_summary["panels"]) == PANELS[:9], "original panel prefix differs from interrupted record")
        conditions, paired_archives, arrays_checked = 0, 0, 0
        expected_files = {INTERRUPTED_RUN+"/"+name for name in ("config.json", "summary.json")}
        expected_files.update(INTERRUPTED_RUN+f"/inputs/{label}.npy" for label in GENOMES)
        for index, (label, old) in enumerate(old_summary["panels"].items()):
            new = summary["panels"][label]
            expected_conditions = CONDITIONS if index < 8 else CONDITIONS[:3]
            require(old["status"] == ("finished" if index < 8 else "interrupted")
                    and tuple(old["conditions"]) == expected_conditions and old["controller"] == new["controller"],
                    "original condition/status/controller prefix differs: "+label)
            expected_raw = {condition: f"{label}/{condition}.npz" for condition in expected_conditions}
            if index < 8:
                expected_raw["paired"] = f"{label}/paired.npz"
                require(old["paired"] == new["paired"], "recovery scientific paired summary differs: "+label)
                paired_archives += 1
            else:
                require("paired" not in old, "original incomplete panel unexpectedly has paired metrics")
            require(old["raw_files"] == expected_raw, "original raw-file manifest differs: "+label)
            for condition in expected_conditions:
                require(old["conditions"][condition] == new["conditions"][condition],
                        "recovery scientific condition summary differs: "+label+"/"+condition)
                conditions += 1
            for relative in expected_raw.values():
                expected_files.add(INTERRUPTED_RUN+"/"+relative)
                original_arrays, recovered_arrays = self.arrays(INTERRUPTED_RUN+"/"+relative), self.arrays(folder+"/"+relative)
                require(set(original_arrays) == set(recovered_arrays), "recovery overlapping array names differ: "+relative)
                for key, before in original_arrays.items():
                    after = recovered_arrays[key]
                    require(before.dtype == after.dtype and before.shape == after.shape and before.tobytes() == after.tobytes(),
                            "recovery overlapping array bits differ: "+relative+":"+key)
                    arrays_checked += 1
        require(set(manifest) == expected_files and len(manifest) == 54 and conditions == 35 and paired_archives == 8,
                "original interrupted evidence count or manifest differs")
        for label in GENOMES:
            relative = f"inputs/{label}.npy"
            require(self.read(INTERRUPTED_RUN+"/"+relative) == self.read(folder+"/"+relative), "recovery copied genome bytes differ: "+label)
        self.verify_interrupted_manifest(manifest_path)
        return dict(original_run=INTERRUPTED_RUN, recovery_run=folder, original_status="interrupted", original_files_unchanged=54,
                    manifest=manifest_path, overlapping_condition_archives=conditions, overlapping_paired_archives=paired_archives,
                    overlapping_arrays_bit_identical=arrays_checked, scientific_summaries_identical=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-commit", default=SOURCE)
    parser.add_argument("--evaluation", help="explicit registered E115-v5 or E115-v5-recovery1 folder; default selects original attempt only")
    parser.add_argument("--interrupted-manifest", help="required preservation SHA-256 manifest for explicit recovery1")
    parser.add_argument("--inventory", action="store_true", help="validate available source/config metadata without opening training checkpoints or evaluation archives")
    args = parser.parse_args()
    report = dict(status="not-audited", mode="inventory" if args.inventory else "full",
                  helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), training=[])
    audit = None
    try:
        audit = Auditor(args.root, args.source_commit)
        report.update(registered_source=audit.source, registered_code_trees=audit.expected_trees)
        sources = {}
        for i in range(3):
            original = audit.original(i)
            sources[f"start{i}"] = original
            for arm in ("mean", "min"):
                result = audit.training(i, arm, original, inventory=args.inventory)
                report["training"].append({key: value for key, value in result.items() if key != "data"})
                if "data" in result:
                    sources[f"{arm}{i}"] = result
        if args.inventory:
            report.update(status="inventory-only", final_evaluation_audited=False,
                          note="Available configs/source bytes checked; no checkpoint, metric, final50 or scientific gate claim.")
            code = 0
        else:
            if len(sources) != 9:
                raise Incomplete("all six finished50 training artifacts are required before the final audit")
            for i in range(3):
                mean, minimum = sources[f"mean{i}"], sources[f"min{i}"]
                compare(mean["initial_optimizer"], minimum["initial_optimizer"], f"seed{i} matched initial optimizer")
                compare(mean["final_key"], minimum["final_key"], f"seed{i} matched generation50 RNG position")
            evaluation, expected_name = select_evaluation(audit, args.evaluation, args.interrupted_manifest)
            if expected_name == RECOVERY_NAME:
                audit.verify_interrupted_manifest(args.interrupted_manifest)
            report["evaluation"] = audit.evaluate(evaluation, sources, expected_name)
            if expected_name == RECOVERY_NAME:
                report["recovery"] = audit.recovery_evidence(evaluation, args.interrupted_manifest)
            report.update(status="passed", final_evaluation_audited=True,
                limitations=["Read-only artifact reconstruction, not an optimizer or physics replay.",
                    "RNG keys and initial states are checked across saved arrays; they are not regenerated from the seed.",
                    "Initial zero Adam moments are verified through recorded metadata; final moments are checked for finite valid state.",
                    "A valid operational gate does not establish causal learning, reversal competence, or broader intelligence."])
            code = 0
    except Incomplete as error:
        report.update(status="incomplete", error=str(error), final_evaluation_audited=False)
        code = 2
    except (AuditError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", final_evaluation_audited=False)
        code = 1
    if audit is not None:
        report["checked_file_sha256"] = audit.hashes
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
