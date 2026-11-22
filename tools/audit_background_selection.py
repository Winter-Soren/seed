"""Independent E136 paired-objective, physics, optimizer and family audit.

No Seed simulator or evaluator is imported. JAX supplies only independently
specified random primitives and float32 optimizer arithmetic. Frozen first
decisions are checked from saved W/x with the E135 NumPy roundoff envelope.
"""
import argparse
import copy
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "HANDOFF.md").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import audit_replay_selection as replay
import audit_scene_transfer as scene
import audit_scene_transfer_physics as physics

encounter, base, source_bytes = replay.encounter, replay.base, replay.source_bytes
require, same, subset, typed = base.require, base.same, base.subset, base.typed
exact, fingerprint, prefixed = replay.exact, replay.fingerprint, replay.prefixed
CFG, MOVES, NEURAL, PLACEMENTS = replay.CFG, replay.MOVES, replay.NEURAL, replay.PLACEMENTS
PROTOCOL = "background-selection-v1"
VIEWS, GENOTYPES = ("isolated", "background"), ("parent", "isolated", "background")
HISTORIES = ("stationary", "reversal")
KEY_ROLES = ("script", "birth", "chamber", "background", "probe", "solo_layout", "solo_action")
CELLS = ((8, 16), (16, 32), (8, 32), (16, 16))
TRAIN_CELLS = np.asarray(((8, 0), (8, 8), (16, 0), (16, 8)), np.int32)
FIELDS = ("Y", "first_food", "first_poison", "no_contact", "survival", "latency")
CHOICE_FIELDS = ("p_food", "p_poison", "p_other", "approach", "conditional", "conditional_food_fraction")
BODY_SHAPES = dict(grid=(16, 16), pos=(1, 2), energy=(1,), alive=(1,), sig=(1,),
                   denergy=(1,), key=(2,), t=(), counts=(1, 2, 2))
TRACE_SHAPES = {**scene.TRACE_SHAPES, "explicit_learning_denergy": (1,)}
SOURCE_PATHS = {"seed/seed.py", "seed/encounter_diagnostic.py", "seed/replay_selection.py",
    "seed/scene_transfer.py", "seed/background_selection.py", "scripts/social_observer_evolution.py",
    "scripts/run_replay_selection.py", "scripts/run_background_selection.py", "scripts/run_experiment.py",
    "scripts/run_cold_donor.py", "scripts/run_shelter_evolution.py"}
estimate = scene.estimate


@lru_cache(maxsize=1)
def random_helpers():
    jax, jnp, script, world, tape = replay.random_helpers()
    cross = (np.asarray([8, 8]) + MOVES) @ np.asarray([16, 1])
    eligible = jnp.asarray(np.setdiff1d(np.arange(256, dtype=np.int32), cross))

    @jax.jit
    def background(key):
        cells = jax.random.choice(key, eligible, (46,), replace=False)
        return jnp.zeros(256, jnp.int32).at[cells[:23]].set(1).at[cells[23:]].set(2).reshape(16, 16)

    @jax.jit
    def first_draws(keys, logits):
        def one(key, logit):
            action_key = jax.random.split(key, 3)[1]
            return action_key, jax.random.categorical(action_key, logit)
        return jax.vmap(one)(keys, logits)

    def derive(keys):
        grid, position = world(keys[5])
        return dict(script=np.asarray(script(keys[0])), background=np.asarray(background(keys[3])),
                    probe_keys=np.asarray(jax.random.split(keys[4], 12)),
                    target_grid=np.asarray(grid), target_position=np.asarray(position))
    return jax, jnp, script, background, tape, first_draws, derive


def trace_arrays(raw, prefix, leading):
    return {name: scene.numeric(raw, prefix + name, leading + shape,
        np.int32 if name in scene.INTEGER_FIELDS else np.uint32 if name in scene.KEY_FIELDS
        else bool if name in scene.BOOL_FIELDS else np.float32) for name, shape in TRACE_SHAPES.items()}


def body_arrays(raw, prefix, leading):
    return {name: scene.numeric(raw, prefix + name, leading + shape,
        np.int32 if name in ("grid", "pos", "t") else np.uint32 if name == "key" else np.float32)
        for name, shape in BODY_SHAPES.items()}


def trace_physics(data, initial, rules, *, script=None, gap=0):
    exact(data["explicit_learning_denergy"], data["incoming_denergy"],
          "explicit native learning input is the same observable physical energy change")
    physics.verify(data, initial, rules, random_helpers()[4], script=script, gap=gap)
    return scene.outcomes(data)


def audit_family(raw, index, genomes, keys, pre, gap):
    require((pre, gap) == CELLS[index % 4], "fixed held-out schedule")
    steps, leading = 2*(pre+16)+gap, (3, 2, 2)
    traces = {"exposure": trace_arrays(raw, "exposure_", leading+(steps,)),
              "first": trace_arrays(raw, "first_", leading+(2, 12)),
              "solo": trace_arrays(raw, "solo_", leading+(50,))}
    bodies = {name: body_arrays(raw, name+"_", leading) for name in ("acquired", "solo_initial", "solo_final")}
    expected = {"base_index", "keys", "pre_contacts", "gap", "initial_rules", "genome_sha256",
                "script", "background", "target_grid", "target_position"}
    expected |= {prefix+"_"+name for prefix in traces for name in TRACE_SHAPES}
    expected |= {prefix+"_"+name for prefix in bodies for name in BODY_SHAPES}
    expected |= {prefix+"_"+name+"_sha256" for prefix in bodies for name in NEURAL}
    expected |= {"acquired_"+name for name in NEURAL}
    require(set(raw) == expected, "complete and exact E136 family evidence schema")
    for name, value in (("base_index", index), ("pre_contacts", pre), ("gap", gap)):
        exact(scene.numeric(raw, name, (), np.int32), np.asarray(value, np.int32), "assigned family " + name)
    exact(scene.numeric(raw, "keys", (7, 2), np.uint32), keys, "seven independent family roles")
    exact(scene.numeric(raw, "initial_rules", (2,), np.int32), np.asarray([0, 1], np.int32), "both hidden initial rules")
    exact(scene.numeric(raw, "genome_sha256", (3, 32), np.uint8),
          np.stack([fingerprint(genome) for genome in genomes]), "pinned parent and both attempt-ten endpoints")
    randomness = random_helpers()[6](keys)
    exact(scene.numeric(raw, "script", (pre+16, 5), np.int32), randomness["script"][:pre+16], "independent balanced script")
    for name, shape in (("background", (16, 16)), ("target_grid", (16, 16)), ("target_position", (1, 2))):
        exact(scene.numeric(raw, name, shape, np.int32), randomness[name], "independently generated " + name)
    cross = np.asarray([8, 8])+MOVES
    require(not np.any(raw["background"][cross[:, 0], cross[:, 1]])
        and np.count_nonzero(raw["background"] == 1) == np.count_nonzero(raw["background"] == 2) == 23,
        "balanced distant items preserve every immediate alternative")
    acquired = {name: scene.numeric(raw, "acquired_"+name, leading+shape, np.float32)
                for name, shape in scene.NEURAL_SHAPES.items()}
    require(np.isin(acquired["mask"], (0, 1)).all() and np.all(np.abs(acquired["W"]) <= 3), "native topology and weight limits")
    for prefix in bodies:
        for name in NEURAL:
            scene.numeric(raw, prefix+"_"+name+"_sha256", leading+(32,), np.uint8)
    for name, value in traces["exposure"].items():
        exact(value[:, :, 0, :2*pre], value[:, :, 1, :2*pre], "matched pre-boundary experience " + name)
    local, sampled = np.empty(leading+(2,)), np.empty(leading+(2,))
    choice = {name: np.empty_like(local) for name in CHOICE_FIELDS}
    decomposition = {name: np.empty(leading) for name in ("approach", "conditional")}
    solo = {name: np.empty(leading) for name in FIELDS}
    n_distractors = int(np.count_nonzero(raw["background"][6:11, 6:11]))
    for genotype in range(3):
        for initial_rule in range(2):
            for history in range(2):
                address, target = (genotype, initial_rule, history), initial_rule ^ history
                exposure = {name: value[address] for name, value in traces["exposure"].items()}
                birth = scene.neutral_body(np.zeros((16, 16), np.int32), np.asarray([[8, 8]], np.int32), keys[2])
                rules = np.full(steps, target, np.int32)
                rules[:2*pre] = initial_rule
                trace_physics(exposure, birth, rules, script=raw["script"], gap=gap)
                carried = {name: value[address] for name, value in bodies["acquired"].items()}
                replay.state_end(carried, exposure, steps)
                snapshot = {name: value[address] for name, value in acquired.items()}
                for name, value in snapshot.items():
                    exact(fingerprint(value), raw["acquired_"+name+"_sha256"][address], "actual saved acquired " + name)
                    exact(raw["solo_initial_"+name+"_sha256"][address], raw["acquired_"+name+"_sha256"][address],
                          "ordinary solo carries the unprobed acquired " + name)
                exact(snapshot["x"][:, :103], exposure["observation"][-1], "acquired sensors reflect the final replay input")
                exact(snapshot["x"][:, -6:-1], exposure["efference"][-1], "acquired motor state contains actual movement")
                exact(snapshot["x"][:, -1], exposure["signal"][-1], "acquired signal carry")
                first = {name: value[address] for name, value in traces["first"].items()}
                exact(first["explicit_learning_denergy"], first["incoming_denergy"], "neutral first-decision feedback")
                compact_first = {name: first[name] for name in scene.TRACE_SHAPES}
                scene.check_frozen_policy(snapshot, compact_first)
                scene.first_physics(compact_first, raw["background"], randomness["probe_keys"], target)
                parts, contributions = scene.first_decomposition(first["probabilities"], target)
                for name, value in parts.items():
                    choice[name][address] = value
                # Native bounded softmax gives strictly positive mass to both focal moves.
                require(np.all(first["probabilities"] > 0), "native softmax keeps conditional color denominator positive")
                choice["conditional_food_fraction"][address] = (parts["conditional"]+1)/2
                for name, value in contributions.items():
                    decomposition[name][address] = value
                for view in range(2):
                    probability = first["probabilities"][view, :, 0]
                    local[address+(view,)] = replay.signed_choice(probability, target)
                    actions = first["action"][view, :, 0]
                    sampled[address+(view,)] = np.mean((actions == PLACEMENTS[:, target]).astype(float)
                        - (actions == PLACEMENTS[:, 1-target]).astype(float))
                if n_distractors == 0:
                    for name, value in first.items():
                        exact(value[0], value[1], "no visible background means identical native first decisions " + name)
                start = {name: value[address] for name, value in bodies["solo_initial"].items()}
                for name, value in scene.neutral_body(raw["target_grid"], raw["target_position"], keys[6]).items():
                    exact(start[name], value, "neutral ordinary solo start " + name)
                require(int(start["t"]) == 0 and not np.any(start["counts"]), "ordinary solo starts with a fresh physical clock")
                trace = {name: value[address] for name, value in traces["solo"].items()}
                outcome = trace_physics(trace, start, np.full(50, target, np.int32))
                final = {name: value[address] for name, value in bodies["solo_final"].items()}
                replay.state_end(final, trace, 50)
                exact(raw["solo_final_mask_sha256"][address], raw["solo_initial_mask_sha256"][address], "solo keeps inherited topology")
                for name, value in outcome.items():
                    solo[name][address] = value
    return dict(base_index=index, n_distractors=n_distractors, local=local, sampled=sampled,
                first_choice=choice, decomposition=decomposition, solo=solo)


def summarize_rows(rows):
    n = len(rows)
    require(n >= 2 and [row["base_index"] for row in rows] == list(range(n)), "all assigned families remain in denominator")
    local = {name: np.stack([row[name] for row in rows]).mean(2) for name in ("local", "sampled")}
    require(all(value.shape == (n, 3, 2, 2) and np.isfinite(value).all() for value in local.values()), "family local dimensions")
    solo = {name: np.stack([row["solo"][name] for row in rows]).mean(2) for name in FIELDS}
    require(all(value.shape == (n, 3, 2) and np.isfinite(value).all() for value in solo.values()), "family solo dimensions")
    require(np.allclose(solo["Y"], solo["first_food"]-solo["first_poison"], atol=0, rtol=0)
        and np.allclose(solo["first_food"]+solo["first_poison"]+solo["no_contact"], 1, atol=0, rtol=0),
        "every assigned life has exactly one signed first-contact outcome")
    choice = {name: np.stack([row["first_choice"][name] for row in rows]).mean(2) for name in CHOICE_FIELDS}
    decomposition = {name: np.stack([row["decomposition"][name] for row in rows]).mean(2) for name in ("approach", "conditional")}
    arms, contrasts = {}, {}
    for genotype, label in enumerate(GENOTYPES):
        arms[label] = dict(histories={})
        for history, hlabel in enumerate(HISTORIES):
            arms[label]["histories"][hlabel] = dict(
                local={view: estimate(local["local"][:, genotype, history, v]) for v, view in enumerate(VIEWS)},
                sampled={view: estimate(local["sampled"][:, genotype, history, v]) for v, view in enumerate(VIEWS)},
                first_choice={view: {name: estimate(value[:, genotype, history, v]) for name, value in choice.items()}
                              for v, view in enumerate(VIEWS)},
                solo=scene.behavior_summary({name: value[:, genotype, history] for name, value in solo.items()}),
                sparse_minus_dense_components={name: estimate(value[:, genotype, history]) for name, value in decomposition.items()})
    for history, label in enumerate(HISTORIES):
        contrasts[label+"_background_absolute_Y"] = estimate(solo["Y"][:, 2, history])
        for comparator, name in ((0, "parent"), (1, "isolated")):
            for field, values in (("solo", solo["Y"]), ("survival", solo["survival"])):
                contrasts[f"{label}_{field}_gain_vs_{name}"] = estimate(values[:, 2, history]-values[:, comparator, history])
            gains = local["local"][:, 2, history]-local["local"][:, comparator, history]
            for view, tag in ((0, "sparse"), (1, "dense")):
                contrasts[f"{label}_{tag}_gain_vs_{name}"] = estimate(gains[:, view])
            contrasts[f"{label}_dense_minus_sparse_interaction_vs_{name}"] = estimate(gains[:, 1]-gains[:, 0])
            for field, values in decomposition.items():
                contrasts[f"{label}_sparse_minus_dense_{field}_component_vs_{name}"] = estimate(values[:, 2, history]-values[:, comparator, history])
        contrasts[label+"_isolated_control_solo_gain_vs_parent"] = estimate(solo["Y"][:, 1, history]-solo["Y"][:, 0, history])
        for view, tag in ((0, "sparse"), (1, "dense")):
            contrasts[f"{label}_isolated_control_{tag}_gain_vs_parent"] = estimate(local["local"][:, 1, history, view]-local["local"][:, 0, history, view])
    result = dict(families=n, assigned_lives_per_genotype_history=2*n, arms=arms, contrasts=contrasts,
        distractors=estimate([row["n_distractors"] for row in rows]),
        zero_distractors=estimate([row["n_distractors"] == 0 for row in rows]))
    result["gates"] = gates(result)
    return result


def gates(summary):
    contrasts = summary["contrasts"]
    decisions = {}
    for label, primary, safeguard in (("stationary_repair", "stationary", "reversal"),
                                      ("reversal_advancement", "reversal", "stationary")):
        key = primary+"_background_absolute_Y"
        criteria = {key: replay.passes(contrasts[key])}
        for comparison in ("parent", "isolated"):
            key = f"{primary}_solo_gain_vs_{comparison}"
            criteria[key] = replay.passes(contrasts[key], .05)
            key = f"{safeguard}_solo_gain_vs_{comparison}"
            criteria[key+"_safeguard"] = bool(contrasts[key]["lower_2se"] > -.05)
            for history in HISTORIES:
                key = f"{history}_survival_gain_vs_{comparison}"
                criteria[key+"_safeguard"] = bool(contrasts[key]["lower_2se"] > -.05)
        if label == "stationary_repair":
            key = "stationary_dense_gain_vs_isolated"
            criteria[key] = replay.passes(contrasts[key], .05)
        decisions[label] = dict(passed=all(criteria.values()), criteria=criteria,
            failed_criteria=[name for name, passed in criteria.items() if not passed])
    return decisions


def candidate_scores(samples, view):
    require(view in VIEWS, "registered selection view")
    probabilities = typed(samples, "probabilities", (16, 4, 2, 2, 2, 12, 5), "f")
    logits = typed(samples, "logits", probabilities.shape, "f")
    encounter.softmax(logits, probabilities)
    require(np.all(np.abs(logits) <= 3), "native bounded training logits")
    signed = np.empty((16, 4, 2, 2, 2), np.float64)
    for initial in range(2):
        for history in range(2):
            signed[:, :, initial, history] = replay.signed_choice(probabilities[:, :, initial, history], initial ^ history)
    both = typed(samples, "view_fitness", (16, 4, 2), "f")
    replay.physical_close(both, signed.mean((2, 3)), "both candidate scores are free five-action expectations")
    fitness = typed(samples, "fitness", (16, 4), "f")
    exact(fitness, both[:, :, VIEWS.index(view)], "only the assigned view is routed into selection")
    _, jnp = source_bytes.random_runtime()
    return np.asarray(jnp.asarray(fitness).mean(1))


def tied_ranks(scores):
    """Derive tie intervals by sorting, independently of the production ranks."""
    scores = np.asarray(scores)
    require(scores.ndim == 1 and len(scores) >= 2 and np.isfinite(scores).all(), "finite population score vector")
    order, positions = np.argsort(scores, kind="stable"), np.empty(len(scores), np.float32)
    begin = 0
    while begin < len(scores):
        end = begin+1
        while end < len(scores) and scores[order[end]] == scores[order[begin]]:
            end += 1
        positions[order[begin:end]] = (begin+end-1)/2
        begin = end
    # Separate float32 division and subtraction match the saved native rank scale.
    return positions/np.float32(len(scores)-1)-np.float32(.5)


def view_ranks(samples):
    both = typed(samples, "view_fitness", (16, 4, 2), "f")
    _, jnp = source_bytes.random_runtime()
    scores = np.stack([np.asarray(jnp.asarray(both[:, :, view]).mean(1)) for view in range(2)])
    return np.stack([tied_ranks(value) for value in scores])


def optimizer_transition(previous, samples, view):
    scores = candidate_scores(samples, view)
    eps = typed(samples, "eps", (16, 155), "f")
    exact(eps[8:], -eps[:8], "antithetic mutation pairs")
    jax, jnp = source_bytes.random_runtime()
    next_key, noise_key, world_key, birth_key = jax.random.split(jnp.asarray(previous["key"]), 4)
    exact(eps[:8], np.asarray(jax.random.normal(noise_key, (8, 155))), "independently reconstructed mutation noise")
    exact(samples["world_keys"], np.asarray(jax.random.split(world_key, 4)), "four fresh family roots per attempt")
    births = np.asarray(jax.random.split(birth_key, 32)).reshape(8, 4, 2)
    exact(samples["development_keys"], np.concatenate((births, births)), "same developmental key for antithetic twins")
    exact(samples["candidate_genomes"], np.asarray(jnp.asarray(previous["mu"])+.05*jnp.asarray(eps)), "candidate genotype arithmetic")
    weights = tied_ranks(scores)
    exact(samples["rank_weights"], weights, "average ranks preserve fitness ties")
    informative = bool(np.any(scores != scores[0]))
    expected = {name: previous[name].copy() for name in ("mu", "m", "v", "t")}
    if informative:
        mu, m, v = (jnp.asarray(previous[name]) for name in ("mu", "m", "v"))
        gradient = (jnp.asarray(weights)[:, None]*jnp.asarray(eps)).sum(0)/(.05*16)-.005*mu
        reference = (weights.astype(np.float64)[:, None]*eps.astype(np.float64)).sum(0)/.8-.005*previous["mu"].astype(np.float64)
        magnitude = (np.abs(weights.astype(np.float64)[:, None]*eps.astype(np.float64))).sum(0)/.8+.005*np.abs(previous["mu"])
        unit = np.finfo(np.float32).eps/2
        require(np.all(np.abs(np.asarray(gradient, np.float64)-reference)
                <= 22*unit/(1-22*unit)*magnitude+22*np.finfo(np.float32).smallest_subnormal), "independent ES reduction roundoff bound")
        step = int(previous["t"])+1
        m, v = .9*m+.1*gradient, .999*v+.001*gradient**2
        mu = mu+.02*(m/(1-.9**step))/(jnp.sqrt(v/(1-.999**step))+1e-8)
        expected.update(mu=np.asarray(mu), m=np.asarray(m), v=np.asarray(v), t=np.asarray(step, previous["t"].dtype))
    expected["key"] = np.asarray(next_key)
    return expected, scores, informative


class Auditor(replay.Auditor):
    def __init__(self, root, source):
        super().__init__(root, source)
        self.paired_inputs = {}
        self.paired_configs = {}
        self.checked_source_commits = set()
        self.training_measurements = {}

    def metadata(self, folder, index, mode, view=None):
        config = self.json(folder+"/config.json")
        development, training = mode.startswith("development"), mode.endswith("training")
        attempts, families = (2 if development else 10, 0) if training else (0, 4 if development else 128)
        train_seed = 2026121600+(0 if development else index+1)
        eval_seed = 2026121700+(0 if development else index+1)
        name = "E136-benchmark-"+view if development and training else "E136-benchmark-eval" if development else (
            f"E136-A{index}-{view}" if training else f"E136-A{index}")
        require(Path(folder).name.endswith("_"+name), "registered run name selects the objective")
        subset(config, dict(protocol=PROTOCOL, mode=mode, seed_index=index, cfg=CFG, l2=.005, view=view,
            expected_attempts=attempts, expected_families=families), "registered E136 design")
        same(config["training_seed"], train_seed if training else None, "independent lineage training master")
        same(config["evaluation_seed"], None if training else eval_seed, "separate held-out master")
        same(config["key_roles"], list(KEY_ROLES), "explicit final role names")
        if training:
            require(config["endpoint_sha256"] is None and config["assignments"] is None
                and not config["train_receipts"], "training configuration precedes its endpoint and final keys")
        else:
            require(isinstance(config["endpoint_sha256"], list) and len(config["endpoint_sha256"]) == 2,
                    "evaluation pins exactly two completed endpoint hashes")
        args = config["args"]
        subset(args, dict(name=name, eval_protocol=PROTOCOL, generations=attempts, eval_worlds=families, pop=16, worlds=4,
            sigma=.05, lr=.02, seed=train_seed if training else eval_seed,
            no_plastic=False, allow_dirty=False, quick=False, resume=None, cfg=None, objective=None, memory_condition=None,
            max_train_seconds=None, checkpoint_every=10), "frozen native invocation")
        require(args.get("controller") in (None, "rules-v0") and args.get("structural_plasticity") in (None, "off")
            and not args.get("persistence", False), "unchanged native controller")
        commit = config["git"]["commit"]
        require(config["git"]["dirty"] is False and re.fullmatch(r"[0-9a-f]{40}", commit)
                and self.code_trees(commit) == self.expected_trees, "clean frozen scientific source")
        require(set(config["source_code"]) == SOURCE_PATHS, "complete frozen transitive source manifest")
        if commit not in self.checked_source_commits:
            context = type("SourceContext", (), {"root": self.root, "source": commit})()
            attributes = source_bytes.Auditor.source_attributes(context, list(config["source_code"]))
            for path, digest in config["source_code"].items():
                blob = base.subprocess.run(["git", "show", commit+":"+path], cwd=self.root,
                    capture_output=True, check=True, timeout=15).stdout
                source_bytes.source_hash_representation(blob, digest, attributes[path])
            self.checked_source_commits.add(commit)
        if self.source_manifest is None:
            self.source_manifest = config["source_code"]
        same(config["source_code"], self.source_manifest, "same development and final scientific source bytes")
        path, digest = base.ORIGINALS[index]
        subset(config["source"], dict(path=path, sha256=digest, training_seed=index), "original E103 parent lineage")
        same(args["start"], "file:"+path, "original parent start")
        content = self.read(path)
        require(hashlib.sha256(content).hexdigest() == digest and self.read(folder+"/genome_start.npy") == content,
                "unchanged original parent bytes")
        parent = np.load(io.BytesIO(content), allow_pickle=False)
        require(parent.shape == (155,) and parent.dtype == np.float32 and np.isfinite(parent).all(), "native 155-number parent")
        report = self.json(folder+"/summary.json")
        subset(report, dict(status="finished", protocol=PROTOCOL, mode=mode, seed_index=index,
            completed_attempts=attempts, completed_families=families), "complete assigned workload")
        require(0 < report["elapsed_seconds"] <= config["max_seconds"], "guarded process time bound")
        require(len(report["attempt_wall_seconds"]) == attempts and len(report["family_wall_seconds"]) == families
            and all(np.isfinite(v) and v >= 0 for v in report["attempt_wall_seconds"]+report["family_wall_seconds"]), "complete timing receipt")
        return config, report, parent

    def training(self, folder, index, development, view):
        config, report, parent = self.metadata(folder, index, "development-training" if development else "training", view)
        pair_id = (development, index)
        same_args = {name: value for name, value in config["args"].items() if name != "name"}
        if view == "isolated":
            self.paired_configs[pair_id] = same_args
        else:
            same(same_args, self.paired_configs[pair_id], "paired arms differ only in registered objective name")
        attempts = config["expected_attempts"]
        require(len(report["attempts"]) == attempts+1 and len(report["samples"]) == attempts, "all attempted optimizer states")
        require(report["attempts"][0]["path"] == "generations/attempt_0000.npz", "fixed starting checkpoint")
        previous = self.artifact(folder, report["attempts"][0])
        require(set(previous) == {"mu", "m", "v", "t", "key"}, "exact optimizer state schema")
        exact(previous["mu"], parent, "both populations start at own parent")
        require(not previous["m"].any() and not previous["v"].any() and int(previous["t"]) == 0, "fresh Adam state")
        jax = random_helpers()[0]
        exact(previous["key"], np.asarray(jax.random.split(jax.random.PRNGKey(config["training_seed"]))[0]), "fixed initial optimizer key")
        if view == "isolated":
            self.register_keys(previous["key"][None], "new lineage optimizer")
        updates = []
        for attempt in range(1, attempts+1):
            started = time.monotonic()
            require(report["attempts"][attempt]["path"] == f"generations/attempt_{attempt:04d}.npz"
                and report["samples"][attempt-1]["path"] == f"generations/samples_{attempt:04d}.npz", "ordered immutable attempt records")
            current = self.artifact(folder, report["attempts"][attempt])
            require(set(current) == set(previous), "consistent optimizer state schema")
            samples = self.artifact(folder, report["samples"][attempt-1])
            expected, scores, informative = optimizer_transition(previous, samples, view)
            both_ranks = view_ranks(samples)
            exact(samples["rank_weights"], both_ranks[VIEWS.index(view)], "assigned objective alone supplies average tied ranks")
            for name, value in expected.items():
                exact(current[name], value, "independent optimizer transition " + name)
            if view == "isolated":
                self.register_keys(current["key"][None], "optimizer continuation")
                self.register_keys(samples["world_keys"], "fresh training family roots")
                self.register_keys(samples["development_keys"][:8], "fresh paired birth keys")
            self.training_batch(samples, register=view == "isolated")
            paired = {name: value for name, value in samples.items() if name.startswith("batch_")
                or name in ("eps", "world_keys", "development_keys")}
            paired["optimizer_key"] = previous["key"]
            if attempt == 1:
                paired.update({"first_"+name: value for name, value in samples.items()
                    if name in ("candidate_genomes", "probabilities", "logits", "view_fitness", "sampled_action", "probe_action_keys",
                                "counts", "survived", "exposure_energy_bounds")})
            identity = (development, index, attempt)
            if view == "isolated":
                self.paired_inputs[identity] = {name: fingerprint(value) for name, value in paired.items()}
            else:
                require(set(paired) == set(self.paired_inputs[identity]), "same candidate primitive and key evidence in both arms")
                for name, value in paired.items():
                    exact(fingerprint(value), self.paired_inputs[identity][name], "paired arm common randomness/workload " + name)
            updates.append(dict(attempt=attempt, informative=informative, optimizer_updates=int(current["t"]),
                changed_parameters=int(np.count_nonzero(current["mu"] != previous["mu"])),
                score_mean=float(scores.mean()), score_best=float(scores.max()), distinct_scores=int(len(np.unique(scores))),
                separated_antithetic_pairs=int(np.count_nonzero(scores[:8] != scores[8:])),
                view_rank_vectors={name: both_ranks[i].tolist() for i, name in enumerate(VIEWS)},
                view_rank_vectors_equal=bool(np.array_equal(both_ranks[0], both_ranks[1])),
                view_rank_difference_norm=float(np.linalg.norm(both_ranks[1].astype(np.float64)-both_ranks[0]))))
            previous = current
            self.training_seconds.append(time.monotonic()-started)
        content = self.read(folder+"/genome_final.npy")
        endpoint = np.load(io.BytesIO(content), allow_pickle=False)
        exact(endpoint, previous["mu"], "endpoint is exactly the final attempted mean, not a selected checkpoint")
        same(report["endpoint_sha256"], hashlib.sha256(content).hexdigest(), "terminal endpoint hash")
        same(report["optimizer_updates"], int(previous["t"]), "all-tied attempts preserve Adam update count")
        self.training_reports[folder] = (endpoint, config, report)
        result = dict(index=index, view=view, run=folder, attempts=updates, endpoint_sha256=report["endpoint_sha256"],
            view_rank_vectors_equal_attempts=sum(row["view_rank_vectors_equal"] for row in updates),
            view_rank_vectors_different_attempts=sum(not row["view_rank_vectors_equal"] for row in updates))
        self.training_measurements[(development, index, view)] = result
        return result

    def manipulation(self, development, index):
        arms = [self.training_measurements[(development, index, view)] for view in VIEWS]
        arrays = [self.training_reports[arm["run"]][0] for arm in arms]
        return dict(development=development, index=index, first_attempt_matching_candidates_and_both_views=True,
            ranks_on_each_realized_population={arm["view"]: dict(
                equal_attempts=arm["view_rank_vectors_equal_attempts"], different_attempts=arm["view_rank_vectors_different_attempts"])
                for arm in arms},
            endpoint_parameters_identical=bool(np.array_equal(fingerprint(arrays[0]), fingerprint(arrays[1]))),
            endpoint_changed_parameters=int(np.count_nonzero(arrays[0] != arrays[1])),
            endpoint_difference_norm=float(np.linalg.norm(arrays[1].astype(np.float64)-arrays[0])))

    def training_batch(self, samples, *, register):
        batch = prefixed(samples, "batch_")
        require(set(batch) == {"world_keys", "role_keys", "schedule_key", "pre_contacts", "gap", "scripts", "backgrounds"}, "matched training batch schema")
        exact(batch["world_keys"], samples["world_keys"], "batch shares optimizer family roots")
        jax, _, script, background, _, draws, _ = random_helpers()
        roles = np.stack([np.asarray(jax.random.split(key, 4)) for key in samples["world_keys"]])
        exact(batch["role_keys"], roles, "independent script/chamber/background/probe roles")
        schedule_key = np.asarray(jax.random.fold_in(samples["world_keys"][0], 0xE136))
        exact(batch["schedule_key"], schedule_key, "independent E136 training schedule key")
        cells = TRAIN_CELLS[np.asarray(jax.random.permutation(schedule_key, 4))]
        exact(batch["pre_contacts"], cells[:, 0], "each training contact count")
        exact(batch["gap"], cells[:, 1], "each training gap")
        if register:
            self.register_keys(roles, "fresh training roles")
            self.register_keys(schedule_key[None], "fresh training schedule")
        for family in range(4):
            expected = np.asarray(script(roles[family, 0])).copy()
            expected[int(cells[family, 0])+16:] = 0
            exact(batch["scripts"][family], expected, "balanced training script with zero padding")
            exact(batch["backgrounds"][family], np.asarray(background(roles[family, 2])), "fresh common training background")
        probe_keys = np.stack([np.asarray(jax.random.split(key, 12)) for key in roles[:, 3]])
        broadcast = np.broadcast_to(probe_keys[None, :, None, None, None], (16, 4, 2, 2, 2, 12, 2))
        action_keys, sampled = draws(broadcast.reshape(-1, 2), samples["logits"].reshape(-1, 5))
        exact(typed(samples, "sampled_action", (16, 4, 2, 2, 2, 12), "i"),
              np.asarray(sampled).reshape(16, 4, 2, 2, 2, 12), "same action randomness across candidates, views and histories")
        exact(typed(samples, "probe_action_keys", (16, 4, 2, 2, 2, 12, 2), "u"),
              np.asarray(action_keys).reshape(16, 4, 2, 2, 2, 12, 2), "native independent first-action keys")
        counts = typed(samples, "counts", (16, 4, 2, 2), "fiu")
        for family, pre in enumerate(cells[:, 0]):
            replay.physical_close(counts[:, family], np.broadcast_to([[int(pre)//2, int(pre)//2], [8, 8]], (16, 2, 2)),
                                  "prescribed exposure contacts cannot themselves be the fitness")
        require(np.shape(samples["survived"]) == (16, 4) and np.all(samples["survived"] == 1), "all assigned candidate exposures survive")
        bounds = typed(samples, "exposure_energy_bounds", (16, 4, 2), "f")
        require(np.all(bounds[..., 0] >= .462-1e-5) and np.all(bounds[..., 1] <= 1.25+1e-5)
            and np.all(bounds[..., 0] <= bounds[..., 1]), "deterministic paired energy bound")

    def evaluation(self, folder, index, development):
        config, report, parent = self.metadata(folder, index, "development-evaluation" if development else "evaluation")
        receipts = config["train_receipts"]
        require(len(receipts) == (2 if development else 6), "both arms of every required lineage pinned before evaluation")
        endpoints = []
        for ordinal, receipt in enumerate(receipts):
            lineage, view = ordinal//2, VIEWS[ordinal % 2]
            require(receipt["seed_index"] == lineage and receipt["view"] == view and receipt["run"] in self.training_reports,
                    "fixed lineage-then-view receipt order")
            endpoint, training_config, training_report = self.training_reports[receipt["run"]]
            same(training_config["view"], view, "receipt objective matches completed population")
            paths = dict(config_sha256="config.json", summary_sha256="summary.json",
                checkpoint_sha256=f"generations/attempt_{2 if development else 10:04d}.npz", endpoint_sha256="genome_final.npy")
            for name, path in paths.items():
                same(receipt[name], hashlib.sha256(self.read(receipt["run"]+"/"+path)).hexdigest(), "immutable endpoint receipt " + name)
            if lineage == index:
                endpoints.append(endpoint)
                same(config["endpoint_sha256"][ordinal % 2], receipt["endpoint_sha256"], "own paired endpoint hashes")
        require(len(endpoints) == 2, "both own descendants selected before final worlds")
        same(report["endpoint_sha256"], config["endpoint_sha256"], "same endpoint pins in terminal receipt")
        for view, endpoint, digest in zip(VIEWS, endpoints, config["endpoint_sha256"]):
            content = self.read(folder+f"/genome_{view}.npy")
            same(hashlib.sha256(content).hexdigest(), digest, "held-out endpoint copy")
            exact(np.load(io.BytesIO(content), allow_pickle=False), endpoint, "held-out genotype identity")
        assignment = self.artifact(folder, config["assignments"])
        require(set(assignment) == {"family_keys", "pre_contacts", "gap", "initial_rules"}, "fixed assignment schema")
        count, jax = config["expected_families"], random_helpers()[0]
        master = jax.random.fold_in(jax.random.PRNGKey(config["evaluation_seed"]), 0xE136)
        family_keys = np.stack([np.asarray(jax.random.split(jax.random.fold_in(master, family), 7)) for family in range(count)])
        table = np.asarray([CELLS[family % 4] for family in range(count)], np.int32)
        for column, name in enumerate(("pre_contacts", "gap")):
            exact(assignment[name], table[:, column], "preassigned complete final " + name)
        exact(assignment["family_keys"], family_keys, "genuinely separate final role keys")
        exact(assignment["initial_rules"], np.asarray([0, 1], np.int32), "both initial mappings per family")
        self.register_keys(family_keys, "held-out lineage/family roles distinct from training")
        require(len(report["raw_files"]) == count, "all assigned raw family artifacts")
        rows = []
        for family, manifest in enumerate(report["raw_files"]):
            started = time.monotonic()
            require(manifest["path"] == f"families/family_{family:04d}.npz", "ordered assigned family evidence")
            raw = self.artifact(folder, manifest)
            rows.append(audit_family(raw, family, [parent]+endpoints, family_keys[family], *map(int, table[family])))
            self.family_seconds.append(time.monotonic()-started)
        return dict(index=index, run=folder, measurements=summarize_rows(rows))


def self_check():
    uniform = np.full((12, 5), .2, np.float32)
    require(replay.signed_choice(uniform, 0) == 0, "uniform five-way first choice")
    perfect = np.zeros((12, 5), np.float32)
    perfect[np.arange(12), PLACEMENTS[:, 0]] = 1
    require(replay.signed_choice(perfect, 0) == 1 and replay.signed_choice(perfect, 1) == -1, "all five action indices and twelve orientations")
    require(replay.signed_choice(perfect, 0)-replay.signed_choice(perfect, 1) == 2, "opposite perfect choices yield signed history effect two")
    require(replay.signed_choice(perfect, 0)-replay.signed_choice(perfect, 0) == 0, "identical policies yield zero same-target history effect")
    probabilities = np.stack((uniform, .5*perfect+.5*uniform))[:, :, None, :]
    parts, split = scene.first_decomposition(probabilities, 0)
    require(abs(sum(split.values())+.5) < 1e-7 and np.all(parts["approach"] > 0), "approach/color split is exactly signed-score loss")
    rows = []
    for index in range(4):
        y = np.full((3, 2, 2), .125)
        y[2] = .5
        local = np.zeros((3, 2, 2, 2))
        local[2] = .3
        rows.append(dict(base_index=index, n_distractors=4, local=local, sampled=local.copy(),
            first_choice={name: np.full_like(local, .5) for name in CHOICE_FIELDS},
            decomposition={name: np.zeros((3, 2, 2)) for name in ("approach", "conditional")},
            solo=dict(Y=y, first_food=(1+y)/2, first_poison=(1-y)/2,
                no_contact=np.zeros_like(y), survival=np.ones_like(y), latency=np.ones_like(y))))
    positive = summarize_rows(rows)
    require(all(value["passed"] for value in positive["gates"].values()), "separate fully positive gates")
    stationary = copy.deepcopy(rows)
    for row in stationary:
        row["solo"]["Y"][2, :, 1] = .125
        row["solo"]["first_food"][2, :, 1] = .5625
        row["solo"]["first_poison"][2, :, 1] = .4375
    decisions = summarize_rows(stationary)["gates"]
    require(decisions["stationary_repair"]["passed"] and not decisions["reversal_advancement"]["passed"], "stationary repair cannot advance reversal")
    equal_control = copy.deepcopy(rows)
    for row in equal_control:
        for name in FIELDS:
            row["solo"][name][1] = row["solo"][name][2]
    require(not summarize_rows(equal_control)["gates"]["reversal_advancement"]["passed"], "gain over own parent is insufficient without fresh control")
    imprecise = estimate([-.5, .5, -.5, .5])
    require(imprecise["se"] > 0 and not replay.passes(imprecise, .05), "family uncertainty is not replaced by clone count")
    zero_contact = scene.behavior_summary({name: np.zeros(4) if name in ("Y", "first_food", "first_poison")
        else np.full(4, 50.) if name == "latency" else np.ones(4) for name in FIELDS})["contact_accuracy"]
    require(zero_contact["mean"] is None and zero_contact["se"] is None and zero_contact["contact_count"] == 0
        and zero_contact["assigned_lives"] == 8, "no-contact accuracy has explicit null and denominator")
    lethal = scene.outcomes(dict(food=np.asarray([[True], [False]]), poison=np.zeros((2, 1), bool),
                                 alive_after=np.zeros((2, 1), np.float32)))
    require(lethal["Y"] == 1 and lethal["survival"] == 0, "first contact retained even on death tick")
    ranks = tied_ranks(np.asarray([1, 1, 2, 2, 3], np.float32))
    exact(ranks, np.asarray([-.375, -.375, .125, .125, .5], np.float32), "average tied ranks retain ties")
    ordered = np.arange(16, dtype=np.float32)
    sparse, dense = tied_ranks(ordered), tied_ranks(-ordered)
    require(not np.array_equal(sparse, dense) and sparse[0] < sparse[-1] and dense[0] > dense[-1],
            "reversing objective scores reverses ranks on the same synthetic candidate population")
    exact(tied_ranks(np.ones(16, np.float32)), np.zeros(16, np.float32), "all-tied population has no rank direction")
    # All constants are fixtures. This check generates no worlds or random samples.
    return dict(status="self-check passed", groups=12, simulator_imported=False, random_samples_generated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-commit")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), allow_nan=False))
        return 0
    if not args.source_commit:
        parser.error("--source-commit is required")
    started, auditor = time.monotonic(), None
    report = dict(protocol=PROTOCOL, mode="development" if args.development else "training" if args.training else "final")
    try:
        auditor = Auditor(args.root, args.source_commit)
        # Read only statuses and filenames until the entire requested bundle is complete.
        dev_train = [auditor.completed("E136-benchmark-"+view, 2, 0) for view in VIEWS]
        dev_eval = auditor.completed("E136-benchmark-eval", 0, 4)
        final_train = [] if args.development else [(i, view, auditor.completed(f"E136-A{i}-{view}", 10, 0)) for i in range(3) for view in VIEWS]
        final_eval = [] if args.development or args.training else [auditor.completed(f"E136-A{i}", 0, 128) for i in range(3)]
        report["training"] = [auditor.training(folder, 0, True, view) for view, folder in zip(VIEWS, dev_train)]
        report["development"] = auditor.evaluation(dev_eval, 0, True)
        report["training"].extend(auditor.training(folder, i, False, view) for i, view, folder in final_train)
        report["manipulation"] = [auditor.manipulation(True, 0)]+([] if args.development else [auditor.manipulation(False, i) for i in range(3)])
        report["runs"] = [auditor.evaluation(folder, i, False) for i, folder in enumerate(final_eval)]
        gates_all = {name: dict(passed=all(row["measurements"]["gates"][name]["passed"] for row in report["runs"]),
            count=sum(row["measurements"]["gates"][name]["passed"] for row in report["runs"]),
            lineages=[row["index"] for row in report["runs"] if row["measurements"]["gates"][name]["passed"]])
            for name in ("stationary_repair", "reversal_advancement")} if final_eval else None
        report.update(status="passed", registered_source=args.source_commit, registered_source_code=auditor.source_manifest,
            final_evaluation_audited=bool(final_eval), scientific_gates=gates_all,
            scientific_gate_passed=None if gates_all is None else gates_all["reversal_advancement"]["passed"],
            limitations=["Candidate development and learning matrix updates during exposure/solo are not independently rerun.",
                "Both training scores, assigned scalar, exact ES transitions, paired key paths, physical traces and all family gates are independently reconstructed.",
                "Frozen first-decision logits are independently bounded from archived acquired W/x; this does not prove how the neural state was acquired.",
                "Ordinary solo Y includes encounter frequency; conditional contact accuracy and first-view occupancy/color effects are descriptive.",
                "Performance after matched acquisition and gaps does not establish autonomous reversal learning from birth."])
        code = 0
    except base.Incomplete as error:
        report.update(status="incomplete", error=str(error), final_evaluation_audited=False)
        code = 2
    except (base.AuditError, OSError, ValueError, KeyError, TypeError, base.subprocess.SubprocessError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", final_evaluation_audited=False)
        code = 1
    elapsed = time.monotonic()-started
    if auditor is not None:
        report["checked_file_sha256"] = auditor.hashes
        report["timing"] = dict(training_attempt_wall_seconds=auditor.training_seconds, family_wall_seconds=auditor.family_seconds,
            elapsed_seconds=elapsed, setup_and_summary_seconds=max(0., elapsed-sum(auditor.training_seconds)-sum(auditor.family_seconds)),
            training_order="development isolated/background, then lineage 0/1/2 isolated/background; attempts ascending",
            family_order="four development, then final lineage 0/1/2; (8,16),(16,32),(8,32),(16,16) cycle")
    encoded = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
        print(json.dumps(dict(status=report["status"], output=str(args.output), scientific_gates=report.get("scientific_gates"),
                             error=report.get("error")), allow_nan=False))
    else:
        print(encoded, end="")
    return code


if __name__ == "__main__":
    sys.exit(main())
