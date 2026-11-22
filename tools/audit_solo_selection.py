"""Independent E138 selection, compact solo physics and final-family audit.

This module never imports a Seed simulator or production optimizer. It reuses
the independently written E134/E135/E136 audit kernels, with E138 assignments,
training physics, objective routing and scientific decisions reconstructed here.
JAX is used only for specified random primitives and float32 ES arithmetic.
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
import audit_background_selection as background
from e138_feedback_opportunities import feedback_opportunities, FIELDS as OPPORTUNITY_FIELDS

replay, scene = background.replay, background.scene
base, source_bytes, encounter = background.base, background.source_bytes, background.encounter
require, same, subset, typed = base.require, base.same, base.subset, base.typed
exact, fingerprint, prefixed = replay.exact, replay.fingerprint, replay.prefixed
CFG, MOVES, PLACEMENTS = replay.CFG, replay.MOVES, replay.PLACEMENTS
estimate = scene.estimate
PROTOCOL = "solo-selection-v1"
OBJECTIVES = ("background", "solo")
GENOTYPES = ("parent", "background", "solo")
VIEWS = ("isolated", "background")
HISTORIES = ("stationary", "reversal")
TRAIN_ROLES = ("script", "chamber", "background", "probe", "solo_layout", "solo_action")
KEY_ROLES = ("script", "birth", "chamber", "background", "probe", "solo_layout", "solo_action")
DEVELOPMENT_TRAINING_RUNS = ("E138-benchmark-background", "E138-benchmark-solo")
DEVELOPMENT_EVAL_NAME = "E138-benchmark-eval"
TRAIN_CELLS = np.asarray(((8, 0), (8, 8), (16, 0), (16, 8)), np.int32)
CELLS = ((8, 16), (16, 32), (8, 32), (16, 16))
FIELDS, CHOICE_FIELDS = background.FIELDS, background.CHOICE_FIELDS
SOURCE_PATHS = (background.SOURCE_PATHS - {"scripts/run_background_selection.py"}
                | {"scripts/run_solo_selection.py", "seed/solo_selection.py"})
BATCH_FIELDS = {"world_keys", "role_keys", "schedule_key", "pre_contacts", "gap", "scripts",
                "backgrounds", "solo_grids", "solo_positions"}
SOLO_FIELDS = {"action", "sampled_action", "food", "poison", "alive_after", "item", "energy",
               "logits", "incoming_denergy", "forced_action", "hebbian_enabled", "predictor_enabled"}
SAMPLE_FIELDS = {"eps", "candidate_genomes", "world_keys", "development_keys", "rank_weights",
                 "fitness", "view_fitness", "probabilities", "logits", "sampled_action",
                 "probe_action_keys", "counts", "survived", "exposure_energy_bounds"}
SAMPLE_FIELDS |= {"batch_" + name for name in BATCH_FIELDS} | {"solo_" + name for name in SOLO_FIELDS}


@lru_cache(maxsize=1)
def solo_random_helpers():
    """Four common tapes, then native-shape categorical draws at all saved logits."""
    jax, jnp = source_bytes.random_runtime()

    @jax.jit
    def family_tapes(keys):
        def one(initial):
            def tick(key, unused):
                next_key, action_key, regrowth_key = jax.random.split(key, 3)
                cell = jax.random.randint(regrowth_key, (1, 2), 0, 16)[0]
                return next_key, (action_key, cell)
            return jax.lax.scan(tick, initial, None, length=50)[1]
        return jax.vmap(one)(keys)

    @jax.jit
    def categorical(keys, logits):
        return jax.vmap(lambda key, logit: jax.random.categorical(key, logit[None, :])[0])(keys, logits)

    return family_tapes, categorical


def first_contact(food, poison):
    """All assigned trajectories, including contacts before an eventual death."""
    food, poison = np.asarray(food, bool), np.asarray(poison, bool)
    require(food.shape == poison.shape and food.ndim >= 1, "paired contact histories")
    require(not np.any(food & poison), "no tick consumes both meanings")
    contact = food | poison
    present = contact.any(-1)
    first = np.argmax(contact, axis=-1)
    first_food = present & np.take_along_axis(food, first[..., None], -1)[..., 0]
    first_poison = present & ~first_food
    return dict(Y=first_food.astype(np.float64) - first_poison,
                first_food=first_food, first_poison=first_poison, no_contact=~present,
                latency=np.where(present, first + 1, food.shape[-1]))


def candidate_scores(samples, objective):
    require(objective in OBJECTIVES, "registered selection objective")
    probability = typed(samples, "probabilities", (16, 4, 2, 2, 12, 5), "f")
    logits = typed(samples, "logits", probability.shape, "f")
    encounter.softmax(logits, probability)
    require(np.all(np.abs(logits) <= 3), "native bounded focal logits")
    signed = np.empty((16, 4, 2, 2), np.float64)
    for initial in range(2):
        for history in range(2):
            signed[:, :, initial, history] = replay.signed_choice(
                probability[:, :, initial, history], initial ^ history)
    outcome = first_contact(typed(samples, "solo_food", (16, 4, 2, 2, 50), "b"),
                            typed(samples, "solo_poison", (16, 4, 2, 2, 50), "b"))
    both = typed(samples, "view_fitness", (16, 4, 2), "f")
    replay.physical_close(both[:, :, 0], signed.mean((2, 3)), "full-view expected focal first choice S")
    exact(both[:, :, 1], outcome["Y"].mean((2, 3)).astype(both.dtype),
          "all-assigned physical first-item Y, including no contact and lethal contact")
    selected = typed(samples, "fitness", (16, 4), "f")
    exact(selected, both[:, :, OBJECTIVES.index(objective)], "only the assigned objective enters ES")
    _, jnp = source_bytes.random_runtime()
    return np.asarray(jnp.asarray(selected).mean(1))


def optimizer_transition(previous, samples, objective):
    scores = candidate_scores(samples, objective)
    eps = typed(samples, "eps", (16, 155), "f")
    exact(eps[8:], -eps[:8], "antithetic mutation pairs")
    jax, jnp = source_bytes.random_runtime()
    next_key, noise_key, world_key, birth_key = jax.random.split(jnp.asarray(previous["key"]), 4)
    exact(eps[:8], np.asarray(jax.random.normal(noise_key, (8, 155))), "independently derived mutation noise")
    exact(typed(samples, "world_keys", (4, 2), "u"), np.asarray(jax.random.split(world_key, 4)), "four fresh family roots")
    births = np.asarray(jax.random.split(birth_key, 32)).reshape(8, 4, 2)
    exact(typed(samples, "development_keys", (16, 4, 2), "u"), np.concatenate((births, births)), "antithetic twins share birth keys")
    exact(typed(samples, "candidate_genomes", (16, 155), "f"),
          np.asarray(jnp.asarray(previous["mu"]) + .05*jnp.asarray(eps)), "candidate genotype arithmetic")
    weights = background.tied_ranks(scores)
    exact(typed(samples, "rank_weights", (16,), "f"), weights, "independently sorted average tied ranks")
    informative = bool(np.any(scores != scores[0]))
    expected = {name: previous[name].copy() for name in ("mu", "m", "v", "t")}
    if informative:
        mu, m, v = (jnp.asarray(previous[name]) for name in ("mu", "m", "v"))
        gradient = (jnp.asarray(weights)[:, None]*jnp.asarray(eps)).sum(0)/(.05*16) - .005*mu
        reference = (weights.astype(np.float64)[:, None]*eps.astype(np.float64)).sum(0)/.8 - .005*previous["mu"].astype(np.float64)
        magnitude = (np.abs(weights.astype(np.float64)[:, None]*eps.astype(np.float64))).sum(0)/.8 + .005*np.abs(previous["mu"])
        unit = np.finfo(np.float32).eps/2
        require(np.all(np.abs(np.asarray(gradient, np.float64) - reference)
                <= 22*unit/(1 - 22*unit)*magnitude + 22*np.finfo(np.float32).smallest_subnormal),
                "independent ES reduction roundoff envelope")
        step = int(previous["t"]) + 1
        m, v = .9*m + .1*gradient, .999*v + .001*gradient**2
        mu = mu + .02*(m/(1 - .9**step))/(jnp.sqrt(v/(1 - .999**step)) + 1e-8)
        expected.update(mu=np.asarray(mu), m=np.asarray(m), v=np.asarray(v), t=np.asarray(step, previous["t"].dtype))
    expected["key"] = np.asarray(next_key)
    return expected, scores, informative


def compact_solo_physics(samples, roles):
    """Reconstruct 256 physical worlds in parallel, retaining no tick grids."""
    shape, leading, trajectories = (16, 4, 2, 2, 50), (16, 4, 2, 2), 256
    data = {}
    for name in SOLO_FIELDS:
        kind = "b" if name in ("food", "poison", "hebbian_enabled", "predictor_enabled") else (
               "i" if name in ("action", "sampled_action", "forced_action", "item") else "f")
        value_shape = shape + (5,) if name == "logits" else shape
        data[name] = typed(samples, "solo_" + name, value_shape, kind).reshape((trajectories, 50) + ((5,) if name == "logits" else ()))
    require(np.all(data["forced_action"] == -1), "every scored solo action is voluntary")
    require(np.all(data["hebbian_enabled"]) and np.all(data["predictor_enabled"]), "native plasticity stays enabled throughout solo")
    require(np.all(np.abs(data["logits"]) <= 3), "native bounded solo logits")
    require(np.isin(data["action"], np.arange(5)).all(), "five-action indexing")
    require(np.isin(data["alive_after"], (0, 1)).all(), "binary all-assigned survival")
    tapes, draw = solo_random_helpers()
    action_keys, regrowth_cells = (np.asarray(value) for value in tapes(roles[:, 5]))
    require(action_keys.shape == (4, 50, 2) and regrowth_cells.shape == (4, 50, 2), "fifty complete family tapes")
    keys = np.broadcast_to(action_keys[None, :, None, None], leading + (50, 2))
    sampled = np.asarray(draw(keys.reshape(-1, 2), data["logits"].reshape(-1, 5))).reshape(trajectories, 50)
    exact(data["sampled_action"], sampled, "native categorical draws from independently derived common action keys")
    exact(data["action"], sampled, "chosen solo action is the native sampled action")
    cells = np.broadcast_to(regrowth_cells[None, :, None, None], leading + (50, 2)).reshape(trajectories, 50, 2)
    grid = np.broadcast_to(samples["batch_solo_grids"][None, :, None, None], leading + (16, 16)).reshape(trajectories, 256).copy()
    pos = np.broadcast_to(samples["batch_solo_positions"][:, 0][None, :, None, None], leading + (2,)).reshape(trajectories, 2).copy()
    rule = np.broadcast_to(np.bitwise_xor.outer(np.arange(2), np.arange(2))[None, None], leading).reshape(-1)
    rows = np.arange(trajectories)
    energy, alive, incoming = np.ones(trajectories, np.float32), np.ones(trajectories, np.float32), np.zeros(trajectories, np.float32)
    max_energy_residual = 0.
    for tick in range(50):
        exact(data["incoming_denergy"][:, tick], incoming, "native incoming energy change at solo tick " + str(tick))
        pos = (pos + MOVES[sampled[:, tick]]*alive[:, None].astype(np.int32)) % 16
        position_index = pos[:, 0]*16 + pos[:, 1]
        item = grid[rows, position_index]
        eat = (alive > 0) & (item > 0)
        food, poison = eat & (item == 1 + rule), eat & (item == 2 - rule)
        exact(data["item"][:, tick], item, "actual encountered item at solo tick " + str(tick))
        exact(data["food"][:, tick], food, "physical food at solo tick " + str(tick))
        exact(data["poison"][:, tick], poison, "physical poison at solo tick " + str(tick))
        predicted = np.clip(energy - np.float32(.003)*alive + np.float32(.25)*food
                            - np.float32(.25)*poison, np.float32(0), np.float32(2))
        actual = data["energy"][:, tick]
        replay.physical_close(actual, predicted, "clipped physical energy at solo tick " + str(tick))
        max_energy_residual = max(max_energy_residual, float(np.max(np.abs(actual.astype(np.float64) - predicted))))
        next_alive = alive*(actual > 0)
        exact(data["alive_after"][:, tick], next_alive, "absorbing death at solo tick " + str(tick))
        # Use the checked device energy at each boundary, as the independent
        # final-family physics kernel does, avoiding accumulated float rounding.
        incoming = (actual - energy)*alive
        energy, alive = actual.copy(), next_alive
        grid[rows, position_index] = np.where(eat, 0, item)
        regrowth_index = cells[:, tick, 0]*16 + cells[:, tick, 1]
        grid[rows, regrowth_index] = np.where(eat, item, grid[rows, regrowth_index])
    outcome = first_contact(data["food"], data["poison"])
    return dict(trajectories=trajectories, ticks=50, checked_transitions=trajectories*50,
                max_energy_residual=max_energy_residual, energy_check_max_relative_tolerance=float(8*np.finfo(np.float32).eps),
                first_food=int(np.count_nonzero(outcome["first_food"])), first_poison=int(np.count_nonzero(outcome["first_poison"])),
                no_contact=int(np.count_nonzero(outcome["no_contact"])), surviving=int(np.count_nonzero(alive)))


def summarize_rows(rows):
    n = len(rows)
    require(n >= 2 and [row["base_index"] for row in rows] == list(range(n)), "all assigned families in denominator")
    local = {name: np.stack([row[name] for row in rows]).mean(2) for name in ("local", "sampled")}
    solo = {name: np.stack([row["solo"][name] for row in rows]).mean(2) for name in FIELDS}
    require(all(value.shape == (n, 3, 2, 2) and np.isfinite(value).all() for value in local.values()), "paired family local dimensions")
    require(all(value.shape == (n, 3, 2) and np.isfinite(value).all() for value in solo.values()), "paired family solo dimensions")
    require(np.array_equal(solo["Y"], solo["first_food"] - solo["first_poison"])
            and np.array_equal(solo["first_food"] + solo["first_poison"] + solo["no_contact"], np.ones((n, 3, 2))),
            "every assigned life has one signed first-contact outcome")
    choice = {name: np.stack([row["first_choice"][name] for row in rows]).mean(2) for name in CHOICE_FIELDS}
    decomposition = {name: np.stack([row["decomposition"][name] for row in rows]).mean(2) for name in ("approach", "conditional")}
    arms, contrasts = {}, {}
    for genotype, label in enumerate(GENOTYPES):
        arms[label] = dict(histories={})
        for history, hlabel in enumerate(HISTORIES):
            arms[label]["histories"][hlabel] = dict(
                local={view: estimate(local["local"][:, genotype, history, v]) for v, view in enumerate(VIEWS)},
                sampled={view: estimate(local["sampled"][:, genotype, history, v]) for v, view in enumerate(VIEWS)},
                first_choice={view: {name: estimate(value[:, genotype, history, v]) for name, value in choice.items()} for v, view in enumerate(VIEWS)},
                solo=scene.behavior_summary({name: value[:, genotype, history] for name, value in solo.items()}),
                sparse_minus_dense_components={name: estimate(value[:, genotype, history]) for name, value in decomposition.items()})
    for history, label in enumerate(HISTORIES):
        contrasts[label + "_solo_absolute_Y"] = estimate(solo["Y"][:, 2, history])
        for comparator, name in ((0, "parent"), (1, "background")):
            for field, values in (("solo", solo["Y"]), ("survival", solo["survival"])):
                contrasts[f"{label}_{field}_gain_vs_{name}"] = estimate(values[:, 2, history] - values[:, comparator, history])
            gains = local["local"][:, 2, history] - local["local"][:, comparator, history]
            for view, tag in ((0, "sparse"), (1, "dense")):
                contrasts[f"{label}_{tag}_gain_vs_{name}"] = estimate(gains[:, view])
            contrasts[f"{label}_dense_minus_sparse_interaction_vs_{name}"] = estimate(gains[:, 1] - gains[:, 0])
            for field, values in decomposition.items():
                contrasts[f"{label}_sparse_minus_dense_{field}_component_vs_{name}"] = estimate(values[:, 2, history] - values[:, comparator, history])
        contrasts[label + "_background_control_solo_gain_vs_parent"] = estimate(solo["Y"][:, 1, history] - solo["Y"][:, 0, history])
        for view, tag in ((0, "sparse"), (1, "dense")):
            contrasts[f"{label}_background_control_{tag}_gain_vs_parent"] = estimate(local["local"][:, 1, history, view] - local["local"][:, 0, history, view])
    result = dict(families=n, assigned_lives_per_genotype_history=2*n, genotype_order=list(GENOTYPES),
                  local_view_order=list(VIEWS), arms=arms, contrasts=contrasts,
                  distractors=estimate([row["n_distractors"] for row in rows]),
                  zero_distractors=estimate([row["n_distractors"] == 0 for row in rows]))
    result["gates"] = gates(result)
    if any("feedback_opportunities" in row for row in rows):
        require(all("feedback_opportunities" in row for row in rows), "complete descriptive opportunity assignment")
        result["feedback_opportunities"] = summarize_opportunities(rows)
    return result


def summarize_opportunities(rows):
    n = len(rows)
    values = {name: np.stack([row["feedback_opportunities"][name] for row in rows]) for name in OPPORTUNITY_FIELDS}
    require(all(value.shape == (n, 3, 2, 2) and value.dtype == bool for value in values.values()), "opportunity life flags")
    families = {name: value.mean(2) for name, value in values.items()}
    arms, contrasts = {}, {}
    for genotype, label in enumerate(GENOTYPES):
        arms[label] = {}
        for history, hlabel in enumerate(HISTORIES):
            arms[label][hlabel] = {name: dict(count=int(value[:, genotype, :, history].sum()),
                assigned_lives=2*n, family_rate=estimate(families[name][:, genotype, history])) for name, value in values.items()}
    for history, hlabel in enumerate(HISTORIES):
        for comparator, name in ((0, "parent"), (1, "background")):
            contrasts[hlabel + "_solo_minus_" + name] = {
                field: estimate(value[:, 2, history] - value[:, comparator, history]) for field, value in families.items()}
        contrasts[hlabel + "_background_minus_parent"] = {
            field: estimate(value[:, 1, history] - value[:, 0, history]) for field, value in families.items()}
    return dict(independent_families=n, assigned_lives_per_genotype_history=2*n, arms=arms, paired_rate_differences=contrasts,
        interpretation="Descriptive all-assigned opportunities through the second contact; no causal learning or retention claim and no gate contribution.")


def gates(summary):
    contrasts, decisions = summary["contrasts"], {}
    for label, primary, safeguard in (("stationary_repair", "stationary", "reversal"),
                                      ("reversal_advancement", "reversal", "stationary")):
        key = primary + "_solo_absolute_Y"
        criteria = {key: replay.passes(contrasts[key])}
        for comparison in ("parent", "background"):
            key = f"{primary}_solo_gain_vs_{comparison}"
            criteria[key] = replay.passes(contrasts[key], .05)
            key = f"{safeguard}_solo_gain_vs_{comparison}"
            criteria[key + "_safeguard"] = bool(contrasts[key]["lower_2se"] > -.05)
            for history in HISTORIES:
                key = f"{history}_survival_gain_vs_{comparison}"
                criteria[key + "_safeguard"] = bool(contrasts[key]["lower_2se"] > -.05)
        decisions[label] = dict(passed=all(criteria.values()), criteria=criteria,
                               failed_criteria=[name for name, passed in criteria.items() if not passed])
    return decisions


class Auditor(background.Auditor):
    def metadata(self, folder, index, mode, view=None):
        config = self.json(folder + "/config.json")
        development, training = mode.startswith("development"), mode.endswith("training")
        attempts, families = (2 if development else 10, 0) if training else (0, 4 if development else 128)
        train_seed, eval_seed = 2026121800 + (0 if development else index + 1), 2026121900 + (0 if development else index + 1)
        name = DEVELOPMENT_TRAINING_RUNS[OBJECTIVES.index(view)] if development and training else (
            DEVELOPMENT_EVAL_NAME if development else f"E138-A{index}-{view}" if training else f"E138-A{index}")
        require(Path(folder).name.endswith("_" + name), "registered E138 run name")
        subset(config, dict(protocol=PROTOCOL, mode=mode, seed_index=index, cfg=CFG, l2=.005, view=view,
                            expected_attempts=attempts, expected_families=families), "registered E138 design")
        same(config["training_seed"], train_seed if training else None, "independent lineage training master")
        same(config["evaluation_seed"], None if training else eval_seed, "distinct held-out master")
        same(config["key_roles"], list(KEY_ROLES), "explicit held-out key roles")
        same(config["genotype_order"], list(GENOTYPES), "explicit parent/proxy/direct genotype order")
        if training:
            require(config["endpoint_sha256"] is None and config["assignments"] is None and not config["train_receipts"],
                    "training precedes endpoints and held-out assignments")
        else:
            require(isinstance(config["endpoint_sha256"], list) and len(config["endpoint_sha256"]) == 2, "exactly two pinned endpoints")
        args = config["args"]
        subset(args, dict(name=name, eval_protocol=PROTOCOL, generations=attempts, eval_worlds=families,
                         pop=16, worlds=4, sigma=.05, lr=.02, seed=train_seed if training else eval_seed,
                         no_plastic=False, allow_dirty=False, quick=False, resume=None, cfg=None, objective=None,
                         memory_condition=None, max_train_seconds=None, checkpoint_every=10), "frozen native invocation")
        require(args.get("controller") in (None, "rules-v0") and args.get("structural_plasticity") in (None, "off")
                and not args.get("persistence", False), "unchanged native controller")
        commit = config["git"]["commit"]
        require(config["git"]["dirty"] is False and re.fullmatch(r"[0-9a-f]{40}", commit), "clean frozen source")
        same(self.code_trees(commit), self.expected_trees, "registered scientific source trees")
        require(set(config["source_code"]) == SOURCE_PATHS, "complete transitive source manifest")
        if commit not in self.checked_source_commits:
            context = type("SourceContext", (), {"root": self.root, "source": commit})()
            attributes = source_bytes.Auditor.source_attributes(context, list(config["source_code"]))
            for path, digest in config["source_code"].items():
                blob = base.subprocess.run(["git", "show", commit + ":" + path], cwd=self.root,
                                           capture_output=True, check=True, timeout=15).stdout
                source_bytes.source_hash_representation(blob, digest, attributes[path])
            self.checked_source_commits.add(commit)
        if self.source_manifest is None:
            self.source_manifest = config["source_code"]
        same(config["source_code"], self.source_manifest, "same scientific kernels for development and held-out work")
        path, digest = base.ORIGINALS[index]
        subset(config["source"], dict(path=path, sha256=digest, training_seed=index), "original E103 parent")
        same(args["start"], "file:" + path, "own original parent start")
        content = self.read(path)
        require(hashlib.sha256(content).hexdigest() == digest and self.read(folder + "/genome_start.npy") == content,
                "unchanged original parent bytes")
        parent = np.load(io.BytesIO(content), allow_pickle=False)
        require(parent.shape == (155,) and parent.dtype == np.float32 and np.isfinite(parent).all(), "native 155-number genome")
        report = self.json(folder + "/summary.json")
        subset(report, dict(status="finished", protocol=PROTOCOL, mode=mode, seed_index=index,
                            completed_attempts=attempts, completed_families=families), "complete assigned workload")
        require(0 < report["elapsed_seconds"] <= config["max_seconds"], "guarded process bound")
        require(len(report["attempt_wall_seconds"]) == attempts and len(report["family_wall_seconds"]) == families
                and all(np.isfinite(v) and v >= 0 for v in report["attempt_wall_seconds"] + report["family_wall_seconds"]), "complete process timing receipt")
        return config, report, parent

    def training(self, folder, index, development, view):
        config, report, parent = self.metadata(folder, index, "development-training" if development else "training", view)
        pair_id = (development, index)
        same_args = {name: value for name, value in config["args"].items() if name != "name"}
        if view == OBJECTIVES[0]:
            self.paired_configs[pair_id] = same_args
        else:
            same(same_args, self.paired_configs[pair_id], "paired arms differ only in assigned objective name")
        attempts = config["expected_attempts"]
        require(len(report["attempts"]) == attempts + 1 and len(report["samples"]) == attempts, "all attempted optimizer artifacts")
        require(report["attempts"][0]["path"] == "generations/attempt_0000.npz", "fixed starting checkpoint")
        previous = self.artifact(folder, report["attempts"][0])
        require(set(previous) == {"mu", "m", "v", "t", "key"}, "exact optimizer schema")
        exact(typed(previous, "mu", (155,), "f"), parent, "both populations start from own parent")
        for name in ("m", "v"):
            require(not typed(previous, name, (155,), "f").any(), "fresh Adam " + name)
        require(int(typed(previous, "t", (), "i")) == 0, "fresh Adam count")
        jax = background.random_helpers()[0]
        exact(typed(previous, "key", (2,), "u"), np.asarray(jax.random.split(jax.random.PRNGKey(config["training_seed"]))[0]), "initial optimizer key")
        if view == OBJECTIVES[0]:
            self.register_keys(previous["key"][None], "new lineage optimizer")
        updates = []
        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            self.trace("attempt-start", run=folder, index=attempt)
            require(report["attempts"][attempt]["path"] == f"generations/attempt_{attempt:04d}.npz"
                    and report["samples"][attempt-1]["path"] == f"generations/samples_{attempt:04d}.npz", "ordered immutable attempt artifacts")
            current, samples = self.artifact(folder, report["attempts"][attempt]), self.artifact(folder, report["samples"][attempt-1])
            require(set(current) == set(previous), "consistent optimizer schema")
            require(set(samples) == SAMPLE_FIELDS, "exact compact candidate evidence schema")
            expected, scores, informative = optimizer_transition(previous, samples, view)
            both_ranks = background.view_ranks(samples)
            exact(samples["rank_weights"], both_ranks[OBJECTIVES.index(view)], "assigned scalar alone supplies tied ranks")
            for name, value in expected.items():
                exact(current[name], value, "independent optimizer transition " + name)
            if view == OBJECTIVES[0]:
                self.register_keys(current["key"][None], "optimizer continuation")
                self.register_keys(samples["world_keys"], "fresh training family roots")
                self.register_keys(samples["development_keys"][:8], "fresh paired birth keys")
            physical = self.training_batch(samples, register=view == OBJECTIVES[0])
            paired = {name: value for name, value in samples.items()
                      if name.startswith("batch_") or name in ("eps", "world_keys", "development_keys")}
            paired["optimizer_key"] = previous["key"]
            if attempt == 1:
                paired.update({"first_" + name: value for name, value in samples.items()
                               if name not in ("fitness", "rank_weights")})
            identity = (development, index, attempt)
            if view == OBJECTIVES[0]:
                self.paired_inputs[identity] = {name: fingerprint(value) for name, value in paired.items()}
            else:
                require(set(paired) == set(self.paired_inputs[identity]), "same paired primitive evidence")
                for name, value in paired.items():
                    exact(fingerprint(value), self.paired_inputs[identity][name], "paired common randomness and first-population identity " + name)
            perturbation_delta = ((both_ranks[1].astype(np.float64) - both_ranks[0])[:, None]
                                  * samples["eps"].astype(np.float64)).sum(0)/(.05*16)
            _, jnp = source_bytes.random_runtime()
            all_scores = np.asarray(jnp.asarray(samples["view_fitness"]).mean(1))
            updates.append(dict(attempt=attempt, informative=informative, optimizer_updates=int(current["t"]),
                changed_parameters=int(np.count_nonzero(current["mu"] != previous["mu"])),
                score_mean=float(scores.mean()), score_best=float(scores.max()), distinct_scores=int(len(np.unique(scores))),
                separated_antithetic_pairs=int(np.count_nonzero(scores[:8] != scores[8:])),
                objective_distinct_scores={name: int(len(np.unique(all_scores[:, i]))) for i, name in enumerate(OBJECTIVES)},
                objective_separated_antithetic_pairs={name: int(np.count_nonzero(all_scores[:8, i] != all_scores[8:, i])) for i, name in enumerate(OBJECTIVES)},
                view_rank_vectors={name: both_ranks[i].tolist() for i, name in enumerate(OBJECTIVES)},
                view_rank_vectors_equal=bool(np.array_equal(both_ranks[0], both_ranks[1])),
                view_rank_difference_norm=float(np.linalg.norm(both_ranks[1].astype(np.float64) - both_ranks[0])),
                rank_weighted_perturbation_difference_norm=float(np.linalg.norm(perturbation_delta)), solo_physics=physical))
            previous = current
            self.training_seconds.append(time.monotonic() - started)
            self.trace("attempt-complete", run=folder, index=attempt, wall_seconds=self.training_seconds[-1])
        content = self.read(folder + "/genome_final.npy")
        endpoint = np.load(io.BytesIO(content), allow_pickle=False)
        exact(endpoint, previous["mu"], "endpoint is last attempted mean, not selected checkpoint")
        same(report["endpoint_sha256"], hashlib.sha256(content).hexdigest(), "terminal endpoint hash")
        same(report["optimizer_updates"], int(previous["t"]), "all-tied attempts preserve Adam count")
        self.training_reports[folder] = (endpoint, config, report)
        result = dict(index=index, view=view, run=folder, attempts=updates, endpoint_sha256=report["endpoint_sha256"],
            view_rank_vectors_equal_attempts=sum(row["view_rank_vectors_equal"] for row in updates),
            view_rank_vectors_different_attempts=sum(not row["view_rank_vectors_equal"] for row in updates))
        self.training_measurements[(development, index, view)] = result
        return result

    def manipulation(self, development, index):
        arms = [self.training_measurements[(development, index, view)] for view in OBJECTIVES]
        arrays = [self.training_reports[arm["run"]][0] for arm in arms]
        return dict(development=development, index=index, first_attempt_matching_candidates_and_both_scores=True,
                    ranks_on_each_realized_population={arm["view"]: dict(equal_attempts=arm["view_rank_vectors_equal_attempts"],
                        different_attempts=arm["view_rank_vectors_different_attempts"]) for arm in arms},
                    endpoint_parameters_identical=bool(np.array_equal(fingerprint(arrays[0]), fingerprint(arrays[1]))),
                    endpoint_changed_parameters=int(np.count_nonzero(arrays[0] != arrays[1])),
                    endpoint_difference_norm=float(np.linalg.norm(arrays[1].astype(np.float64) - arrays[0])))

    def training_batch(self, samples, *, register):
        batch = prefixed(samples, "batch_")
        require(set(batch) == BATCH_FIELDS, "exact matched training batch schema")
        exact(batch["world_keys"], samples["world_keys"], "batch roots match optimizer")
        jax, _, script, background_grid, _, draws, _ = background.random_helpers()
        roles = np.stack([np.asarray(jax.random.split(key, 6)) for key in samples["world_keys"]])
        exact(typed(batch, "role_keys", (4, 6, 2), "u"), roles, "six independent training roles")
        schedule_key = np.asarray(jax.random.fold_in(samples["world_keys"][0], 0xE138))
        exact(typed(batch, "schedule_key", (2,), "u"), schedule_key, "independent E138 schedule key")
        cells = TRAIN_CELLS[np.asarray(jax.random.permutation(schedule_key, 4))]
        exact(typed(batch, "pre_contacts", (4,), "i"), cells[:, 0], "balanced pre-contact schedule")
        exact(typed(batch, "gap", (4,), "i"), cells[:, 1], "balanced gap schedule")
        if register:
            self.register_keys(roles, "fresh training roles")
            self.register_keys(schedule_key[None], "fresh training schedule")
        typed(batch, "scripts", (4, 32, 5), "i")
        typed(batch, "backgrounds", (4, 16, 16), "i")
        typed(batch, "solo_grids", (4, 16, 16), "i")
        typed(batch, "solo_positions", (4, 1, 2), "i")
        world = replay.random_helpers()[3]
        for family in range(4):
            expected_script = np.asarray(script(roles[family, 0])).copy()
            expected_script[int(cells[family, 0]) + 16:] = 0
            exact(batch["scripts"][family], expected_script, "balanced common script with zero padding")
            exact(batch["backgrounds"][family], np.asarray(background_grid(roles[family, 2])), "fresh common focal background")
            grid, pos = world(roles[family, 4])
            exact(batch["solo_grids"][family], np.asarray(grid), "independently derived ordinary grid")
            exact(batch["solo_positions"][family], np.asarray(pos), "independently derived ordinary starting position")
        probe_keys = np.stack([np.asarray(jax.random.split(key, 12)) for key in roles[:, 3]])
        broadcast = np.broadcast_to(probe_keys[None, :, None, None], (16, 4, 2, 2, 12, 2))
        action_keys, sampled = draws(broadcast.reshape(-1, 2), samples["logits"].reshape(-1, 5))
        exact(typed(samples, "sampled_action", (16, 4, 2, 2, 12), "i"), np.asarray(sampled).reshape(16, 4, 2, 2, 12), "common native focal action randomness")
        exact(typed(samples, "probe_action_keys", (16, 4, 2, 2, 12, 2), "u"), np.asarray(action_keys).reshape(16, 4, 2, 2, 12, 2), "independent focal action keys")
        counts = typed(samples, "counts", (16, 4, 2, 2), "fiu")
        for family, pre in enumerate(cells[:, 0]):
            replay.physical_close(counts[:, family], np.broadcast_to([[int(pre)//2, int(pre)//2], [8, 8]], (16, 2, 2)), "prescribed contacts are not selectable fitness")
        require(np.all(typed(samples, "survived", (16, 4), "fbiu") == 1), "all assigned acquired exposures survive")
        bounds = typed(samples, "exposure_energy_bounds", (16, 4, 2), "f")
        require(np.all(bounds[..., 0] >= .462 - 1e-5) and np.all(bounds[..., 1] <= 1.25 + 1e-5)
                and np.all(bounds[..., 0] <= bounds[..., 1]), "deterministic paired exposure energy bound")
        return compact_solo_physics(samples, roles)

    def evaluation(self, folder, index, development):
        config, report, parent = self.metadata(folder, index, "development-evaluation" if development else "evaluation")
        receipts = config["train_receipts"]
        require(len(receipts) == (2 if development else 6), "both arms of every lineage pinned before evaluation")
        endpoints = []
        for ordinal, receipt in enumerate(receipts):
            lineage, view = ordinal//2, OBJECTIVES[ordinal % 2]
            require(receipt["seed_index"] == lineage and receipt["view"] == view and receipt["run"] in self.training_reports, "fixed lineage-then-objective receipt order")
            endpoint, train_config, train_report = self.training_reports[receipt["run"]]
            same(train_config["view"], view, "receipt objective matches completed population")
            paths = dict(config_sha256="config.json", summary_sha256="summary.json",
                         checkpoint_sha256=f"generations/attempt_{2 if development else 10:04d}.npz", endpoint_sha256="genome_final.npy")
            for name, path in paths.items():
                same(receipt[name], hashlib.sha256(self.read(receipt["run"] + "/" + path)).hexdigest(), "immutable endpoint receipt " + name)
            if lineage == index:
                endpoints.append(endpoint)
                same(config["endpoint_sha256"][ordinal % 2], receipt["endpoint_sha256"], "own paired endpoint hashes")
        require(len(endpoints) == 2, "both own descendants fixed before final worlds")
        same(report["endpoint_sha256"], config["endpoint_sha256"], "same endpoint pins in terminal receipt")
        for view, endpoint, digest in zip(OBJECTIVES, endpoints, config["endpoint_sha256"]):
            content = self.read(folder + f"/genome_{view}.npy")
            same(hashlib.sha256(content).hexdigest(), digest, "held-out endpoint copy")
            exact(np.load(io.BytesIO(content), allow_pickle=False), endpoint, "held-out genotype identity")
        assignment = self.artifact(folder, config["assignments"])
        require(set(assignment) == {"family_keys", "pre_contacts", "gap", "initial_rules"}, "fixed assignment schema")
        assignment_started = time.monotonic()
        count, jax = config["expected_families"], background.random_helpers()[0]
        master = jax.random.fold_in(jax.random.PRNGKey(config["evaluation_seed"]), 0xE138)
        family_keys = np.stack([np.asarray(jax.random.split(jax.random.fold_in(master, family), 7)) for family in range(count)])
        table = np.asarray([CELLS[family % 4] for family in range(count)], np.int32)
        for column, name in enumerate(("pre_contacts", "gap")):
            exact(assignment[name], table[:, column], "preassigned complete held-out " + name)
        exact(assignment["family_keys"], family_keys, "independent E138 evaluation key derivation")
        exact(assignment["initial_rules"], np.asarray([0, 1], np.int32), "both initial mappings")
        self.register_keys(family_keys, "held-out lineage/family roles distinct from training")
        self.assignment_seconds[folder] = time.monotonic() - assignment_started
        require(len(report["raw_files"]) == count, "all assigned raw family archives")
        rows = []
        for family, manifest in enumerate(report["raw_files"]):
            started = time.monotonic()
            self.trace("family-start", run=folder, index=family)
            require(manifest["path"] == f"families/family_{family:04d}.npz", "ordered assigned family evidence")
            raw = self.artifact(folder, manifest)
            row = background.audit_family(raw, family, [parent] + endpoints, family_keys[family], *map(int, table[family]))
            row["feedback_opportunities"] = feedback_opportunities(
                raw["solo_observation"], raw["solo_action"], raw["solo_alive_before"])
            rows.append(row)
            self.family_seconds.append(time.monotonic() - started)
            self.trace("family-complete", run=folder, index=family, wall_seconds=self.family_seconds[-1])
        return dict(index=index, run=folder, measurements=summarize_rows(rows))


def self_check():
    background.self_check()
    outcome = first_contact(np.asarray([[False, True, False], [False, False, False], [False, False, False]]),
                            np.asarray([[True, False, False], [False, False, False], [False, True, False]]))
    exact(outcome["Y"], np.asarray([-1., 0., -1.]), "first contact, no-contact zero, later food cannot replace first poison")
    lethal = first_contact(np.asarray([[True, False]]), np.asarray([[False, False]]))
    require(lethal["Y"][0] == 1, "no survival filter on first-contact objective")
    rows = []
    for index in range(4):
        y = np.full((3, 2, 2), .125)
        y[2] = .5
        local = np.zeros((3, 2, 2, 2))
        rows.append(dict(base_index=index, n_distractors=4, local=local, sampled=local.copy(),
            first_choice={name: np.full_like(local, .5) for name in CHOICE_FIELDS},
            decomposition={name: np.zeros((3, 2, 2)) for name in ("approach", "conditional")},
            solo=dict(Y=y, first_food=(1 + y)/2, first_poison=(1 - y)/2,
                      no_contact=np.zeros_like(y), survival=np.ones_like(y), latency=np.ones_like(y))))
    positive = summarize_rows(rows)
    require(all(value["passed"] for value in positive["gates"].values()), "positive solo gains suffice with descriptive local S unchanged")
    stationary = copy.deepcopy(rows)
    for row in stationary:
        row["solo"]["Y"][2, :, 1] = .125
        row["solo"]["first_food"][2, :, 1] = .5625
        row["solo"]["first_poison"][2, :, 1] = .4375
    decisions = summarize_rows(stationary)["gates"]
    require(decisions["stationary_repair"]["passed"] and not decisions["reversal_advancement"]["passed"], "repair and reversal remain separate claims")
    equal = copy.deepcopy(rows)
    for row in equal:
        for name in FIELDS:
            row["solo"][name][1] = row["solo"][name][2]
    require(not summarize_rows(equal)["gates"]["reversal_advancement"]["passed"], "both parent and matched-control improvements required")
    _, jnp = source_bytes.random_runtime()
    positions = np.arange(31, dtype=np.float32)/2
    scaled = np.asarray(jnp.asarray(positions)/15 - .5)
    require(np.all(np.abs(scaled.astype(np.float64) - (np.arange(31)/30 - .5)) <= 1e-7),
            "all population-sixteen half-ranks close to independent exact rational")
    require(not any(name == "seed" or name.startswith("seed.") for name in sys.modules), "audit imports no production simulator")
    return dict(status="self-check passed", groups=19, simulator_imported=False, random_samples_generated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-commit")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", "--progress-path", dest="progress", type=Path,
                        help="Exclusive JSONL stage timing receipt")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), allow_nan=False))
        return 0
    if not args.source_commit:
        parser.error("--source-commit is required")
    started, auditor = time.monotonic(), None
    progress = args.progress.open("x", encoding="utf-8", newline="\n") if args.progress else None
    if progress is not None:
        progress.write(json.dumps(dict(event="audit-start", source=args.source_commit)) + "\n")
        progress.flush()
    report = dict(protocol=PROTOCOL, mode="development" if args.development else "training" if args.training else "final")
    try:
        auditor = Auditor(args.root, args.source_commit, progress)
        # Status-only admission precedes every read of requested scientific data.
        dev_train = [auditor.completed(name, 2, 0) for name in DEVELOPMENT_TRAINING_RUNS]
        dev_eval = auditor.completed(DEVELOPMENT_EVAL_NAME, 0, 4)
        final_train = [] if args.development else [(i, view, auditor.completed(f"E138-A{i}-{view}", 10, 0)) for i in range(3) for view in OBJECTIVES]
        final_eval = [] if args.development or args.training else [auditor.completed(f"E138-A{i}", 0, 128) for i in range(3)]
        report["training"] = [auditor.reconstruct("training", folder, auditor.training, 0, True, view) for view, folder in zip(OBJECTIVES, dev_train)]
        report["development"] = auditor.reconstruct("evaluation", dev_eval, auditor.evaluation, 0, True)
        report["training"].extend(auditor.reconstruct("training", folder, auditor.training, i, False, view) for i, view, folder in final_train)
        report["manipulation"] = [auditor.manipulation(True, 0)] + ([] if args.development else [auditor.manipulation(False, i) for i in range(3)])
        report["runs"] = [auditor.reconstruct("evaluation", folder, auditor.evaluation, i, False) for i, folder in enumerate(final_eval)]
        gates_all = {name: dict(passed=all(row["measurements"]["gates"][name]["passed"] for row in report["runs"]),
            count=sum(row["measurements"]["gates"][name]["passed"] for row in report["runs"]),
            lineages=[row["index"] for row in report["runs"] if row["measurements"]["gates"][name]["passed"]])
            for name in ("stationary_repair", "reversal_advancement")} if final_eval else None
        report.update(status="passed", registered_source=args.source_commit, registered_source_code=auditor.source_manifest,
            final_evaluation_audited=bool(final_eval), scientific_gates=gates_all,
            scientific_gate_passed=None if gates_all is None else gates_all["reversal_advancement"]["passed"],
            evidence_scope=dict(candidate_physical_steps_per_attempt=12800, final_families_per_lineage=128,
                final_lives_per_genotype_history=256, compact_candidates_store_tick_grids=False,
                physical_energy_relative_envelope=float(8*np.finfo(np.float32).eps),
                final_neural_logit_check="inherited independent E135 NumPy roundoff envelope at frozen readout only"),
            limitations=["Candidate development and neural learning during exposure/solo are not independently rerun.",
                "Training solo movement, item consumption, regrowth, energy, death, incoming feedback and categorical draws are reconstructed from compact evidence; logits are not independently recomputed from full neural states.",
                "The native kernel has one physical denergy input to learning; no separate explicit-learning trace is claimed in compact training evidence.",
                "Final acquired W/x support independent frozen first-decision checks, but do not prove how those neural states were acquired.",
                "Solo Y includes encounter frequency. Conditional contact accuracy and focal occupancy/color effects remain descriptive.",
                "Performance after matched acquisition and a gap does not establish autonomous reversal learning from birth."])
        code = 0
    except base.Incomplete as error:
        report.update(status="incomplete", error=str(error), final_evaluation_audited=False)
        code = 2
    except (base.AuditError, OSError, ValueError, KeyError, TypeError, base.subprocess.SubprocessError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", final_evaluation_audited=False)
        code = 1
    elapsed = time.monotonic() - started
    if auditor is not None:
        report["checked_file_sha256"] = auditor.hashes
        report["timing"] = dict(training_attempt_wall_seconds=auditor.training_seconds, family_wall_seconds=auditor.family_seconds,
            elapsed_seconds=elapsed, setup_and_summary_seconds=max(0., elapsed - sum(auditor.training_seconds) - sum(auditor.family_seconds)),
            runs=auditor.run_timing, non_run_wall_seconds=max(0., elapsed - sum(row["wall_seconds"] for row in auditor.run_timing)),
            training_order="development background/solo, then lineage 0/1/2 background/solo; attempts ascending",
            family_order="four development, then final lineage 0/1/2; (8,16),(16,32),(8,32),(16,16) cycle")
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
        print(json.dumps(dict(status=report["status"], output=str(args.output), scientific_gates=report.get("scientific_gates"), error=report.get("error")), allow_nan=False))
    else:
        print(encoded, end="")
    if progress is not None:
        progress.write(json.dumps(dict(event="audit-end", status=report["status"], wall_seconds=elapsed)) + "\n")
        progress.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
