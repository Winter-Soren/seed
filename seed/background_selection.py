"""E136 paired native-genome selection on isolated or ordinary-background choices.

Both objectives execute the same two-view kernel. Only the host-side selection
of the returned fitness differs; no objective enters the compiled computation.
"""
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from . import encounter_diagnostic as encounter
from . import replay_selection as replay
from . import scene_transfer as scene
from . import seed


PROTOCOL = "background-selection-v1"
CFG = encounter.CFG
ASSIGNMENT_TAG = 0xE136
GENOTYPES = ("parent", "isolated", "background")
HISTORIES = ("stationary", "reversal")
VIEWS = ("isolated", "background")
KEY_ROLES = ("script", "birth", "chamber", "background", "probe", "solo_layout", "solo_action")
TRAIN_KEY_ROLES = ("script", "chamber", "background", "probe")
TRAIN_CELLS = replay.TRAIN_CELLS.copy()
GRID_STEPS = replay.GRID_STEPS
CHOICE_FIELDS = scene.CHOICE_FIELDS


def training_batch(world_keys):
    """Build four rule-blind histories and scenes, shared across objectives."""
    world_keys = replay._keys(world_keys, 4)
    role_keys = np.stack([np.asarray(jax.random.split(key, len(TRAIN_KEY_ROLES))) for key in world_keys])
    schedule_key = np.asarray(jax.random.fold_in(world_keys[0], ASSIGNMENT_TAG))
    cells = TRAIN_CELLS[np.asarray(jax.random.permutation(schedule_key, len(TRAIN_CELLS)))]
    scripts = np.zeros((4, 32, 5), np.int32)
    backgrounds = []
    for i, (pre, _) in enumerate(cells):
        script = encounter.contact_script(role_keys[i, 0], int(pre))
        scripts[i, :len(script)] = script
        backgrounds.append(np.asarray(scene._background()(jnp.asarray(role_keys[i, 2]))))
    return dict(world_keys=world_keys.copy(), role_keys=role_keys, schedule_key=schedule_key,
                scripts=scripts, backgrounds=np.stack(backgrounds), pre_contacts=cells[:, 0], gap=cells[:, 1])


def _score_family(flat, script, birth_key, chamber_key, background, probe_key, pre_contacts, gap):
    """Return both expected five-action scores from identical acquired clones.

    Probabilities and logits have initial-rule/history/view/placement/action axes.
    The native first moves are free, their learning writes are off, and their
    resulting states are discarded. Neither first move changes a later assay.
    """
    probe_keys = jax.random.split(probe_key, len(encounter.PLACEMENTS))

    def initial_rule(rule):
        birth = encounter._birth(flat, "plastic", birth_key, chamber_key, CFG)
        boundary, prior = encounter._chamber("plastic", CFG)(flat, birth, script[:pre_contacts], rule)
        pre_counts = jnp.stack((prior["food"].sum(), prior["poison"].sum())).astype(jnp.float32)
        pre_bounds = jnp.stack((prior["unclipped_energy"].min(), prior["unclipped_energy"].max()))

        def history_score(history):
            target = jnp.bitwise_xor(rule, history)
            delayed, post_counts, bounds = replay._training_history(
                flat, boundary, script[pre_contacts:pre_contacts + 16], target, gap)
            _, trace = scene._first_decisions()(flat, delayed, background, probe_keys, target)
            return dict(probabilities=trace["probabilities"][..., 0, :],
                        logits=trace["logits"][..., 0, :],
                        sampled_action=trace["sampled_action"][..., 0],
                        probe_action_keys=trace["action_key"],
                        counts=jnp.stack((pre_counts, post_counts)), survived=delayed["alive"][0],
                        exposure_energy_bounds=jnp.stack((jnp.minimum(pre_bounds[0], bounds[0]),
                                                         jnp.maximum(pre_bounds[1], bounds[1]))))
        return jax.lax.map(history_score, jnp.arange(2, dtype=jnp.int32))

    result = jax.lax.map(initial_rule, jnp.arange(2, dtype=jnp.int32))
    probabilities = result["probabilities"]
    placements = jnp.asarray(encounter.PLACEMENTS)
    targets = jnp.bitwise_xor(jnp.arange(2)[:, None], jnp.arange(2)[None, :])
    good = placements[:, targets].transpose(1, 2, 0)[:, :, None, :, None]
    bad = placements[:, 1 - targets].transpose(1, 2, 0)[:, :, None, :, None]
    food = jnp.take_along_axis(probabilities, good, axis=-1)[..., 0]
    poison = jnp.take_along_axis(probabilities, bad, axis=-1)[..., 0]
    # Average initial rules, histories and placements, leaving the view axis.
    return dict(view_fitness=(food - poison).mean((0, 1, 3)), probabilities=probabilities,
                logits=result["logits"], sampled_action=result["sampled_action"],
                probe_action_keys=result["probe_action_keys"], counts=result["counts"].mean((0, 1)),
                survived=result["survived"].min(),
                exposure_energy_bounds=jnp.stack((result["exposure_energy_bounds"][..., 0].min(),
                                                 result["exposure_energy_bounds"][..., 1].max())))


@lru_cache(None)
def _population_cell(pre_contacts, gap):
    @jax.jit
    def run(candidates, script, birth_keys, chamber_key, background, probe_key):
        return jax.lax.map(lambda pair: _score_family(
            pair[0], script, pair[1], chamber_key, background, probe_key, pre_contacts, gap),
            (candidates, birth_keys))
    return run


def make_evaluator(view, c=CFG):
    """Select one scalar only after the common two-view computation returns."""
    encounter._config(c)
    if view not in VIEWS:
        raise ValueError("training objective must be isolated or background")
    view_index = VIEWS.index(view)

    def evaluate(candidates, batch, birth_keys):
        candidates, birth_keys = np.asarray(candidates), np.asarray(birth_keys)
        if (candidates.ndim != 2 or not len(candidates) or candidates.shape[1:] != (155,)
                or candidates.dtype != np.float32 or not np.isfinite(candidates).all()
                or birth_keys.shape != (len(candidates), 4, 2) or birth_keys.dtype != np.uint32):
            raise ValueError("candidate and antithetic birth arrays have invalid dimensions")
        if sorted(zip(np.asarray(batch["pre_contacts"]).tolist(), np.asarray(batch["gap"]).tolist())) != list(map(tuple, TRAIN_CELLS)):
            raise ValueError("training must contain each registered schedule cell once")
        if (np.shape(batch["role_keys"]) != (4, 4, 2) or np.asarray(batch["role_keys"]).dtype != np.uint32
                or np.shape(batch["scripts"]) != (4, 32, 5)
                or np.asarray(batch["scripts"]).dtype != np.int32
                or np.shape(batch["backgrounds"]) != (4, c.G, c.G)
                or np.asarray(batch["backgrounds"]).dtype != np.int32):
            raise ValueError("training materialization differs from the four-family contract")
        records = []
        for i in range(4):
            pre, gap = int(batch["pre_contacts"][i]), int(batch["gap"][i])
            records.append(_population_cell(pre, gap)(
                jnp.asarray(candidates), jnp.asarray(batch["scripts"][i]), jnp.asarray(birth_keys[:, i]),
                jnp.asarray(batch["role_keys"][i, 1]), jnp.asarray(batch["backgrounds"][i]),
                jnp.asarray(batch["role_keys"][i, 3])))
        result = jax.tree.map(lambda *values: jnp.stack(values, axis=1), *records)
        bounds = np.asarray(result["exposure_energy_bounds"])
        if (not np.isfinite(np.asarray(result["view_fitness"])).all() or not np.isfinite(bounds).all()
                or bounds[..., 0].min() <= 0 or bounds[..., 1].max() >= c.e_max
                or not np.all(np.asarray(result["survived"]) == 1)):
            raise ValueError("candidate replay violated finite scores or guaranteed survival")
        return dict(result, fitness=result["view_fitness"][..., view_index])
    return evaluate


def evaluate_family(parent, isolated, background, keys, index, pre_contacts, gap, c=CFG):
    """Assay three acquired genomes on common fresh scenes and free solo grids.

    Final first-choice axes are genotype, initial rule, history, view, placement,
    native agent, action. Solo arrays have genotype, initial rule, history, turn
    and each native field's trailing axes. All assigned lives remain present.
    """
    encounter._config(c)
    keys = replay._keys(keys, len(KEY_ROLES))
    pre_contacts = replay._integer(pre_contacts, (8, 16), "old-contact count")
    gap = replay._integer(gap, (16, 32), "held-out gap")
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < 128:
        raise ValueError("family index leaves the registered panel")
    flats = [encounter._genome(flat, "plastic", c) for flat in (parent, isolated, background)]
    script = encounter.contact_script(keys[0], pre_contacts)
    prescript, postscript = map(jnp.asarray, (script[:pre_contacts], script[pre_contacts:]))
    scene_background = scene._background()(jnp.asarray(keys[3]))
    probe_keys = jax.random.split(jnp.asarray(keys[4]), len(encounter.PLACEMENTS))
    grid, position = seed.init_world(jnp.asarray(keys[5]), c)
    row = dict(base_index=np.asarray(index, np.int32), keys=keys.copy(),
               pre_contacts=np.asarray(pre_contacts, np.int32), gap=np.asarray(gap, np.int32),
               initial_rules=np.asarray([0, 1], np.int32), script=script,
               genome_sha256=np.stack([encounter.array_hash(flat) for flat in flats]),
               background=np.asarray(scene_background), target_grid=np.asarray(grid), target_position=np.asarray(position))
    genotypes = []
    for flat in flats:
        birth = encounter._birth(flat, "plastic", jnp.asarray(keys[1]), jnp.asarray(keys[2]), c)
        rules = []
        for initial in (0, 1):
            boundary, prior = encounter._chamber("plastic", c)(flat, birth, prescript, initial)
            histories = []
            for history in (0, 1):
                target = initial ^ history
                acquired, post = encounter._chamber("plastic", c)(flat, boundary, postscript, target)
                delayed, empty = replay._empty_gap(gap)(flat, acquired, target)
                exposure = replay._join_trace(prior, post, empty)
                _, first = scene._first_decisions()(flat, delayed, scene_background, probe_keys, target)
                # The solo starts from the unprobed acquired state, not either
                # first-decision result. Every solo movement uses forced=-1.
                fresh = encounter._fresh_body(delayed, grid, position, jnp.asarray(keys[6]), c)
                final, solo = encounter._free_grid("plastic", c, steps=GRID_STEPS)(flat, fresh, target)
                record = {}
                replay._prefix(record, "exposure_", replay._compact_trace(exposure))
                replay._prefix(record, "acquired_", encounter._state_evidence(delayed))
                replay._prefix(record, "acquired_", {name: np.asarray(delayed[name]) for name in encounter.NEURAL})
                replay._prefix(record, "first_", replay._compact_trace(first))
                replay._prefix(record, "solo_initial_", encounter._state_evidence(fresh))
                replay._prefix(record, "solo_final_", encounter._state_evidence(final))
                replay._prefix(record, "solo_", replay._compact_trace(solo))
                histories.append(record)
            rules.append(encounter._stack(histories))
        genotypes.append(encounter._stack(rules))
    row.update(encounter._stack(genotypes))
    validate_family(row)
    return row


def validate_family(row):
    """Local contract checks; the independent auditor reconstructs physics."""
    probabilities, logits = row["first_probabilities"], row["first_logits"]
    if probabilities.shape != (3, 2, 2, 2, 12, 1, 5) or logits.shape != probabilities.shape:
        raise ValueError("first-choice genotype/rule/history/view/placement/action axes differ")
    for prob, logit in zip(probabilities.reshape(-1, 5), logits.reshape(-1, 5)):
        encounter.validate_probabilities(prob, logit)
    for name, value in row.items():
        array = np.asarray(value)
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError("nonfinite E136 family field " + name)
    energy = row["exposure_energy"]
    if (not encounter._identical(energy, row["exposure_unclipped_energy"])
            or energy.min() < .462 - 1e-5 or energy.max() > 1.25 + 1e-5
            or not np.all(row["exposure_alive_after"] == 1)):
        raise ValueError("matched exposure violated the deterministic survival bound")
    for prefix in ("first_", "solo_"):
        if (not np.all(row[prefix + "forced_action"] == -1)
                or not encounter._identical(row[prefix + "action"], row[prefix + "sampled_action"])):
            raise ValueError("scored action was imposed in " + prefix)
    for flag in ("hebbian_enabled", "predictor_enabled"):
        if np.any(row["first_" + flag]) or not np.all(row["solo_" + flag]):
            raise ValueError("first-choice or solo learning-write timing differs")
    for prefix in ("exposure_", "first_", "solo_"):
        if not encounter._identical(row[prefix + "explicit_learning_denergy"], row[prefix + "incoming_denergy"]):
            raise ValueError("explicit learning feedback differs in " + prefix)
    observation = row["first_observation"]
    if not encounter._identical(observation, np.broadcast_to(observation[0, 0, 0], observation.shape).copy()):
        raise ValueError("genotypes or histories received different immediate assay observations")
    item_features = 2 * (2 * CFG.R + 1) ** 2
    if not encounter._identical(observation[:, :, :, 0, ..., item_features:],
                                observation[:, :, :, 1, ..., item_features:]):
        raise ValueError("the view intervention changed a non-item feature")
    for field in ("grid_before", "position_before", "energy_before", "incoming_denergy", "signal_before", "key_before"):
        values = row["first_" + field]
        if not encounter._identical(values[:, :, :, 0], values[:, :, :, 1]):
            raise ValueError("first-choice physical state differs between views in " + field)
    for name in encounter.NEURAL:
        if not encounter._identical(row["solo_initial_" + name + "_sha256"], row["acquired_" + name + "_sha256"]):
            raise ValueError("solo did not start from the unprobed acquired " + name)


signed_local = replay.signed_local


def summary_input(row):
    """Reduce within each family, retaining rules, histories and both views."""
    local = np.empty((3, 2, 2, 2), np.float64)
    sampled = np.empty_like(local)
    choice = {field: np.empty_like(local) for field in CHOICE_FIELDS}
    decomposition = {field: np.empty((3, 2, 2), np.float64) for field in ("approach", "conditional")}
    probabilities = row["first_probabilities"][..., 0, :].astype(np.float64)
    actions = row["first_action"][..., 0]
    ordinal = np.arange(len(encounter.PLACEMENTS))
    for initial in (0, 1):
        for history in (0, 1):
            target = initial ^ history
            probs = probabilities[:, initial, history]
            p_food = probs[..., ordinal, encounter.PLACEMENTS[:, target]]
            p_poison = probs[..., ordinal, encounter.PLACEMENTS[:, 1 - target]]
            signed, approach = p_food - p_poison, p_food + p_poison
            conditional = np.divide(signed, approach, out=np.zeros_like(signed), where=approach > 0)
            local[:, initial, history] = signed.mean(-1)
            for field, values in zip(CHOICE_FIELDS, (p_food, p_poison, 1 - approach, approach, conditional)):
                choice[field][:, initial, history] = values.mean(-1)
            # Exact symmetric product decomposition of isolated minus full S;
            # the second component describes conditional color preference.
            decomposition["approach"][:, initial, history] = ((approach[:, 0] - approach[:, 1]) * conditional.mean(1)).mean(-1)
            decomposition["conditional"][:, initial, history] = ((conditional[:, 0] - conditional[:, 1]) * approach.mean(1)).mean(-1)
            decisions = actions[:, initial, history]
            sampled[:, initial, history] = ((decisions == encounter.PLACEMENTS[:, target]).astype(float)
                                            - (decisions == encounter.PLACEMENTS[:, 1 - target]).astype(float)).mean(-1)
    behavior = scene._behavior(row, "solo_")
    hub = CFG.G // 2
    visible = slice(hub - CFG.R, hub + CFG.R + 1)
    return dict(base_index=int(row["base_index"]), local=local, sampled=sampled,
                first_choice=choice, decomposition=decomposition,
                n_distractors=int(np.count_nonzero(row["background"][visible, visible])),
                solo_Y=behavior["Y"], first_food=behavior["first_food"], first_poison=behavior["first_poison"],
                no_contact=behavior["no_contact"], contact_latency=behavior["latency"], solo_survival=behavior["survival"])
