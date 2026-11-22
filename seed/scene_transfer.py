"""Paired, first-observation-only scene transfer with native free decisions."""
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from . import encounter_diagnostic as encounter
from . import replay_selection as replay
from . import seed


PROTOCOL = "scene-transfer-v1"
ASSIGNMENT_TAG = 0xE135
KEY_ROLES = ("script", "birth", "chamber", "background", "probe_action", "continuation", "solo_layout", "solo_action")
MASTER_SEEDS = (2026121500, 2026121501, 2026121502, 2026121503)
ARMS = ("parent", "descendant")
HISTORIES = ("stationary", "reversal")
VIEWS = ("masked", "full")
CELLS = ((8, 16), (16, 32), (8, 32), (16, 16))
CFG = encounter.CFG
GRID_STEPS = 50
FIELDS = ("Y", "first_food", "first_poison", "no_contact", "survival", "latency")
CHOICE_FIELDS = ("p_food", "p_poison", "p_other", "approach", "conditional")


def schedule(index):
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < 128:
        raise ValueError("E135 family index must lie within the frozen panel")
    return CELLS[index % len(CELLS)]


def family_keys(master, index):
    if master not in MASTER_SEEDS:
        raise ValueError("E135 master is not registered")
    schedule(index)
    key = jax.random.fold_in(jax.random.PRNGKey(master), ASSIGNMENT_TAG)
    return np.asarray(jax.random.split(jax.random.fold_in(key, index), len(KEY_ROLES)))


@lru_cache(None)
def _background(c=CFG):
    hub = np.full(2, c.G // 2)
    moves = np.asarray(seed.MOVES)
    cross = (hub + moves) @ np.asarray([c.G, 1])
    candidates = jnp.asarray(np.setdiff1d(np.arange(c.G * c.G, dtype=np.int32), cross))
    items = jnp.repeat(jnp.asarray([1, 2], dtype=jnp.int32), c.n_items - 1)

    @jax.jit
    def generate(key):
        cells = jax.random.choice(key, candidates, items.shape, replace=False)
        return jnp.zeros(c.G * c.G, jnp.int32).at[cells].set(items).reshape(c.G, c.G)
    return generate


@lru_cache(None)
def _first_decisions(c=CFG):
    pos = jnp.full((1, 2), c.G // 2, jnp.int32)
    item_features = 2 * (2 * c.R + 1) ** 2

    @jax.jit
    def run(flat, acquired, background, keys, rule):
        genome = encounter._unravel(c)(flat)

        def rotation(placement, key):
            focal = encounter._pair_grid(pos, placement, c)
            initial = encounter._fresh_body(acquired, background + focal, pos, key, c)
            full = seed.observe(initial["grid"], pos, initial["alive"], initial["sig"], initial["energy"],
                                initial["denergy"], c)
            isolated = seed.observe(focal, pos, initial["alive"], initial["sig"], initial["energy"],
                                    initial["denergy"], c)
            masked = full.at[:, :item_features].set(isolated[:, :item_features])
            return jax.vmap(lambda view: encounter._step(initial, genome, rule, -1, "plastic", c,
                hebbian=False, predictor=False, observation=view))(jnp.stack((masked, full)))

        return jax.vmap(rotation, out_axes=1)(jnp.asarray(encounter.PLACEMENTS), keys)
    return run


def _compact(trace):
    return {name: np.asarray(value) for name, value in trace.items() if name not in replay.TRACE_OMIT}


def _prefixed(prefix, fields):
    return {prefix + name: value for name, value in fields.items()}


def _evaluate_history(flat, boundary, prior, script, rule, gap, background, probe_keys, continuation,
                      grid, position, action_key):
    acquired, later = encounter._chamber("plastic", CFG)(flat, boundary, script, rule)
    acquired, empty = replay._empty_gap(gap)(flat, acquired, rule)
    exposure = replay._join_trace(_compact(prior), _compact(later), _compact(empty))
    first_states, decisions = _first_decisions()(flat, acquired, background, probe_keys, rule)
    decisions = _compact(decisions)
    scene = []
    for view in range(len(VIEWS)):
        first = {name: value[view, continuation:continuation + 1] for name, value in decisions.items()}
        initial_after_first = jax.tree.map(lambda state: state[view, continuation], first_states)
        _, rest = encounter._free_grid("plastic", CFG, steps=GRID_STEPS - 1)(flat, initial_after_first, rule)
        scene.append(replay._join_trace(first, _compact(rest)))
    fresh = encounter._fresh_body(acquired, grid, position, action_key, CFG)
    _, reference = encounter._free_grid("plastic", CFG, steps=GRID_STEPS)(flat, fresh, rule)
    return {
        **_prefixed("exposure_", exposure),
        **_prefixed("acquired_", {name: np.asarray(acquired[name]) for name in encounter.NEURAL}),
        **_prefixed("acquired_", {name + "_sha256": encounter.array_hash(acquired[name]) for name in encounter.NEURAL}),
        **_prefixed("first_", decisions),
        **_prefixed("scene_", encounter._stack(scene)),
        **_prefixed("scene_initial_", {name: np.stack((acquired[name], acquired[name])) for name in encounter.NEURAL}),
        **_prefixed("reference_", _compact(reference)),
        **_prefixed("reference_initial_", {name: np.asarray(fresh[name]) for name in encounter.NEURAL}),
    }


def build_family(keys, index, parent_genome, descendant_genome):
    pre_contacts, gap = schedule(index)
    keys = replay._keys(keys, len(KEY_ROLES))
    script = encounter.contact_script(keys[0], pre_contacts)
    prescript, postscript = map(jnp.asarray, (script[:pre_contacts], script[pre_contacts:]))
    background = _background()(keys[3])
    probe_keys = jax.random.split(jnp.asarray(keys[4]), len(encounter.PLACEMENTS))
    continuation = int(jax.random.randint(keys[5], (), 0, len(encounter.PLACEMENTS)))
    grid, position = seed.init_world(jnp.asarray(keys[6]), CFG)
    flats = [encounter._genome(genome, "plastic", CFG) for genome in (parent_genome, descendant_genome)]
    row = dict(base_index=np.asarray(index, np.int32), keys=keys.copy(),
               pre_contacts=np.asarray(pre_contacts, np.int32), gap=np.asarray(gap, np.int32),
               continuation=np.asarray(continuation, np.int32), initial_rules=np.asarray([0, 1], np.int32),
               genome_sha256=np.stack([encounter.array_hash(flat) for flat in flats]), script=script,
               background=np.asarray(background), reference_grid=np.asarray(grid), reference_pos=np.asarray(position))
    controllers = []
    for flat in flats:
        birth = encounter._birth(flat, "plastic", jnp.asarray(keys[1]), jnp.asarray(keys[2]), CFG)
        rules = []
        for initial in (0, 1):
            boundary, prior = encounter._chamber("plastic", CFG)(flat, birth, prescript, initial)
            histories = [_evaluate_history(flat, boundary, prior, postscript, initial ^ history, gap,
                         background, probe_keys, continuation, grid, position, jnp.asarray(keys[7]))
                         for history in (0, 1)]
            rules.append(encounter._stack(histories))
        controllers.append(encounter._stack(rules))
    row.update(encounter._stack(controllers))
    _validate_family(row)
    return row


def _validate_family(row):
    for name, values in row.items():
        if values.dtype.kind == "f" and not np.isfinite(values).all():
            raise ValueError("nonfinite E135 field " + name)
    if (not np.all(row["exposure_alive_after"]) or row["exposure_energy"].min() < .462 - 1e-5
            or row["exposure_energy"].max() > 1.25 + 1e-5
            or not np.array_equal(row["exposure_energy"], row["exposure_unclipped_energy"])):
        raise ValueError("matched exposure violated the deterministic survival bound")
    for prefix in ("first_", "scene_", "reference_"):
        if np.any(row[prefix + "forced_action"] != -1) or not np.array_equal(
                row[prefix + "action"], row[prefix + "sampled_action"]):
            raise ValueError("E135 scored a non-native choice in " + prefix)
    for switch in ("hebbian_enabled", "predictor_enabled"):
        if (np.any(row["first_" + switch]) or np.any(row["scene_" + switch][..., 0])
                or not np.all(row["scene_" + switch][..., 1:]) or not np.all(row["reference_" + switch])):
            raise ValueError("E135 write timing differs from protocol")
    probabilities = row["first_probabilities"]
    if probabilities.shape != (2, 2, 2, 2, 12, 1, 5):
        raise ValueError("E135 native action axes differ from protocol")
    for prob, logits in zip(probabilities.reshape(-1, 5), row["first_logits"].reshape(-1, 5)):
        encounter.validate_probabilities(prob, logits)


def _behavior(row, prefix):
    food = row[prefix + "food"][..., 0].astype(bool)
    poison = row[prefix + "poison"][..., 0].astype(bool)
    contact = food | poison
    first = np.argmax(contact, axis=-1)[..., None]
    first_food = np.take_along_axis(food, first, axis=-1)[..., 0]
    first_poison = np.take_along_axis(poison, first, axis=-1)[..., 0]
    return dict(Y=first_food.astype(float) - first_poison.astype(float), first_food=first_food,
                first_poison=first_poison, no_contact=~contact.any(-1),
                survival=row[prefix + "alive_after"][..., -1, 0],
                latency=np.where(contact.any(-1), first[..., 0], GRID_STEPS))


def summary_input(row):
    local = np.empty((2, 2, 2, 2), np.float64)
    sampled = np.empty_like(local)
    choice = {field: np.empty_like(local) for field in CHOICE_FIELDS}
    decomposition = {field: np.empty((2, 2, 2), np.float64) for field in ("approach", "conditional")}
    probabilities = row["first_probabilities"][..., 0, :].astype(np.float64)
    actions = row["first_action"][..., 0]
    ordinal = np.arange(len(encounter.PLACEMENTS))
    other_moves = np.all(np.arange(5)[None, :, None] != encounter.PLACEMENTS[:, None, :], axis=-1)
    for initial in (0, 1):
        for history in (0, 1):
            rule = initial ^ history
            probs = probabilities[:, initial, history]
            p_food = probs[..., ordinal, encounter.PLACEMENTS[:, rule]]
            p_poison = probs[..., ordinal, encounter.PLACEMENTS[:, 1 - rule]]
            p_other = np.where(other_moves, probs, 0).sum(-1)
            signed = p_food - p_poison
            approach = p_food + p_poison
            conditional = np.divide(signed, approach, out=np.zeros_like(signed), where=approach > 0)
            local[:, initial, history] = signed.mean(-1)
            for field, values in zip(CHOICE_FIELDS, (p_food, p_poison, p_other, approach, conditional)):
                choice[field][:, initial, history] = values.mean(-1)
            decomposition["approach"][:, initial, history] = ((approach[:, 0] - approach[:, 1]) * conditional.mean(1)).mean(-1)
            decomposition["conditional"][:, initial, history] = ((conditional[:, 0] - conditional[:, 1]) * approach.mean(1)).mean(-1)
            decisions = actions[:, initial, history]
            food = decisions == encounter.PLACEMENTS[:, rule]
            poison = decisions == encounter.PLACEMENTS[:, 1 - rule]
            sampled[:, initial, history] = (food.astype(float) - poison.astype(float)).mean(-1)
    hub = CFG.G // 2
    visible = slice(hub - CFG.R, hub + CFG.R + 1)
    return dict(base_index=int(row["base_index"]), n_distractors=np.count_nonzero(row["background"][visible, visible]),
                local=local, sampled=sampled, first_choice=choice, decomposition=decomposition,
                scene=_behavior(row, "scene_"), reference=_behavior(row, "reference_"))


def _estimate(values):
    values = np.asarray(values, np.float64)
    n = len(values)
    mean, se = float(values.mean()), float(values.std(ddof=1) / np.sqrt(n))
    return dict(n=n, mean=mean, se=se, lower_2se=mean - 2 * se, upper_2se=mean + 2 * se)


def _behavior_summary(arrays):
    scores = {field: _estimate(values) for field, values in arrays.items()}
    food = arrays["first_food"]
    contact = food + arrays["first_poison"]
    n = len(food)
    food_count, total_count = int(2 * food.sum()), int(2 * contact.sum())
    mean = food_count / total_count if total_count else None
    se = _estimate((food - mean * contact) / contact.mean())["se"] if total_count else None
    scores["contact_accuracy"] = dict(n=n, mean=mean, se=se, food_count=food_count, contact_count=total_count,
                                      assigned_lives=2 * n, descriptive=True)
    return scores


def summarize_rows(rows):
    rows = list(rows)
    if len(rows) < 2 or [row["base_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("E135 summary needs complete consecutive family identities")
    counts = np.asarray([row["n_distractors"] for row in rows])
    values = {name: np.stack([row[name] for row in rows]).mean(axis=2) for name in ("local", "sampled")}
    behavior = {name: {field: np.stack([row[name][field] for row in rows]).mean(axis=2) for field in FIELDS}
                for name in ("scene", "reference")}
    choice = {field: np.stack([row["first_choice"][field] for row in rows]).mean(axis=2) for field in CHOICE_FIELDS}
    decomposition = {field: np.stack([row["decomposition"][field] for row in rows]).mean(axis=2)
                     for field in ("approach", "conditional")}
    histories = {}
    for h, history in enumerate(HISTORIES):
        masked, full = (values["local"][:, 1, h] - values["local"][:, 0, h]).T
        scene_masked, scene_full = (behavior["scene"]["Y"][:, 1, h] - behavior["scene"]["Y"][:, 0, h]).T
        report = dict(masked_gain=_estimate(masked), full_gain=_estimate(full), attenuation=_estimate(masked - full),
                      scene_gain_masked=_estimate(scene_masked), scene_gain_full=_estimate(scene_full),
                      scene_attenuation=_estimate(scene_masked - scene_full),
                      reference_gain=_estimate(behavior["reference"]["Y"][:, 1, h] - behavior["reference"]["Y"][:, 0, h]))
        for name, array in values.items():
            report[name] = {arm: {view: _estimate(array[:, g, h, v]) for v, view in enumerate(VIEWS)}
                            for g, arm in enumerate(ARMS)}
        report["first_choice"] = {arm: {view: {field: _estimate(array[:, g, h, v]) for field, array in choice.items()}
                                       for v, view in enumerate(VIEWS)} for g, arm in enumerate(ARMS)}
        report["decomposition"] = {field: _estimate(array[:, 1, h] - array[:, 0, h]) for field, array in decomposition.items()}
        report["scene"] = {arm: {view: _behavior_summary({field: array[:, g, h, v] for field, array in behavior["scene"].items()})
                                 for v, view in enumerate(VIEWS)} for g, arm in enumerate(ARMS)}
        report["reference"] = {arm: _behavior_summary({field: array[:, g, h] for field, array in behavior["reference"].items()})
                               for g, arm in enumerate(ARMS)}
        histories[history] = report
    return dict(families=len(rows), distractors=_estimate(counts), zero_distractors=_estimate(counts == 0), histories=histories)


def gate(summary):
    stationary = summary["histories"]["stationary"]
    isolated, attenuation, full = (stationary[name] for name in ("masked_gain", "attenuation", "full_gain"))
    isolated_replicates = isolated["mean"] >= .05 and isolated["lower_2se"] > 0
    attenuation_confirmed = attenuation["mean"] >= .05 and attenuation["lower_2se"] > 0
    passed = isolated_replicates and attenuation_confirmed
    return dict(isolated_replicates=isolated_replicates, attenuation_confirmed=attenuation_confirmed,
                passed=passed, practical_loss_excluded=isolated_replicates and attenuation["upper_2se"] < .05,
                advantage_erased=passed and full["upper_2se"] <= 0,
                partial_attenuation=passed and full["lower_2se"] > 0,
                reference_deficit=stationary["reference_gain"]["upper_2se"] < 0)
