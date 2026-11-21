"""Conditional E126: matched from-blank ES on paired randomized reversals."""
from functools import lru_cache
import hashlib
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from seed import direct, seed
from scripts import cold_donor, shelter_evolution, social_learning as social
from scripts import social_observer_evolution as optimizer, social_reversal as reversal


PROTOCOL = "reversal-evolution-v1"
CFG = seed.Cfg(M=1)
CONTROLLERS = ("rules-v0", direct.FORMAT)
FLIP_TIMES = reversal.FLIP_TIMES
ASSIGNMENT_TAG = 0xE126
ARMS = ("plastic_stationary", "plastic_reversal", "lesion_stationary", "lesion_reversal",
        "direct_stationary", "direct_reversal")
EVENTS = social.EVENTS
PREFIX_FIELDS = (*reversal.TRACE_FIELDS, "hebbian_enabled", "predictor_enabled")
es_step = optimizer.es_step
average_centered_ranks = optimizer.average_centered_ranks
evaluation_keys = shelter_evolution.evaluation_keys


def _config(c, controller="rules-v0"):
    if controller not in CONTROLLERS or c.M != 1 or c.T != 300 or c.R < 1:
        raise ValueError("reversal evolution requires its M1/T300 world and registered controller")


def array_hash(value):
    return np.frombuffer(hashlib.sha256(np.ascontiguousarray(value).tobytes()).digest(), np.uint8).copy()


def candidate_hashes(candidates):
    return np.stack([array_hash(value) for value in np.asarray(candidates)])


def initialize_state(controller, training_seed, c=CFG):
    _config(c, controller)
    shelter_evolution.evaluation_keys(training_seed, 0)
    mean = None
    if controller == direct.FORMAT:
        _, initial_key = jax.random.split(jax.random.PRNGKey(int(training_seed)))
        mean = direct.init_genome(c, initial_key)
    return seed.es_init(c, mean=mean, seed=int(training_seed))


def training_assignment(world_keys):
    world_keys = np.asarray(world_keys)
    if (world_keys.shape != (4, 2) or world_keys.dtype != np.uint32
            or len({key.tobytes() for key in world_keys}) != 4):
        raise ValueError("training requires four distinct base-world keys")
    key = jax.random.fold_in(jnp.asarray(world_keys[0]), ASSIGNMENT_TAG)
    permutation = np.asarray(jax.random.permutation(key, 4), np.int32)
    return dict(assignment_key=np.asarray(key), permutation=permutation,
                flip_steps=np.asarray(FLIP_TIMES, np.int32)[permutation])


def sampling(state, pop=128, E=4, sigma=.1):
    if (not isinstance(pop, Integral) or isinstance(pop, bool) or pop < 2 or pop % 2
            or E != 4 or not np.isfinite(sigma) or sigma <= 0):
        raise ValueError("sampling requires an even population, four paired bases and positive sigma")
    next_key, noise_key, world_key, birth_key = jax.random.split(state["key"], 4)
    half = jax.random.normal(noise_key, (pop // 2, state["mu"].size))
    candidates = state["mu"] + sigma * jnp.concatenate((half, -half))
    worlds = np.asarray(jax.random.split(world_key, E))
    births = jax.random.split(birth_key, pop // 2 * E).reshape(pop // 2, E, 2)
    return dict(noise_key=np.asarray(noise_key), half_eps_sha256=array_hash(half),
                candidate_sha256=candidate_hashes(candidates), next_key=np.asarray(next_key),
                world_keys=worlds, development_keys=np.asarray(jnp.concatenate((births, births))),
                **training_assignment(worlds))


def evaluation_schedule(master, count):
    if not isinstance(count, Integral) or isinstance(count, bool) or count < 4 or count % 4:
        raise ValueError("held-out bases must balance all four registered flip times")
    shelter_evolution.evaluation_keys(master, 0)
    key = jax.random.fold_in(jax.random.PRNGKey(int(master)), ASSIGNMENT_TAG)
    permutation = np.asarray(jax.random.permutation(key, int(count)), np.int32)
    choices = np.repeat(np.asarray(FLIP_TIMES, np.int32), count // 4)
    return dict(key=np.asarray(key), permutation=permutation, flip_steps=choices[permutation])


def _windows_jax(result, flip):
    events = result["events"]
    food, poison = events["food"].astype(jnp.int32), events["poison"].astype(jnp.int32)
    counts, first, first_poison, missing, latency = [], [], [], [], []
    for start, stop in ((flip - 50, flip), (flip, flip + 50), (flip + 50, flip + 100)):
        f, p = food[start:stop, 0], poison[start:stop, 0]
        encounters = (f + p) > 0
        exists, index = encounters.any(), jnp.argmax(encounters)
        counts.append(jnp.stack((f.sum(), p.sum())))
        first.append(exists & (f[index] > 0))
        first_poison.append(exists & (p[index] > 0))
        missing.append(~exists)
        latency.append(jnp.where(exists, index + 1, 51))
    return dict(window_counts=jnp.stack(counts), window_first_food=jnp.stack(first),
                window_first_poison=jnp.stack(first_poison), window_no_meal=jnp.stack(missing),
                window_first_step=jnp.stack(latency), whole_return=(food - poison).sum(),
                survival=jnp.stack((events["alive_before"][flip, 0], events["alive_after"][flip + 49, 0],
                                    events["alive_after"][flip + 99, 0], events["alive_after"][-1, 0])).astype(bool))


@lru_cache(None)
def _group_scorer(controller, c, flip):
    _config(c, controller)
    unravel = shelter_evolution._unravel(c)

    def one(flat, world_key, birth_key):
        genome = unravel(flat) if controller == "rules-v0" else None
        weights = direct.decode(flat, c) if controller == direct.FORMAT else None
        lives = [seed.rollout(genome, world_key, birth_key, rule, c, plastic=weights is None,
                              predictor_plastic=weights is None, direct_weights=weights,
                              flip_step=flip, record=False, diagnostics=True) for rule in (0, 1)]
        metrics = jax.tree.map(lambda *values: jnp.stack(values), *[_windows_jax(life, flip) for life in lives])
        late = metrics["window_counts"][:, 2, 0] - metrics["window_counts"][:, 2, 1]
        return dict(fitness=jnp.min(late).astype(jnp.float32),
                    counts=lives[0]["counts"] + lives[1]["counts"],
                    survived=metrics["survival"][:, -1].astype(jnp.float32).mean(),
                    rule_late_return=late, **{"rule_" + name: value for name, value in metrics.items()},
                    **{name: jnp.stack([life["events"][name].astype(bool) for life in lives]) for name in EVENTS})

    return jax.jit(jax.vmap(one, in_axes=(0, None, 0)))


@lru_cache(None)
def make_evaluator(controller="rules-v0", c=CFG):
    """Four cached static-flip scorers; only one base-world group is live at once."""
    _config(c, controller)
    size = seed.genome_size(c) if controller == "rules-v0" else direct.genome_size(c)

    def evaluate(candidates, world_keys, development_keys):
        if (candidates.ndim != 2 or candidates.shape[1] != size or candidates.dtype != jnp.float32
                or np.asarray(development_keys).shape != (len(candidates), 4, 2)):
            raise ValueError("candidate encoding or development-key batch differs from the paired protocol")
        assignment = training_assignment(world_keys)
        groups = []
        for index, flip in enumerate(assignment["flip_steps"]):
            result = _group_scorer(controller, c, int(flip))(
                candidates, world_keys[index], development_keys[:, index])
            groups.append(jax.block_until_ready(result))
        return jax.tree.map(lambda *values: jnp.stack(values, axis=1), *groups)

    return evaluate


def pack_evidence(result):
    """Keep compact scalar reductions plus little-bitorder raw binary event evidence."""
    shape = np.asarray(result[EVENTS[0]]).shape
    if len(shape) != 5 or shape[1:3] != (4, 2) or shape[-1] != 1:
        raise ValueError("candidate event evidence requires population/base/rule/time/individual axes")
    evidence = {name: np.asarray(value).copy() for name, value in result.items() if name not in EVENTS}
    evidence["event_shape"] = np.asarray(shape, np.int64)
    for name in EVENTS:
        value = np.asarray(result[name])
        if value.shape != shape or not np.isin(value, (0, 1)).all():
            raise ValueError("candidate events must be matching binary arrays")
        evidence[name + "_packed"] = np.packbits(value.astype(bool).reshape(-1), bitorder="little")
    return evidence


def unpack_events(evidence):
    shape = np.asarray(evidence["event_shape"])
    if shape.shape != (5,) or shape.dtype.kind not in "iu" or np.any(shape < 1) or tuple(shape[1:3]) != (4, 2) or shape[-1] != 1:
        raise ValueError("invalid packed event dimensions")
    count = int(np.prod(shape))
    result = {}
    for name in EVENTS:
        packed = np.asarray(evidence[name + "_packed"])
        if packed.dtype != np.uint8 or packed.shape != ((count + 7) // 8,):
            raise ValueError("packed event bytes differ from the declared dimensions")
        bits = np.unpackbits(packed, bitorder="little")
        if np.any(bits[count:]):
            raise ValueError("packed event padding must be zero")
        result[name] = bits[:count].astype(bool).reshape(tuple(shape))
    return result


def validate_training_evidence(evidence, flip_steps, c=CFG):
    events = unpack_events(evidence)
    f, p, before, after = (events[name] for name in EVENTS)
    if (f.shape[3] != c.T or not before[:, :, :, 0].all() or np.any(f & p)
            or np.any((f | p) & ~before) or np.any(after & ~before)
            or not np.array_equal(before[:, :, :, 1:], after[:, :, :, :-1])
            or sorted(np.asarray(flip_steps).tolist()) != list(FLIP_TIMES)):
        raise ValueError("invalid candidate event timing or assigned flip distribution")
    late = np.stack([(f[:, b, :, step + 50:step + 100].sum((2, 3))
                      - p[:, b, :, step + 50:step + 100].sum((2, 3)))
                     for b, step in enumerate(flip_steps)], axis=1)
    if (not np.array_equal(evidence["rule_late_return"], late)
            or not np.array_equal(evidence["fitness"], late.min(axis=2))
            or not np.array_equal(evidence["survived"], after[:, :, :, -1, 0].mean(2))):
        raise ValueError("candidate objective or survival differs from every assigned raw life")


def prefix_hash(life, stop):
    digest = hashlib.sha256()
    for name in PREFIX_FIELDS:
        value = np.ascontiguousarray(np.asarray(life[name])[:stop])
        digest.update(name.encode() + b"\0")
        digest.update(value.dtype.str.encode() + b"\0")
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    return np.frombuffer(digest.digest(), np.uint8).copy()


def _record_life(result, flip):
    raw = jax.tree.map(np.asarray, result)
    row = {}
    reversal._record_life(raw, "", row, flip)
    for name in ("W", "x", "Wp"):
        row["probe_" + name + "_sha256"] = array_hash(raw["_reversal_probe"][name][0])
    row["preflip_sha256"] = prefix_hash(row, flip)
    row["prelesion_sha256"] = prefix_hash(row, flip + 1)
    return row


def _life(row, arm, rule):
    prefix = arm + "_"
    return {name[len(prefix):]: np.asarray(value)[rule] for name, value in row.items() if name.startswith(prefix)}


def evaluate_world(plastic_flat, direct_flat, keys, index, flip_step, c=CFG):
    _config(c)
    plastic_flat, direct_flat, keys = np.asarray(plastic_flat), np.asarray(direct_flat), np.asarray(keys)
    if (plastic_flat.shape != (seed.genome_size(c),) or direct_flat.shape != (direct.genome_size(c),)
            or plastic_flat.dtype != np.float32 or direct_flat.dtype != np.float32
            or not np.isfinite(plastic_flat).all() or not np.isfinite(direct_flat).all()
            or keys.shape != (2, 2) or keys.dtype != np.uint32 or np.array_equal(*keys)
            or not isinstance(index, Integral) or isinstance(index, bool) or index < 0 or flip_step not in FLIP_TIMES):
        raise ValueError("invalid frozen endpoints, independent base keys or assigned flip")
    genome = shelter_evolution._unravel(c)(plastic_flat)
    weights = direct.decode(jnp.asarray(direct_flat), c)
    grid, position = seed.init_world(keys[0], c)
    row = dict(keys=keys.copy(), base_index=np.asarray(index, np.int32), flip_step=np.asarray(flip_step, np.int32),
               steps=np.asarray(c.T, np.int32), grid_size=np.asarray(c.G, np.int32), radius=np.asarray(c.R, np.int32),
               hidden_units=np.asarray(c.H, np.int32), initial_energy=np.asarray(c.e0, np.float32),
               initial_grid=np.asarray(grid, np.uint8), initial_position=np.asarray(position, np.int32),
               plastic_genome_sha256=array_hash(plastic_flat), direct_genome_sha256=array_hash(direct_flat),
               direct_weights_sha256=array_hash(weights))
    for arm in ARMS:
        is_direct, lesion, reversal_world = arm.startswith("direct"), arm.startswith("lesion"), arm.endswith("reversal")
        lives = []
        for rule in (0, 1):
            result = seed.rollout(None if is_direct else genome, keys[0], keys[1], rule, c,
                plastic=not is_direct, predictor_plastic=not is_direct,
                direct_weights=weights if is_direct else None, flip_step=int(flip_step) if reversal_world else None,
                record=True, diagnostics=True, _world_protocol=PROTOCOL,
                _hebbian_off_step=int(flip_step) + 1 if lesion else None, _reversal_probe_step=int(flip_step) + 1)
            lives.append(_record_life(result, int(flip_step)))
        row.update({arm + "_" + name: np.stack([life[name] for life in lives]) for name in lives[0]})
    validate_world(row, c)
    return row


def validate_world(row, c=CFG):
    _config(c)
    flip = int(row["flip_step"])
    keys = np.asarray(row["keys"])
    if (flip not in FLIP_TIMES or keys.shape != (2, 2) or keys.dtype != np.uint32
            or np.array_equal(*keys) or int(row["base_index"]) < 0):
        raise ValueError("invalid world assignment or keys")
    for name, value in (("steps", c.T), ("grid_size", c.G), ("radius", c.R), ("hidden_units", c.H), ("initial_energy", c.e0)):
        if float(row[name]) != value:
            raise ValueError("world metadata differs from its controller configuration")
    for name in ("plastic_genome_sha256", "direct_genome_sha256", "direct_weights_sha256"):
        if np.asarray(row[name]).shape != (32,) or np.asarray(row[name]).dtype != np.uint8:
            raise ValueError("endpoint fingerprints must be SHA256 byte arrays")
    initial_grid = cold_donor._array(row, "initial_grid", (c.G, c.G), integer=True)
    initial_position = cold_donor._array(row, "initial_position", (1, 2), integer=True)
    if not np.isin(initial_grid, (0, 1, 2)).all() or np.any(initial_position < 0) or np.any(initial_position >= c.G):
        raise ValueError("invalid fresh initial world")
    _, inputs, _, neurons, _ = seed.dims(c)
    for arm in ARMS:
        for rule in (0, 1):
            life = _life(row, arm, rule)
            food, poison, before, after = social._events(life, "", c.T)
            shapes = dict(grid=(c.T, c.G, c.G), position=(c.T, 1, 2), energy=(c.T, 1),
                signal=(c.T, 1), alive=(c.T, 1), surprise=(c.T,), observation=(c.T, 1, inputs),
                action=(c.T, 1), action_key=(c.T, 2), efference=(c.T, 1, 5),
                activity=(c.T, 1, neurons), denergy=(c.T, 1), hebbian_enabled=(c.T,), predictor_enabled=(c.T,))
            for name, shape in shapes.items():
                cold_donor._array(life, name, shape, integer=name in ("grid", "position", "action", "action_key"))
            if (not np.isin(life["grid"], (0, 1, 2)).all() or np.any(life["position"] < 0)
                    or np.any(life["position"] >= c.G) or np.any(life["energy"] < 0)
                    or np.any(life["energy"] > c.e_max) or np.any(np.abs(life["signal"]) > 1)
                    or np.any(life["action"] < 0) or np.any(life["action"] > 4)):
                raise ValueError("trajectory leaves its physical bounds")
            for name, value in (("energy", c.e0), ("alive", 1.), ("denergy", 0.)):
                if not np.array_equal(life["initial_" + name], np.asarray([value], np.float32)):
                    raise ValueError("every assigned life must start in a fresh body")
            previous_grid = np.concatenate((initial_grid[None], life["grid"][:-1]))
            position = life["position"][:, 0]
            encountered = previous_grid[np.arange(c.T), position[:, 0], position[:, 1]]
            rules = np.where(np.arange(c.T) >= flip, 1 - rule, rule) if arm.endswith("reversal") else np.full(c.T, rule)
            if (not np.array_equal(food[:, 0], (encountered == 1 + rules) & before[:, 0])
                    or not np.array_equal(poison[:, 0], (encountered == 2 - rules) & before[:, 0])):
                raise ValueError("recorded encounters do not follow the current hidden rule")
            expected = reversal.window_outcomes(food, poison, before, after, flip)
            if any(not np.array_equal(life[name], value) for name, value in expected.items()):
                raise ValueError("window results differ from actual assigned events")
            if float(life["fitness"]) != float(food.sum() - poison.sum()):
                raise ValueError("whole-life return differs from encounters")
            for name, length in (("preflip_sha256", flip), ("prelesion_sha256", flip + 1)):
                if not np.array_equal(life[name], prefix_hash(life, length)):
                    raise ValueError("prefix fingerprint differs from the recorded trajectory")
            is_direct, lesion = arm.startswith("direct"), arm.startswith("lesion")
            expected_hebb = np.zeros(c.T, bool) if is_direct else np.arange(c.T) < (flip + 1 if lesion else c.T)
            if (not np.array_equal(life["hebbian_enabled"], expected_hebb)
                    or not np.array_equal(life["predictor_enabled"], np.full(c.T, not is_direct))):
                raise ValueError("lesion timing or predictor mode differs from the assigned intervention")
            for stage in ("initial", "probe", "final"):
                for name in ("W", "x", "Wp"):
                    digest = np.asarray(life[stage + "_" + name + "_sha256"])
                    if digest.shape != (32,) or digest.dtype != np.uint8:
                        raise ValueError("invalid actual neural-state fingerprint")
            if lesion and not np.array_equal(life["probe_W_sha256"], life["final_W_sha256"]):
                raise ValueError("lesioned weights changed after the first postflip outcome")
            if is_direct and any(not np.array_equal(life[stage + "_W_sha256"], row["direct_weights_sha256"])
                                 for stage in ("initial", "probe", "final")):
                raise ValueError("direct controller weights changed within its lifetime")
    for rule in (0, 1):
        for controller in ("plastic", "lesion", "direct"):
            stationary, changed = (_life(row, controller + "_" + condition, rule) for condition in ("stationary", "reversal"))
            if not np.array_equal(stationary["preflip_sha256"], changed["preflip_sha256"]):
                raise ValueError("stationary/reversal histories differ before their hidden boundary")
        for condition in ("stationary", "reversal"):
            intact, lesion = (_life(row, arm + "_" + condition, rule) for arm in ("plastic", "lesion"))
            for name in ("prelesion_sha256", "probe_W_sha256", "probe_x_sha256", "probe_Wp_sha256"):
                if not np.array_equal(intact[name], lesion[name]):
                    raise ValueError("lesion changed the neural state or first postflip outcome before its assigned tick")


def summary_input(row):
    c = CFG._replace(G=int(row["grid_size"]), R=int(row["radius"]), H=int(row["hidden_units"]), e0=float(row["initial_energy"]))
    validate_world(row, c)
    names = ("base_index", "flip_step", "keys", "plastic_genome_sha256", "direct_genome_sha256")
    result = {name: np.asarray(row[name]).copy() for name in names}
    for arm in ARMS:
        for field in (*["window_" + name for name in reversal.WINDOW_FIELDS], "boundary_survival", "fitness"):
            result[arm + "_" + field] = np.asarray(row[arm + "_" + field]).copy()
    return result


def _stats(values):
    result = cold_donor._mean_se(values)
    mean, se = result["mean"], result["se"]
    return {**result, "lower_2se": None if se is None else mean - 2 * se,
            "upper_2se": None if se is None else mean + 2 * se}


def _arm_vectors(rows, arm):
    prefix = arm + "_"
    counts = [np.stack([cold_donor._array(row, prefix + "window_" + name, (2, 3), integer=True)
                        for row in rows]) for name in ("food_count", "poison_count")]
    first, poison, missing = [np.stack([cold_donor._array(row, prefix + "window_" + name, (2, 3), binary=True)
                                      for row in rows]) for name in ("first_food", "first_poison", "no_meal")]
    survival = np.stack([cold_donor._array(row, prefix + "boundary_survival", (2, 4), binary=True) for row in rows])
    whole = np.stack([cold_donor._array(row, prefix + "fitness", (2,)) for row in rows])
    if (any(np.any(value < 0) or np.any(value > 50) for value in counts)
            or np.any(counts[0] + counts[1] > 50)
            or not np.all(first.astype(int) + poison.astype(int) + missing.astype(int) == 1)
            or not np.array_equal(missing, counts[0] + counts[1] == 0)
            or np.any(first & (counts[0] == 0)) or np.any(poison & (counts[1] == 0))
            or np.any(np.abs(whole) > 300)):
        raise ValueError("invalid assigned-window outcomes in compact summary")
    return dict(counts=counts, first=first, poison=poison, missing=missing, survival=survival,
                whole=whole, J=(counts[0][:, :, 2] - counts[1][:, :, 2]).min(1),
                first_late=first[:, :, 2].mean(1), joint=survival[:, :, -1].all(1).astype(float))


def summarize(rows):
    rows = [summary_input(row) if "plastic_reversal_food" in row else row for row in rows]
    if not rows:
        raise ValueError("at least one complete paired base world is required")
    seen_indices, seen_keys = set(), set()
    for row in rows:
        index = cold_donor._scalar(row, "base_index", integer=True)
        keys = cold_donor._array(row, "keys", (2, 2), integer=True)
        tokens = [key.tobytes() for key in keys]
        if (index in seen_indices or len(set(tokens)) != 2
                or seen_keys.intersection(tokens) or int(row["flip_step"]) not in FLIP_TIMES):
            raise ValueError("independent base worlds cannot repeat indices or random keys")
        seen_indices.add(index)
        seen_keys.update(tokens)
        for name in ("plastic_genome_sha256", "direct_genome_sha256"):
            if not np.array_equal(row[name], rows[0][name]):
                raise ValueError("evaluation endpoint changed across worlds")
    vectors, arms = {}, {}
    for arm in ARMS:
        value = vectors[arm] = _arm_vectors(rows, arm)
        windows = {}
        for index, name in enumerate(reversal.WINDOWS):
            food = value["counts"][0][:, :, index].mean(1)
            poison = value["counts"][1][:, :, index].mean(1)
            windows[name] = dict(first_food=_stats(value["first"][:, :, index].mean(1)),
                first_poison=_stats(value["poison"][:, :, index].mean(1)),
                no_meal=_stats(value["missing"][:, :, index].mean(1)), food_count=_stats(food),
                poison_count=_stats(poison), fitness=_stats(food - poison),
                contact_accuracy=cold_donor._ratio_se(food, food + poison))
        arms[arm] = dict(late_min_return=_stats(value["J"]), late_first_food=_stats(value["first_late"]),
            joint_survival=_stats(value["joint"]), mean_survival=_stats(value["survival"][:, :, -1].mean(1)),
            per_rule_survival={str(rule): _stats(value["survival"][:, rule, -1]) for rule in (0, 1)},
            survival_points={name: _stats(value["survival"][:, :, i].mean(1)) for i, name in enumerate(reversal.SURVIVAL_POINTS)},
            whole_return=_stats(value["whole"].mean(1)), windows=windows)
    p, d, lesion = (vectors[name] for name in ("plastic_reversal", "direct_reversal", "lesion_reversal"))
    pj, dj, first = (arms["plastic_reversal"]["late_min_return"], arms["direct_reversal"]["late_min_return"],
                     arms["plastic_reversal"]["late_first_food"])
    defined = len(rows) > 1
    combined = float(np.hypot(pj["se"], dj["se"])) if defined else None
    difference = pj["mean"] - dj["mean"]
    comparison = dict(difference=difference, combined_se=combined, paired_difference=_stats(p["J"] - d["J"]),
                      passed=difference > 2 * combined if defined else None)
    lesion_difference = _stats(p["J"] - lesion["J"])
    mechanism = dict(difference=lesion_difference, passed=lesion_difference["lower_2se"] > 0 if defined else None)
    survival_difference = _stats(p["joint"] - d["joint"])
    survival_absolute = p["joint"].mean() >= .5 if defined else None
    survival_relative = survival_difference["lower_2se"] > -.10 if defined else None
    survival = dict(difference=survival_difference, absolute_passed=bool(survival_absolute) if defined else None,
                    relative_passed=survival_relative, passed=bool(survival_absolute and survival_relative) if defined else None)
    positive = pj["lower_2se"] > 0 if defined else None
    accuracy = first["mean"] >= .60 and first["mean"] - .50 > 2 * first["se"] if defined else None
    performance = comparison["passed"] and positive and accuracy and mechanism["passed"] if defined else None
    return dict(protocol=PROTOCOL, base_worlds=len(rows), rule_replays_per_base=2, lives_per_arm=2 * len(rows),
                world_keys_sha256=array_hash(np.stack([row["keys"] for row in rows])).tobytes().hex(),
                plastic_genome_sha256=np.asarray(rows[0]["plastic_genome_sha256"], np.uint8).tobytes().hex(),
                direct_genome_sha256=np.asarray(rows[0]["direct_genome_sha256"], np.uint8).tobytes().hex(),
                flip_counts={str(flip): sum(int(row["flip_step"]) == flip for row in rows) for flip in FLIP_TIMES},
                arms=arms, plastic_vs_direct=comparison, plastic_vs_lesion=mechanism, survival=survival,
                positive_return_passed=positive, late_accuracy_passed=accuracy,
                stationary_lesion_difference=_stats(vectors["plastic_stationary"]["J"] - vectors["lesion_stationary"]["J"]),
                gate=dict(performance_passed=performance, survival_passed=survival["passed"],
                          passed=performance and survival["passed"] if defined else None))


def aggregate(summaries):
    summaries = list(summaries)
    if len(summaries) != 3 or any(row.get("protocol") != PROTOCOL or row.get("base_worlds") != 128
                                 or row.get("flip_counts") != {str(flip): 32 for flip in FLIP_TIMES}
                                 for row in summaries):
        raise ValueError("the formal rung requires three complete 128-base paired panels")
    panels = [row.get("world_keys_sha256", "") for row in summaries]
    if len(set(panels)) != 3 or any(len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value) for value in panels):
        raise ValueError("the formal rung requires three distinct recorded world panels")
    performance = [row["gate"]["performance_passed"] for row in summaries]
    survival = [row["gate"]["survival_passed"] for row in summaries]
    if any(type(value) is not bool for value in performance + survival):
        raise ValueError("every final panel must have a defined gate")
    return dict(passed=sum(performance) >= 2 and all(survival), passing_seeds=sum(performance),
                required_passing_seeds=2, performance_passed=performance, survival_passed=survival,
                all_seed_survival_passed=all(survival))
