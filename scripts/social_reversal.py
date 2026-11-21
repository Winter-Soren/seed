"""E125 saved-controller measurement in paired stationary and randomized-reversal worlds."""
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from scripts import cold_donor, shelter_evolution, social_belief as belief, social_learning as social
from seed import seed


PROTOCOL = "reversal-random-v1"
CFG = seed.Cfg(M=1)
ARMS = ("observed", "belief_reset", "birth_parent", "direct_baseline")
CONDITIONS = ("stationary", "reversal")
WINDOWS = ("pre", "early", "late")
SURVIVAL_POINTS = ("at_flip", "after_early", "after_late", "end")
FLIP_TIMES = (100, 125, 150, 175)
ASSIGNMENT_TAG = 0xE125
KEY_ORDER = social.KEY_ORDER
family_keys = social.family_keys
TRACE_FIELDS = ("grid", "position", "energy", "signal", "alive", "surprise", *social.EVENTS,
                "observation", "action", "action_key", "efference", "activity", "denergy")
WINDOW_FIELDS = ("first_food", "first_poison", "no_meal", "food_count", "poison_count", "fitness", "first_meal_step")


def assignment_key(master):
    shelter_evolution.evaluation_keys(master, 0)
    return np.asarray(jax.random.fold_in(jax.random.PRNGKey(int(master)), ASSIGNMENT_TAG))


def assignments_manifest(master, count):
    if not isinstance(count, Integral) or isinstance(count, bool) or count < 4 or (count != 4 and count % 8):
        raise ValueError("assignment count must be four development families or a multiple of eight")
    if count == 4:
        base = np.array([(index % 2, step) for index, step in enumerate(FLIP_TIMES)], np.int32)
    else:
        base = np.repeat(np.array([(rule, step) for rule in (0, 1) for step in FLIP_TIMES], np.int32), count // 8, axis=0)
    key = assignment_key(master)
    permutation = np.asarray(jax.random.permutation(key, int(count)), np.int32)
    return dict(key=key, permutation=permutation, table=base[permutation])


def schedule(master, count):
    return assignments_manifest(master, count)["table"]


def window_outcomes(food, poison, alive_before, alive_after, flip_step):
    food, poison = np.asarray(food, bool), np.asarray(poison, bool)
    before, after = np.asarray(alive_before, bool), np.asarray(alive_after, bool)
    if (food.ndim != 2 or food.shape[1] != 1 or any(value.shape != food.shape for value in (poison, before, after))
            or flip_step < 50 or flip_step + 100 > len(food) or np.any(food & poison)
            or np.any((food | poison) & ~before)):
        raise ValueError("invalid event stream or registered analysis window")
    windows = {name: [] for name in WINDOW_FIELDS}
    for start, stop in ((flip_step - 50, flip_step), (flip_step, flip_step + 50), (flip_step + 50, flip_step + 100)):
        first = social.first_meal(food[start:stop], poison[start:stop])
        for name in ("first_food", "first_poison", "no_meal", "first_meal_step"):
            windows[name].append(first[name])
        food_count, poison_count = int(food[start:stop].sum()), int(poison[start:stop].sum())
        windows["food_count"].append(food_count)
        windows["poison_count"].append(poison_count)
        windows["fitness"].append(food_count - poison_count)
    return {**{"window_" + name: np.asarray(value, bool if name in ("first_food", "first_poison", "no_meal") else np.int32)
               for name, value in windows.items()},
            "boundary_survival": np.asarray([before[flip_step, 0], after[flip_step + 49, 0],
                                               after[flip_step + 99, 0], after[-1, 0]], bool)}


def _same_bytes(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def _record_preparation(row, arm, observer, frames, q):
    row[arm + "_frames"] = frames
    row[arm + "_q"] = np.asarray(q, np.float32)
    row[arm + "_watch_motor"] = np.asarray(observer["trace"]["motor"])
    row[arm + "_watch_denergy"] = np.asarray(observer["trace"]["denergy"])
    for name in ("W", "x", "Wp"):
        for stage, source in (("initial", "initial"), ("final", "final")):
            row[f"{arm}_watch_{stage}_{name}_sha256"] = social._hash(observer[source][name])


def _record_life(result, prefix, row, flip_step):
    result = jax.tree.map(np.asarray, result)
    for name, source in (("grid", "grid"), ("position", "pos"), ("energy", "energy"),
                         ("signal", "sig"), ("alive", "alive"), ("surprise", "surprise")):
        row[prefix + name] = result["traj"][source].astype(np.uint8) if name == "grid" else result["traj"][source]
    for name in social.EVENTS:
        row[prefix + name] = result["events"][name].astype(bool)
    for name, value in result["_world"].items():
        row[prefix + name] = value
    for stage in ("initial", "final"):
        for name in ("W", "x", "Wp"):
            row[prefix + stage + "_" + name + "_sha256"] = social._hash(result["_world_" + stage][name][0])
    for name in ("energy", "alive", "denergy"):
        row[prefix + "initial_" + name] = result["_world_initial"][name]
    if "_belief" in result:
        row.update({prefix + "belief_" + name: value for name, value in result["_belief"].items()})
    row[prefix + "fitness"] = np.asarray(result["fitness"], np.float32)
    row.update({prefix + name: value for name, value in window_outcomes(
        row[prefix + "food"], row[prefix + "poison"], row[prefix + "alive_before"],
        row[prefix + "alive_after"], flip_step).items()})


def evaluate_family(flat, direct_weights, lambda_value, keys, index, rule, flip_step, c=CFG):
    flat, direct_weights, keys = np.asarray(flat), np.asarray(direct_weights), np.asarray(keys)
    _, inputs, _, neurons, _ = seed.dims(c)
    if (c.M != 1 or c.T != 300 or c.R < 1 or flat.shape != (seed.genome_size(c),) or flat.dtype != np.float32
            or direct_weights.shape != (neurons, neurons) or direct_weights.dtype != np.float32
            or not np.isfinite(flat).all() or not np.isfinite(direct_weights).all()
            or keys.shape != (4, 2) or keys.dtype != np.uint32 or len({key.tobytes() for key in keys}) != 4
            or not isinstance(index, Integral) or isinstance(index, bool) or index < 0
            or rule not in (0, 1) or flip_step not in FLIP_TIMES or lambda_value not in belief.LAMBDA_GRID):
        raise ValueError("invalid saved controller, family assignment or fixed reversal protocol")
    row = dict(original_genome=flat.copy(), direct_weights=direct_weights.copy(), keys=keys.copy(),
               family_index=np.asarray(index, np.int32), rule=np.asarray(rule, np.int32),
               flip_step=np.asarray(flip_step, np.int32), lambda_value=np.asarray(lambda_value, np.float64),
               steps=np.asarray(c.T, np.int32), agents=np.asarray(1, np.int32),
               watch_start=np.asarray(c.T // 2, np.int32),
               grid_size=np.asarray(c.G, np.int32), radius=np.asarray(c.R, np.int32),
               hidden_units=np.asarray(c.H, np.int32), initial_energy=np.asarray(c.e0, np.float32))
    row.update(social._record_teacher(flat, keys, int(rule), c))
    movie, row["camera_positions"] = social.make_movie(row, c)
    row["observed_frames"] = movie
    row.update(belief.decode_movie(movie, lambda_value, c))
    observer = social.watch(flat, keys[3], movie, c)
    birth = belief._birth_observer(flat, keys[3], c)
    empty = np.zeros((0, inputs), np.float32)
    preparations = dict(observed=observer, belief_reset=observer, birth_parent=birth)
    for arm, prepared in preparations.items():
        _record_preparation(row, arm, prepared, empty if arm == "birth_parent" else movie,
                            float(row["q"]) if arm == "observed" else .5)
    grid, position = seed.init_world(keys[2], c)
    row.update(solo_initial_grid=np.asarray(grid, np.uint8), solo_initial_position=np.asarray(position, np.int32))
    genome = shelter_evolution._unravel(c)(flat)
    for arm in ARMS:
        for condition in CONDITIONS:
            options = dict(record=True, diagnostics=True, _world_protocol=PROTOCOL,
                           flip_step=int(flip_step) if condition == "reversal" else None)
            if arm == "direct_baseline":
                options.update(direct_weights=jnp.asarray(direct_weights), plastic=False, predictor_plastic=False)
            elif arm in ("observed", "belief_reset"):
                final = preparations[arm]["final"]
                options.update(_initial_weights=final["W"][None],
                               _initial_brain=dict(x=final["x"][None], Wp=final["Wp"][None]),
                               _belief_q=jnp.asarray([row[arm + "_q"]]))
            result = seed.rollout(genome, keys[2], keys[3], int(rule), c, **options)
            _record_life(result, arm + "_" + condition + "_", row, int(flip_step))
            del result
        row[arm + "_preflip_identical"] = np.asarray(_preflip_identity(row, arm), bool)
    validate_family(row, c)
    return row


def _preflip_identity(row, arm):
    boundary = int(row["flip_step"])
    left, right = arm + "_stationary_", arm + "_reversal_"
    for name in TRACE_FIELDS:
        if not _same_bytes(row[left + name][:boundary], row[right + name][:boundary]):
            return False
    if arm in ("observed", "belief_reset"):
        for name in ("q", "action", "bias", "unbiased_logits", "biased_logits", "efference", "denergy"):
            if not _same_bytes(row[left + "belief_" + name][:boundary], row[right + "belief_" + name][:boundary]):
                return False
    # The boundary's observation and action precede its changed reward.
    for name in ("observation", "action", "action_key", "efference", "activity", "alive_before"):
        if not _same_bytes(row[left + name][boundary], row[right + name][boundary]):
            return False
    return all(_same_bytes(row[left + "initial_" + name + "_sha256"], row[right + "initial_" + name + "_sha256"])
               for name in ("W", "x", "Wp"))


def _config_from_row(row):
    return CFG._replace(G=int(row["grid_size"]), R=int(row["radius"]), H=int(row["hidden_units"]),
                        T=int(row["steps"]), e0=float(row["initial_energy"]))


def validate_family(row, c=CFG):
    _, inputs, _, neurons, _ = seed.dims(c)
    if (int(row["steps"]) != 300 or c.T != 300 or int(row["agents"]) != 1 or c.M != 1
            or int(row["rule"]) not in (0, 1) or int(row["flip_step"]) not in FLIP_TIMES
            or float(row["lambda_value"]) not in belief.LAMBDA_GRID):
        raise ValueError("raw family differs from the registered reversal protocol")
    for name, expected in (("grid_size", c.G), ("radius", c.R), ("hidden_units", c.H), ("initial_energy", c.e0)):
        if float(row[name]) != expected:
            raise ValueError("raw configuration differs from its engine")
    cold_donor._scalar(row, "family_index", integer=True)
    keys = cold_donor._array(row, "keys", (4, 2), integer=True)
    if keys.dtype != np.uint32 or len({key.tobytes() for key in keys}) != 4:
        raise ValueError("family roles must have distinct recorded random keys")
    original = cold_donor._array(row, "original_genome", (seed.genome_size(c),))
    direct = cold_donor._array(row, "direct_weights", (neurons, neurons))
    if original.dtype != np.float32 or direct.dtype != np.float32:
        raise ValueError("saved controllers must retain their original float32 representation")
    social._events(row, "teacher_", c.T)
    movie, camera = social.make_movie(row, c)
    if not _same_bytes(movie, row["observed_frames"]) or not _same_bytes(camera, row["camera_positions"]):
        raise ValueError("teacher preparation differs from the actual natural movie")
    for name, expected in belief.decode_movie(movie, float(row["lambda_value"]), c).items():
        if not _same_bytes(expected, row[name]):
            raise ValueError("prepared belief differs from public movie evidence")
    for arm in ARMS[:-1]:
        length = 0 if arm == "birth_parent" else c.T // 2
        if not _same_bytes(row[arm + "_frames"], movie if length else np.zeros((0, inputs), np.float32)):
            raise ValueError("wrong preparation information or update budget")
        motors = cold_donor._array(row, arm + "_watch_motor", (length, 5))
        expected = np.zeros_like(motors)
        expected[:, 0] = 1
        if not _same_bytes(motors, expected) or np.any(row[arm + "_watch_denergy"]):
            raise ValueError("observation must remain passive with own stay feedback and no reward")
        for name in ("W", "x", "Wp"):
            for stage in ("initial", "final"):
                key = f"{arm}_watch_{stage}_{name}_sha256"
                if np.asarray(row[key]).dtype != np.uint8 or np.asarray(row[key]).shape != (32,):
                    raise ValueError("invalid recorded preparation fingerprint")
            if not _same_bytes(row[f"{arm}_watch_initial_{name}_sha256"], row[f"observed_watch_initial_{name}_sha256"]):
                raise ValueError("plastic arms must share the same newborn")
            if arm == "belief_reset" and not _same_bytes(row[f"{arm}_watch_final_{name}_sha256"], row[f"observed_watch_final_{name}_sha256"]):
                raise ValueError("belief reset must preserve acquired neural state")
            if arm == "birth_parent" and not _same_bytes(row[f"{arm}_watch_initial_{name}_sha256"], row[f"{arm}_watch_final_{name}_sha256"]):
                raise ValueError("unexposed birth cannot acquire an observer state")
        expected_q = float(row["q"]) if arm == "observed" else .5
        if float(row[arm + "_q"]) != expected_q:
            raise ValueError("belief intervention differs from the registered preparation")
    initial_grid = cold_donor._array(row, "solo_initial_grid", (c.G, c.G), integer=True)
    initial_pos = cold_donor._array(row, "solo_initial_position", (1, 2), integer=True)
    if not np.isin(initial_grid, (0, 1, 2)).all() or np.any(initial_pos < 0) or np.any(initial_pos >= c.G):
        raise ValueError("invalid shared solo world")
    for arm in ARMS:
        for condition in CONDITIONS:
            prefix = arm + "_" + condition + "_"
            food, poison, alive, after = social._events(row, prefix, c.T)
            shapes = dict(grid=(c.T, c.G, c.G), position=(c.T, 1, 2), energy=(c.T, 1),
                          signal=(c.T, 1), alive=(c.T, 1), surprise=(c.T,), observation=(c.T, 1, inputs),
                          action=(c.T, 1), action_key=(c.T, 2), efference=(c.T, 1, 5),
                          activity=(c.T, 1, neurons), denergy=(c.T, 1))
            for name, shape in shapes.items():
                cold_donor._array(row, prefix + name, shape, integer=name in ("grid", "position", "action", "action_key"))
            grid, pos, energy = (np.asarray(row[prefix + name]) for name in ("grid", "position", "energy"))
            if (not np.isin(grid, (0, 1, 2)).all() or np.any(pos < 0) or np.any(pos >= c.G)
                    or np.any(energy < 0) or np.any(energy > c.e_max) or not np.array_equal(after, row[prefix + "alive"])):
                raise ValueError("solo trajectory violates body or world bounds")
            previous_grid = np.concatenate((initial_grid[None], grid[:-1]))
            previous_pos = np.concatenate((initial_pos[None], pos[:-1]))
            items = previous_grid[np.arange(c.T), pos[:, 0, 0], pos[:, 0, 1]]
            rules = np.full(c.T, int(row["rule"]))
            if condition == "reversal":
                rules[int(row["flip_step"]):] = 1 - rules[int(row["flip_step"]):]
            if (not np.array_equal(food[:, 0], (items == rules + 1) & alive[:, 0])
                    or not np.array_equal(poison[:, 0], (items == 2 - rules) & alive[:, 0])):
                raise ValueError("encounter labels differ from the hidden world boundary")
            actions = np.asarray(row[prefix + "action"])[:, 0]
            if np.any(actions < 0) or np.any(actions > 4):
                raise ValueError("invalid sampled action")
            expected_pos = (previous_pos[:, 0] + np.asarray(seed.MOVES)[actions] * alive) % c.G
            if (not np.array_equal(expected_pos, pos[:, 0])
                    or not np.array_equal(row[prefix + "efference"], np.eye(5, dtype=np.float32)[actions, None])
                    or not np.array_equal(row[prefix + "activity"][:, :, -6:-1], row[prefix + "efference"])):
                raise ValueError("actual movement or neural feedback differs from sampled action")
            previous_energy = np.concatenate((np.full((1, 1), c.e0, np.float32), energy[:-1]))
            if not np.array_equal(row[prefix + "denergy"], (energy - previous_energy) * alive):
                raise ValueError("solo reward is not the body's actual energy change")
            for name, expected in window_outcomes(food, poison, alive, after, int(row["flip_step"])).items():
                if not _same_bytes(expected, row[prefix + name]):
                    raise ValueError("window endpoint differs from all-assigned encounter evidence")
            if float(row[prefix + "fitness"]) != float(food.sum() - poison.sum()):
                raise ValueError("lifetime fitness differs from the recorded encounters")
            for name, expected in (("energy", c.e0), ("alive", 1.), ("denergy", 0.)):
                if not np.array_equal(row[prefix + "initial_" + name], np.asarray([expected], np.float32)):
                    raise ValueError("each solo condition must start with a fresh body")
            for stage in ("initial", "final"):
                for name in ("W", "x", "Wp"):
                    value = np.asarray(row[prefix + stage + "_" + name + "_sha256"])
                    if value.dtype != np.uint8 or value.shape != (32,):
                        raise ValueError("invalid actual solo neural-state fingerprint")
            if arm == "direct_baseline":
                expected_W = social._hash(direct)
                if (not _same_bytes(row[prefix + "initial_W_sha256"], expected_W)
                        or not _same_bytes(row[prefix + "final_W_sha256"], expected_W)
                        or not _same_bytes(row[prefix + "initial_Wp_sha256"], row[prefix + "final_Wp_sha256"])):
                    raise ValueError("direct baseline weights or predictor changed")
            else:
                for name in ("W", "x", "Wp"):
                    if not _same_bytes(row[prefix + "initial_" + name + "_sha256"], row[f"{arm}_watch_final_{name}_sha256"]):
                        raise ValueError("solo initialization differs from the assigned neural preparation")
            if arm in ("observed", "belief_reset"):
                # This shared validator sees the same local items and fixed q; no boundary enters the readout.
                proxy = {**row, prefix + "q": row[arm + "_q"]}
                belief._validate_readout(proxy, prefix, grid, pos, energy, c)
        identical = _preflip_identity(row, arm)
        if not identical or not bool(row[arm + "_preflip_identical"]):
            raise ValueError("stationary and reversal differ before the hidden outcome change")


def summary_input(row):
    """Drop large traces only after validating them; keep every assigned family's paired endpoints."""
    validate_family(row, _config_from_row(row))
    names = ("keys", "family_index", "rule", "flip_step", "lambda_value")
    compact = {name: np.asarray(row[name]).copy() for name in names}
    for arm in ARMS:
        compact[arm + "_preflip_identical"] = np.asarray(row[arm + "_preflip_identical"]).copy()
        for condition in CONDITIONS:
            prefix = arm + "_" + condition + "_"
            for name in (*("window_" + value for value in WINDOW_FIELDS), "boundary_survival", "fitness"):
                compact[prefix + name] = np.asarray(row[prefix + name]).copy()
    return compact


def _stats(values):
    values = list(values)
    return cold_donor._mean_se(values) if values else dict(mean=None, se=None)


def _condition_summary(rows, prefix):
    windows = {}
    for index, name in enumerate(WINDOWS):
        food = np.asarray([int(row[prefix + "window_food_count"][index]) for row in rows])
        poison = np.asarray([int(row[prefix + "window_poison_count"][index]) for row in rows])
        windows[name] = {field: _stats([row[prefix + "window_" + field][index] for row in rows])
                         for field in WINDOW_FIELDS}
        windows[name].update(accuracy=cold_donor._ratio_se(food, food + poison) if rows else dict(mean=None, se=None),
                              food_encounters=int(food.sum()), poison_encounters=int(poison.sum()),
                              encounters=int((food + poison).sum()), assigned_families=len(rows))
    return dict(families=len(rows), windows=windows, fitness=_stats([row[prefix + "fitness"] for row in rows]),
                survival={name: _stats([row[prefix + "boundary_survival"][index] for row in rows])
                          for index, name in enumerate(SURVIVAL_POINTS)})


def transfer_gate(late, recovery, accuracy_difference, survival_difference):
    defined = all(value["se"] is not None for value in (late, recovery, accuracy_difference, survival_difference))
    checks = dict(late_level_passed=bool(late["mean"] >= .60),
                  late_chance_passed=bool(late["mean"] - .50 > 2 * late["se"]) if defined else None,
                  recovery_passed=bool(recovery["mean"] > 2 * recovery["se"]) if defined else None,
                  accuracy_noninferiority_passed=bool(accuracy_difference["mean"] - 2 * accuracy_difference["se"] > -.10) if defined else None,
                  survival_noninferiority_passed=bool(survival_difference["mean"] - 2 * survival_difference["se"] > -.10) if defined else None)
    return dict(passed=all(checks.values()) if defined else None, **checks)


def _validate_compact(rows):
    if not rows:
        raise ValueError("a reversal panel requires assigned families")
    seen, indices = set(), set()
    selected_lambda = float(rows[0]["lambda_value"])
    for row in rows:
        index = cold_donor._scalar(row, "family_index", integer=True)
        if (index in indices or int(row["rule"]) not in (0, 1) or int(row["flip_step"]) not in FLIP_TIMES
                or float(row["lambda_value"]) != selected_lambda or selected_lambda not in belief.LAMBDA_GRID):
            raise ValueError("inconsistent or duplicated family assignment")
        indices.add(index)
        keys = cold_donor._array(row, "keys", (4, 2), integer=True)
        if keys.dtype != np.uint32:
            raise ValueError("compact family keys must retain uint32 representation")
        for key in keys:
            if key.tobytes() in seen:
                raise ValueError("family random keys were reused")
            seen.add(key.tobytes())
        for arm in ARMS:
            if not bool(row[arm + "_preflip_identical"]):
                raise ValueError("unverified pre-flip identity cannot enter the summary")
            for condition in CONDITIONS:
                prefix = arm + "_" + condition + "_"
                for name in WINDOW_FIELDS:
                    cold_donor._array(row, prefix + "window_" + name, (3,),
                                       binary=name in ("first_food", "first_poison", "no_meal"),
                                       integer=name in ("food_count", "poison_count", "fitness", "first_meal_step"))
                cold_donor._array(row, prefix + "boundary_survival", (4,), binary=True)
                cold_donor._scalar(row, prefix + "fitness")
                food, poison = row[prefix + "window_food_count"], row[prefix + "window_poison_count"]
                if (np.any(food < 0) or np.any(poison < 0) or np.any(food + poison > 50)
                        or not np.array_equal(row[prefix + "window_fitness"], food - poison)
                        or not np.all(row[prefix + "window_first_food"].astype(int)
                                      + row[prefix + "window_first_poison"].astype(int)
                                      + row[prefix + "window_no_meal"].astype(int) == 1)
                        or not np.array_equal(row[prefix + "window_no_meal"], food + poison == 0)):
                    raise ValueError("compact windows have inconsistent contact accounting")
                latency = row[prefix + "window_first_meal_step"]
                survival = np.asarray(row[prefix + "boundary_survival"], int)
                if (np.any(latency < 1) or np.any(latency > 51)
                        or not np.array_equal(latency == 51, row[prefix + "window_no_meal"])
                        or np.any(np.diff(survival) > 0)):
                    raise ValueError("compact missed-contact or survival timing is inconsistent")
            left, right = arm + "_stationary_", arm + "_reversal_"
            if (any(row[left + "window_" + field][0] != row[right + "window_" + field][0] for field in WINDOW_FIELDS)
                    or row[left + "boundary_survival"][0] != row[right + "boundary_survival"][0]):
                raise ValueError("paired histories differ before the analysis boundary")


def summarize(rows):
    rows = list(rows)
    if rows and "teacher_grid" in rows[0]:
        rows = [summary_input(row) for row in rows]
    _validate_compact(rows)
    arms = {}
    for arm in ARMS:
        conditions = {}
        for condition in CONDITIONS:
            prefix = arm + "_" + condition + "_"
            conditions[condition] = _condition_summary(rows, prefix)
            conditions[condition]["end_survivor_only"] = _condition_summary(
                [row for row in rows if row[prefix + "boundary_survival"][-1]], prefix)
            conditions[condition]["alive_at_flip_only"] = _condition_summary(
                [row for row in rows if row[prefix + "boundary_survival"][0]], prefix)
        stat, rev = arm + "_stationary_", arm + "_reversal_"
        differences = {name + "_first_food_difference": _stats([
            float(row[rev + "window_first_food"][index]) - float(row[stat + "window_first_food"][index]) for row in rows])
                       for index, name in enumerate(WINDOWS)}
        recovery = _stats([float(row[rev + "window_first_food"][2]) - float(row[rev + "window_first_food"][1])
                           - float(row[stat + "window_first_food"][2]) + float(row[stat + "window_first_food"][1])
                           for row in rows])
        survival = _stats([float(row[rev + "boundary_survival"][-1]) - float(row[stat + "boundary_survival"][-1]) for row in rows])
        differences.update(recovery_difference=recovery, end_survival_difference=survival)
        gate = transfer_gate(conditions["reversal"]["windows"]["late"]["first_food"], recovery,
                              differences["late_first_food_difference"], survival)
        arms[arm] = dict(conditions=conditions, reversal_minus_stationary=differences, gate=gate)
    comparisons = {}
    for name, left, right in (("within_social", "observed", "belief_reset"),
                               ("within_birth", "birth_parent", "direct_baseline")):
        comparisons[name] = {}
        for condition in CONDITIONS:
            lp, rp = left + "_" + condition + "_", right + "_" + condition + "_"
            windows = {name: {field + "_difference": _stats([
                float(row[lp + "window_" + field][index]) - float(row[rp + "window_" + field][index]) for row in rows])
                             for field in ("first_food", "fitness")}
                       for index, name in enumerate(WINDOWS)}
            comparisons[name][condition] = dict(windows=windows, end_survival_difference=_stats([
                float(row[lp + "boundary_survival"][-1]) - float(row[rp + "boundary_survival"][-1]) for row in rows]))
    return dict(families=len(rows), primary="observed", selected_lambda=float(rows[0]["lambda_value"]), arms=arms,
                gate=arms["observed"]["gate"], **comparisons,
                assignment_counts={f"{rule}:{step}": sum(int(row["rule"]) == rule and int(row["flip_step"]) == step for row in rows)
                                   for rule in (0, 1) for step in FLIP_TIMES},
                neutral_selected_birth_duplicates="birth_parent",
                preflip_identity=dict(all_recorded_fields_identical=True,
                    limitation="Per-frame W and Wp were not recorded; their intermediate identity is not independently replayed."))
